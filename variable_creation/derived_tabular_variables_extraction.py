import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import os

def preprocess_water_bodies(water_path):
    """
    Loads and preprocesses waterbody vector data, combining line and surface hydrography into a single layer.
    
    Args:
        water_path (str): Path to the GeoPackage containing hydrography layers.

    Returns:
        GeoDataFrame: Merged waterbody polygons (rivers buffered by width, plus surfaces).
    """
    # Load the linear hydrography (e.g. rivers and streams)
    lineaire = gpd.read_file(water_path, layer='reseauhydro_lineaire')
    # Load the surface hydrography (e.g. lakes, ponds)
    surface = gpd.read_file(water_path, layer='Surface en eau')
    
    # Buffer the linear features by half their width (column 'Largeur/2')
    lineaire['geometry'] = lineaire.apply(
        lambda row: row['geometry'].buffer(row['Largeur/2']) if not np.isnan(row['Largeur/2']) else row['geometry'],
        axis=1
    )
    
    # Merge line and surface features into one GeoDataFrame
    water = gpd.GeoDataFrame(pd.concat([lineaire, surface]).reset_index(drop=True))
    return water

def convert_raster_mask_to_polygon(raster_path):
    """
    Converts a binary raster mask (e.g., woodland mask) into vector polygons.

    Args:
        raster_path (str): Path to the raster mask file.

    Returns:
        GeoDataFrame: Polygons representing the "on" (value=1) regions in the mask.
    """
    with rasterio.open(raster_path) as src:
        image = src.read(1)  # Read the first band (assuming single-band mask)
        mask = image == 1    # Mask of areas with value 1 (e.g., woodland)
        transform = src.transform  # Raster-to-world transform

    # Generate polygons from contiguous "1" pixel regions
    shapes_generator = shapes(image, mask=mask, transform=transform)
    polygons = [shape(geom) for geom, value in shapes_generator if value == 1]
    
    # Convert to a GeoDataFrame in the raster's CRS
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=src.crs)
    return gdf

def distance_to_closest_polygon(gdf, points, col_name):
    """
    Computes the minimum distance from each point to any polygon in a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): Polygons to measure distance to.
        points (GeoDataFrame): Points for which distances are calculated.
        col_name (str): Name of the output column for distances.

    Returns:
        GeoDataFrame: Points with an added column for minimum distance to polygons.
    """
    # For each point, calculate the minimum distance to any polygon
    points[col_name] = points.geometry.apply(lambda point: gdf.distance(point).min())
    return points

def wood_density(point_geom, wood_geom, buffer_size):
    """
    Computes the proportion of forested area (wood density) within a buffer around a point.

    Args:
        point_geom (shapely.geometry.Point): The center of the buffer.
        wood_geom (GeoDataFrame): Polygons representing forested areas.
        buffer_size (int): Buffer radius (meters).

    Returns:
        float: Fraction of the buffer area covered by forest.
    """
    # Create a circular buffer around the point
    buffer = point_geom.buffer(buffer_size)
    # Calculate intersection with forest polygons
    intersection = wood_geom.intersection(buffer)
    # Sum the area of all forested parts within the buffer
    wood_surface = intersection.area.sum()
    # Area of the buffer itself
    buffer_surface = buffer.area
    # Return the density as a ratio (handle case of zero area)
    return wood_surface / buffer_surface if buffer_surface > 0 else 0

def stats_raster(point_geom, raster_path, buffer_size, nodata_value=-32767.0):
    """
    Computes mean and std of raster values within a buffer around a point, ignoring nodata.

    Args:
        point_geom (shapely.geometry.Point): Center of the buffer.
        raster_path (str): Path to the raster file.
        buffer_size (int): Buffer radius in meters.
        nodata_value (float): Value representing no data in the raster.

    Returns:
        tuple: (mean, std) for the region in the buffer, or (None, None) if no overlap.
    """
    # Create a circular buffer around the point
    buffer = point_geom.buffer(buffer_size)
    with rasterio.open(raster_path) as src:
        # Build raster extent as a polygon
        raster_bounds = shape({
            'type': 'Polygon',
            'coordinates': [[
                [src.bounds.left, src.bounds.top],
                [src.bounds.right, src.bounds.top],
                [src.bounds.right, src.bounds.bottom],
                [src.bounds.left, src.bounds.bottom],
                [src.bounds.left, src.bounds.top]
            ]]
        })
        # Clip buffer to raster extent (avoid sampling outside image)
        clipped_buffer = buffer.intersection(raster_bounds)
        if clipped_buffer.is_empty:
            return None, None

        # Use rasterstats to calculate zonal stats for the buffer area
        from rasterstats import zonal_stats
        stats = zonal_stats(
            [mapping(clipped_buffer)], raster_path, stats="mean std", nodata=nodata_value, all_touched=False
        )[0]
        return stats["mean"], stats["std"]

def get_pixel_values(raster_path, points):
    """
    Sample raster pixel values at the location of each point.

    Args:
        raster_path (str): Path to the raster file.
        points (GeoDataFrame): Points to sample.

    Returns:
        list: List of pixel values at each point location.
    """
    with rasterio.open(raster_path) as src:
        # For each point, sample its (x, y) coordinates in the raster
        values = list(src.sample([(p.x, p.y) for p in points.geometry]))
    # Return the first band value for each sample
    return [v[0] for v in values]