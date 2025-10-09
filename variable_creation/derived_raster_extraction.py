import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask
from scipy.ndimage import distance_transform_edt, binary_opening, binary_closing
from rasterio.transform import from_origin
from pyproj import Transformer
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
import osmnx as ox
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from utils import vector_to_raster

class WaterDistanceMap:
    """
    Class to create a water mask and compute the distance map to the nearest water body,
    using multispectral imagery and vector data describing water zones and rivers.
    """
    def __init__(self, image_path, water_path):
        """
        Initialize the WaterDistanceMap class.

        Parameters:
        - image_path (str): Path to the multispectral image file.
        - water_path (str): Path to the GeoPackage (.gpkg) file containing water zones and river data.
        """
        self.image_path = image_path
        self.water_path = water_path

    @staticmethod
    def add_buffer_to_geometry(gdf):
        """
        Adds a buffer to each geometry in the GeoDataFrame using the 'Largeur/2' column.
        Buffers allow the representation of river widths as polygons.

        Parameters:
        - gdf (GeoDataFrame): Must include a 'Largeur/2' column and geometry.

        Returns:
        - GeoDataFrame: Updated with buffered geometries.
        """
        # Check that the buffer column exists
        if 'Largeur/2' not in gdf.columns:
            raise ValueError("Column 'Largeur/2' is missing in the GeoDataFrame.")
        # Apply the buffer, skipping NaNs
        gdf['geometry'] = gdf.apply(
            lambda row: row['geometry'].buffer(row['Largeur/2']) if not np.isnan(row['Largeur/2']) else row['geometry'],
            axis=1
        )
        return gdf
        
    def create_water_mask(self, site, output_dir=None, save_mask=True):
        """
        Create a binary water mask raster from vector data (water surfaces and rivers).

        Parameters:
        - site (str): Site identifier, must match the vector data's 'Site' field.
        - output_dir (str): Directory where the water mask will be saved.
        - save_mask (bool): If True, save the generated raster to disk.

        Returns:
        - water_raster (numpy.ndarray): Binary water mask.
        - height (int): Raster height (pixels).
        - width (int): Raster width (pixels).
        - crs (CRS): Coordinate Reference System of the raster.
        - transform (Affine): Affine geotransform of the raster.
        """
        # Only allow known site names to avoid mismatches
        if site not in ['Timbertiere', 'Fao', 'Roudoudour', 'Louroux', 'Cisse']:
            raise ValueError("Invalid value received for `site`, should be either 'Timbertiere', 'Fao', 'Roudoudour', 'Louroux' or 'Cisse'.")

        # Load image metadata to define the raster grid for output
        with rasterio.open(self.image_path) as src:
            profile = src.profile
            transform = src.transform
            width = src.width
            height = src.height
            crs = src.crs

        # Read water polygons, filter by site, and prepare as raster input
        water_gdf = gpd.read_file(self.water_path, layer="Surface en eau")
        water_gdf = water_gdf[water_gdf['Site'] == site]

        # Read rivers, filter by site, and buffer by half-width for polygonization
        rivers_gdf = gpd.read_file(self.water_path, layer="reseauhydro_lineaire")
        rivers_gdf = rivers_gdf[rivers_gdf['Site'] == site]
        rivers_gdf = self.add_buffer_to_geometry(rivers_gdf)

        # Rasterize both water surfaces and buffered rivers into a binary mask
        water_raster = vector_to_raster([water_gdf, rivers_gdf], transform, width, height)

        # Save to disk if requested
        if save_mask:
            mask_path = os.path.join(output_dir, f'water_mask_{site.lower()}.tif')
            with rasterio.open(mask_path, 'w', driver='GTiff', height=height, width=width,
                               count=1, dtype='uint8', crs=crs, transform=transform) as dst:
                dst.write(water_raster, 1)
            print(f"Water raster has been created and saved to {mask_path}")

        return water_raster, height, width, crs, transform

    def create_water_distance_map(self, site, output_dir, save_mask=True):
        """
        Compute a distance raster showing the distance from each pixel to the nearest water body.

        Parameters:
        - site (str): Site identifier; must be recognized in the vector data.
        - output_dir (str): Directory to save the output.
        - save_mask (bool): If True, save the binary water mask before creating the distance map.

        Returns:
        - distance_raster (numpy.ndarray): Distance to nearest water in pixels (same size as input image).
        """
        # Validate site
        if site not in ['Timbertiere', 'Fao', 'Roudoudour', 'Louroux', 'Cisse']:
            raise ValueError("Invalid value received for `site`, should be either 'Timbertiere', 'Fao', 'Roudoudour', 'Louroux' or 'Cisse'.")

        # Step 1: Create the water mask raster
        water_raster, height, width, crs, transform = self.create_water_mask(site, output_dir, save_mask)

        # Step 2: Calculate distance from each pixel to nearest water (Euclidean distance)
        distance_raster = distance_transform_edt(water_raster == 0)  # True where not water

        # Step 3: Save the distance raster to disk
        distance_map_path = os.path.join(output_dir, f'water_distance_map_{site.lower()}.tif')
        with rasterio.open(distance_map_path, 'w', driver='GTiff', height=height, width=width,
                           count=1, dtype='float32', crs=crs, transform=transform) as dst:
            dst.write(distance_raster, 1)

        print(f"Distance raster has been created and saved to {distance_map_path}")

        return distance_raster


class HedgerowMask:
    """
    Class for generating a hedgerow mask, using NDVI and height (DHM),
    and removing buildings using OpenStreetMap data.
    """
    def __init__(self, image_path, dhm_path):
        """
        Initialize the HedgerowMask class.

        Parameters:
        - image_path (str): Path to the multispectral image file.
        - dhm_path (str): Path to the Digital Height Model (DHM) file.
        """
        self.image_path = image_path
        self.dhm_path = dhm_path

    def load_align_image_and_dhm(self):
        """
        Load and align the multispectral image and DHM, ensuring same extent and pixel grid.

        Returns:
        - multispectral_image (numpy.ndarray): The multispectral image array.
        - dhm_aligned (numpy.ndarray): The aligned DHM array.
        - multispectral_transform (Affine): The affine transform of the multispectral image.
        - multispectral_width (int): The width of the multispectral image.
        - multispectral_height (int): The height of the multispectral image.
        - multispectral_bounds (BoundingBox): The bounding box of the multispectral image.
        - multispectral_meta (dict): Metadata of the multispectral image.
        - multispectral_crs (CRS): Coordinate reference system of the multispectral image.
        """
        # Step 1: Load the multispectral image and record its spatial parameters
        with rasterio.open(self.image_path) as src:
            multispectral_image = src.read()
            multispectral_transform = src.transform
            multispectral_crs = src.crs
            multispectral_bounds = src.bounds
            multispectral_meta = src.meta
            multispectral_width = src.width
            multispectral_height = src.height

        # Step 2: Load the DHM (height model)
        with rasterio.open(self.dhm_path) as dhm_src:
            dhm = dhm_src.read(1)  
            dhm_transform = dhm_src.transform
            dhm_crs = dhm_src.crs

        # Step 3: "Force" the DHM CRS to match the image (often needed if metadata is inconsistent)
        dhm_crs_forced = multispectral_crs

        # Step 4: Allocate an empty array for the reprojected DHM, with the same size as the image
        dhm_aligned = np.empty((multispectral_height, multispectral_width), dtype=rasterio.float32)

        # Step 5: Calculate the affine transform for the output grid (same as image)
        out_transform, out_width, out_height = calculate_default_transform(
            multispectral_crs, multispectral_crs, multispectral_width, multispectral_height,
            *multispectral_bounds)

        # Step 6: Reproject the DHM to match the image spatial grid
        reproject(
            source=dhm,
            destination=dhm_aligned,
            src_transform=dhm_transform,
            src_crs=dhm_crs_forced,
            dst_transform=out_transform,
            dst_crs=multispectral_crs,
            resampling=Resampling.bilinear
        )

        return (multispectral_image, dhm_aligned, multispectral_transform, 
                multispectral_width, multispectral_height, multispectral_bounds, 
                multispectral_meta, multispectral_crs)

    def create_NDVI(self, multispectral_image):
        """
        Compute Normalized Difference Vegetation Index (NDVI) from the multispectral image.

        Parameters:
        - multispectral_image (numpy.ndarray): The input image array.

        Returns:
        - ndvi (numpy.ndarray): Array of NDVI values.
        """
        # NDVI = (NIR - Red) / (NIR + Red)
        # The exact band assignments may differ by sensor
        red_band = multispectral_image[5] if multispectral_image.shape[0] > 5 else multispectral_image[2]
        nir_band = multispectral_image[9] if multispectral_image.shape[0] > 5 else multispectral_image[4]
        # Compute NDVI, taking care with division (avoid divide by zero)
        ndvi = (nir_band.astype(float) - red_band.astype(float)) / (nir_band + red_band + 1e-6)
        return ndvi

    def load_buildings(self, multispectral_bounds, multispectral_crs, src_crs='EPSG:2154', target_crs='EPSG:4326'):
        """
        Download and prepare building outlines from OpenStreetMap for the region.

        Parameters:
        - multispectral_bounds (BoundingBox): Bounding box of the target image.
        - multispectral_crs (CRS): CRS of the image (for reprojecting buildings).
        - src_crs (str): CRS of the image bounds (default: Lambert 93).
        - target_crs (str): CRS for OSM data (default: WGS84).

        Returns:
        - gdf_buildings (GeoDataFrame): Buildings geometries in the image CRS.
        """
        # Transform the bounding box from local/projected CRS to WGS84 (lat/lon) for OSM requests
        transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
        min_lon, min_lat = transformer.transform(multispectral_bounds.left, multispectral_bounds.bottom)
        max_lon, max_lat = transformer.transform(multispectral_bounds.right, multispectral_bounds.top)

        # Download OSM building footprints within the bounding box
        gdf_buildings = ox.geometries_from_bbox(north=max_lat, south=min_lat, east=max_lon, west=min_lon, tags={'building': True})

        # Reproject to match the image CRS if needed
        if gdf_buildings.crs != multispectral_crs:
            gdf_buildings = gdf_buildings.to_crs(multispectral_crs)
        # Optionally buffer buildings slightly to ensure overlap
        gdf_buildings = gdf_buildings.buffer(3)
        return gdf_buildings

    def create_hedgerow_mask(self, ndvi_threshold=0.3, height_threshold=2.5):
        """
        Create a binary mask identifying hedgerows (linear features with vegetation and height).

        Parameters:
        - ndvi_threshold (float): Minimum NDVI to classify as vegetation.
        - height_threshold (float): Minimum height for a pixel to be considered a hedgerow.

        Returns:
        - final_mask (numpy.ndarray): The binary hedgerow mask.
        - multispectral_meta (dict): Metadata for saving the mask.
        """
        # Step 1: Load and align the input image and DHM (height model)
        (multispectral_image, dhm_aligned, multispectral_transform, 
         multispectral_width, multispectral_height, multispectral_bounds, 
         multispectral_meta, multispectral_crs) = self.load_align_image_and_dhm()

        # Step 2: Compute NDVI to identify vegetated areas
        ndvi = self.create_NDVI(multispectral_image)

        # Step 3: Combine NDVI and height thresholds to find likely hedgerows
        vegetation_mask = (ndvi > ndvi_threshold) & (dhm_aligned > height_threshold)

        # Step 4: Clean up the mask using morphological opening and closing
        cleaned_mask_1 = binary_opening(vegetation_mask, structure=np.ones((3, 3)))
        cleaned_mask_2 = binary_closing(cleaned_mask_1, structure=np.ones((3, 3)))

        # Step 5: Remove buildings using OSM data to avoid confusion with hedgerows
        try:
            gdf_buildings = self.load_buildings(multispectral_bounds, multispectral_crs)
            buildings_mask = vector_to_raster([gdf_buildings], multispectral_transform, multispectral_width, multispectral_height)
            final_mask = np.where(buildings_mask, 0, cleaned_mask_2)
        except:
            # If building data cannot be downloaded or processed, skip this step
            final_mask = cleaned_mask_2

        return final_mask, multispectral_meta

    def save_mask(self, mask, meta, mask_path):
        """
        Save a binary mask as a GeoTIFF raster file.

        Parameters:
        - mask (numpy.ndarray): The mask to save.
        - meta (dict): Metadata (copied from the source image) for correct georeferencing.
        - mask_path (str): Path to the output raster file.
        """
        # Update metadata for a single-band, 8-bit unsigned output
        out_meta = meta.copy()
        out_meta.update({"driver": "GTiff", "count": 1, "dtype": 'uint8'})
        # Remove nodata if present (not needed for binary masks)
        if 'nodata' in out_meta:
            out_meta.pop('nodata')
        # Write the mask array to disk
        with rasterio.open(mask_path, "w", **out_meta) as dest:
            dest.write(mask.astype(np.uint8), 1)
        print(f"Mask has been created and saved to {mask_path}")