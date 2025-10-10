import numpy as np
import pandas as pd
import os
import rasterio

def read_image(image_path):
    """
    Open and read a raster image file using rasterio.
    
    Parameters:
    - image_path: str, Path to the raster image file.
    
    Returns:
    - rasterio.io.DatasetReader: Rasterio object for accessing image data.
    """
    return rasterio.open(image_path)
    

def _save_patch(patch, profile, index, base_name, output_dir, multispectrale=True):
    """
    Save the extracted patch to a file.
    
    Parameters:
    - patch: np.ndarray, Patch to save.
    - profile: dict, Metadata profile for the patch.
    - index: int, Patch number to use in the filename.
    - base_name: str, Base name for the patch file.
    - multispectrale: bool, If True, treat the patch as multispectral.
    """

    if multispectrale and patch.shape[0] > patch.shape[-1]:
        patch = np.transpose(patch, (2, 0, 1)) 

    # print(patch.shape)
    count = patch.shape[0] if multispectrale else 1
    height = patch.shape[1] if len(patch.shape) > 2 else patch.shape[0]
    width = patch.shape[2] if len(patch.shape) > 2 else patch.shape[1]
    
    assert 1 <= count <= 20
    assert 32 <= height <= 2048
    assert 32 <= width <= 2048
    
    file_path = os.path.join(output_dir, f'{base_name}_{index}.tif')
    with rasterio.open(file_path, 'w', 
                       driver='GTiff', 
                       height=height, 
                       width=width, 
                       count=count, dtype=np.float32, 
                       crs=profile['crs'], 
                       transform=profile['transform']) as dst:
        dst.write(patch, None if multispectrale else 1)



def remove_nan(image_paths, labels):
    return list(np.array(image_paths)[pd.Index(labels).notna()])

def get_patch_list(base_dir, patch_type, sites=None):
    """
    Get a list of all patch files in the specified patch type directory, optionally filtering by site.

    Args:
        base_dir (str): The root directory (train_patches, test_patches, or val_patches).
        patch_type (str): The type of patch directory to look for (e.g., 'dsm_patches', 'dtm_patches').
        sites (list, optional): A list of site names to filter by (e.g., ['fao', 'roudoudour', 'timbertiere']).
                                If None, no site filtering is applied.

    Returns:
        list: A list of paths to the patch files.
    """
    patch_list = []
    for root, dirs, files in os.walk(base_dir):
        # Check if the directory matches the patch type
        if os.path.basename(root) == patch_type:
            # Check if filtering by sites is required
            if sites:
                # Check if the site name (e.g., 'fao') is part of the path
                if any(site in root for site in sites):
                    patch_list.extend([os.path.join(root, file) for file in files])
            else:
                # No site filtering, add all files
                patch_list.extend([os.path.join(root, file) for file in files])
    return patch_list

def is_invalid_patch(file_path):
    """Check if the patch contains only NaN values or has a constant value."""
    try:
        with rasterio.open(file_path) as src:
            data = src.read()
            if np.all(np.isnan(data)) or np.all(data == data[0, 0, 0]):
                return True
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return False


