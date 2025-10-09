import numpy as np
import pandas as pd
import os
import random
import rasterio
import tensorflow as tf
import torch
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

def overlay_masks(img, mask, bands=[5, 3, 1]):
    # Ensure img is a NumPy array
    img = np.array(img)

    if len(img.shape) == 2:
        # Grayscale image
        rgb = np.stack([img, img, img], axis=-1)
    else:
        num_bands = img.shape[0]  # Assuming the shape is (bands, height, width)
        if num_bands == 1:
            # Grayscale image
            gray = img[0]
            rgb = np.stack([gray, gray, gray], axis=-1)
        elif num_bands == 3:
            # RGB image
            rgb = np.transpose(img, (1, 2, 0))  # Assuming shape is (3, height, width)
        elif num_bands > 3:
            # Use specified bands
            r = img[bands[0]]
            g = img[bands[1]]
            b = img[bands[2]]
            rgb = np.stack([r, g, b], axis=-1)
        else:
            raise ValueError("Unsupported number of bands. Expected 1, 3, or more.")

    # Normalize the RGB image
    rgb = np.clip(rgb, 0, 255)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255
    rgb = rgb.astype(np.uint8)
    
    # Convert to PyTorch tensors
    rgb_tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
    mask_tensor = torch.from_numpy(mask.astype(bool))
    
    # Overlay the mask on the image
    img_with_mask = draw_segmentation_masks(rgb_tensor, masks=mask_tensor, alpha=0.4, colors=[(255, 0, 0)])
    
    # Convert back to NumPy array
    overlaid_mask = np.transpose(np.array(img_with_mask), (1, 2, 0))
    
    return overlaid_mask


def inspect_dataset(dataset, dataset_name, num_samples=1, for_autoencoder=False):
    print(f"Inspecting {dataset_name}...")

    if for_autoencoder:
        for i, image in enumerate(dataset.take(num_samples)):
            print(f"\nSample {i + 1}:")
            print(f"Image shape: {image.shape}")
            print(f"Image data type: {image.dtype}")
            print(f"Image max value: {tf.reduce_max(image).numpy()}")
            print(f"Image min value: {tf.reduce_min(image).numpy()}")

    else:
        for i, (image, label) in enumerate(dataset.take(num_samples)):
            print(f"\nSample {i + 1}:")
            print(f"Image shape: {image.shape}")
            print(f"Label: {label.numpy()}")
            print(f"Image data type: {image.dtype}")
            print(f"Image max value: {tf.reduce_max(image).numpy()}")
            print(f"Image min value: {tf.reduce_min(image).numpy()}")        
    
    print(f"\nFinished inspecting {dataset_name}.")

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


def plot_history(acc_history, val_acc_history, loss_history, val_loss_history):
    # Create a figure with a row of two subplots
    plt.figure(figsize=(14, 5))
    
    # Plot accuracy on the first subplot
    plt.subplot(1, 2, 1)
    plt.plot(acc_history)
    plt.plot(val_acc_history)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    
    # Plot loss on the second subplot
    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.plot(val_loss_history)
    plt.title('Model Loss')
    plt.ylim((0, 5))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    
    # Show the plots
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def custom_train_test_split(img_paths, dsm_paths, dtm_paths, water_paths, labels, split=0.5):
    """
    Splits image paths, DSM paths, DTM paths, water paths, and labels into training and testing sets.
    
    Args:
        img_paths (list): List of file paths to the image inputs.
        dsm_paths (list): List of file paths to DSM inputs.
        dtm_paths (list): List of file paths to DTM inputs.
        water_paths (list): List of file paths to water data inputs.
        labels (list): List of labels corresponding to each input set.
        split (float): Ratio of data to be used for training. Defaults to 0.5.

    Returns:
        tuple: (X_train, train_labels, X_test, test_labels) where:
            - X_train (tuple): Tuple containing lists of paths for training (image, DSM, DTM, water).
            - train_labels (list): Labels for the training set.
            - X_test (tuple): Tuple containing lists of paths for testing (image, DSM, DTM, water).
            - test_labels (list): Labels for the testing set.
    """

    # Randomly shuffle indices to split the dataset
    selected_indices = random.sample(range(len(img_paths)), len(img_paths))
    train_indices = selected_indices[:int(split * len(img_paths))]
    test_indices = selected_indices[int(split * len(img_paths)):]

    # Prepare training dataset from train indices
    train_image_paths = [img_paths[i] for i in train_indices]
    train_dsm_paths = [dsm_paths[i] for i in train_indices]
    train_dtm_paths = [dtm_paths[i] for i in train_indices]
    train_water_paths = [water_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    # Prepare testing dataset from test indices
    test_image_paths = [img_paths[i] for i in test_indices]
    test_dsm_paths = [dsm_paths[i] for i in test_indices]
    test_dtm_paths = [dtm_paths[i] for i in test_indices]
    test_water_paths = [water_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    # Combine paths into tuples for easy handling
    X_train = (train_image_paths, train_dsm_paths, train_dtm_paths, train_water_paths)
    X_test = (test_image_paths, test_dsm_paths, test_dtm_paths, test_water_paths)

    return X_train, train_labels, X_test, test_labels


def compute_insect_share(sequence, title):
    labels = []
    for i in range(100000):
        try:
            batch = sequence.__getitem__(i)
            labels += batch[-1].tolist()
        except:
            break 
    print(f"{title}:{round(100*np.sum(labels)/len(labels), 3)}")


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

def get_pseudo_patch_list(base_dir, patch_type, sites=None, campaigns=["spring", "summer"]):
        
    patch_list = []
    
    for campaign in campaigns:
        for site in sites:
            for root, dirs, files in os.walk(os.path.join(base_dir, campaign, site, patch_type)):
                patch_list.extend([os.path.join(base_dir, campaign, site, patch_type, file) for file in files])
                
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

def clean_pseudo_dataset(root_dir="../donnees_terrain/donnees_modifiees/pseudo_patches/", campaigns=['spring', 'summer'], sites=["fao", "cisse", "louroux", 'timbertiere', "roudoudour"]):
    n = 0
    for campaign in campaigns:
        for site in sites:
            n_site = 0
            print(f"Processing site: {site}")
    
            image_dir = os.path.join(root_dir, campaign, site, "pseudo_labels_image_patches")
            dsm_dir = os.path.join(root_dir, campaign, site, "pseudo_labels_dsm_patches")
            dtm_dir = os.path.join(root_dir, campaign, site, "pseudo_labels_dtm_patches")
            water_dir = os.path.join(root_dir, campaign, site, "pseudo_labels_water_patches")
    
            if not os.path.exists(image_dir) or not os.path.exists(dsm_dir):
                print(f"Skipping {site} due to missing directories")
                continue
            
            # Use glob to get full paths
            image_paths = glob.glob(os.path.join(image_dir, "*.tif"))
            dsm_paths = glob.glob(os.path.join(dsm_dir, "*.tif"))
    
            for i in tqdm(range(len(image_paths))):
                filename = os.path.basename(image_paths[i])
                parts = filename.split('_')
    
                if len(parts) < 5:
                    print(f"Skipping invalid filename: {filename}")
                    continue
    
                x, y = parts[3], parts[4][:-4]
    
                dtm_path = dtm_dir + f"/{campaign}_{site}_dtm_patch_{x}_{y}.tif"
                water_path = water_dir + f"/{campaign}_{site}_water_patch_{x}_{y}.tif"
    
                if is_invalid_patch(image_paths[i]) or is_invalid_patch(dsm_paths[i]):
                    n += 1
                    n_site += 1
                    os.remove(image_paths[i])
                    os.remove(dsm_paths[i])
                    os.remove(dtm_path) 
                    os.remove(water_path)
            print(f"{n} invalid patches for site {site}.")
    
    print(f"{n} invalid patches deleted.")

def convert_autoencoder_gen_to_dataset(gen, batch_size, buffer_size):
    batch = gen.__getitem__(0)
    imgs = tf.constant(batch[0])
    dems = tf.constant(batch[1])
    for i in tqdm(range(1, len(gen))):
        batch = gen.__getitem__(i)
        img = batch[0]
        dem = batch[1]
        imgs = tf.concat([imgs, img], axis=0)
        dems = tf.concat([dems, dem], axis=0)
    
    dataset = tf.data.Dataset.from_tensor_slices((imgs, dems))
    # train_dataset = train_dataset.map(lambda image, dsm, dtm: preprocess(image, dsm, dtm, augment_data=False), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size)    
    return dataset

