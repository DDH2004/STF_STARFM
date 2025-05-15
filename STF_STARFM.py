"""
STF_STARFM: A Python implementation of the STARFM (Spatial and Temporal Adaptive Reflectance Fusion Model) algorithm.

This module provides functions to load and process satellite imagery, divide images into tiles, 
apply the STARFM algorithm for fusing images, and save the fused output. The STARFM algorithm 
is used for blending high-resolution and low-resolution satellite images to generate fused images 
with enhanced spatial and temporal resolution.

Functions:
- load_stacked_image: Load and stack image bands from file paths.
- extract_tile: Extract a specific tile from an image.
- fuse_tile: Fuse a single tile using the STARFM algorithm.
- fuse_tiles: Fuse all tiles in an image using the STARFM algorithm.
- save_fused_image: Save the fused image to a file.
- main: Main function to orchestrate the image fusion process.

Dependencies:
- numpy
- rasterio
- PIL (Pillow)
- tqdm
- starfm4py (custom library for STARFM operations)

Example usage:
    python STF_STARFM.py
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from tqdm import tqdm

from starfm4py import stack_bands, fuse_starfm


def load_stacked_image(path_list, bands):
    """
    Load and stack image bands from a list of file paths.

    Args:
        path_list (list of str): List of file paths to the image bands.
        bands (list of str): List of band names to stack.

    Returns:
        np.ndarray: Stacked image array with shape (H, W, bands).
    """
    stacked = stack_bands(path_list, bands=bands)
    if stacked.shape[0] <= 5:  # Assuming shape (bands, H, W), transpose to (H, W, bands)
        stacked = stacked.transpose(1, 2, 0)
    return stacked


def extract_tile(image, i, j, tile_size):
    """
    Extract a tile from the image.

    Args:
        image (np.ndarray): Input image array with shape (H, W, bands).
        i (int): Starting row index of the tile.
        j (int): Starting column index of the tile.
        tile_size (int): Size of the tile.

    Returns:
        np.ndarray: Extracted tile with shape (tile_size, tile_size, bands).
    """
    return image[i:i + tile_size, j:j + tile_size, :]


def fuse_tile(master_tile, slave_tile, pred_tile):
    """
    Fuse a single tile using the STARFM algorithm.

    Args:
        master_tile (np.ndarray): Master tile array with shape (tile_size, tile_size, bands).
        slave_tile (np.ndarray): Slave tile array with shape (tile_size, tile_size, bands).
        pred_tile (np.ndarray): Prediction tile array with shape (tile_size, tile_size, bands).

    Returns:
        np.ndarray: Fused tile array with shape (tile_size, tile_size, bands).
    """
    tile_output = np.zeros_like(master_tile)
    for band_idx in range(master_tile.shape[-1]):
        fused_band = fuse_starfm(
            master_tile[..., band_idx],
            slave_tile[..., band_idx],
            pred_tile[..., band_idx]
        )
        tile_output[..., band_idx] = fused_band
    return tile_output


def fuse_tiles(master, slave, pred, tile_size):
    """
    Fuse all tiles in the image using the STARFM algorithm.

    Args:
        master (np.ndarray): Master image array with shape (H, W, bands).
        slave (np.ndarray): Slave image array with shape (H, W, bands).
        pred (np.ndarray): Prediction image array with shape (H, W, bands).
        tile_size (int): Size of the tiles to process.

    Returns:
        np.ndarray: Fused image array with shape (H, W, bands).
    """
    fused_image = np.zeros_like(master)
    height, width, bands = master.shape

    for i in tqdm(range(0, height, tile_size), desc="Rows"):
        for j in range(0, width, tile_size):
            tile_h = min(tile_size, height - i)
            tile_w = min(tile_size, width - j)

            master_tile = extract_tile(master, i, j, tile_size)
            slave_tile = extract_tile(slave, i, j, tile_size)
            pred_tile = extract_tile(pred, i, j, tile_size)

            fused_tile = fuse_tile(master_tile, slave_tile, pred_tile)
            fused_image[i:i + tile_h, j:j + tile_w, :] = fused_tile[:tile_h, :tile_w, :]

    return fused_image


def save_fused_image(image_array, output_path):
    """
    Save the fused image to a file.

    Args:
        image_array (np.ndarray): Fused image array with shape (H, W, bands).
        output_path (str): Path to save the output image.
    """
    Image.fromarray(image_array.astype(np.uint8)).save(output_path)


def main(master_paths, slave_paths, pred_paths, bands, tile_size, output_path):
    """
    Main function to load images, fuse them, and save the output.

    Args:
        master_paths (list of str): List of file paths for the master image bands.
        slave_paths (list of str): List of file paths for the slave image bands.
        pred_paths (list of str): List of file paths for the prediction image bands.
        bands (list of str): List of band names to process.
        tile_size (int): Size of the tiles to process.
        output_path (str): Path to save the fused output image.
    """
    print("Loading images...")
    master = load_stacked_image(master_paths, bands)
    slave = load_stacked_image(slave_paths, bands)
    pred = load_stacked_image(pred_paths, bands)

    print("Fusing tiles...")
    fused = fuse_tiles(master, slave, pred, tile_size)

    print("Saving fused image...")
    save_fused_image(fused, output_path)
    print("Fusion complete.")


if __name__ == '__main__':
    # Example usage - update these paths accordingly
    master_paths = ["/path/to/master_red.tif", "/path/to/master_green.tif", ...]
    slave_paths = ["/path/to/slave_red.tif", "/path/to/slave_green.tif", ...]
    pred_paths = ["/path/to/pred_red.tif", "/path/to/pred_green.tif", ...]
    bands = ['red', 'green', 'blue', 'nir']
    tile_size = 256
    output_path = "fused_output.png"

    main(master_paths, slave_paths, pred_paths, bands, tile_size, output_path)
