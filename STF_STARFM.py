import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from tqdm import tqdm

from starfm4py import stack_bands, fuse_starfm


def load_stacked_image(path_list, bands):
    """Load and stack image bands."""
    stacked = stack_bands(path_list, bands=bands)
    if stacked.shape[0] <= 5:  # Assuming shape (bands, H, W), transpose to (H, W, bands)
        stacked = stacked.transpose(1, 2, 0)
    return stacked


def extract_tile(image, i, j, tile_size):
    return image[i:i + tile_size, j:j + tile_size, :]


def fuse_tile(master_tile, slave_tile, pred_tile):
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
    Image.fromarray(image_array.astype(np.uint8)).save(output_path)


def main(master_paths, slave_paths, pred_paths, bands, tile_size, output_path):
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