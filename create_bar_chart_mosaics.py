"""
Create mosaics from PNG bar chart images and save as WebP format.

Run this script in your SimBA conda environment:
    conda activate simba
    python create_bar_chart_mosaics.py
"""
import os
import sys

# Try to import required libraries
try:
    import cv2
    USE_CV2 = True
except ImportError:
    try:
        from PIL import Image
        USE_PIL = True
        USE_CV2 = False
    except ImportError:
        print("Error: Neither OpenCV (cv2) nor PIL/Pillow is available.")
        print("Please install one of them:")
        print("  conda install opencv")
        print("  or")
        print("  pip install pillow")
        sys.exit(1)

import numpy as np

# Input and output directories
input_dir = '/Users/simon/Desktop/envs/simba/simba/test_output'
output_dir = '/Users/simon/Desktop/envs/simba/simba/test_output'

# Get all PNG files
png_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
print(f"Found {len(png_files)} PNG files:")
for f in png_files:
    print(f"  - {f}")

if not png_files:
    print("No PNG files found!")
    sys.exit(1)

# Target size for each image in the mosaic
target_size = (640, 480)

# Load and resize all images
images = []
for png_file in png_files:
    img_path = os.path.join(input_dir, png_file)
    try:
        if USE_CV2:
            img = cv2.imread(img_path)
            if img is not None:
                resized = cv2.resize(img, target_size)
                images.append(resized)
                print(f"✓ Loaded and resized: {png_file} ({img.shape} -> {resized.shape})")
            else:
                print(f"✗ Failed to load: {png_file}")
        else:
            img = Image.open(img_path)
            original_size = img.size
            resized = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(resized)
            images.append(img_array)
            print(f"✓ Loaded and resized: {png_file} ({original_size} -> {target_size})")
    except Exception as e:
        print(f"✗ Failed to load {png_file}: {e}")

if not images:
    print("No images could be loaded!")
    sys.exit(1)

print(f"\nLoaded {len(images)} images")

# Create mosaics in different grid layouts
def create_mosaic(images, cols, rows, name_suffix=""):
    """Create a mosaic with specified number of columns and rows"""
    # Take only the images that fit in the grid
    grid_images = images[:cols * rows]
    
    # Fill empty slots with white images if needed
    while len(grid_images) < cols * rows:
        if USE_CV2:
            grid_images.append(np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255)
        else:
            grid_images.append(np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255)
    
    # Create rows
    rows_list = []
    for i in range(rows):
        row_start = i * cols
        row_end = row_start + cols
        row_images = grid_images[row_start:row_end]
        row = np.hstack(row_images)
        rows_list.append(row)
    
    # Stack rows vertically
    mosaic = np.vstack(rows_list)
    return mosaic

# Create different mosaic layouts
mosaics = []

# Mosaic 1: 3x4 grid (12 images)
if len(images) >= 12:
    print("\nCreating 3x4 mosaic (12 images)...")
    mosaic_3x4 = create_mosaic(images, cols=3, rows=4, name_suffix="3x4")
    mosaics.append(("bar_chart_mosaic_3x4", mosaic_3x4))
    print(f"✓ Created 3x4 mosaic: {mosaic_3x4.shape}")

# Mosaic 2: 4x3 grid (12 images)
if len(images) >= 12:
    print("\nCreating 4x3 mosaic (12 images)...")
    mosaic_4x3 = create_mosaic(images, cols=4, rows=3, name_suffix="4x3")
    mosaics.append(("bar_chart_mosaic_4x3", mosaic_4x3))
    print(f"✓ Created 4x3 mosaic: {mosaic_4x3.shape}")

# Mosaic 3: 2x6 grid (all 11 images + 1 empty)
print("\nCreating 2x6 mosaic (all images)...")
mosaic_2x6 = create_mosaic(images, cols=2, rows=6, name_suffix="2x6")
mosaics.append(("bar_chart_mosaic_2x6", mosaic_2x6))
print(f"✓ Created 2x6 mosaic: {mosaic_2x6.shape}")

# Mosaic 4: 6x2 grid (all 11 images + 1 empty)
print("\nCreating 6x2 mosaic (all images)...")
mosaic_6x2 = create_mosaic(images, cols=6, rows=2, name_suffix="6x2")
mosaics.append(("bar_chart_mosaic_6x2", mosaic_6x2))
print(f"✓ Created 6x2 mosaic: {mosaic_6x2.shape}")

# Mosaic 5: 3x4 grid with first 11 images (one empty slot)
print("\nCreating 3x4 mosaic (11 images + 1 empty)...")
mosaic_3x4_partial = create_mosaic(images[:11], cols=3, rows=4, name_suffix="3x4_partial")
mosaics.append(("bar_chart_mosaic_3x4_partial", mosaic_3x4_partial))
print(f"✓ Created 3x4 partial mosaic: {mosaic_3x4_partial.shape}")

# Save all mosaics as WebP
print("\n" + "=" * 60)
print("Saving mosaics as WebP...")
for name, mosaic in mosaics:
    webp_path = os.path.join(output_dir, f"{name}.webp")
    try:
        if USE_CV2:
            # OpenCV supports WebP
            success = cv2.imwrite(webp_path, mosaic, [cv2.IMWRITE_WEBP_QUALITY, 90])
            if success:
                file_size = os.path.getsize(webp_path) / 1024  # KB
                print(f"✓ Saved: {webp_path} ({file_size:.1f} KB)")
            else:
                print(f"✗ Failed to save: {webp_path}")
        else:
            # PIL/Pillow supports WebP
            mosaic_img = Image.fromarray(mosaic.astype(np.uint8))
            mosaic_img.save(webp_path, 'WEBP', quality=90)
            file_size = os.path.getsize(webp_path) / 1024  # KB
            print(f"✓ Saved: {webp_path} ({file_size:.1f} KB)")
    except Exception as e:
        print(f"✗ Failed to save {webp_path}: {e}")

print("\n" + "=" * 60)
print("✓ All mosaics created and saved as WebP!")
