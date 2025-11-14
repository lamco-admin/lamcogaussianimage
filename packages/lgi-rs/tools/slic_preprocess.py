#!/usr/bin/env python3
"""
SLIC Superpixel Preprocessing for LGI Encoder

Generates Gaussian initialization from SLIC segmentation

Usage:
    python tools/slic_preprocess.py <image_path> <n_segments> [output_json]

Example:
    python tools/slic_preprocess.py kodak-dataset/kodim02.png 500 slic_init.json
"""

import sys
import json
import numpy as np
from skimage.segmentation import slic
from skimage.io import imread
from skimage.color import rgb2lab

def generate_slic_gaussians(image_path, n_segments=100, output_path="slic_init.json"):
    """Generate Gaussian parameters from SLIC superpixels"""

    print(f"Loading image: {image_path}")
    image = imread(image_path)

    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image, image, image], axis=-1)

    height, width = image.shape[:2]
    print(f"Image size: {width}×{height}")

    # Run SLIC segmentation
    print(f"Running SLIC (n_segments={n_segments}, compactness=10)...")
    segments = slic(
        image,
        n_segments=n_segments,
        compactness=10,  # Balance color vs spatial proximity
        sigma=1,          # Gaussian smoothing
        start_label=0,
    )

    actual_segments = segments.max() + 1
    print(f"Created {actual_segments} superpixels")

    # Extract Gaussian parameters from each superpixel
    gaussians = []

    for segment_id in range(actual_segments):
        mask = (segments == segment_id)
        pixels = np.argwhere(mask)

        if len(pixels) == 0:
            continue

        # Centroid (mean position)
        cy, cx = pixels.mean(axis=0)

        # Mean color
        segment_colors = image[mask].astype(float) / 255.0
        r, g, b = segment_colors.mean(axis=0)

        # Compute covariance for Gaussian shape
        if len(pixels) > 1:
            # Position covariance (in pixel space)
            cov = np.cov(pixels.T)

            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eig(cov)

            # Sort eigenvalues (descending)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Gaussian scales (normalized to [0,1])
            # sqrt(eigenvalue) gives std dev in pixels
            scale_x = float(np.sqrt(eigenvalues[0])) / width
            scale_y = float(np.sqrt(eigenvalues[1])) / height

            # Rotation from major eigenvector
            # Note: eigenvector is (y, x) ordering from argwhere
            rotation = float(np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0]))
        else:
            # Single pixel - use small isotropic
            scale_x = 0.01
            scale_y = 0.01
            rotation = 0.0

        gaussians.append({
            "position": [float(cx / width), float(cy / height)],
            "scale": [scale_x, scale_y],
            "rotation": rotation,
            "color": [float(r), float(g), float(b)],
            "segment_id": int(segment_id),
            "pixel_count": int(len(pixels)),
        })

    # Save to JSON
    output = {
        "source_image": image_path,
        "image_width": int(width),
        "image_height": int(height),
        "n_segments_requested": n_segments,
        "n_gaussians": len(gaussians),
        "gaussians": gaussians,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Saved {len(gaussians)} Gaussian inits to {output_path}")

    # Statistics
    pixel_counts = [g["pixel_count"] for g in gaussians]
    print(f"\nSuperpixel statistics:")
    print(f"  Mean pixels/segment: {np.mean(pixel_counts):.1f}")
    print(f"  Min/max pixels: {np.min(pixel_counts)} / {np.max(pixel_counts)}")

    return gaussians

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python slic_preprocess.py <image_path> <n_segments> [output_json]")
        print("Example: python slic_preprocess.py kodim02.png 500")
        sys.exit(1)

    image_path = sys.argv[1]
    n_segments = int(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) > 3 else "slic_init.json"

    generate_slic_gaussians(image_path, n_segments, output_path)
