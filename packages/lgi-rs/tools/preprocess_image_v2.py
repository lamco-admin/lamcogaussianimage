#!/usr/bin/env python3
"""
LGI Image Preprocessing Pipeline v2 - Production Implementation

Generates analysis maps and metadata for intelligent Gaussian placement.

Output naming:
    Input:  kodim02.png
    Output: kodim02.json (metadata)
            kodim02_placement.npy, kodim02_entropy.npy, etc. (maps)

File handling modes:
    --mode overwrite (default): Always regenerate
    --mode skip: Skip if JSON exists
    --mode update: Only recompute if parameters changed
    --mode prompt: Ask user

Usage:
    python tools/preprocess_image_v2.py kodim02.png
    python tools/preprocess_image_v2.py kodim02.png --mode skip
    python tools/preprocess_image_v2.py kodim02.png --n-segments 1000 --use-gpu
"""

import sys
import os
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import cv2
import mahotas as mh
import skimage
from skimage import io, segmentation, morphology
from skimage.color import rgb2gray

class ImagePreprocessor:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.has_cuda = False

        if use_gpu:
            self.has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
            if not self.has_cuda:
                print("⚠️  GPU requested but CUDA not available, using CPU")
                self.use_gpu = False

    def preprocess(self, image_path, params):
        """
        Full preprocessing pipeline
        
        Returns: metadata dict
        """
        image_path = Path(image_path).resolve()
        base_name = image_path.stem  # kodim02
        output_dir = image_path.parent  # Same directory as image

        print(f"Preprocessing: {image_path.name}")
        print(f"Output: {base_name}.json + {base_name}_*.npy")

        # Load image
        image = io.imread(str(image_path))
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[2] == 4:
            image = image[:, :, :3]

        height, width = image.shape[:2]
        file_size = image_path.stat().st_size

        # Compute checksum
        with open(image_path, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Analysis
        print("  [1/8] Entropy map...")
        entropy_map = self._compute_entropy(image, params['entropy_tile_size'])
        np.save(output_dir / f"{base_name}_entropy.npy", entropy_map)

        print("  [2/8] Gradient map...")
        gradient_map = self._compute_gradient(image)
        np.save(output_dir / f"{base_name}_gradient.npy", gradient_map)

        print("  [3/8] Texture map (Haralick)...")
        texture_map = self._compute_texture(image)
        np.save(output_dir / f"{base_name}_texture.npy", texture_map)

        print(f"  [4/8] SLIC segments (n={params['n_segments']})...")
        segments = segmentation.slic(
            image, n_segments=params['n_segments'],
            compactness=10, sigma=1, channel_axis=-1
        )
        np.save(output_dir / f"{base_name}_segments.npy", segments)

        print("  [5/8] Saliency map...")
        saliency_map = self._compute_saliency(image)
        np.save(output_dir / f"{base_name}_saliency.npy", saliency_map)

        print("  [6/8] Distance transform...")
        distance_map = self._compute_distance(segments)
        np.save(output_dir / f"{base_name}_distance.npy", distance_map)

        print("  [7/8] Skeleton...")
        skeleton = self._compute_skeleton(segments)
        np.save(output_dir / f"{base_name}_skeleton.npy", skeleton)

        print("  [8/8] Placement map...")
        placement_map = self._generate_placement_map(
            entropy_map, gradient_map, texture_map,
            saliency_map, distance_map, params
        )
        np.save(output_dir / f"{base_name}_placement.npy", placement_map)

        # Create metadata
        metadata = {
            "source": {
                "filename": image_path.name,
                "path": str(image_path),
                "format": image_path.suffix[1:].upper(),
                "width": int(width),
                "height": int(height),
                "channels": int(image.shape[2]),
                "file_size_bytes": int(file_size),
                "checksum_sha256": checksum
            },
            "preprocessing": {
                "version": "2.0.0",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "libraries": {
                    "opencv": cv2.__version__,
                    "mahotas": mh.__version__,
                    "scikit_image": skimage.__version__,
                },
                "parameters": params,
                "gpu_used": self.use_gpu and self.has_cuda
            },
            "analysis_maps": {
                "placement_map": f"{base_name}_placement.npy",
                "entropy_map": f"{base_name}_entropy.npy",
                "gradient_map": f"{base_name}_gradient.npy",
                "texture_map": f"{base_name}_texture.npy",
                "saliency_map": f"{base_name}_saliency.npy",
                "distance_map": f"{base_name}_distance.npy",
                "skeleton": f"{base_name}_skeleton.npy",
                "segments": f"{base_name}_segments.npy"
            },
            "statistics": {
                "global_entropy": float(entropy_map.mean()),
                "mean_gradient": float(gradient_map.mean()),
                "texture_percentage": float((texture_map > 0.5).sum() / texture_map.size * 100),
                "actual_segments": int(segments.max() + 1)
            }
        }

        # Save JSON
        json_path = output_dir / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Complete: {json_path.name}")
        return metadata

    def _compute_entropy(self, image, tile_size):
        gray = rgb2gray(image)
        h, w = gray.shape
        entropy_map = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                tile = gray[y:min(y+tile_size,h), x:min(x+tile_size,w)]
                hist, _ = np.histogram(tile, bins=256, range=(0, 1))
                hist = hist[hist > 0].astype(float)
                if len(hist) > 0:
                    hist = hist / hist.sum()
                    entropy = -np.sum(hist * np.log2(hist))
                    entropy_map[y:min(y+tile_size,h), x:min(x+tile_size,w)] = entropy

        return entropy_map / (entropy_map.max() + 1e-8)

    def _compute_gradient(self, image):
        gray = (rgb2gray(image) * 255).astype(np.uint8)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x.astype(float)**2 + grad_y.astype(float)**2)
        return (gradient / (gradient.max() + 1e-8)).astype(np.float32)

    def _compute_texture(self, image):
        gray = (rgb2gray(image) * 255).astype(np.uint8)
        h, w = gray.shape
        texture_map = np.zeros((h, w), dtype=np.float32)
        tile_size = 32

        for y in range(0, h - tile_size, tile_size // 2):
            for x in range(0, w - tile_size, tile_size // 2):
                tile = gray[y:y+tile_size, x:x+tile_size]
                try:
                    haralick = mh.features.haralick(tile)
                    contrast = haralick[:, 1].mean()
                    energy = haralick[:, 8].mean()
                    texture_score = contrast / (energy + 0.01)
                    texture_map[y:y+tile_size, x:x+tile_size] = texture_score
                except:
                    pass

        return texture_map / (texture_map.max() + 1e-8)

    def _compute_saliency(self, image):
        image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency = detector.computeSaliency(image_uint8)
        if not success:
            return np.ones(image.shape[:2], dtype=np.float32)
        return (saliency / (saliency.max() + 1e-8)).astype(np.float32)

    def _compute_distance(self, segments):
        dist_map = np.zeros(segments.shape, dtype=np.float32)
        for seg_id in range(segments.max() + 1):
            mask = (segments == seg_id).astype(np.uint8)
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            dist_map = np.maximum(dist_map, dist)
        return dist_map / (dist_map.max() + 1e-8)

    def _compute_skeleton(self, segments):
        skeleton_map = np.zeros(segments.shape, dtype=np.float32)
        for seg_id in range(min(segments.max() + 1, 500)):  # Limit for speed
            mask = (segments == seg_id)
            if mask.sum() > 100:
                skel = morphology.skeletonize(mask)
                skeleton_map[skel] = 1.0
        return skeleton_map

    def _generate_placement_map(self, entropy, gradient, texture, saliency, distance, params):
        w = params.get('weights', {})
        placement = (
            w.get('entropy', 0.3) * entropy +
            w.get('gradient', 0.25) * gradient +
            w.get('texture', 0.2) * texture +
            w.get('saliency', 0.15) * saliency +
            w.get('distance', 0.10) * distance
        )
        return (placement / placement.sum()).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="LGI Image Preprocessing v2")
    parser.add_argument("image_path", help="Input image")
    parser.add_argument("--n-segments", type=int, default=500)
    parser.add_argument("--entropy-tile-size", type=int, default=16)
    parser.add_argument("--mode", choices=['overwrite', 'skip', 'update', 'prompt'], default='overwrite')
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image_path).resolve()
    json_path = image_path.with_suffix('.json')

    # File handling
    if json_path.exists():
        if args.mode == 'skip':
            print(f"Skipping (JSON exists): {json_path.name}")
            return
        elif args.mode == 'prompt':
            response = input(f"{json_path.name} exists. Overwrite? [y/N]: ")
            if response.lower() != 'y':
                return
        elif args.mode == 'update':
            # Check if parameters changed
            with open(json_path) as f:
                old_meta = json.load(f)
            old_params = old_meta.get('preprocessing', {}).get('parameters', {})
            new_params = {'n_segments': args.n_segments, 'entropy_tile_size': args.entropy_tile_size}
            if old_params == new_params:
                print(f"Skipping (parameters unchanged): {json_path.name}")
                return

    # Run preprocessing
    preprocessor = ImagePreprocessor(use_gpu=args.use_gpu)
    params = {
        'n_segments': args.n_segments,
        'entropy_tile_size': args.entropy_tile_size
    }

    metadata = preprocessor.preprocess(str(image_path), params)

    print("\n" + "="*70)
    print(f"✅ {image_path.stem}.json created")
    print(f"   Maps: {len(metadata['analysis_maps'])} .npy files")
    print(f"   Stats: entropy={metadata['statistics']['global_entropy']:.4f}, "
          f"texture={metadata['statistics']['texture_percentage']:.1f}%")
    print("="*70)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
