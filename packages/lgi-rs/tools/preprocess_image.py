#!/usr/bin/env python3
"""
LGI Image Preprocessing Pipeline - Comprehensive Implementation

Analyzes image and generates placement probability map for intelligent Gaussian positioning.

Libraries (latest versions):
- opencv-contrib-python 4.12.0.88: Morphology, distance transforms, features
- mahotas 1.4.18: Texture analysis (Haralick, LBP, TAS)
- scikit-image 0.25.2: SLIC, skeletonization, filters

Outputs:
- placement_map.npy: Combined probability map for Gaussian placement [HxW]
- entropy_map.npy: Per-pixel complexity scores
- gradient_map.npy: Edge strength magnitudes
- saliency_map.npy: Visual importance weights
- texture_map.npy: Texture classification (texture=1, smooth=0)
- segments.npy: SLIC superpixel labels
- distance_map.npy: Distance transform (region centers)
- skeleton.npy: Medial axis skeleton
- metadata.json: Analysis statistics and parameters

Usage:
    python tools/preprocess_image.py <image_path> [options]

Options:
    --n-segments INT        SLIC superpixel count (default: 500)
    --entropy-tile-size INT Tile size for entropy (default: 16)
    --use-gpu              Enable GPU acceleration (if available)
    --output-dir PATH       Output directory (default: preprocessing/)
    --verbose              Detailed logging
"""

import sys
import os
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import cv2
import mahotas as mh
import skimage
from skimage import io, segmentation, morphology, filters, feature, exposure
from skimage.color import rgb2gray
from scipy import ndimage
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Comprehensive image analysis for LGI encoding

    Generates placement probability map combining:
    - Entropy (local complexity)
    - Gradient magnitude (edges)
    - Texture classification (Haralick)
    - Saliency (visual importance)
    - SLIC segments (boundary-aware)
    - Distance transforms (region centers)
    - Skeleton (structural spine)
    """

    def __init__(
        self,
        n_segments: int = 500,
        entropy_tile_size: int = 16,
        use_gpu: bool = False,
        verbose: bool = False,
    ):
        self.n_segments = n_segments
        self.entropy_tile_size = entropy_tile_size
        self.use_gpu = use_gpu
        self.verbose = verbose

        # Check GPU availability
        if use_gpu:
            self.has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
            if self.has_cuda:
                logger.info(f"CUDA available: {cv2.cuda.getCudaEnabledDeviceCount()} devices")
            else:
                logger.warning("GPU requested but CUDA not available, falling back to CPU")
                self.use_gpu = False

        logger.info("ImagePreprocessor initialized")
        logger.info(f"  Libraries: OpenCV {cv2.__version__}, Mahotas {mh.__version__}, scikit-image {skimage.__version__}")

    def preprocess(self, image_path: str, output_dir: str = "preprocessing") -> Dict[str, Any]:
        """
        Full preprocessing pipeline

        Args:
            image_path: Path to input image
            output_dir: Directory for output files

        Returns:
            Dictionary with analysis results and file paths
        """
        logger.info(f"=== Preprocessing: {image_path} ===")

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load image
        image = self._load_image(image_path)
        height, width = image.shape[:2]

        logger.info(f"Image loaded: {width}×{height}, {image.shape[2]} channels")

        # Run all analyses
        results = {}

        # 1. Entropy analysis
        logger.info("Computing entropy map...")
        entropy_map = self._compute_entropy_map(image)
        np.save(output_dir / "entropy_map.npy", entropy_map)
        results['entropy_map'] = str(output_dir / "entropy_map.npy")
        results['global_entropy'] = float(entropy_map.mean())

        # 2. Gradient analysis
        logger.info("Computing gradient map...")
        gradient_map = self._compute_gradient_map(image)
        np.save(output_dir / "gradient_map.npy", gradient_map)
        results['gradient_map'] = str(output_dir / "gradient_map.npy")
        results['mean_gradient'] = float(gradient_map.mean())

        # 3. Texture classification (Mahotas)
        logger.info("Analyzing texture (Haralick features)...")
        texture_map = self._compute_texture_map(image)
        np.save(output_dir / "texture_map.npy", texture_map)
        results['texture_map'] = str(output_dir / "texture_map.npy")
        results['texture_percentage'] = float((texture_map > 0.5).sum() / texture_map.size * 100)

        # 4. SLIC segmentation
        logger.info(f"Computing SLIC segments (n={self.n_segments})...")
        segments = self._compute_slic_segments(image)
        np.save(output_dir / "segments.npy", segments)
        results['segments'] = str(output_dir / "segments.npy")
        results['actual_segments'] = int(segments.max() + 1)

        # 5. Saliency map
        logger.info("Computing saliency map...")
        saliency_map = self._compute_saliency_map(image)
        np.save(output_dir / "saliency_map.npy", saliency_map)
        results['saliency_map'] = str(output_dir / "saliency_map.npy")

        # 6. Distance transform (region centers)
        logger.info("Computing distance transform...")
        distance_map = self._compute_distance_transform(image, segments)
        np.save(output_dir / "distance_map.npy", distance_map)
        results['distance_map'] = str(output_dir / "distance_map.npy")

        # 7. Skeleton/medial axis
        logger.info("Computing skeleton...")
        skeleton = self._compute_skeleton(image, segments)
        np.save(output_dir / "skeleton.npy", skeleton)
        results['skeleton'] = str(output_dir / "skeleton.npy")

        # 8. Generate combined placement probability map
        logger.info("Generating placement probability map...")
        placement_map = self._generate_placement_map(
            entropy_map,
            gradient_map,
            texture_map,
            saliency_map,
            distance_map,
        )
        np.save(output_dir / "placement_map.npy", placement_map)
        results['placement_map'] = str(output_dir / "placement_map.npy")

        # 9. Save metadata
        metadata = {
            "source_image": str(image_path),
            "image_width": int(width),
            "image_height": int(height),
            "preprocessing_version": "1.0.0",
            "library_versions": {
                "opencv": cv2.__version__,
                "mahotas": mh.__version__,
                "scikit-image": skimage.__version__,
            },
            "analysis": results,
            "parameters": {
                "n_segments": self.n_segments,
                "entropy_tile_size": self.entropy_tile_size,
                "gpu_enabled": self.use_gpu,
            },
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Preprocessing complete!")
        logger.info(f"   Output: {output_dir}/")
        logger.info(f"   Placement map: {placement_map.shape}, range [{placement_map.min():.6f}, {placement_map.max():.6f}]")

        return metadata

    def _load_image(self, path: str) -> np.ndarray:
        """Load image in RGB format"""
        image = io.imread(path)

        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image, image, image], axis=-1)

        if image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]

        return image

    def _compute_entropy_map(self, image: np.ndarray) -> np.ndarray:
        """
        Compute local entropy map (tile-based)

        Uses histogram-based entropy in sliding window
        """
        gray = rgb2gray(image)
        height, width = gray.shape
        tile_size = self.entropy_tile_size

        entropy_map = np.zeros((height, width), dtype=np.float32)

        # Tile-based entropy
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)

                tile = gray[y:y_end, x:x_end]

                # Histogram-based entropy
                hist, _ = np.histogram(tile, bins=256, range=(0, 1))
                hist = hist.astype(float) / hist.sum()
                hist = hist[hist > 0]  # Remove zeros

                entropy = -np.sum(hist * np.log2(hist))

                entropy_map[y:y_end, x:x_end] = entropy

        # Normalize to [0, 1]
        if entropy_map.max() > 0:
            entropy_map = entropy_map / entropy_map.max()

        return entropy_map

    def _compute_gradient_map(self, image: np.ndarray) -> np.ndarray:
        """
        Compute gradient magnitude map

        Uses Sobel operator (OpenCV for speed, optional GPU)
        """
        gray = rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8)

        if self.use_gpu and self.has_cuda:
            # GPU path
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(gray_uint8)

            gpu_grad_x = cv2.cuda.createSobelFilter(
                cv2.CV_8U, cv2.CV_16S, 1, 0, ksize=3
            )
            gpu_grad_y = cv2.cuda.createSobelFilter(
                cv2.CV_8U, cv2.CV_16S, 0, 1, ksize=3
            )

            grad_x = gpu_grad_x.apply(gpu_img).download()
            grad_y = gpu_grad_y.apply(gpu_img).download()
        else:
            # CPU path
            grad_x = cv2.Sobel(gray_uint8, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_uint8, cv2.CV_16S, 0, 1, ksize=3)

        # Magnitude
        gradient = np.sqrt(grad_x.astype(float)**2 + grad_y.astype(float)**2)

        # Normalize to [0, 1]
        if gradient.max() > 0:
            gradient = gradient / gradient.max()

        return gradient.astype(np.float32)

    def _compute_texture_map(self, image: np.ndarray) -> np.ndarray:
        """
        Classify texture vs smooth regions using Haralick features

        Uses Mahotas for texture analysis
        """
        gray = rgb2gray(image)
        gray_uint8 = (gray * 255).astype(np.uint8)

        height, width = gray.shape
        texture_map = np.zeros((height, width), dtype=np.float32)

        # Compute Haralick features in tiles
        tile_size = 32  # Larger tiles for texture analysis

        for y in range(0, height - tile_size, tile_size // 2):
            for x in range(0, width - tile_size, tile_size // 2):
                tile = gray_uint8[y:y+tile_size, x:x+tile_size]

                try:
                    # Haralick texture features (13 features × 4 directions)
                    haralick = mh.features.haralick(tile)

                    # Use mean contrast and energy as texture indicators
                    contrast = haralick[:, 1].mean()  # Contrast
                    energy = haralick[:, 8].mean()     # Energy (uniformity)

                    # High contrast + low energy = texture
                    # Low contrast + high energy = smooth
                    texture_score = contrast / (energy + 0.01)

                    texture_map[y:y+tile_size, x:x+tile_size] = texture_score

                except Exception as e:
                    # Tile too uniform, mark as smooth
                    texture_map[y:y+tile_size, x:x+tile_size] = 0.0

        # Normalize
        if texture_map.max() > 0:
            texture_map = texture_map / texture_map.max()

        return texture_map

    def _compute_slic_segments(self, image: np.ndarray) -> np.ndarray:
        """
        SLIC superpixel segmentation

        Perceptually meaningful regions respecting boundaries
        """
        segments = segmentation.slic(
            image,
            n_segments=self.n_segments,
            compactness=10,  # Balance color vs spatial proximity
            sigma=1,          # Gaussian smoothing
            start_label=0,
            channel_axis=-1,
        )

        logger.info(f"  SLIC created {segments.max() + 1} segments")

        return segments

    def _compute_saliency_map(self, image: np.ndarray) -> np.ndarray:
        """
        Visual saliency detection

        Uses spectral residual method (OpenCV)
        """
        # Convert to uint8 for OpenCV
        image_uint8 = (image * 255).astype(np.uint8)

        # OpenCV saliency (spectral residual)
        saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency = saliency_detector.computeSaliency(image_uint8)

        if not success:
            logger.warning("Saliency detection failed, using uniform")
            return np.ones(image.shape[:2], dtype=np.float32)

        # Normalize
        if saliency.max() > 0:
            saliency = saliency / saliency.max()

        return saliency.astype(np.float32)

    def _compute_distance_transform(
        self,
        image: np.ndarray,
        segments: np.ndarray
    ) -> np.ndarray:
        """
        Distance transform for finding region centers

        Uses OpenCV for speed (optional GPU)
        """
        # Create binary masks for each segment
        height, width = image.shape[:2]
        distance_map = np.zeros((height, width), dtype=np.float32)

        n_segments = segments.max() + 1

        for seg_id in range(n_segments):
            mask = (segments == seg_id).astype(np.uint8)

            if self.use_gpu and self.has_cuda:
                # GPU distance transform
                gpu_mask = cv2.cuda_GpuMat()
                gpu_mask.upload(mask)
                gpu_dist = cv2.cuda.createDistanceTransform(cv2.DIST_L2, 3)
                dist = gpu_dist.apply(gpu_mask).download()
            else:
                # CPU distance transform
                dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

            # Keep maximum distance per pixel (across all segments)
            distance_map = np.maximum(distance_map, dist)

        # Normalize
        if distance_map.max() > 0:
            distance_map = distance_map / distance_map.max()

        return distance_map

    def _compute_skeleton(
        self,
        image: np.ndarray,
        segments: np.ndarray
    ) -> np.ndarray:
        """
        Compute skeleton/medial axis

        Structural spine for placement along thin structures
        Uses scikit-image (OpenCV lacks skeleton)
        """
        skeleton_map = np.zeros(image.shape[:2], dtype=np.float32)

        n_segments = segments.max() + 1

        for seg_id in range(n_segments):
            mask = (segments == seg_id)

            # Skip very small segments
            if mask.sum() < 100:
                continue

            # Skeletonize
            skel = morphology.skeletonize(mask)

            skeleton_map[skel] = 1.0

        return skeleton_map

    def _generate_placement_map(
        self,
        entropy_map: np.ndarray,
        gradient_map: np.ndarray,
        texture_map: np.ndarray,
        saliency_map: np.ndarray,
        distance_map: np.ndarray,
    ) -> np.ndarray:
        """
        Generate combined placement probability map

        Combines all analysis signals with empirically tuned weights
        """
        # Weights (tunable)
        w_entropy = 0.3    # Complexity
        w_gradient = 0.25  # Edges
        w_texture = 0.2    # Texture regions
        w_saliency = 0.15  # Visual importance
        w_distance = 0.10  # Region centers

        # Combine
        placement = (
            w_entropy * entropy_map +
            w_gradient * gradient_map +
            w_texture * texture_map +
            w_saliency * saliency_map +
            w_distance * distance_map
        )

        # Normalize to probability distribution (sum = 1.0)
        placement = placement / placement.sum()

        logger.info(f"Placement map: min={placement.min():.6e}, max={placement.max():.6e}, sum={placement.sum():.6f}")

        return placement.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(
        description="LGI Image Preprocessing - Generate placement maps and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("image_path", help="Input image file")
    parser.add_argument("--n-segments", type=int, default=500,
                       help="SLIC superpixel count (default: 500)")
    parser.add_argument("--entropy-tile-size", type=int, default=16,
                       help="Entropy tile size (default: 16)")
    parser.add_argument("--use-gpu", action="store_true",
                       help="Enable GPU acceleration (requires CUDA)")
    parser.add_argument("--output-dir", default="preprocessing",
                       help="Output directory (default: preprocessing/)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create preprocessor
    preprocessor = ImagePreprocessor(
        n_segments=args.n_segments,
        entropy_tile_size=args.entropy_tile_size,
        use_gpu=args.use_gpu,
        verbose=args.verbose,
    )

    # Run preprocessing
    results = preprocessor.preprocess(args.image_path, args.output_dir)

    # Print summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Image: {args.image_path}")
    print(f"Size: {results.get('image_width')}×{results.get('image_height')}")
    print(f"\nAnalysis:")
    print(f"  Global entropy: {results.get('global_entropy', 0):.4f}")
    print(f"  Mean gradient: {results.get('mean_gradient', 0):.4f}")
    print(f"  Texture: {results.get('texture_percentage', 0):.1f}%")
    print(f"  SLIC segments: {results.get('actual_segments', 0)}")
    print(f"\nOutputs:")
    print(f"  {args.output_dir}/")
    print(f"  - placement_map.npy (HxW probability map)")
    print(f"  - entropy_map.npy, gradient_map.npy, texture_map.npy")
    print(f"  - saliency_map.npy, distance_map.npy, skeleton.npy")
    print(f"  - segments.npy (SLIC labels)")
    print(f"  - metadata.json")
    print("="*70)

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    main()
