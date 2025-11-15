# Comprehensive Python Tools, Scripts & Utilities Catalog
## Gaussian Image Codec Repository

---

## SECTION 1: PRIMARY PREPROCESSING TOOLS
### Modern Preprocessing Pipeline (lgi-rs/tools/)

These are the main preprocessing scripts for the current Rust-based encoder.

---

#### 1.1: preprocess_image.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image.py`

**Purpose:** Comprehensive image preprocessing pipeline that analyzes images and generates placement probability maps for intelligent Gaussian positioning. Main preprocessing script for Image-GS encoding.

**Key Features:**
- Tile-based entropy analysis (histogram-based)
- Sobel gradient magnitude mapping
- Haralick texture feature analysis (Mahotas)
- SLIC superpixel segmentation
- Visual saliency detection (spectral residual)
- Distance transforms for region centers
- Medial axis skeleton computation
- Combined placement probability map generation

**Input:** Single image file (PNG, JPG, etc.)

**Output Files Generated:**
- `entropy_map.npy` - Per-pixel complexity scores
- `gradient_map.npy` - Edge strength magnitudes
- `texture_map.npy` - Texture classification
- `saliency_map.npy` - Visual importance weights
- `segments.npy` - SLIC superpixel labels
- `distance_map.npy` - Distance transform (region centers)
- `skeleton.npy` - Medial axis skeleton
- `placement_map.npy` - Combined probability map [0-1] normalized
- `metadata.json` - Analysis statistics and parameters

**Key Functions:**
- `__init__(n_segments, entropy_tile_size, use_gpu, verbose)` - Initialize preprocessor
- `preprocess(image_path, output_dir)` - Full preprocessing pipeline
- `_load_image(path)` - RGB image loading
- `_compute_entropy_map(image)` - Histogram-based entropy (tile size 16 default)
- `_compute_gradient_map(image)` - Sobel operator (CPU/GPU)
- `_compute_texture_map(image)` - Haralick features (32x32 tiles)
- `_compute_slic_segments(image)` - SLIC with 500 segments default
- `_compute_saliency_map(image)` - Spectral residual method
- `_compute_distance_transform(image, segments)` - Per-segment distance transform
- `_compute_skeleton(image, segments)` - Skeletonization via scikit-image
- `_generate_placement_map(entropy, gradient, texture, saliency, distance)` - Weighted combination

**Configuration Parameters:**
- `n_segments`: SLIC superpixel count (default: 500)
- `entropy_tile_size`: Tile size for entropy (default: 16)
- `use_gpu`: Enable CUDA acceleration (boolean)
- `output_dir`: Output directory path
- `verbose`: Detailed logging

**Weights for Placement Map:**
- Entropy: 0.3 (local complexity)
- Gradient: 0.25 (edges)
- Texture: 0.2 (texture regions)
- Saliency: 0.15 (visual importance)
- Distance: 0.10 (region centers)

**Dependencies:**
- opencv-contrib-python 4.12.0.88
- mahotas 1.4.18 (Haralick, LBP, TAS)
- scikit-image 0.25.2 (SLIC, skeletonization)
- numpy, scipy, PIL

**GPU Support:** CUDA optional via cv2.cuda_*

**Command Line Usage:**
```bash
python tools/preprocess_image.py <image_path> [--n-segments INT] [--entropy-tile-size INT] [--use-gpu] [--output-dir PATH] [--verbose]
```

---

#### 1.2: preprocess_image_v2.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image_v2.py`

**Purpose:** Production-ready preprocessing v2. Streamlined version with improved file handling modes and metadata organization.

**Key Differences from v1:**
- Output naming convention: `{image_stem}.json` + `{image_stem}_{map_type}.npy`
- File handling modes: overwrite, skip, update, prompt
- SHA256 checksums for images
- Compact metadata structure
- Proper file path resolution
- Parameter change detection

**Output Files:**
- `{image_stem}.json` - Main metadata
- `{image_stem}_placement.npy` - Placement probability map
- `{image_stem}_entropy.npy`, `_gradient.npy`, `_texture.npy`
- `{image_stem}_saliency.npy`, `_distance.npy`, `_skeleton.npy`
- `{image_stem}_segments.npy` - SLIC labels

**Metadata Structure:**
- `source`: filename, path, format, dimensions, channels, file size, SHA256 checksum
- `preprocessing`: version, timestamp, libraries, parameters, GPU usage
- `analysis_maps`: mapping of map names to filenames
- `statistics`: global entropy, mean gradient, texture %, segment count

**Key Functions:**
- Same as v1 but with simplified names
- `_compute_entropy(image, tile_size)`
- `_compute_gradient(image)`
- `_compute_texture(image)`
- `_compute_saliency(image)`
- `_compute_distance(segments)`
- `_compute_skeleton(segments)` - Limited to 500 segments for speed
- `_generate_placement_map(entropy, gradient, texture, saliency, distance, params)`

**Configuration Parameters:**
- `n_segments`: SLIC superpixel count (default: 500)
- `entropy_tile_size`: Tile size for entropy (default: 16)
- `mode`: File handling mode - 'overwrite' (default), 'skip', 'update', 'prompt'
- `use_gpu`: Enable CUDA acceleration

**Command Line Usage:**
```bash
python tools/preprocess_image_v2.py <image_path> [--n-segments INT] [--entropy-tile-size INT] [--mode {overwrite|skip|update|prompt}] [--use-gpu]
```

**Special Features:**
- Parameter change detection: skips reprocessing if parameters unchanged
- Safe mode: asks user before overwriting
- Skip mode: uses existing JSON if available
- Automatic output directory selection (same as input image)

---

#### 1.3: slic_preprocess.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/slic_preprocess.py`

**Purpose:** Generate Gaussian initialization parameters from SLIC superpixels. Used for initializing Gaussian positions, scales, and rotations directly from superpixel analysis.

**Key Concept:** Each superpixel becomes an initial Gaussian with derived parameters:
- Position: superpixel centroid
- Scale: covariance eigenvalues in pixel space
- Rotation: eigenvector orientation
- Color: mean superpixel color

**Input:** 
- Image file path
- Number of SLIC segments (n_segments parameter)
- Optional output JSON path

**Output:**
- JSON file with Gaussian initialization data
- Structure:
  ```json
  {
    "source_image": "path/to/image.png",
    "image_width": 512,
    "image_height": 512,
    "n_segments_requested": 500,
    "n_gaussians": 487,
    "gaussians": [
      {
        "position": [x_normalized, y_normalized],
        "scale": [sx_normalized, sy_normalized],
        "rotation": angle_radians,
        "color": [r, g, b],
        "segment_id": integer,
        "pixel_count": count
      },
      ...
    ]
  }
  ```

**Key Functions:**
- `generate_slic_gaussians(image_path, n_segments, output_path)` - Main function
  - Loads image (handles grayscale/RGB/RGBA)
  - Runs SLIC with compactness=10, sigma=1
  - For each segment:
    - Computes centroid as position
    - Extracts mean color
    - Computes covariance matrix
    - Eigendecomposition for scales and rotation
    - Normalizes scales to [0,1]
  - Returns list of Gaussian parameters

**Parameters:**
- `image_path`: Path to input image
- `n_segments`: SLIC superpixel count (default: 100)
- `output_path`: JSON output path (default: "slic_init.json")

**Statistics Provided:**
- Mean pixels per segment
- Min/max pixel counts
- Total number of Gaussians created

**Command Line Usage:**
```bash
python slic_preprocess.py <image_path> <n_segments> [output_json]
python slic_preprocess.py kodim02.png 500 slic_init.json
```

**Dependencies:**
- scikit-image (slic, imread, rgb2lab)
- numpy

---

## SECTION 2: IMAGE PROCESSING UTILITIES
### Legacy Image-GS Package (lgi-legacy/image-gs/utils/)

---

#### 2.1: image_utils.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/image_utils.py`

**Purpose:** Image loading, saving, format conversion, visualization utilities for Gaussian rendering.

**Key Functions:**

**Metrics & Quality:**
- `get_psnr(image1, image2, max_value=1.0)` - Peak Signal-to-Noise Ratio calculation
  - Returns inf if MSE < 1e-7
  - Input: torch tensors [0,1] range

**Coordinate Systems:**
- `get_grid(h, w, x_lim=[0,1], y_lim=[0,1])` - Create coordinate grids
  - Returns mesh grid of normalized coordinates
  - Used for Gaussian evaluation at pixel positions

**Image Operations:**
- `compute_image_gradients(image)` - Compute Sobel gradients
  - Returns (gy, gx) - Sobel gradients in y and x directions
  - Uses scipy.ndimage.sobel
  - Input: numpy array [C, H, W]
  - Output: normalized gradients

- `load_images(load_path, downsample_ratio=None, gamma=None)` - Load image dataset
  - Supports single file or directory
  - Formats: JPEG, JPG, PNG
  - Modes: L (grayscale), RGB, RGBA
  - Returns: (concatenated images, channel list, filename list)
  - Handles downsampling, gamma correction
  - Output: [C, H, W] format, float32, [0,1] range

- `to_output_format(image, gamma)` - Convert internal to output format
  - Handles torch/numpy conversion
  - Clamps to [0,1]
  - Applies inverse gamma
  - Converts to uint8
  - Handles shape [H,W] or [C,H,W]

- `save_image(image, save_path, gamma=None, zoom=None)` - Save processed image
  - Uses to_output_format
  - Optional zoom/resize
  - PIL backend

- `separate_image_channels(images, input_channels)` - Split concatenated channels
  - Splits [C, H, W] into multiple [Ci, H, W] arrays per image

**Visualization:**
- `visualize_gaussians(filepath, xy, scale, rot, feat, img_h, img_w, input_channels, alpha=0.8, gamma=None)` - Render Gaussians as ellipses
  - Draws elliptical disks for each Gaussian
  - Uses matplotlib
  - Parameters: positions (xy), scales, rotations, features (colors)
  - GAUSSIAN_ZOOM = 5x magnification
  - GAUSSIAN_COLOR = green (#80ed99)
  - Saves PNG files (one per channel)

- `visualize_added_gaussians(filepath, images, old_xy, new_xy, input_channels, size=500, every_n=5, alpha=0.8, gamma=None)` - Track Gaussian additions during optimization
  - Overlays old positions (red) vs new positions (green) on image
  - Samples every_n Gaussians for clarity
  - Saves visualization PNGs

**Constants:**
- ALLOWED_IMAGE_FILE_FORMATS: [".jpeg", ".jpg", ".png"]
- ALLOWED_IMAGE_TYPES: {"RGB": 3, "RGBA": 3, "L": 1}
- PLOT_DPI: 72.0
- GAUSSIAN_ZOOM: 5
- GAUSSIAN_COLOR: "#80ed99" (green)

**Font Configuration:**
- Uses Linux Libertine font (assets/fonts/linux_libertine/LinLibertine_R.ttf)
- matplotlib.rcParams customized for publication quality

**Gamma Correction:**
- Linear to sRGB: `power(image, 1/gamma)`
- sRGB to linear: `power(image, gamma)`

---

#### 2.2: quantization_utils.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/quantization_utils.py`

**Purpose:** Bit precision control for Gaussian parameters using straight-through estimators.

**Key Functions:**
- `ste_quantize(x: torch.Tensor, num_bits: int = 16) -> torch.Tensor` - Quantize with gradients
  - Implements straight-through estimator (STE)
  - Quantization forward: round to integer, clamp to [0, 2^bits - 1]
  - Gradient restoration backward: preserves original gradients
  - Allows training with quantized parameters
  - Default: 16-bit precision
  - Range: 4-32 bits supported

**Reference:** https://arxiv.org/abs/1308.3432 (Bengio et al., 2013)

**Application:** Used in model.py for controlling bit precision of:
- Position: pos_bits
- Scale: scale_bits
- Rotation: rot_bits
- Features: feat_bits

---

#### 2.3: saliency_utils.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/saliency_utils.py`

**Purpose:** Visual saliency detection using EMLNet (Eye Movement Loss NET).

**Key Functions:**
- `get_smap(image, path, filter_size=15) -> np.ndarray` - Compute saliency map
  - Uses EMLNet with ResNet50 backbone
  - Dual-stream architecture (ImageNet + Places)
  - Input: RGB torch tensor on CUDA
  - Output: saliency map [0,1] as uint8

**EMLNet Architecture:**
- `imagenet_model`: ResNet50 pretrained on ImageNet
- `places_model`: ResNet50 pretrained on Places
- `decoder_model`: Upsampling decoder
- Output resolution: 480x640
- Upsampled back to input resolution

**Processing Steps:**
1. Resize image to 480x640
2. Extract features from both models
3. Decode with both feature streams
4. Resize back to original dimensions
5. Post-process: Gaussian filter (sigma=15), normalize to [0,1]

**Post-processing:**
- Gaussian smoothing (filter_size parameter)
- Min-max normalization

**Model Files Required:**
- `{path}/emlnet/res_imagenet.pth`
- `{path}/emlnet/res_places.pth`
- `{path}/emlnet/res_decoder.pth`

**References:**
- https://arxiv.org/abs/1805.01047
- https://github.com/SenJia/EML-NET-Saliency

---

#### 2.4: misc_utils.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/misc_utils.py`

**Purpose:** Miscellaneous utilities for configuration management, random seeds, and checkpoints.

**Key Functions:**

**Directory Management:**
- `clean_dir(path)` - Remove directory if exists (recursive)
- `get_latest_ckpt_step(load_path)` - Find latest checkpoint
  - Parses checkpoint filenames: `{name}-{step}.pt`
  - Returns highest step number (-1 if none found)

**Random Seeding:**
- `set_random_seed(seed)` - Reproducible results
  - Sets: random, numpy, torch, torch.cuda seeds
  - Disables cuDNN benchmarking
  - Enables deterministic mode

**Configuration:**
- `load_cfg(cfg_path: str, parser: ArgumentParser) -> ArgumentParser` - Load YAML config
  - Reads YAML file
  - Converts values to argparse arguments
  - Validates no None values
  - Boolean flags: `--flag` action="store_true"
  - Other types: `--param type(value)`

- `save_cfg(path: str, args, mode="w")` - Save config
  - Outputs header comment
  - YAML dump of args

---

#### 2.5: flip.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/flip.py`

**Purpose:** FLIP loss implementation - perceptual error metric for image quality (NVIDIA).

**Classes:**

**HDRFLIPLoss (nn.Module)**
- Computes HDR-FLIP error between HDR images
- Methods:
  - `forward(test, reference, pixels_per_degree, tone_mapper, start_exposure, stop_exposure)` - Compute error map
  - Supports tone mappers: "reinhard", "hable", "aces" (default)
  - Handles multiple exposures automatically
  - Returns per-pixel error tensor [0,1]

**LDRFLIPLoss (nn.Module)**
- Computes LDR-FLIP error between LDR images
- Input: [N,C,H,W] in sRGB [0,1]
- Method:
  - `forward(test, reference, pixels_per_degree)` - Compute error

**Color Pipelines:**
1. Spatial filtering (frequency-dependent contrast sensitivity)
2. Color space transforms (sRGB -> YCxCz -> Lab)
3. Perceptual uniformity (Hunt adjustment)
4. Error redistribution (color metric)
5. Feature detection (edge + point)

**Key Algorithms:**
- Color space transforms: sRGB, linear RGB, XYZ, YCxCz, Lab
- Spatial CSF: Achromatic (A), Red-Green (RG), Blue-Yellow (BY)
- Tone mapping: Reinhard, Hable, ACES
- Feature detection: Gaussian derivatives (edges and points)
- HyAB color distance metric
- Hunt color adaptation

**Parameters:**
- `qc = 0.7` - Color pipeline exponent
- `qf = 0.5` - Feature pipeline exponent
- `pc = 0.4` - Color redistribution multiplier
- `pt = 0.95` - Target redistribution threshold

**References:**
- FLIP: A Difference Evaluator for Alternating Images (High Performance Graphics 2020)
- HDR-FLIP: Visualizing Errors in Rendered HDR (Eurographics 2021)
- Ray Tracing Gems II Chapter 2021
- Authors: Andersson, Nilsson, Akenine-Moller, Oskarsson, Astrom, Fairchild

**NVIDIA Copyright:** BSD-3-Clause License (2020-2024)

---

## SECTION 3: TRAINING & OPTIMIZATION MODULES
### Main Model Files

---

#### 3.1: main.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/main.py`

**Purpose:** Entry point for Image-GS training and rendering pipeline.

**Key Functions:**
- `get_gaussian_cfg(args)` - Build Gaussian configuration string
  - Format: `num-{count}_scale-{scale}_bits-{pos}-{scale}-{rot}-{feat}_top-{k}_mode-ratio`
  - Validates bit precision: 4-32 bits per parameter

- `get_log_dir(args)` - Construct experiment logging directory
  - Path: `{log_root}/{exp_name}/{gaussian_cfg}_{loss_cfg}_{optional_flags}`
  - Includes: loss weights, downsampling, learning rate schedule, progressive optimization flags

- `main(args)` - Main training/rendering loop
  - Initializes GaussianSplatting2D model
  - Calls model.optimize() for training
  - Calls model.render() for evaluation

**Workflow:**
1. Load YAML config (cfgs/default.yaml)
2. Parse command-line arguments
3. Create model with configuration
4. Either train or render based on args.eval flag

---

#### 3.2: model.py (Partial)
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/model.py`

**Purpose:** Core GaussianSplatting2D model implementation.

**Class: GaussianSplatting2D(nn.Module)**

**Initialization Methods:**
- `_init_logging(args)` - Set up logging, directories, checkpoints
- `_init_target(args)` - Load target image(s)
- `_init_bit_precision(args)` - Set quantization bits
- `_init_gaussians(args)` - Initialize Gaussian parameters
- `_init_loss(args)` - Configure loss functions
- `_init_optimization(args)` - Set up optimizer and learning rate schedule
- `_init_pos_scale_feat(args)` - Initialize positions, scales, features

**Key Attributes:**
- Log directory structure: checkpoints/, train/, eval/
- Checkpoint saving: every N steps
- Evaluation: every N steps
- Image saving: every N steps
- Gaussian visualization: optional per-iteration ellipse rendering

**Training/Inference:**
- `optimize()` - Main training loop
- `render()` - Render images from Gaussians

**Imports Used:**
- fused_ssim: Fast SSIM computation
- lpips: LPIPS perceptual loss
- pytorch_msssim: Multi-scale SSIM
- gsplat: 2D Gaussian projection and rasterization

---

## SECTION 4: GAUSSIAN SPLATTING KERNELS
### Core Rendering Primitives

---

#### 4.1: project_gaussians_2d_scale_rot.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/gsplat/gsplat/project_gaussians_2d_scale_rot.py`

**Purpose:** Project 2D Gaussians to image plane with scale and rotation.

**Function:**
- `project_gaussians_2d_scale_rot(means2d, scales2d, rotation, img_height, img_width, tile_bounds)` - Project parameters
  - Input: 2D positions, 2D scales, rotations
  - Output: 2D coordinates, radii, conics (2D covariance), tiles hit
  - CUDA-backed forward/backward passes

**Autograd Function: _ProjectGaussians2dScaleRot**
- Forward: CUDA kernel for projection (via gsplat.cuda._C)
- Backward: Computes gradients w.r.t. position, scale, rotation
- Returns: (xys, radii, conics, num_tiles_hit)

**Integration:** Used in rendering pipeline for Gaussian parameterization and projection.

---

#### 4.2: rasterize_sum.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/gsplat/gsplat/rasterize_sum.py`

**Purpose:** Rasterize 2D Gaussians with tiling for efficient rendering.

**Function:**
- `rasterize_gaussians_sum(xys, radii, conics, num_tiles_hit, colors, img_height, img_width, BLOCK_H=16, BLOCK_W=16, topk_norm=False)` - Render image
  - Input: Gaussian properties, image dimensions, block size
  - Output: Rendered image tensor
  - Supports standard and top-k normalization modes

**Autograd Function: _RasterizeGaussiansSum**
- Forward:
  - Bins Gaussians into tiles
  - Sorts Gaussians by tile and depth
  - Rasterizes with optional top-k normalization
  - CUDA kernel execution
  
- Backward:
  - Computes gradients for all inputs
  - Returns: (v_xy, None, v_conic, None, v_colors, None, None, None, None, None)

**Tiling Strategy:**
- Default block size: 16x16 pixels
- Computes tile bounds from image dimensions
- Creates unique intersection IDs for sorting
- Efficient batch processing

**Normalization Modes:**
- Standard: Direct Gaussian weighted sum
- Top-k: Limit to top-k Gaussians per pixel (controlled by topk_norm flag)

---

#### 4.3: rasterize_no_tiles.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/gsplat/gsplat/rasterize_no_tiles.py`

**Purpose:** Simple Gaussian rasterization without tiling (for comparison/testing).

**Functions:**

**1. rasterize_gaussians_no_tiles()**
- Input: xys, conics, colors, image dimensions
- Output: Rendered image
- No tiling overhead
- Simpler but potentially slower for many Gaussians

**2. rasterize_gaussians_simple()**
- Direct rasterization with scale/rotation
- Input: positions, scales, rotations, features
- Less optimized, used for debugging

**Autograd Functions:**
- _RasterizeGaussiansNoTiles: Standard rasterization
  - Stores: xys, conics, colors, pixel_topk (for backward)
  - Returns: (v_xy, v_conic, v_colors, None, None)

- _RasterizeGaussiansSimple: Scale/rotation variant
  - Stores: xys, scale, rot, feat, pixel_topk
  - Returns: (v_xy, v_scale, v_rot, v_feat, None, None)

---

#### 4.4: utils.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/gsplat/gsplat/utils.py`

**Purpose:** Utility functions for Gaussian binning, sorting, and covariance handling.

**Key Functions:**

**Binning & Sorting:**
- `map_gaussian_to_intersects(num_points, num_intersects, xys, radii, cum_tiles_hit, tile_bounds)` - Map Gaussians to tile intersections
  - Returns: (isect_ids, gaussian_ids)
  - Not differentiable

- `get_tile_bin_edges(num_intersects, num_tiles, isect_ids_sorted)` - Create tile bins
  - Returns: tile_bins with (lower, upper) ranges per tile
  - Not differentiable

- `bin_and_sort_gaussians(num_points, num_intersects, xys, radii, cum_tiles_hit, tile_bounds)` - Complete binning/sorting
  - Returns: (isect_ids_unsorted, gaussian_ids_unsorted, isect_ids_sorted, gaussian_ids_sorted, tile_bins)

**Covariance:**
- `compute_cov2d_bounds(cov2d)` - Compute 2D covariance bounds
  - Input: [batch, 3] upper triangular values
  - Output: (conics, radii) for efficient Gaussian evaluation

**Cumulative Operations:**
- `compute_cumulative_intersects(num_tiles_hit)` - Compute cumulative tile hits
  - Returns: (num_intersects, cum_tiles_hit)
  - Uses torch.cumsum for GPU efficiency

**Data Flow:**
1. Project Gaussians to 2D (project_gaussians_2d_scale_rot)
2. Compute covariance bounds (compute_cov2d_bounds)
3. Bin and sort Gaussians (bin_and_sort_gaussians)
4. Rasterize with sorted order (rasterize_gaussians_sum)

---

#### 4.5: __init__.py (gsplat package)
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/gsplat/gsplat/__init__.py`

**Purpose:** Package initialization and API exports.

**Exported Functions:**
- project_gaussians_2d_scale_rot
- rasterize_gaussians_sum
- rasterize_gaussians_no_tiles
- bin_and_sort_gaussians
- compute_cumulative_intersects
- compute_cov2d_bounds
- get_tile_bin_edges
- map_gaussian_to_intersects

**Deprecated Classes (Backward Compatibility):**
- MapGaussiansToIntersects
- ComputeCumulativeIntersects
- ComputeCov2dBounds
- GetTileBinEdges
- BinAndSortGaussians
- ProjectGaussians2dScaleRot
- RasterizeGaussiansSum

---

## SECTION 5: PERFORMANCE & QUALITY METRICS
### SSIM & Loss Functions

---

#### 5.1: fused_ssim/__init__.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-tools/fused-ssim/fused_ssim/__init__.py`

**Purpose:** Fast SSIM computation via custom CUDA kernels.

**Key Class: FusedSSIMMap(torch.autograd.Function)**
- Custom autograd function for SSIM
- Forward: Calls CUDA kernel `fusedssim()`
- Backward: Calls CUDA kernel `fusedssim_backward()`
- Supports two padding modes:
  - "same": Full SSIM map
  - "valid": Trimmed 5 pixels on each side

**Main Function:**
- `fused_ssim(img1, img2, padding="same", train=True)` - Compute SSIM loss
  - Constant stabilization: C1=0.01^2, C2=0.03^2
  - Returns: scalar SSIM value (mean of map)
  - Training mode: computes gradients
  - Inference mode: no gradients

**Dependencies:**
- fused_ssim_cuda: Custom CUDA extension module
  - `fusedssim(C1, C2, img1, img2, train)` - Forward kernel
  - `fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)` - Backward kernel

**SSIM Formula:**
- SSIM = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1^2 + mu2^2 + C1)*(sigma1^2 + sigma2^2 + C2))

**Outputs (Forward):**
- ssim_map: Per-pixel SSIM values
- dm_dmu1, dm_dsigma1_sq, dm_dsigma12: Intermediate derivatives for backward

---

#### 5.2: fused-ssim tests
**Test Files:**

**test.py** - Correctness & benchmark tests
- Path: `/home/user/lamcogaussianimage/packages/lgi-tools/fused-ssim/tests/test.py`
- Compares against:
  - Reference SSIM (standard implementation)
  - pytorch_msssim (pytorch-mssim library)
- Tests:
  - Forward correctness (assert torch.isclose)
  - Backward correctness (gradient matching)
  - Padding modes: "same" vs "valid"
  - Benchmarks: Forward, backward, inference time

**genplot.py** - Performance benchmarking
- Path: `/home/user/lamcogaussianimage/packages/lgi-tools/fused-ssim/tests/genplot.py`
- Benchmark script:
  - Varies image sizes: 50x50 to 1500x1500
  - 5 channels, 5 batch samples
  - Compares pytorch_mssim vs fused_ssim
  - Generates plots: training_time.png, inference_time.png
  - Output to: ../images/

**train_image.py** - Optimization example
- Path: `/home/user/lamcogaussianimage/packages/lgi-tools/fused-ssim/tests/train_image.py`
- Demonstrates SSIM-based optimization:
  - Loads albert.jpg from ../images/
  - Initializes random predicted image
  - Optimizes with Adam to maximize SSIM
  - Target: SSIM > 0.9999
  - Saves: predicted.jpg

---

## SECTION 6: LEGACY & TESTING CODE
### Simplified CPU Implementations

---

#### 6.1: gaussian_2d_cpu.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs-cpu/gaussian_2d_cpu.py`

**Purpose:** CPU-compatible simplified Image-GS implementation for testing and education.

**Classes:**

**Gaussian2D:**
- Represents single 2D Gaussian
- Attributes: position (x,y), scale (sx,sy), rotation angle, color (r,g,b), opacity

**ImageGS(nn.Module):**
- Simplified Image-GS model
- Parameters:
  - positions: [num_gaussians, 2]
  - scales: [num_gaussians, 2]
  - rotations: [num_gaussians]
  - colors: [num_gaussians, 3]
  - opacities: [num_gaussians]

**Key Methods:**
- `compute_gaussian_2d(x, y, pos, scale, rotation)` - Evaluate Gaussian at point
  - Applies rotation matrix
  - Exponential decay in local coords
  - Returns value [0,1]

- `render(resolution)` - Render full image
  - Creates coordinate grids
  - Evaluates all Gaussians at all pixels
  - Blends with opacity
  - Returns [C, H, W] image tensor

**Features:**
- Pure PyTorch (no CUDA required)
- Educational simplicity
- Suitable for CPU-only environments

---

#### 6.2: test_gaussian.py
**Full Path:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs-cpu/test_gaussian.py`

**Purpose:** Test suite for CPU Gaussian implementation.

**Test Functions:**

**create_test_image():**
- Generates synthetic test image (256x256x3)
- Gradients: red (horizontal), green (vertical), blue (constant)
- Shapes: white circle, colored squares
- Used for validation testing

**test_basic_reconstruction():**
- Tests image reconstruction with varying Gaussian counts
- Tests: 100, 500, 1000, 2000 Gaussians
- Trains for 1000 iterations per configuration
- Outputs:
  - comparison.png: Grid of results
  - rendered_{N}.png: Individual renders
  - test_target.png: Ground truth

**test_progressive_optimization():**
- Tests gradual Gaussian addition during training
- Stages:
  1. 50 Gaussians, 500 iterations
  2. +50 (100 total), 500 iterations
  3. +100 (200 total), 500 iterations
  4. +300 (500 total), 1000 iterations
- Outputs:
  - progressive_optimization.png: Multi-stage results
  - final_progressive.png: Final render
  - final_gaussians.png: Gaussian visualization

**Dependencies:**
- torch, numpy, PIL, matplotlib
- Local imports: gaussian_2d_cpu, ImageGSTrainer, utility functions

---

## SECTION 7: CONFIGURATION & INITIALIZATION
### Setup Files

---

#### 7.1: setup.py files
**Locations:**
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/gsplat/setup.py`
- `/home/user/lamcogaussianimage/packages/lgi-tools/fused-ssim/setup.py`

**Purpose:** Python package build and distribution configuration.

**gsplat setup.py:**
- Builds C++ CUDA extensions for Gaussian operations
- Compiles custom CUDA kernels
- Creates Python bindings via pybind11

**fused-ssim setup.py:**
- Builds CUDA extension for SSIM computation
- Custom SSIM forward/backward kernels

---

## SECTION 8: INTEGRATION SUMMARY

### Preprocessing Pipeline Flow:
```
Input Image
    ↓
preprocess_image_v2.py (or v1)
    ├→ Entropy Map (tile-based histogram)
    ├→ Gradient Map (Sobel)
    ├→ Texture Map (Haralick features)
    ├→ Saliency Map (spectral residual)
    ├→ SLIC Segments
    ├→ Distance Transform
    └→ Skeleton Map
        ↓
Combined Placement Map
    ↓
Output: .npy files + metadata.json
```

### Optional SLIC Initialization:
```
Input Image
    ↓
slic_preprocess.py
    ├→ SLIC Segmentation (500 superpixels)
    ├→ Per-segment: Centroid, Covariance, Color
    ├→ Eigendecomposition (scales, rotation)
    └→ Normalization
        ↓
Output: slic_init.json
    └→ Gaussian positions, scales, rotations, colors
```

### Training Pipeline:
```
Preprocessed Image + Placement Map
    ↓
main.py (entry point)
    ↓
GaussianSplatting2D (model.py)
    ├→ Initialize Gaussians (slic_init.json or random)
    ├→ Forward pass:
    │   ├→ project_gaussians_2d_scale_rot (project.py)
    │   ├→ rasterize_gaussians_sum (rasterize_sum.py)
    │   └→ render image
    ├→ Compute loss:
    │   ├→ L1 loss
    │   ├→ SSIM loss (fused_ssim)
    │   ├→ LPIPS loss
    │   └→ Optional: FLIP loss
    ├→ Backpropagation (with ste_quantize)
    ├→ Optimizer step (learning rate schedule)
    └→ Checkpoints + Visualization
        ↓
Output: Trained model, rendered images, metrics
```

### Rust Integration:
- All .py files preprocess images for Rust encoder
- Placement maps guide Gaussian positioning
- SLIC initialization provides starting parameters
- Output formats (.npy, .json) are Rust-compatible

---

## SECTION 9: DEPENDENCY SUMMARY

### Direct Python Dependencies:
**Core:**
- torch, torchvision
- numpy, scipy
- scikit-image
- OpenCV (opencv-contrib-python)
- Mahotas (Haralick features)
- PIL/Pillow
- matplotlib
- PyYAML

**Custom Extensions:**
- gsplat.cuda (custom C++/CUDA bindings)
- fused_ssim_cuda (custom CUDA kernels)
- lpips (perceptual loss)
- pytorch_msssim (multi-scale SSIM)

### Optional:**
- CUDA toolkit (for GPU acceleration)
- cuDNN (for CUDA optimization)

---

## SECTION 10: KEY ALGORITHMS & METHODS

### Preprocessing:
1. **Entropy**: Histogram-based Shannon entropy in sliding tiles
2. **Gradient**: Sobel operator (with optional GPU acceleration)
3. **Texture**: Haralick features (contrast & energy)
4. **Saliency**: Spectral residual method or EMLNet
5. **Segmentation**: SLIC with color+spatial compactness
6. **Distance**: Euclidean distance transform per segment
7. **Skeleton**: Morphological skeletonization

### Placement Map Combination:
- Weighted linear combination: 0.3*entropy + 0.25*gradient + 0.2*texture + 0.15*saliency + 0.10*distance
- Normalized to probability distribution (sum=1)

### Gaussian Rendering:
1. **Projection**: 2D Gaussian projection with scale & rotation
2. **Tiling**: Bin Gaussians into 16x16 tile grid
3. **Sorting**: Depth sort per tile via intersection IDs
4. **Rasterization**: Efficient tile-based rendering
5. **Blending**: Alpha composition with opacity

### Quality Metrics:
1. **PSNR**: Peak Signal-to-Noise Ratio
2. **SSIM**: Structural Similarity (fused CUDA kernel)
3. **LPIPS**: Learned Perceptual Image Patch Similarity
4. **MS-SSIM**: Multi-scale SSIM
5. **FLIP**: Perceptual error metric (HDR + LDR)

---

## FILE LOCATIONS REFERENCE

**Preprocessing Tools:**
- `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image.py`
- `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image_v2.py`
- `/home/user/lamcogaussianimage/packages/lgi-rs/tools/slic_preprocess.py`

**Utilities:**
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/image_utils.py`
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/quantization_utils.py`
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/saliency_utils.py`
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/misc_utils.py`
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/flip.py`

**Training & Model:**
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/main.py`
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/model.py`

**Gaussian Splatting:**
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/gsplat/gsplat/project_gaussians_2d_scale_rot.py`
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/gsplat/gsplat/rasterize_sum.py`
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/gsplat/gsplat/rasterize_no_tiles.py`
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/gsplat/gsplat/utils.py`
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/gsplat/gsplat/__init__.py`

**SSIM & Loss:**
- `/home/user/lamcogaussianimage/packages/lgi-tools/fused-ssim/fused_ssim/__init__.py`
- `/home/user/lamcogaussianimage/packages/lgi-tools/fused-ssim/tests/test.py`
- `/home/user/lamcogaussianimage/packages/lgi-tools/fused-ssim/tests/genplot.py`
- `/home/user/lamcogaussianimage/packages/lgi-tools/fused-ssim/tests/train_image.py`

**CPU Testing:**
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs-cpu/gaussian_2d_cpu.py`
- `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs-cpu/test_gaussian.py`

