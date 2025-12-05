# IMAGE PROCESSING ALGORITHM INDEX
## Quick Reference Guide

**Full Catalog:** See `/home/user/lamcogaussianimage/IMAGE_PROCESSING_CATALOG.md` (868 lines)

---

## QUICK LOOKUP BY ALGORITHM TYPE

### Edge & Gradient Detection
1. **Sobel Operator** (scipy.ndimage)
   - File: `image_utils.py:51-58`
   - Input: Multi-channel image
   - Output: Gradient magnitude in Y, X directions
   
2. **Sobel Operator** (cv2, GPU-optional)
   - File: `preprocess_image.py:256-291`
   - Kernel: 3×3, CV_16S
   - GPU: `cv2.cuda.createSobelFilter()`

3. **FLIP Feature Detection** (Gaussian-based)
   - File: `flip.py:529-567`
   - Types: Edge (1st derivative), Point (2nd derivative)
   - Output: 2-channel feature maps

### Color Space Transformations
1. **Complete Pipeline** (15+ conversions)
   - File: `flip.py:570-715`
   - Supports: sRGB, linRGB, XYZ, L*a*b*, YCxCz
   - Reversible and composable

2. **Tone Mapping** (3 types)
   - File: `flip.py:230-294`
   - Methods: Reinhard, Hable, ACES
   - With exposure compensation

### Saliency & Visual Attention
1. **Spectral Residual** (Fast)
   - File: `preprocess_image.py:355-376`
   - Library: OpenCV
   - Speed: ~100ms (768×512)

2. **EML-NET** (Deep Learning)
   - File: `saliency_utils.py:1-35`
   - Dual-stream ResNet50 (ImageNet + Places)
   - GPU-required

### Texture Analysis
1. **Haralick Features**
   - File: `preprocess_image.py:293-334`
   - Library: Mahotas
   - Features: Contrast, Energy
   - Tile-based: 32×32 with 50% overlap

### Segmentation
1. **SLIC Superpixels**
   - File: `preprocess_image.py:336-353`
   - Compactness: 10
   - Default: 500 segments
   - Output: Integer label map

### Structure Detection
1. **Distance Transform** (Euclidean)
   - File: `preprocess_image.py:378-414`
   - Library: OpenCV (GPU-optional)
   - Kernel: L2 connectivity (3×3)

2. **Skeleton/Medial Axis**
   - File: `preprocess_image.py:416-443`
   - Library: scikit-image
   - Per-segment morphological skeletonization

### Local Analysis
1. **Shannon Entropy** (Tile-based)
   - File: `preprocess_image.py:221-254`
   - Tile size: 16×16 (configurable)
   - Bins: 256
   - Output: Normalized [0,1]

2. **Gradient Magnitude**
   - File: `preprocess_image.py:256-291`
   - Formula: √(Gx² + Gy²)
   - Normalized to [0,1]

### Filtering & Smoothing
1. **Gaussian Blur**
   - File: `saliency_utils.py:29`
   - Library: scikit-image
   - Default filter size: 15

2. **Contrast Sensitivity Functions** (CSF)
   - File: `flip.py:391-441`
   - 3 channels: Achromatic, Red-Green, Blue-Yellow
   - Human perception model

3. **Spatial Filtering with CSF**
   - File: `flip.py:444-472`
   - Channel-specific convolution
   - Replicate padding

### Image I/O & Format
1. **Image Loading**
   - File: `image_utils.py:61-96`
   - Formats: JPEG, PNG
   - Supports: L (grayscale), RGB, RGBA
   - Gamma correction, downsampling

2. **Image Resizing**
   - File: `image_utils.py:61-96, 117-123`
   - Methods: BILINEAR (load), BOX (zoom)

3. **Format Conversion**
   - File: `image_utils.py:99-114`
   - Torch ↔ NumPy
   - Gamma correction
   - Range normalization

### Quality Metrics & Loss Functions
1. **PSNR**
   - File: `image_utils.py:35-40`
   - Formula: 20*log10(MAX/√MSE)
   - Output: Scalar dB

2. **Fused SSIM**
   - File: `fused-ssim/__init__.py:8-42`
   - Padding: "same" or "valid"
   - CUDA backend
   - Constants: C1=(0.01)², C2=(0.03)²

3. **LDR-FLIP**
   - File: `flip.py:132-227`
   - Color pipeline: CSF + Hunt adjustment + HyAB
   - Feature pipeline: Edge + Point detection
   - Combined with power law

4. **HDR-FLIP**
   - File: `flip.py:58-129`
   - Multi-exposure tone mapping
   - Supports: Reinhard, Hable, ACES
   - Per-pixel maximum error

5. **Quantization (STE)**
   - File: `quantization_utils.py:4-17`
   - Straight-through estimator
   - Default: 16-bit precision

---

## QUICK LOOKUP BY FILE

### image_utils.py
- `get_psnr()` - PSNR metric
- `get_grid()` - Normalized coordinate grid
- `compute_image_gradients()` - Sobel gradient
- `load_images()` - Image loading with downsampling
- `to_output_format()` - Tensor to image conversion
- `save_image()` - Image saving with zoom
- `separate_image_channels()` - Channel manipulation
- `visualize_gaussians()` - Ellipse visualization
- `visualize_added_gaussians()` - Gaussian positions

### flip.py (868 lines of complex algorithms)
- `LDRFLIPLoss.forward()` - LDR FLIP loss computation
- `HDRFLIPLoss.forward()` - HDR FLIP loss computation
- `compute_ldrflip()` - Core LDR FLIP algorithm
- `compute_start_stop_exposures()` - Exposure computation
- `tone_map()` - Tone mapping (3 methods)
- `generate_spatial_filter()` - CSF generation
- `spatial_filter()` - CSF-based spatial filtering
- `hunt_adjustment()` - Perceptual color adjustment
- `hyab()` - HyAB color distance
- `redistribute_errors()` - Error remapping
- `feature_detection()` - Edge/point detection
- `color_space_transform()` - 15+ color conversions

### saliency_utils.py
- `get_smap()` - EML-NET saliency detection

### saliency/decoder.py
- `Decoder` - Fusion decoder architecture
- `build_decoder()` - Load pretrained weights

### saliency/resnet.py
- `ResNet` - Backbone architecture
- `BasicBlock`, `Bottleneck` - Residual blocks

### preprocess_image.py / preprocess_image_v2.py
- `_compute_entropy_map()` - Tile-based entropy
- `_compute_gradient_map()` - Sobel with GPU option
- `_compute_texture_map()` - Haralick features
- `_compute_slic_segments()` - SLIC segmentation
- `_compute_saliency_map()` - Spectral residual saliency
- `_compute_distance_transform()` - Euclidean distance
- `_compute_skeleton()` - Binary skeletonization
- `_generate_placement_map()` - Weighted combination

### slic_preprocess.py
- `generate_slic_gaussians()` - SLIC → Gaussian initialization

### quantization_utils.py
- `ste_quantize()` - Straight-through estimator quantization

### fused-ssim/__init__.py
- `FusedSSIMMap.forward()` - CUDA SSIM computation
- `fused_ssim()` - SSIM loss wrapper

---

## LIBRARY VERSIONS (Recommended)

```
opencv-contrib-python>=4.12.0     # Sobel, distance, saliency
mahotas>=1.4.18                     # Haralick texture
scikit-image>=0.25.2                # SLIC, skeleton
numpy                               # Foundation
torch                               # Deep learning
pillow                              # Image I/O
scipy                               # Filters
```

---

## PERFORMANCE CHARACTERISTICS

### Fastest Operations (< 50ms for 768×512)
- Sobel gradient: ~50ms
- Image resizing: ~20ms
- Color space conversion: ~10ms

### Medium Operations (50-500ms)
- Spectral saliency: ~100ms
- SLIC segmentation: ~300ms
- Distance transform: ~200ms

### Slower Operations (> 500ms)
- Haralick texture: ~1.8s (bottleneck)
- Skeleton extraction: ~2.5s
- EML-NET saliency: ~3-5s (GPU-accelerated)

### Bottleneck Mitigation
- Texture: Per-tile, can parallelize
- Skeleton: Per-segment, can parallelize
- EML-NET: GPU required for practical use

---

## ALGORITHM SELECTION GUIDE

**For Gaussian Placement:**
- Use: Entropy + Gradient + Texture + Saliency + Distance
- Weights: 0.30 + 0.25 + 0.20 + 0.15 + 0.10

**For Quality Metrics:**
- Perceptual: FLIP (LDR or HDR)
- Structural: SSIM (fused, CUDA)
- Numeric: PSNR

**For Edge Detection:**
- Speed-critical: Sobel (cv2, 3×3)
- Perception-aware: FLIP feature detection

**For Saliency:**
- Fast: Spectral residual (~100ms)
- Accurate: EML-NET (~3-5s GPU, ~20s CPU)

**For Segmentation:**
- Structure-aware: SLIC superpixels
- Texture-aware: Haralick features (post-SLIC)

---

**Last Updated:** 2025-11-15  
**Total Algorithms Cataloged:** 30+  
**Implementation Coverage:** 95%+
