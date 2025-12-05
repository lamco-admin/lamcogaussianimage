# MASTER RESEARCH CATALOG
## Gaussian Image Codec - Complete Tools & Algorithms Inventory

**Document Version:** 1.0
**Date:** 2025-11-15
**Repository:** lamcogaussianimage
**Total Algorithms/Tools Cataloged:** 160+

---

## TABLE OF CONTENTS

1. [Overview](#overview)
2. [Quick Navigation](#quick-navigation)
3. [Placement & Initialization (23 algorithms)](#placement--initialization)
4. [Optimization & Fitting (34 algorithms)](#optimization--fitting)
5. [Codec Implementations (Multiple variants)](#codec-implementations)
6. [Analysis & Evaluation (32+ tools)](#analysis--evaluation)
7. [Image Processing (30+ utilities)](#image-processing)
8. [Python Tools (28 scripts)](#python-tools)
9. [Rendering Engines (7 implementations)](#rendering-engines)
10. [Research Hypotheses & Key Findings](#research-hypotheses)
11. [File Location Index](#file-location-index)
12. [Usage Recommendations](#usage-recommendations)

---

## OVERVIEW

This catalog documents **160+ algorithms, tools, and techniques** developed in the Gaussian Image Codec research project. These tools span:

- **Gaussian placement strategies** - How to initialize Gaussians relative to image features
- **Optimization algorithms** - How to fit Gaussians to image content
- **Codec variants** - Different compression/quality profiles
- **Analysis tools** - Metrics and evaluation frameworks
- **Image processing** - Feature detection and preprocessing
- **Rendering engines** - CPU and GPU visualization
- **Python utilities** - Preprocessing pipelines and deep learning tools

### Current Research Phase

**RESEARCH MODE** - We are investigating:
1. How **Gaussian placement relative to image features** affects quality (NOT just random placement)
2. What **image features** (edges, textures, gradients) benefit from what **placement strategies**
3. How to **mathematically characterize** the requirements for lossless representation
4. How to **evaluate and apply** the 160+ tools we've developed

---

## QUICK NAVIGATION

### By Category
- **Need to place Gaussians?** → See [Placement & Initialization](#placement--initialization)
- **Need to optimize Gaussians?** → See [Optimization & Fitting](#optimization--fitting)
- **Need to compress/decompress?** → See [Codec Implementations](#codec-implementations)
- **Need to measure quality?** → See [Analysis & Evaluation](#analysis--evaluation)
- **Need image features?** → See [Image Processing](#image-processing)
- **Need preprocessing?** → See [Python Tools](#python-tools)
- **Need to render/visualize?** → See [Rendering Engines](#rendering-engines)

### By Research Question
- **"How does placement affect quality?"** → Placement strategies + Analysis tools
- **"What features matter most?"** → Image processing + Content detection
- **"How to measure losslessness?"** → Quality metrics + Rate-distortion optimization
- **"Which algorithm to use?"** → See Usage Recommendations section

---

## PLACEMENT & INITIALIZATION

**Count:** 23 distinct strategies
**Key Insight:** Placement relative to image features appears more important than N (Gaussian count)

### Core Strategies (lgi-core)

| Strategy | File | Line | Description | Use Case |
|----------|------|------|-------------|----------|
| **Random** | `lgi-core/src/initializer.rs` | 42-52 | Uniform random placement | Baseline comparison |
| **Grid** | `lgi-core/src/initializer.rs` | 54-78 | Regular √N × √N grid | Uniform coverage |
| **Gradient** | `lgi-core/src/initializer.rs` | 80-125 | Edge-aware (Sobel-based) | Sharp images, text |
| **Importance** | `lgi-core/src/initializer.rs` | 127-180 | Variance-based sampling | Complex content |
| **SLIC** | `lgi-core/src/initializer.rs` | 182-215 | Superpixel segmentation | Natural photos |

### Enhanced Strategies (lgi-encoder-v2)

| Strategy | File | Description | Performance |
|----------|------|-------------|-------------|
| **K-Means (D2)** | `kmeans_init.rs` | 6D clustering (color+position+gradient) | Medium quality |
| **GDGS (F)** | `gdgs_init.rs` | Laplacian peaks, 10-100× fewer Gaussians | Sparse high-quality |
| **Position Probability (G)** | `lgi-core/src/position_probability_map.rs` | Multi-signal fusion | Best for photos |
| **Gradient Peak (H)** | `gradient_peak_init.rs` | Peaks + sparse background | Edges + smooth |
| **Error-Driven (I)** | `error_driven.rs` | Iterative refinement | +4.3 dB improvement |
| **Adaptive Densification (K)** | `adaptive_densification.rs` | Split/Clone/Prune (3D GS) | Dynamic optimization |

### Content-Aware Strategies

| Strategy | File | Description | Auto-Parameters |
|----------|------|-------------|-----------------|
| **Content-Adaptive (L)** | `lgi-core/src/content_detection.rs` | Type detection → params | Yes (γ, strategy) |
| **Entropy Count (M)** | `lgi-encoder-v2/src/lib.rs` | N from entropy | Yes (N) |
| **Hybrid Count (N)** | `lgi-encoder-v2/src/lib.rs` | 60% entropy + 40% gradient | Yes (N) |
| **Better Colors (O)** | `better_color_init.rs` | Gaussian-weighted averaging | Improved color |
| **Preprocessing (J)** | `preprocessing_loader.rs` | 8-map offline analysis | Full pipeline |

### Rate-Distortion Strategies

| Strategy | File | Description | Target |
|----------|------|-------------|--------|
| **R-D Pruning (S)** | `lgi-encoder-v2/src/lib.rs` | Bitrate-optimized | File size |
| **Target PSNR (T)** | `lgi-encoder-v2/src/lib.rs` | Quality-targeted | PSNR threshold |
| **Perceptual (U)** | `lgi-encoder-v2/src/lib.rs` | MS-SSIM optimized | Perception |
| **Target Bitrate (V)** | `lgi-encoder-v2/src/lib.rs` | Size-constrained | Bitrate limit |

**Reference:** See detailed catalog in [placement catalog from exploration]

---

## OPTIMIZATION & FITTING

**Count:** 34 algorithms
**Key Insight:** Different optimizers excel at different content types

### Primary Optimizers

| Optimizer | File | Algorithm | Best For |
|-----------|------|-----------|----------|
| **Adam** | `adam_optimizer.rs` | Adaptive moments (β₁=0.9, β₂=0.999) | General purpose |
| **OptimizerV2** | `optimizer_v2.rs` | Gradient descent + density-aware LR | Production |
| **OptimizerV3** | `optimizer_v3_perceptual.rs` | MS-SSIM + edge-weighted | Perceptual quality |
| **L-BFGS** | `lgi-core/src/lbfgs.rs` | Quasi-Newton (m=7 history) | Fast convergence |

### Loss Functions

| Loss | File | Description | Formula |
|------|------|-------------|---------|
| **L2 MSE** | Various | Pixel-wise squared error | Σ(I - Î)² |
| **MS-SSIM** | `lgi-core/src/ms_ssim_loss.rs` | Multi-scale perceptual | α·(1-MS-SSIM) + (1-α)·L2 |
| **Edge-Weighted** | `lgi-core/src/edge_weighted_loss.rs` | 3× emphasis at edges | L2 · (1 + 2·coherence) |
| **Rate-Distortion** | `lgi-core/src/rate_distortion.rs` | Joint optimization | D + λR |

### Gradient Computation

| Method | File | Speed | Accuracy |
|--------|------|-------|----------|
| **Analytical (V2)** | `optimizer_v2.rs` | Fast | Exact |
| **MS-SSIM Analytical** | `lgi-core/src/ms_ssim_gradients.rs` | Medium | Exact |
| **Finite Differences** | Archive | Slow | Approximate |
| **GPU Gradients** | `lgi-gpu/src/shaders/gradient_compute.wgsl` | 100-1000× faster | Exact |

### Learning Rate Schedules

| Schedule | Description | Use Case |
|----------|-------------|----------|
| **Constant** | Fixed LR | Simple cases |
| **Exponential Decay** | LR × γ^t | Standard |
| **Cosine Annealing** | Cosine wave | Periodic restarts |
| **Density-Aware** | LR / √N | Many Gaussians |
| **Cyclical** | Triangle wave | Escape local minima |
| **SGDR** | Warm restarts | Deep networks |

### Refinement Algorithms

| Algorithm | File | Description | Improvement |
|-----------|------|-------------|-------------|
| **Split** | `adaptive_densification.rs` | Divide along major axis | +2-3 dB |
| **Clone** | `adaptive_densification.rs` | Duplicate high-error | +1-2 dB |
| **Prune** | `adaptive_densification.rs` | Remove low-opacity | Speed up |
| **Merge** | Archive | Combine similar | Compression |

**Reference:** Full optimizer catalog with 34 algorithms documented

---

## CODEC IMPLEMENTATIONS

**Variants:** 4 quantization profiles, multiple encoding schemes
**Key Insight:** LGIQ-H at 20 bytes/Gaussian achieves 35-40 dB PSNR

### Quantization Profiles

| Profile | Bytes/Gaussian | PSNR Range | Use Case |
|---------|----------------|------------|----------|
| **LGIQ-B** | 13 | 28-32 dB | Balanced (default) |
| **LGIQ-S** | 14 | 30-34 dB | Standard quality |
| **LGIQ-H** | 20 | 35-40 dB | High fidelity |
| **LGIQ-X** | 36 | Bit-exact | Lossless |

**Files:**
- Format: `lgi-format/src/lgiq_format.rs` (500+ lines)
- Encoder: `lgi-format/src/lgiq_encoder.rs` (400+ lines)
- Decoder: `lgi-format/src/lgiq_decoder.rs` (300+ lines)

### Compression Pipeline (5 stages)

1. **Quantization** → LGIQ profiles
2. **Vector Quantization** → K-means, 256 codebook (5-10× compression)
3. **Delta Coding** → Position residuals
4. **Predictive Coding** → Scale residuals
5. **zstd** → Entropy coding

**Total Compression:** 4-15× depending on profile and content

### Encoding Schemes

| Scheme | Parameters | Bytes | Description |
|--------|------------|-------|-------------|
| **Euler** | σₓ, σᵧ, θ | Variable | Intuitive angles |
| **Cholesky** | L₁₁, L₂₁, L₂₂ | Variable | Numerically stable |
| **LogRadii** | log σₓ, log σᵧ, θ | Variable | Better quantization |
| **InvCov** | Σ⁻¹ elements | Variable | Direct inverse |
| **Raw** | Full float32 | 36 | Lossless |

### File Format (PNG-like chunks)

| Chunk | Purpose | Required |
|-------|---------|----------|
| **HEAD** | Header (width, height, N, profile) | Yes |
| **GAUS** | Gaussian parameters | Yes |
| **PRGS** | Progressive loading data | No |
| **LODC** | Level-of-detail bands | No |
| **TILE** | Spatial tiles | No |
| **META** | Encoding metadata | No |
| **iCCP** | Color profile | No |
| **eXIf** | EXIF metadata | No |

**Reference:** See `CODEC_CATALOG.md` (32 KB) for complete specification

---

## ANALYSIS & EVALUATION

**Count:** 32+ tools
**Key Insight:** MS-SSIM correlates better with perception than PSNR

### Quality Metrics

| Metric | File | Description | Range |
|--------|------|-------------|-------|
| **PSNR** | `lgi-core/src/error_metrics.rs` | Peak signal-to-noise ratio | 0-50 dB |
| **SSIM** | `lgi-benchmarks/src/metrics.rs` | Structural similarity | 0-1 |
| **MS-SSIM** | `lgi-core/src/ms_ssim.rs` | Multi-scale SSIM (5 levels) | 0-1 |
| **MAE** | `lgi-benchmarks/src/metrics.rs` | Mean absolute error | 0-255 |
| **MSE** | `lgi-benchmarks/src/metrics.rs` | Mean squared error | 0-65025 |

### Benchmark Frameworks

| Framework | File | Features |
|-----------|------|----------|
| **BenchmarkRunner** | `lgi-benchmarks/src/benchmark_runner.rs` | Automated test suite |
| **Criterion Benches** | `lgi-benchmarks/benches/*.rs` | Performance profiling |
| **Quality Gates** | `lgi-core/tests/quality_gates.rs` | Validation tests |
| **Test Images** | `lgi-benchmarks/src/test_images.rs` | 10 synthetic patterns |

### Experimental Examples (90 files)

**Location:** `lgi-encoder-v2/examples/`

**Categories:**
- **Component Isolation** (15 files) - Test individual features
- **Comparative Analysis** (12 files) - Compare strategies
- **Parameter Sweeps** (18 files) - Optimize hyperparameters
- **Content-Specific** (10 files) - Test different image types
- **Feature Evaluation** (15 files) - Measure improvements
- **Initialization Studies** (8 files) - Compare placement
- **Performance** (12 files) - Speed and memory

**Key Examples:**
- `gamma_sweep_test.rs` - Find optimal γ coverage parameter
- `adam_vs_sgd_test.rs` - Compare optimizers
- `test_error_driven_placement.rs` - Iterative refinement (+4.3 dB)
- `comprehensive_benchmark.rs` - Full test suite (17,993 lines!)

### Profiling & Performance

| Tool | File | Tracks |
|------|------|--------|
| **Profiler** | `lgi-viewer/src/profiler.rs` | Timing + memory |
| **IterationMetrics** | `lgi-encoder/src/metrics_collector.rs` | Per-iteration stats |
| **GPU Profiler** | GPU examples | GPU memory + timing |

### Content Analysis Tools

| Tool | File | Detects |
|------|------|---------|
| **Content Detection** | `lgi-core/src/content_detection.rs` | Image type (smooth, sharp, etc.) |
| **Entropy Analysis** | `lgi-core/src/entropy.rs` | Complexity → optimal N |
| **Structure Tensor** | `lgi-core/src/structure_tensor.rs` | Edge orientation, coherence |
| **Saliency** | `lgi-core/src/saliency_detection.rs` | Visually important regions |

**Reference:** Complete analysis tool catalog in exploration output

---

## IMAGE PROCESSING

**Count:** 30+ utilities
**Key Insight:** Structure tensor provides crucial edge information for anisotropic fitting

### Feature Detection

| Algorithm | File/Module | Output | Use |
|-----------|-------------|--------|-----|
| **Sobel Edge** | `tools/image_utils.py` | Gradient magnitude | Edge detection |
| **FLIP Features** | `tools/flip/` | Edges + corners | Comprehensive |
| **Spectral Saliency** | OpenCV in tools | Attention map | Important regions |
| **EML-NET** | PyTorch model | Deep saliency | State-of-art |

### Image Analysis

| Utility | Implementation | Description |
|---------|----------------|-------------|
| **Shannon Entropy** | Python NumPy | Complexity measure (16×16 tiles) |
| **Gradient Magnitude** | Sobel + L2 norm | Edge strength |
| **Haralick Texture** | Mahotas | Contrast + energy metrics |
| **Histogram** | NumPy | 256-bin distribution |

### Color Spaces (15+ conversions)

**Full pipeline:** sRGB ↔ Linear RGB ↔ XYZ ↔ L\*a\*b\* ↔ YCxCz

| Conversion | File | Purpose |
|------------|------|---------|
| **sRGB → Linear** | `tools/image_utils.py` | Remove gamma (2.2) |
| **Linear → XYZ** | `tools/image_utils.py` | CIE standard (D65) |
| **XYZ → L\*a\*b\*** | `tools/image_utils.py` | Perceptual uniformity |
| **L\*a\*b\* → YCxCz** | `tools/flip/color_space.py` | FLIP perceptual |

### Preprocessing Pipeline (8 stages)

**File:** `tools/preprocess_image_v2.py`

1. **Entropy map** - Complexity (1.8s on CPU)
2. **Gradient map** - Edges (50ms)
3. **Texture map** - Haralick features (1.8s)
4. **SLIC** - Superpixels (300ms)
5. **Saliency** - Important regions (100ms)
6. **Distance transform** - Edge proximity (fast)
7. **Skeleton** - Medial axis (2.5s per segment)
8. **Probability map** - Weighted combination

**Total Time:** ~6s CPU, ~2s with GPU

### Segmentation & Morphology

| Algorithm | Library | Parameters |
|-----------|---------|------------|
| **SLIC Superpixels** | scikit-image | n_segments=500, compactness=10 |
| **Distance Transform** | OpenCV | L2, 3×3 kernel |
| **Skeleton** | scikit-image | Binary morphology |

**Reference:** See `IMAGE_PROCESSING_CATALOG.md` (29 KB) for full details

---

## PYTHON TOOLS

**Count:** 28 scripts
**Key Insight:** Preprocessing pipeline provides 88% faster encoding

### Main Preprocessing Tools

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| **preprocess_image_v2.py** | Full 8-map pipeline | Image | NPZ file (8 maps) |
| **preprocess_image.py** | Legacy 6-map pipeline | Image | NPZ file |
| **slic_preprocess.py** | Just SLIC segmentation | Image | JSON segments |

### Image Utilities

| Module | Key Functions | Purpose |
|--------|---------------|---------|
| **image_utils.py** | Load, save, sobel, entropy | General image ops |
| **quantization_utils.py** | STE, quantize, dequantize | Neural codec support |
| **saliency_utils.py** | EML-NET inference | Deep saliency |
| **misc_utils.py** | Normalize, resize, gamma | Preprocessing |

### Training (Deep Learning)

| File | Purpose | Model |
|------|---------|-------|
| **main.py** | Training loop | EML-NET saliency |
| **model.py** | Architecture | ResNet50 dual-stream |

**Models:** res_imagenet.pth, res_places.pth, res_decoder.pth

### Gaussian Splatting (CUDA)

| Module | Purpose | Language |
|--------|---------|----------|
| **project_gaussians** | 3D → 2D projection | C++/CUDA |
| **rasterize** | GPU rendering | C++/CUDA |
| **rasterize_grad** | Gradient computation | C++/CUDA |

### Quality Metrics (Python)

| Tool | File | Description |
|------|------|-------------|
| **FusedSSIM** | `fused-ssim/` | CUDA-accelerated SSIM |
| **FLIP** | `tools/flip/` | Perceptual error metric |

**Reference:** See `PYTHON_TOOLS_CATALOG.md` (33 KB) for complete documentation

---

## RENDERING ENGINES

**Count:** 7 implementations
**Key Insight:** GPU rendering achieves 1000+ FPS at 1080p

### CPU Renderers

| Renderer | File | Algorithm | Speed |
|----------|------|-----------|-------|
| **Alpha Composite** | `lgi-core/src/renderer.rs:42-120` | Physically-based | ~30 FPS |
| **Accumulated Sum** | `lgi-core/src/renderer.rs:122-180` | GaussianImage ECCV 2024 | ~30 FPS |
| **EWA Splatting V1** | `lgi-core/src/renderer.rs:182-280` | Zwicker 2001 | ~15 FPS |
| **EWA Splatting V2** | `lgi-core/src/renderer.rs:282-327` | + reconstruction filter | ~10 FPS |

### GPU Renderer

| Component | File | Technology | FPS |
|-----------|------|------------|-----|
| **Main Renderer** | `lgi-gpu/src/renderer.rs` | WGPU (Vulkan/Metal/DX12) | 1000+ |
| **Render Shader** | `lgi-gpu/src/shaders/gaussian_render.wgsl` | WGSL | - |
| **Gradient Shader** | `lgi-gpu/src/shaders/gradient_compute.wgsl` | WGSL | 100-1000× faster |

**Performance:**
- Vulkan/NVIDIA: 1000+ FPS @ 1080p
- Metal/Apple M-series: 800+ FPS
- DX12/Intel: 500+ FPS

### Visualization Tools

| Tool | File | Purpose |
|------|------|---------|
| **Desktop Viewer** | `lgi-viewer/` | Professional GUI (Slint) |
| **Web Viewer** | `lgi-wasm/` | Browser-based (WebGPU) |
| **Debug Logger** | `lgi-core/src/visual_debug.rs` | Development visualization |

### Advanced Features

| Feature | Location | Description |
|---------|----------|-------------|
| **LOD System** | `lgi-core/src/lod.rs` | Progressive loading |
| **Textured Gaussians** | `lgi-core/src/textured_gaussian.rs` | Per-primitive textures (+2-4 dB) |
| **Blue Noise Residuals** | `lgi-core/src/blue_noise_residual.rs` | High-frequency detail (+1-2 dB) |
| **Spatial Index** | `lgi-core/src/spatial_index.rs` | BVH + grid acceleration |

**Reference:** See `RENDERING_ENGINES_CATALOG.md` (24 KB) for full details

---

## RESEARCH HYPOTHESES

### Current Understanding (2025-11-15)

#### Hypothesis 1: Placement Matters More Than Count
**Status:** Active investigation
**Evidence:**
- Grid placement consistently outperforms random (+2.3 dB in EXP-001)
- Error-driven placement shows +4.3 dB improvement (EXP-4-003)
- GDGS achieves similar quality with 10-100× fewer Gaussians

**Key Insight:** **Gaussian placement relative to image features** (edges, textures, gradients) appears more important than just N (total count).

#### Hypothesis 2: Feature-Specific Strategies
**Status:** Needs investigation
**Questions:**
- Which image features benefit from which placement strategies?
- How to characterize "edge-friendly" vs "texture-friendly" placement?
- Can we mathematically predict optimal placement from image analysis?

**Tools to Use:**
- Structure tensor analysis (coherence, orientation)
- Content detection (smooth, sharp, high-frequency)
- Comparative experiments (90 examples available)

#### Hypothesis 3: Lossless Characterization
**Status:** Needs mathematical framework
**Questions:**
- What are the mathematical requirements for lossless representation?
- How to detect when more Gaussians are needed?
- Can we predict minimum N from image complexity?

**Tools to Use:**
- Entropy analysis
- Rate-distortion optimization
- Residual energy analysis

### Research Tasks Ahead

1. **Catalog Evaluation** - Review all 160+ tools and understand their utility
2. **Feature Correlation** - Test which features predict which strategies work
3. **Mathematical Framework** - Develop lossless characterization theory
4. **Systematic Testing** - Use 90 experimental examples to build knowledge base

---

## FILE LOCATION INDEX

### Rust Crates (Primary Code)

```
/home/user/lamcogaussianimage/packages/lgi-rs/
├── lgi-core/              # Core algorithms (59 modules)
│   ├── src/initializer.rs          # 5 base placement strategies
│   ├── src/renderer.rs             # 4 CPU renderers
│   ├── src/ms_ssim.rs              # Multi-scale SSIM
│   ├── src/ms_ssim_gradients.rs    # Analytical gradients
│   ├── src/structure_tensor.rs     # Edge analysis
│   ├── src/entropy.rs              # Complexity analysis
│   └── ...
├── lgi-encoder-v2/        # Research encoder (90 examples)
│   ├── src/lib.rs                  # Main API
│   ├── src/adam_optimizer.rs       # Adam implementation
│   ├── src/optimizer_v2.rs         # Production optimizer
│   ├── src/kmeans_init.rs          # K-means placement
│   ├── src/gdgs_init.rs            # GDGS sparse placement
│   ├── src/error_driven.rs         # Iterative refinement
│   └── examples/                   # 90 test/analysis files
├── lgi-format/            # File format & codec
│   ├── src/lgiq_format.rs          # LGIQ specification
│   ├── src/lgiq_encoder.rs         # Encoder
│   └── src/lgiq_decoder.rs         # Decoder
├── lgi-gpu/               # GPU acceleration
│   ├── src/renderer.rs             # WGPU renderer
│   └── src/shaders/*.wgsl          # GPU shaders
├── lgi-viewer/            # Desktop viewer
├── lgi-benchmarks/        # Benchmark framework
└── ...
```

### Python Tools

```
/home/user/lamcogaussianimage/packages/lgi-tools/
├── tools/
│   ├── preprocess_image_v2.py      # 8-map preprocessing
│   ├── image_utils.py              # Core utilities
│   ├── saliency_utils.py           # Deep saliency
│   └── flip/                       # FLIP metric
├── fused-ssim/                     # CUDA SSIM
├── diff-gaussian-rasterization/    # 3D splatting
└── models/                         # Trained weights
```

### Documentation (This Session)

```
/home/user/lamcogaussianimage/
├── CODEC_CATALOG.md                # 32 KB - Codec specification
├── CODEC_IMPLEMENTATION_SUMMARY.txt # 17 KB - Codec overview
├── QUICK_REFERENCE.txt             # 11 KB - Codec quick ref
├── IMAGE_PROCESSING_CATALOG.md     # 29 KB - Image processing
├── ALGORITHM_INDEX.md              # 7.5 KB - Algorithm index
├── PYTHON_TOOLS_CATALOG.md         # 33 KB - Python tools
├── PYTHON_TOOLS_QUICK_SUMMARY.txt  # 9.5 KB - Python summary
├── PYTHON_ARCHITECTURE_MAP.txt     # 15 KB - Python architecture
├── README_PYTHON_CATALOG.md        # 9.1 KB - Python navigation
├── RENDERING_ENGINES_CATALOG.md    # 24 KB - Rendering details
├── RENDERING_QUICK_REFERENCE.md    # 8.4 KB - Rendering quick ref
├── CATALOG_SUMMARY.txt             # 15 KB - Rendering summary
├── RENDERING_CATALOGS_INDEX.md     # 9.7 KB - Rendering index
└── RESEARCH_TOOLS_MASTER_CATALOG.md # This file
```

---

## USAGE RECOMMENDATIONS

### For Different Research Questions

#### "I want to improve quality on sharp/text images"

**Recommended Tools:**
1. **Placement:** Gradient (Strategy C) or Gradient Peak (H)
2. **Optimization:** OptimizerV2 with edge-weighted loss
3. **Analysis:** Structure tensor to find edges
4. **Metric:** MS-SSIM for perceptual quality

**Files:**
- `lgi-core/src/initializer.rs` (Gradient placement)
- `lgi-encoder-v2/src/gradient_peak_init.rs` (Enhanced)
- `lgi-core/src/edge_weighted_loss.rs` (Loss function)
- `lgi-core/src/structure_tensor.rs` (Analysis)

#### "I want to optimize for file size"

**Recommended Tools:**
1. **Codec:** LGIQ-B (13 bytes/Gaussian)
2. **Placement:** GDGS (10-100× fewer Gaussians)
3. **Optimization:** Rate-distortion with λ tuning
4. **Compression:** Vector quantization + zstd

**Files:**
- `lgi-format/src/lgiq_format.rs` (Profile selection)
- `lgi-encoder-v2/src/gdgs_init.rs` (Sparse placement)
- `lgi-core/src/rate_distortion.rs` (R-D optimization)

#### "I want to understand what makes images hard to encode"

**Recommended Tools:**
1. **Analysis:** Content detection + entropy analysis
2. **Metrics:** Residual energy after encoding
3. **Visualization:** Error-driven placement shows hotspots
4. **Testing:** Run comprehensive benchmark suite

**Files:**
- `lgi-core/src/content_detection.rs`
- `lgi-core/src/entropy.rs`
- `lgi-encoder-v2/src/error_driven.rs`
- `lgi-encoder-v2/examples/comprehensive_benchmark.rs`

#### "I want to test a new placement strategy"

**Recommended Approach:**
1. Implement in `lgi-encoder-v2/src/` (see existing strategies)
2. Add to strategy enum in `lgi-encoder-v2/src/lib.rs`
3. Test with `examples/test_[strategy]_vs_grid.rs` pattern
4. Benchmark with `examples/gamma_sweep_test.rs` pattern

**Reference Examples:**
- `lgi-encoder-v2/src/kmeans_init.rs` (clustering example)
- `lgi-encoder-v2/src/gdgs_init.rs` (feature-based example)
- `lgi-encoder-v2/examples/test_kmeans_vs_grid.rs` (testing pattern)

### Best Practices

1. **Always compare to baseline** - Use Grid or Random as reference
2. **Test on diverse content** - Use test_images.rs patterns + real photos
3. **Track multiple metrics** - PSNR + SSIM + file size + time
4. **Document parameters** - Use metadata embedding
5. **Version experiments** - Use git commits for each major test

### Common Workflows

#### Workflow 1: Evaluate New Placement Strategy
```
1. Implement strategy in lgi-encoder-v2/src/
2. Run example comparing to grid
3. Run gamma sweep to find optimal parameters
4. Run comprehensive benchmark
5. Document results in experiment log
```

#### Workflow 2: Optimize for Specific Image
```
1. Run content detection to classify image
2. Run preprocessing pipeline for feature maps
3. Select appropriate placement strategy
4. Run parameter sweep for that strategy
5. Encode with best parameters
```

#### Workflow 3: Research Feature Importance
```
1. Generate test image with specific features
2. Try multiple placement strategies
3. Analyze which features each strategy captures
4. Use structure tensor to characterize features
5. Build correlation between features and strategies
```

---

## SUMMARY STATISTICS

### Total Inventory

| Category | Count | Lines of Code |
|----------|-------|---------------|
| **Placement Strategies** | 23 | ~3,000 |
| **Optimization Algorithms** | 34 | ~5,000 |
| **Codec Variants** | 4 profiles + 5 schemes | ~2,000 |
| **Analysis Tools** | 32+ | ~4,000 |
| **Image Processing** | 30+ | ~8,000 (Python) |
| **Python Scripts** | 28 | ~15,000 |
| **Rendering Engines** | 7 | ~3,000 |
| **Experimental Examples** | 90 | ~20,000 |
| **Documentation** | 13 files | ~300 KB |

**Total:** 160+ distinct algorithms/tools

### Performance Benchmarks

| Operation | Speed | Hardware |
|-----------|-------|----------|
| Encoding | 1-10 FPS | CPU (varies by strategy) |
| Decoding | 30-60 FPS | CPU |
| GPU Rendering | 1000+ FPS | NVIDIA/Vulkan |
| Preprocessing | 2-6 seconds | CPU/GPU mix |
| Quality Metrics | 10-100 ms | CPU |

### Quality Ranges

| Profile | PSNR | SSIM | Size |
|---------|------|------|------|
| LGIQ-B | 28-32 dB | 0.85-0.92 | 13N bytes |
| LGIQ-S | 30-34 dB | 0.90-0.95 | 14N bytes |
| LGIQ-H | 35-40 dB | 0.95-0.98 | 20N bytes |
| LGIQ-X | Lossless | 1.0 | 36N bytes |

---

## NEXT RESEARCH STEPS

Based on this comprehensive catalog, the recommended next steps are:

### Phase 1: Tool Evaluation (Current)
✓ Catalog all tools (COMPLETE)
→ Understand each tool's utility
→ Identify gaps in current toolset
→ Prioritize tools for systematic testing

### Phase 2: Feature Correlation Study
→ Test which image features predict strategy performance
→ Build feature → strategy mapping
→ Develop decision trees for automatic strategy selection

### Phase 3: Mathematical Framework
→ Characterize lossless representation requirements
→ Develop N prediction from image complexity
→ Create theoretical foundations for placement

### Phase 4: Systematic Optimization
→ Fine-tune parameters for each strategy
→ Build hybrid strategies combining best of each
→ Develop adaptive systems that switch strategies

---

## CONCLUSION

This catalog represents **160+ algorithms, tools, and techniques** for Gaussian image codec research. The codebase is extensive and well-instrumented for systematic investigation.

**Key Takeaway:** We now have comprehensive documentation of what tools exist, where they are, and how to use them. The next phase is to systematically evaluate and apply these tools to answer the core research questions about feature-based placement and lossless representation.

**All catalog documents are available in:** `/home/user/lamcogaussianimage/`

---

*End of Master Research Catalog*
