# GAUSSIAN IMAGE CODEC REPOSITORY - COMPREHENSIVE CATALOG

## REPOSITORY STRUCTURE

### Main Packages
- `/packages/lgi-rs/` - Rust implementations (primary)
- `/packages/lgi-legacy/` - Legacy Python implementations
- `/packages/lgi-tools/` - Utility tools

### Key Rust Modules
- `lgi-format` - File format and chunk handling
- `lgi-encoder` - Primary encoder (v1)
- `lgi-encoder-v2` - Research encoder (v2, multi-variant)
- `lgi-core` - Core algorithms and utilities
- `lgi-cli` - Command-line interface
- `lgi-viewer` - Viewer with async encoding
- `lgi-ffi` - Foreign Function Interface
- `lgi-gpu` - GPU acceleration
- `lgi-wasm` - WebAssembly support
- `lgi-pyramid` - Pyramid/LOD system
- `archive/v1/` - Archive of version 1

---

## CODEC VARIANTS AND PROFILES

### 1. QUANTIZATION PROFILES (4 Total)
**File:** `/packages/lgi-rs/lgi-format/src/quantization.rs`
**Lines:** 1-536

#### LGIQ-B: Balanced Profile
- **Bytes per Gaussian:** 13 bytes (byte-aligned)
- **Quality Range:** 28-32 dB PSNR
- **Layout:**
  - Position: 16-bit × 2 (4 bytes)
  - Scale: 12-bit × 2 (3 bytes packed, log₂ encoded)
  - Rotation: 12-bit (2 bytes for alignment)
  - Color RGB: 8-bit × 3 (3 bytes)
  - Opacity: 8-bit (1 byte)
- **Quantization Methods:**
  - `quantize_balanced()` - Lines 125-163
  - `dequantize_balanced()` - Lines 165-203
- **Features:**
  - Log₂ scale encoding (σ_min=0.001)
  - Position clamping [0, 1]
  - Rotation normalization [-π, π]

#### LGIQ-S: Standard Profile (Enhanced)
- **Bytes per Gaussian:** 14 bytes (byte-aligned)
- **Quality Range:** 30-34 dB PSNR
- **Layout:**
  - Position: 16-bit × 2 (4 bytes)
  - Scale: 14-bit × 2 (4 bytes, higher precision)
  - Rotation: 14-bit (2 bytes)
  - Color + Opacity: 10-bit × 4 (4 bytes packed)
- **Quantization Methods:**
  - `quantize_small()` - Lines 221-264
  - `dequantize_small()` - Lines 266-303
- **Features:**
  - Higher precision scales (14-bit vs 12-bit)
  - Higher precision colors (10-bit vs 8-bit)
  - HDR color support

#### LGIQ-H: High Quality Profile
- **Bytes per Gaussian:** 20 bytes (float16)
- **Quality Range:** 35-40 dB PSNR
- **Layout:**
  - All 9 parameters as IEEE 754 float16:
    - Position X/Y: 2 bytes each
    - Scale X/Y: 2 bytes each
    - Rotation: 2 bytes
    - Color R/G/B: 2 bytes each
    - Opacity: 2 bytes
    - Padding: 2 bytes (reserved)
- **Quantization Methods:**
  - `quantize_high()` - Lines 326-345
  - `dequantize_high()` - Lines 347-369
- **Features:**
  - Full float16 precision (3-4 decimal digits)
  - Dynamic range: 10^-8 to 65504
  - Perfect for high-quality applications

#### LGIQ-X: Lossless Profile
- **Bytes per Gaussian:** 36 bytes (float32)
- **Quality:** Bit-exact reconstruction (∞ dB PSNR)
- **Layout:**
  - All 9 parameters as 32-bit float:
    - 4 bytes × 9 = 36 bytes total
- **Quantization Methods:**
  - `quantize_lossless()` - Lines 373-388
  - `dequantize_lossless()` - Lines 390-407
- **Features:**
  - Full precision IEEE 754 float32
  - Bit-exact reversibility
  - Maximum file size

**Batch Operations:**
- `quantize_all()` - Lines 411-417
- `dequantize_all()` - Lines 420-427

---

## COMPRESSION CONFIGURATION

**File:** `/packages/lgi-rs/lgi-format/src/compression.rs`
**Lines:** 1-240

### CompressionConfig Structure
```rust
struct CompressionConfig {
    quantization: QuantizationProfile,
    enable_vq: bool,
    vq_codebook_size: usize,
    enable_zstd: bool,
    zstd_level: i32,
}
```

### Compression Presets

#### Balanced (Default)
- **Quantization:** LGIQ-B
- **VQ:** Enabled (256 codebook)
- **zstd:** Level 9
- **Expected Ratio:** 4-10×
- **Code:** `CompressionConfig::balanced()` - Lines 34-42

#### Small (Maximum Compression)
- **Quantization:** LGIQ-B
- **VQ:** Enabled (128 codebook, smaller)
- **zstd:** Level 19 (higher compression)
- **Expected Ratio:** Maximum
- **Code:** `CompressionConfig::small()` - Lines 44-53

#### High Quality
- **Quantization:** LGIQ-H
- **VQ:** Disabled
- **zstd:** Level 9
- **Quality Focus:** Better PSNR
- **Code:** `CompressionConfig::high_quality()` - Lines 55-64

#### Lossless
- **Quantization:** LGIQ-X
- **VQ:** Disabled
- **zstd:** Level 19
- **Code:** `CompressionConfig::lossless()` - Lines 66-75

#### Uncompressed
- **Quantization:** LGIQ-X (no compression)
- **VQ:** Disabled
- **zstd:** Disabled
- **Code:** `CompressionConfig::uncompressed()` - Lines 77-86

### Multi-Stage Compression Pipeline
1. **Quantization** (LGIQ profiles)
2. **Vector Quantization** (optional, codebook + indices)
3. **zstd Compression** (optional, high-level streaming compression)

**Compression Statistics:**
- `struct CompressionStats` - Lines 151-167
- `struct CompressionBreakdown` - Lines 170-183
- `print()` method - Lines 192-207

---

## FILE FORMATS AND SERIALIZATION

### 1. LGI Container Format
**File:** `/packages/lgi-rs/lgi-core/src/container_format.rs`
**Lines:** 1-231

#### Parameter Encoding Modes
```rust
enum ParamEncoding {
    Euler = 0,          // σx, σy, θ (standard)
    Cholesky = 1,       // L11, L21, L22 (PSD guarantee)
    LogRadii = 2,       // log(σx), log(σy), θ
    InvCov = 3,         // Inverse covariance
    Raw = 255,          // Uncompressed f32 (lossless)
}
```

#### Compression Methods (Combinable)
```rust
enum CompressionMethod {
    None = 0,
    Zstd = 1,           // zstd compression
    Lz4 = 2,            // LZ4 compression
    Brotli = 3,         // Brotli compression
    Delta = 16,         // Delta coding (positions)
    Predictive = 32,    // Predictive coding (scales)
    VectorQuant = 64,   // Vector quantization
}
```

#### Bit Depth Options
```rust
enum BitDepth {
    Bit8 = 8,
    Bit10 = 10,
    Bit12 = 12,
    Bit14 = 14,
    Bit16 = 16,
    Float32 = 32,       // Lossless
}
```

#### Feature Flags
- `HAS_TEXTURES` - Per-Gaussian textures
- `HAS_LOD` - Level-of-detail bands
- `HAS_SPATIAL_INDEX` - Spatial indexing
- `HAS_PROGRESSIVE` - Progressive loading
- `HAS_TILING` - Tiled layout
- `HAS_METADATA` - Metadata chunks
- `IS_HDR` - HDR color space
- `HAS_BLUE_NOISE` - Blue-noise residuals

#### LGI Header Structure
```
version_major: u16
version_minor: u16
canvas_width: u32
canvas_height: u32
colorspace: u16
bitdepth: u8
alpha_mode: u8
param_encoding: u16
compression_flags: u16
gaussian_count: u32
feature_flags: u32
background_color: u32
index_offset: u64
```

#### Pre-configured Profiles
- `lossless_uncompressed()` - Lines 94-102
- `lossless_compressed()` - Lines 104-112
- `lossy_baseline()` - Lines 114-122 (LGIQ-B)
- `lossy_standard()` - Lines 124-132 (LGIQ-S + delta + zstd)
- `lossy_high_fidelity()` - Lines 134-144 (LGIQ-H + predictive)

### 2. Chunk-Based I/O
**File:** `/packages/lgi-rs/lgi-format/src/chunk.rs`
**Lines:** 1-196

#### Chunk Structure (PNG-like)
```
Length: u32 (big-endian)
Type: [u8; 4] (ASCII fourCC)
Data: [u8; length]
CRC32: u32 (big-endian, type + data)
```

#### Chunk Types
```rust
enum ChunkType {
    Head,   // File header
    Gaus,   // Gaussian data
    Meta,   // JSON metadata
    Inde,   // Spatial index
    Iend,   // End marker
}
```

**I/O Methods:**
- `Chunk::write()` - Lines 82-98
- `Chunk::read()` - Lines 101-129
- `compute_crc()` - Lines 132-137

### 3. Advanced Chunk Types (container_format.rs)
```
HEAD  - Header
GAUS  - Gaussian data
TILE  - Tile boundaries
LODC  - LOD bands
PRGS  - Progressive ordering
INDE  - Spatial index
META  - Metadata (JSON)
iCCP  - ICC profile
eXIf  - EXIF data
tEXt  - Text comments
```

### 4. Gaussian Data Storage
**File:** `/packages/lgi-rs/lgi-format/src/gaussian_data.rs`
**Lines:** 1-318

#### Storage Variants
```rust
enum GaussianData {
    Uncompressed(Vec<GaussianVector>),
    Quantized {
        profile: QuantizationProfile,
        quantized: Vec<QuantizedGaussian>,
        zstd_compressed: Option<Vec<u8>>,
    },
    VqCompressed {
        codebook: Vec<GaussianVector>,
        indices: Vec<u8>,
        zstd_compressed: Option<Vec<u8>>,
    },
}
```

#### Creation Methods
- `from_gaussians()` - Lines 45-52 (uncompressed)
- `from_gaussians_quantized()` - Lines 55-82 (with quantization profile)
- `from_gaussians_vq()` - Lines 85-101 (VQ only)
- `from_gaussians_vq_zstd()` - Lines 104-122 (VQ + zstd)

#### Conversion and Analysis
- `to_gaussians()` - Lines 125-162
- `compression_ratio()` - Lines 223-232
- `compressed_size()` - Lines 198-220
- `uncompressed_size()` - Lines 192-195

---

## ENCODER IMPLEMENTATIONS

### 1. Primary Encoder (v1)
**File:** `/packages/lgi-rs/lgi-encoder/src/lib.rs`
**Lines:** 1-104

#### Main Structure
```rust
pub struct Encoder {
    config: EncoderConfig,
}
```

#### Methods
- `new()` - Default encoder
- `with_config()` - Custom configuration
- `encode()` - Main encoding entry point (lines 49-71)
- `encode_with_progress()` - Progress callback support (lines 73-96)

### 2. Encoder Configuration
**File:** `/packages/lgi-rs/lgi-encoder/src/config.rs`
**Lines:** 1-133

#### EncoderConfig Parameters
```rust
pub struct EncoderConfig {
    init_strategy: InitStrategy,
    initial_scale: f32,
    max_iterations: usize,
    lr_position: f32,
    lr_scale: f32,
    lr_rotation: f32,
    lr_color: f32,
    lr_opacity: f32,
    loss_l2_weight: f32,
    loss_ssim_weight: f32,
    convergence_tolerance: f32,
    early_stopping_patience: usize,
    lr_decay: f32,
    lr_decay_steps: usize,
    enable_qa_training: bool,
    qa_start_iteration: usize,
    qa_codebook_size: usize,
}
```

#### Configuration Presets
- `fast()` - Lines 86-92 (500 iterations)
- `balanced()` (default) - Lines 94-97 (2000 iterations)
- `high_quality()` - Lines 99-107 (5000 iterations)
- `ultra()` - Lines 109-119 (10000 iterations, ultra-precise)

### 3. Encoder V2 (Research-Based)
**File:** `/packages/lgi-rs/lgi-encoder-v2/src/lib.rs`
**Lines:** 1-300+

#### Key Features
- Log-Cholesky covariance (PSD-guaranteed)
- Geodesic EDT (anti-bleeding)
- Structure tensor alignment (edge-aware)
- MS-SSIM loss (perceptual)
- Rate-distortion optimization

#### EncoderV2 Structure
```rust
pub struct EncoderV2 {
    target: ImageBuffer<f32>,
    structure_tensor: StructureTensorField,
    geodesic_edt: GeodesicEDT,
}
```

#### Initialization Strategies (7 variants)

##### 1. Grid-based (Default)
- `initialize_gaussians()` - Lines 83-85
- `initialize_gaussians_guided()` - Lines 88-90
- **Method:** Uniform grid placement with structure tensor alignment

##### 2. Auto Gaussian Count
- `auto_gaussian_count()` - Lines 96-100
- **Method:** Entropy-based adaptive counting

##### 3. Hybrid Gaussian Count
- `hybrid_gaussian_count()` - Lines 109-144
- **Method:** 60% entropy + 40% gradient-based

##### 4. GDGS (Gradient Domain Gaussian Splatting)
- `initialize_gdgs()` - Lines 161-169
- **File:** `/packages/lgi-rs/lgi-encoder-v2/src/gdgs_init.rs`
- **Method:** Places Gaussians at Laplacian peaks (10-100× fewer for same quality)

##### 5. Gradient Peak Strategy (Strategy H)
- `initialize_gradient_peaks()` - Lines 178-188
- **File:** `/packages/lgi-rs/lgi-encoder-v2/src/gradient_peak_init.rs`
- **Method:** 80% at gradient maxima, 20% background grid

##### 6. K-means Clustering (Strategy D)
- `initialize_kmeans()` - Lines 196-204
- **File:** `/packages/lgi-rs/lgi-encoder-v2/src/kmeans_init.rs`
- **Method:** Cluster pixels by (color, position, gradient)

##### 7. SLIC Superpixels (Strategy E)
- `initialize_slic()` - Lines 223-228
- **File:** `/packages/lgi-rs/lgi-encoder-v2/src/slic_init.rs`
- **Method:** Load preprocessing output, respects object boundaries

##### 8. Preprocessing-Guided (Advanced)
- `initialize_from_preprocessing()` - Lines 250-301
- **File:** `/packages/lgi-rs/lgi-encoder-v2/src/preprocessing_loader.rs`
- **Method:** Python preprocessing + placement probability map

### 4. Commercial Encoder (Full-Featured)
**File:** `/packages/lgi-rs/lgi-encoder-v2/src/commercial_encoder.rs`
**Lines:** 1-230+

#### CommercialConfig
```rust
pub struct CommercialConfig {
    target_gaussian_count: usize,
    quality_target: f32,
    max_iterations: usize,
    use_textures: bool,
    texture_size: usize,
    texture_threshold: f32,
    use_residuals: bool,
    residual_threshold: f32,
    use_guided_filter: bool,
    use_triggers: bool,
}
```

#### Encoding Pipeline
1. **Initialize** - Grid or guided filter (lines 104-111)
2. **Optimize Base** - OptimizerV2 (lines 116-122)
3. **Render & Analyze** - Base quality metrics (lines 125-127)
4. **Analytical Triggers** - 7 quality metrics (lines 130-146)
5. **Add Textures** - Adaptive texture mapping (lines 149-173)
6. **Render Textured** - Final quality (lines 176-178)
7. **Blue-Noise Residuals** - High-frequency detail (lines 181-200)

### 5. Encoder V3 (Adaptive)
**File:** `/packages/lgi-rs/lgi-encoder-v2/src/encoder_v3_adaptive.rs`
**Lines:** 1-80

#### Content-Adaptive Approach
```rust
pub struct EncoderV3Adaptive {
    target: ImageBuffer<f32>,
    tensor_field: StructureTensorField,
    content_type: ContentType,
}
```

#### Content Detection (ContentType enum)
- `Smooth` - gamma = 1.2
- `Sharp` - gamma = 0.4
- `HighFrequency` - gamma = 0.4
- `Photo` - gamma = 0.4

#### Process
1. Detect content type
2. Apply content-adaptive gamma
3. Initialize with conditional anisotropy
4. Optimize with OptimizerV2
5. Apply analytical triggers
6. Classify into LOD bands

---

## OPTIMIZATION STRATEGIES

### 1. Optimizer V1
**File:** `/packages/lgi-rs/lgi-encoder/src/optimizer.rs`

#### OptimizerState
```rust
pub struct OptimizerState {
    gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
    // gradient tracking, etc.
}
```

#### Optimization Method
- Gradient-based parameter updates
- Per-parameter learning rates
- Loss function (L2 + SSIM)
- Early stopping with patience

### 2. Optimizer V2 (Enhanced)
**File:** `/packages/lgi-rs/lgi-encoder-v2/src/optimizer_v2.rs`

#### OptimizerV2 Features
```rust
pub struct OptimizerV2 {
    learning_rate_position: f32,
    learning_rate_scale: f32,
    learning_rate_color: f32,
    learning_rate_rotation: f32,
    gpu_renderer: Option<GpuRendererV2>,
    use_ms_ssim: bool,
    use_edge_weighted: bool,
    tensor_field: Option<StructureTensorField>,
}
```

#### Creation Methods
- `new_with_gpu()` - Lines 70-85 (GPU acceleration)
- `new_with_ms_ssim()` - Lines 87-93 (perceptual loss)
- `new_with_gpu_and_ms_ssim()` - Lines 95-100 (combined)

#### Loss Functions
- MS-SSIM loss (perceptual)
- Edge-weighted loss
- L2 loss (default)
- Gradient computation: 454× faster with GPU

### 3. Optimizer V3 (Perceptual)
**File:** `/packages/lgi-rs/lgi-encoder-v2/src/optimizer_v3_perceptual.rs`

#### Features
- Perceptual loss weighting
- Edge-aware gradient computation
- Multi-resolution analysis

### 4. Adaptive Optimization (v1)
**File:** `/packages/lgi-rs/lgi-encoder/src/adaptive.rs`

#### AdaptiveThresholdController
- Weight thresholding for Gaussian culling
- Opacity thresholding (skip transparent)
- Contribution estimation
- Gaussian partitioning (active vs. culled)

#### LifecycleManager
- Health score tracking
- Gradient history (stagnation detection)
- Dynamic merge/prune decisions

### 5. Optimization V2 (v1)
**File:** `/packages/lgi-rs/lgi-encoder/src/optimizer_v2.rs` (lines 1-200)

#### Advanced Features
- L-BFGS optimizer support
- Quantization-aware (QA) training
- Adaptive Gaussian count
- Rate-distortion optimization

---

## LOSS FUNCTIONS

### 1. Loss Function Trait
**File:** `/packages/lgi-rs/lgi-encoder/src/loss.rs`
**Lines:** 1-140

```rust
pub trait LossFunction {
    fn compute(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32;
    fn gradient(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> ImageBuffer<f32>;
}
```

### 2. L2 Loss (MSE)
- **Structure:** `struct L2Loss`
- **Implementation:** Lines 18-51
- **Method:** Mean squared error per channel
- **Gradient:** `2(rendered - target)`

### 3. SSIM Loss (Perceptual)
- **Structure:** `struct SSIMLoss`
- **Window Size:** Configurable (default 11)
- **Implementation:** Lines 73-140

### 4. MS-SSIM Loss
**File:** `/packages/lgi-rs/lgi-core/src/ms_ssim_loss.rs`
- Multi-scale SSIM
- Perceptually aligned gradients

### 5. Edge-Weighted Loss
**File:** `/packages/lgi-rs/lgi-core/src/edge_weighted_loss.rs`
- Weights gradients by local structure
- Higher loss at edges/discontinuities

---

## INITIALIZATION STRATEGIES

**File:** `/packages/lgi-rs/lgi-core/src/initializer.rs`

### InitStrategy Enum
```rust
pub enum InitStrategy {
    Random,      // Uniform random placement
    Grid,        // Uniform grid
    Gradient,    // Edge-aware (default)
    Importance,  // Variance-based sampling
    SLIC,        // Superpixel-based
}
```

### Method Details

#### 1. Random Initialization
- `init_random()` - Lines 76-110
- Seed support for reproducibility
- Color sampling from nearby pixels

#### 2. Grid Initialization
- `init_grid()` - Lines 113-146
- Uniform spacing
- Color from grid positions

#### 3. Gradient-Based (Default)
- `init_gradient()` - Lines 149+
- Edge-aware placement
- Structure tensor alignment

#### 4. Importance Sampling
- `init_importance()` - Variance-weighted
- High variance regions get more Gaussians

#### 5. SLIC Superpixels
- `init_slic()` - Boundary-respecting
- Pre-computed superpixels

---

## VECTOR QUANTIZATION

**File:** `/packages/lgi-rs/lgi-encoder/src/vector_quantization.rs`
**Lines:** 1-270

### GaussianVector
```rust
pub struct GaussianVector {
    pub data: [f32; 9],  // [μx, μy, σx, σy, θ, r, g, b, α]
}
```

### VectorQuantizer
```rust
pub struct VectorQuantizer {
    pub codebook: Vec<GaussianVector>,
    pub codebook_size: usize,
    pub trained: bool,
}
```

### K-means Implementation
- **Initialization:** K-means++ (lines 121-151)
- **Assignment:** Nearest centroid (line 94)
- **Update:** Centroid recalculation (line 98)
- **Convergence:** Distance threshold 1e-6 (line 106)
- **Training:** Lines 80-118

### Methods
- `new()` - Create quantizer
- `train()` - K-means training
- `quantize_all()` - Encode all Gaussians
- `dequantize()` - Decode from codebook + indices
- `distance_squared()` - Vector distance metric

### Compression Gains
- Codebook size: 256 (8-bit indices)
- 5-10× compression with <1 dB quality loss
- From GaussianImage ECCV 2024

---

## RENDERING ENGINES

### 1. Renderer V2 (CPU, Validation)
**File:** `/packages/lgi-rs/lgi-encoder-v2/src/renderer_v2.rs`
**Lines:** 1-88

#### Features
- Weighted average renderer
- Gaussian evaluation with cutoff (3.5 sigma)
- Mahalanobis distance computation
- Per-pixel normalization

#### Algorithm
```
For each pixel (px, py):
  weight_sum = 0
  color_sum = [0, 0, 0]
  For each gaussian g:
    dist² = Mahalanobis distance
    if dist² > 12.25: continue
    w = opacity × exp(-0.5 × dist²)
    weight_sum += w
    color_sum += w × color
  output = color_sum / weight_sum
```

### 2. Renderer GPU
**File:** `/packages/lgi-rs/lgi-encoder-v2/src/renderer_gpu.rs`

#### Features
- GPU acceleration (WGPU/Vulkan)
- 454× speedup vs CPU
- Used in OptimizerV2

### 3. Renderer V3 (Textured)
**File:** `/packages/lgi-rs/lgi-encoder-v2/src/renderer_v3_textured.rs`

#### Features
- Per-Gaussian texture mapping
- Texture blending
- Production-grade quality

---

## FILE I/O AND SERIALIZATION

### 1. LGI Reader (v1)
**File:** `/packages/lgi-rs/lgi-format/src/reader.rs`
**Lines:** 1-200

#### LgiReader Methods
- `read_file()` - Lines 18-23
- `read()` - Lines 26-75
- `read_header()` - Lines 78-94
- `read_header_file()` - Lines 97-101

#### Reading Process
1. Validate magic number
2. Read chunks sequentially
3. Parse HEAD chunk (header)
4. Parse GAUS chunk (Gaussian data)
5. Parse optional META/INDE chunks
6. Find IEND marker

### 2. LGI Writer (v1)
**File:** `/packages/lgi-rs/lgi-format/src/writer.rs`

#### Creation Methods
- `new()` - Default writer
- `new_lossless()` - Lines 32-42 (f32, no quantization)
- `new_compressed()` - Lines 44-56 (LGIQ-B + zstd + delta)
- `new_progressive()` - Lines 58-71 (importance-ordered)

#### Writing Process
- `write_file()` - Lines 74-118
- `write_header()` - Lines 120-135
- `write_gaus_chunk()` - Baseline quantized
- `write_gaus_chunk_lossless()` - Full precision
- `write_gaus_chunk_compressed()` - With compression
- `write_prgs_chunk()` - Progressive ordering

### 3. File Reader V2
**File:** `/packages/lgi-rs/lgi-encoder-v2/src/file_reader.rs`
**Lines:** 1-200

#### ProgressiveResult Structure
- header: LGIHeader
- gaussians: Vec of Gaussians
- importance_order: Optional ordering

#### Methods
- `read_file()` - Lines 24-29
- `read_file_progressive()` - Lines 32-75
- `read_header()` - Lines 77-127
- `read_gaus_chunk()` - Multiple variants
- `read_prgs_chunk()` - Progressive metadata

### 4. File Writer V2
**File:** `/packages/lgi-rs/lgi-encoder-v2/src/file_writer.rs`
**Lines:** 1-200

---

## ADVANCED COMPRESSION TECHNIQUES

### 1. Progressive Encoding
**File:** `/packages/lgi-rs/lgi-core/src/progressive.rs`

#### Methods
- `order_by_importance()` - Importance ordering
- `reorder_gaussians()` - Reorder for streaming

#### Use Cases
- Streaming/progressive loading
- ROI-based transmission
- Variable quality decoding

### 2. Compression Utilities
**File:** `/packages/lgi-rs/lgi-core/src/compression_utils.rs`

#### Techniques
- Delta coding (positions)
- Predictive coding (scales)
- Entropy coding preparation

### 3. Predictive Coding
**File:** `/packages/lgi-rs/lgi-core/src/predictive_coding.rs`

#### Method
- Predict scale from neighboring Gaussians
- Encode delta (residual)
- Improves zstd compression

### 4. Rate-Distortion Optimization
**File:** `/packages/lgi-rs/lgi-core/src/rate_distortion.rs`

#### Features
- Optimal Gaussian count selection
- Quality target specification
- Automatic parameter tuning

---

## ADVANCED FEATURES

### 1. Textured Gaussians
**File:** `/packages/lgi-rs/lgi-core/src/textured_gaussian.rs`

#### TexturedGaussian2D Structure
- Base Gaussian parameters
- Optional texture map (8×8, 16×16, etc.)
- Texture variance tracking

#### Methods
- `from_gaussian()` - Convert base Gaussian
- `should_add_texture()` - Variance threshold check
- `extract_texture_from_image()` - Adaptive extraction

### 2. Blue-Noise Residuals
**File:** `/packages/lgi-rs/lgi-core/src/blue_noise_residual.rs`

#### Features
- Detects residual regions after Gaussian fitting
- Blue-noise dithering pattern
- High-frequency detail recovery

#### Methods
- `detect_residual_regions()` - Identify areas needing residuals
- `apply_to_image()` - Apply residual pattern

### 3. Analytical Triggers
**File:** `/packages/lgi-rs/lgi-core/src/analytical_triggers.rs`

#### 7 Quality Metrics
1. **SED** - Spectral Energy Drop (frequency domain)
2. **ERR** - Entropy-Residual Ratio
3. **LCC** - Laplacian Consistency
4. **DCT** - DCT coefficient analysis
5. **Frequency response** - Multi-band
6. **Texture detection** - Content-based
7. **Edge metrics** - Boundary preservation

#### Usage
- Quality gate decisions
- Trigger-based adaptations
- Automatic feature enabling

### 4. LOD System
**File:** `/packages/lgi-rs/lgi-core/src/lod_system.rs`

#### LODSystem Structure
- Gaussian classification into LOD bands
- Progressive loading support
- Level selection based on view distance

### 5. Tiling
**File:** `/packages/lgi-rs/lgi-core/src/tiling.rs`

#### Features
- Tile-based layout
- Spatial locality
- Cache efficiency

### 6. Spatial Indexing
**File:** `/packages/lgi-rs/lgi-core/src/spatial_index.rs`

#### Structures
- BVH (bounding volume hierarchy)
- Grid-based spatial partitioning
- O(log n) lookup

### 7. Viewport Culling
**File:** `/packages/lgi-rs/lgi-core/src/viewport_culling.rs`

#### Features
- Screen-space culling
- Visibility determination
- Render optimization

### 8. Structure Tensor
**File:** `/packages/lgi-rs/lgi-core/src/structure_tensor.rs`

#### StructureTensorField Computation
- Gradient computation
- Eigenvalue/eigenvector analysis
- Edge orientation detection
- Coherence measurement

#### Usage
- Edge-aware initialization
- Anisotropic Gaussian shaping
- Content analysis

### 9. Geodesic EDT
**File:** `/packages/lgi-rs/lgi-core/src/geodesic_edt.rs`

#### Features
- Boundary-respecting distance transform
- Anti-bleeding (no distance across edges)
- Used for scale clamping

### 10. Guided Filter
**File:** `/packages/lgi-rs/lgi-core/src/guided_filter.rs`

#### Features
- Edge-preserving smoothing
- Color initialization refinement
- Structure-aware filtering

---

## COMPREHENSIVE FILE MAPPING

### LGI Format Package (`lgi-format`)
| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| Compression | compression.rs | 1-240 | Config, presets, stats |
| Quantization | quantization.rs | 1-536 | LGIQ profiles |
| Chunk I/O | chunk.rs | 1-196 | PNG-like chunks |
| Gaussian Data | gaussian_data.rs | 1-318 | Storage variants |
| Header | header.rs | 1-134 | File header |
| Reader | reader.rs | 1-200 | File reading |
| Writer | writer.rs | - | File writing |

### Encoder Package (`lgi-encoder`)
| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| Main | lib.rs | 1-104 | Encoder struct |
| Config | config.rs | 1-133 | Configuration |
| Loss | loss.rs | 1-140 | Loss functions |
| Optimizer | optimizer.rs | - | Gradient descent |
| Optimizer V2 | optimizer_v2.rs | 1-200+ | Advanced optimization |
| Adaptive | adaptive.rs | 1-100+ | Adaptive strategies |
| Vector Quantization | vector_quantization.rs | 1-270 | K-means VQ |
| Autodiff | autodiff.rs | - | Gradient computation |
| Metrics | metrics_collector.rs | - | Performance tracking |

### Encoder V2 Package (`lgi-encoder-v2`)
| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| Main | lib.rs | 1-300+ | EncoderV2 struct, 8 init strategies |
| Commercial | commercial_encoder.rs | 1-230+ | Full-featured encoder |
| Adaptive V3 | encoder_v3_adaptive.rs | 1-80 | Content-adaptive |
| Optimizer V2 | optimizer_v2.rs | 1-100+ | Enhanced optimizer |
| Optimizer V3 | optimizer_v3_perceptual.rs | - | Perceptual loss |
| Renderer V2 | renderer_v2.rs | 1-88 | CPU renderer |
| Renderer GPU | renderer_gpu.rs | - | GPU acceleration |
| Renderer V3 | renderer_v3_textured.rs | - | Textured rendering |
| File Reader | file_reader.rs | 1-200 | LGI file reading |
| File Writer | file_writer.rs | 1-200 | LGI file writing |
| Error-Driven | error_driven.rs | - | Error-based placement |
| GDGS Init | gdgs_init.rs | - | Gradient domain |
| Gradient Peak Init | gradient_peak_init.rs | - | Peak-based (H) |
| K-means Init | kmeans_init.rs | - | Clustering (D) |
| SLIC Init | slic_init.rs | - | Superpixels (E) |
| Preprocessing | preprocessing_loader.rs | - | Python integration |

### Core Package (`lgi-core`) - 59 modules
| Module | Purpose |
|--------|---------|
| initializer.rs | 5 initialization strategies |
| structure_tensor.rs | Edge detection |
| geodesic_edt.rs | Boundary-respecting distance |
| container_format.rs | Advanced LGI format spec |
| quantization.rs | (duplicate of lgi-format) |
| progressive.rs | Progressive ordering |
| compression_utils.rs | Delta/predictive coding |
| rate_distortion.rs | Rate-distortion opt |
| renderer.rs | Core rendering |
| textured_gaussian.rs | Texture mapping |
| blue_noise_residual.rs | Residual patterns |
| analytical_triggers.rs | 7 quality metrics |
| lod_system.rs | Level-of-detail |
| tiling.rs | Tiled layout |
| spatial_index.rs | BVH/grid indexing |
| viewport_culling.rs | Screen-space culling |
| guided_filter.rs | Edge-preserving filter |
| ms_ssim.rs | Multi-scale SSIM |
| ms_ssim_loss.rs | Perceptual loss |
| edge_weighted_loss.rs | Edge-weighted loss |
| saliency_detection.rs | Salient regions |
| content_detection.rs | Content type analysis |
| adaptive_splitting.rs | Dynamic splitting |
| [55+ more specialized modules] | Image processing, optimization, metrics |

---

## SUMMARY STATISTICS

### Codec Variants
- **Quantization Profiles:** 4 (LGIQ-B, S, H, X)
- **Compression Methods:** 6 (Zstd, LZ4, Brotli, Delta, Predictive, VQ)
- **Parameter Encodings:** 5 (Euler, Cholesky, LogRadii, InvCov, Raw)
- **Bit Depths:** 6 (8, 10, 12, 14, 16, 32-bit)

### Initialization Strategies
- **Core:** 5 (Random, Grid, Gradient, Importance, SLIC)
- **V2 Specialized:** 4 (GDGS, Gradient Peak, K-means, SLIC)
- **Total Distinct:** 8 strategies

### Optimizers
- **V1:** Basic gradient descent
- **V2:** Enhanced with GPU, MS-SSIM, edge-weighting
- **V3:** Perceptual loss optimization
- **Adaptive:** Threshold-based culling and lifecycle management

### Loss Functions
- **L2 (MSE):** Baseline
- **SSIM:** Window-based structural similarity
- **MS-SSIM:** Multi-scale perceptual loss
- **Edge-Weighted:** Structure-aware loss

### Advanced Features
- **Textures:** Per-Gaussian texture mapping
- **Residuals:** Blue-noise high-frequency patterns
- **LOD:** Level-of-detail bands
- **Progressive:** Streaming and importance-ordered loading
- **Spatial Index:** BVH, grid-based access
- **Content Detection:** Auto-adaptive gamma
- **Analytical Triggers:** 7-metric quality assessment

---

## CODEC QUALITY TARGETS

| Profile | PSNR Range | Bytes/Gaussian | Quality Focus |
|---------|-----------|-----------------|---|
| LGIQ-B | 28-32 dB | 13 | Balanced |
| LGIQ-S | 30-34 dB | 14 | Standard |
| LGIQ-H | 35-40 dB | 20 | High quality |
| LGIQ-X | Lossless | 36 | Bit-exact |
| Textured | +2-4 dB | +50-200 | Detail recovery |
| + Residuals | +1-2 dB | +10-50 | High-frequency |

---

## COMPRESSION PIPELINE EXAMPLE

```
Input Gaussians
  ↓
[Quantization Profile: LGIQ-B]
  ↓ (13 bytes/Gaussian)
[Vector Quantization: 256 codebook]
  ↓ (codebook + 8-bit indices)
[Delta Coding: Positions]
  ↓ (predict from neighbors)
[Predictive Coding: Scales]
  ↓ (predict from neighbors)
[zstd Compression: Level 9]
  ↓ (high-entropy streaming)
Output LGI File (.lgi)
  ↓
[Chunk Format: HEAD + GAUS + META + IEND]
  ↓
Final Compressed File
```

**Expected Compression:** 4-10× (balanced), up to 15× (aggressive)
**Decoding Speed:** 1000+ FPS (GPU)

---

## KEY IMPLEMENTATION FILES (ABSOLUTE PATHS)

### Core Codec
1. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-format/src/quantization.rs` - 4 quantization profiles
2. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-format/src/compression.rs` - Compression config
3. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-format/src/gaussian_data.rs` - Storage modes
4. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-core/src/container_format.rs` - File format spec

### Encoders
5. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-encoder/src/lib.rs` - Encoder V1
6. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-encoder-v2/src/lib.rs` - Encoder V2
7. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-encoder-v2/src/commercial_encoder.rs` - Full-featured

### I/O
8. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-format/src/reader.rs` - Reader V1
9. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-encoder-v2/src/file_reader.rs` - Reader V2
10. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-encoder-v2/src/file_writer.rs` - Writer V2

### Optimization
11. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-encoder/src/optimizer.rs` - Optimizer V1
12. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-encoder-v2/src/optimizer_v2.rs` - Optimizer V2
13. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-encoder/src/loss.rs` - Loss functions
14. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-core/src/ms_ssim_loss.rs` - Perceptual loss

### Vector Quantization
15. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-encoder/src/vector_quantization.rs` - VQ codec

### Rendering
16. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-encoder-v2/src/renderer_v2.rs` - CPU renderer
17. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-encoder-v2/src/renderer_v3_textured.rs` - Textured

### Advanced Features
18. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-core/src/textured_gaussian.rs` - Textures
19. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-core/src/blue_noise_residual.rs` - Residuals
20. `/home/user/lamcogaussianimage/packages/lgi-rs/lgi-core/src/analytical_triggers.rs` - Quality metrics

