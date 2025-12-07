# FORMAT & COMPRESSION CATALOG - Gaussian Image Codec
## File Format, Quantization, and Compression Pipeline

**Document Version:** 2.0
**Date:** 2025-12-07
**Source:** `packages/lgi-rs/lgi-format/src/` (2,116 lines)
**Based On:** Nov 15 CODEC_CATALOG.md (updated with current code)

---

## TABLE OF CONTENTS

1. [Overview](#overview)
2. [Quantization Profiles (LGIQ)](#quantization-profiles)
3. [Compression Pipeline](#compression-pipeline)
4. [File Format Structure](#file-format-structure)
5. [Chunk-Based I/O](#chunk-based-io)
6. [Gaussian Data Storage](#gaussian-data-storage)
7. [File Location Index](#file-location-index)

---

## OVERVIEW

The lgi-format crate implements the LGI file format, quantization profiles, and compression pipeline. This layer converts optimized Gaussians into compressed binary files.

### Architecture Flow

```
Gaussians (from encoder)
    ↓
Quantization (LGIQ-B/S/H/X profiles)
    ↓
Vector Quantization (optional, codebook compression)
    ↓
zstd Compression (optional, entropy coding)
    ↓
Chunk-based I/O (PNG-like structure)
    ↓
.lgi file on disk
```

---

## QUANTIZATION PROFILES

**File:** `lgi-format/src/quantization.rs`
**Lines:** 535
**Purpose:** 4 quantization profiles trading quality for size

### 1. LGIQ-B (Balanced Profile)

**Bytes Per Gaussian:** 13 bytes (byte-aligned)
**Quality Range:** 28-32 dB PSNR
**Code:** Lines 125-203

#### Bit Layout
```
Position X:  16-bit (2 bytes)
Position Y:  16-bit (2 bytes)
Scale X:     12-bit (1.5 bytes) ┐
Scale Y:     12-bit (1.5 bytes) ├─ 3 bytes packed
Rotation:    12-bit (1.5 bytes) ┘ + 2 bytes alignment = 5 total
Color R:     8-bit (1 byte)
Color G:     8-bit (1 byte)
Color B:     8-bit (1 byte)
Opacity:     8-bit (1 byte)
────────────────────────────────
Total:       13 bytes
```

#### Encoding Features
- Position: [0, 1] normalized, 16-bit precision
- Scale: log₂ encoding (σ_min = 0.001), 12-bit
- Rotation: [-π, π] normalized, 12-bit
- Color: Linear RGB, 8-bit per channel
- Opacity: [0, 1], 8-bit

#### Methods
- `quantize_balanced(gaussian)` - Line 125
- `dequantize_balanced(bytes)` - Line 165

---

### 2. LGIQ-S (Standard/Small Profile)

**Bytes Per Gaussian:** 14 bytes (byte-aligned)
**Quality Range:** 30-34 dB PSNR
**Code:** Lines 221-303

#### Bit Layout
```
Position X:  16-bit (2 bytes)
Position Y:  16-bit (2 bytes)
Scale X:     14-bit (higher precision)
Scale Y:     14-bit
Rotation:    14-bit
────────────────────────────────
Total:       14 bytes (with packing)
```

#### Enhancements over LGIQ-B
- Higher scale precision (14-bit vs 12-bit)
- Higher color precision (10-bit vs 8-bit)
- HDR color support

#### Methods
- `quantize_small(gaussian)` - Line 221
- `dequantize_small(bytes)` - Line 266

---

### 3. LGIQ-H (High Quality Profile)

**Bytes Per Gaussian:** 20 bytes
**Quality Range:** 35-40 dB PSNR
**Code:** Lines 326-369

#### Bit Layout (IEEE 754 float16)
```
All 9 parameters as f16:
Position X:  f16 (2 bytes)
Position Y:  f16 (2 bytes)
Scale X:     f16 (2 bytes)
Scale Y:     f16 (2 bytes)
Rotation:    f16 (2 bytes)
Color R:     f16 (2 bytes)
Color G:     f16 (2 bytes)
Color B:     f16 (2 bytes)
Opacity:     f16 (2 bytes)
Padding:     2 bytes (reserved)
────────────────────────────────
Total:       20 bytes
```

#### Precision
- Float16: 3-4 decimal digits
- Dynamic range: 10⁻⁸ to 65,504
- Suitable for high-quality applications

#### Methods
- `quantize_high(gaussian)` - Line 326
- `dequantize_high(bytes)` - Line 347

---

### 4. LGIQ-X (Lossless Profile)

**Bytes Per Gaussian:** 36 bytes
**Quality:** Bit-exact reconstruction (∞ dB PSNR)
**Code:** Lines 373-407

#### Bit Layout (IEEE 754 float32)
```
All 9 parameters as f32:
9 parameters × 4 bytes = 36 bytes
```

#### Features
- Full precision IEEE 754 float32
- Bit-exact reversibility
- No quantization loss
- Maximum file size

#### Methods
- `quantize_lossless(gaussian)` - Line 373
- `dequantize_lossless(bytes)` - Line 390

---

### Batch Operations

**Functions:**
- `quantize_all(gaussians, profile)` - Line 411
- `dequantize_all(bytes, profile)` - Line 420

Process entire Gaussian vectors at once.

---

## COMPRESSION PIPELINE

**File:** `lgi-format/src/compression.rs`
**Lines:** 239
**Purpose:** Multi-stage compression combining quantization, VQ, and zstd

### CompressionConfig Structure

```rust
pub struct CompressionConfig {
    pub quantization: QuantizationProfile,  // LGIQ-B/S/H/X
    pub enable_vq: bool,                    // Vector quantization
    pub vq_codebook_size: usize,            // 128-256 typical
    pub enable_zstd: bool,                  // Entropy coding
    pub zstd_level: i32,                    // 1-22 (9 default)
}
```

### Compression Presets

#### 1. Balanced (Default)
**Function:** `CompressionConfig::balanced()`
- Quantization: LGIQ-B (13 bytes)
- VQ: Enabled (256 codebook)
- zstd: Level 9
- Expected ratio: 4-10×

#### 2. Small (Maximum Compression)
**Function:** `CompressionConfig::small()`
- Quantization: LGIQ-B
- VQ: Enabled (128 codebook - smaller)
- zstd: Level 19 (highest compression)
- Expected ratio: Maximum

#### 3. High Quality
**Function:** `CompressionConfig::high_quality()`
- Quantization: LGIQ-H (20 bytes)
- VQ: Disabled (preserve quality)
- zstd: Level 9
- Focus: Quality over size

#### 4. Lossless
**Function:** `CompressionConfig::lossless()`
- Quantization: LGIQ-X (36 bytes)
- VQ: Disabled
- zstd: Level 19
- Bit-exact reconstruction

#### 5. Uncompressed
**Function:** `CompressionConfig::uncompressed()`
- Quantization: LGIQ-X
- VQ: Disabled
- zstd: Disabled
- Raw float32 storage

### Compression Stages

**Stage 1: Quantization**
- Input: Vec<Gaussian2D> (float32)
- Output: Vec<u8> (quantized bytes)
- Module: quantization.rs

**Stage 2: Vector Quantization (Optional)**
- Input: Quantized Gaussians
- Output: Codebook + indices
- Compression: 5-10× typical
- Module: lgi-core/vector_quantization.rs

**Stage 3: zstd Entropy Coding (Optional)**
- Input: Quantized or VQ data
- Output: Compressed byte stream
- Compression: 2-4× additional
- Module: zstd Rust crate

**Total Compression:** 4-15× depending on configuration and content

---

## FILE FORMAT STRUCTURE

**File:** `lgi-format/src/header.rs` + `lgi-core/src/container_format.rs`
**Lines:** 133 (header) + container format code

### LGI File Header

```
Magic:              [u8; 4]   "LGI\0"
Version Major:      u16
Version Minor:      u16
Canvas Width:       u32
Canvas Height:      u32
Colorspace:         u16
Bit Depth:          u8
Alpha Mode:         u8
Param Encoding:     u16       (Euler/Cholesky/LogRadii/InvCov/Raw)
Compression Flags:  u16       (None/Zstd/Lz4/Brotli/Delta/Predictive/VQ)
Gaussian Count:     u32
Feature Flags:      u32       (Textures/LOD/SpatialIndex/Progressive/etc.)
Background Color:   u32       (RGBA packed)
Index Offset:       u64       (for random access)
```

### Parameter Encoding Modes

```rust
pub enum ParamEncoding {
    Euler = 0,          // σx, σy, θ (standard, intuitive)
    Cholesky = 1,       // L11, L21, L22 (PSD guarantee)
    LogRadii = 2,       // log(σx), log(σy), θ (better quantization)
    InvCov = 3,         // Σ⁻¹ elements (fast rendering)
    Raw = 255,          // Uncompressed f32 (lossless)
}
```

**Current Implementation:** Euler (default)
**Future:** Log-Cholesky for better numerical properties

### Compression Methods (Bit Flags, Combinable)

```rust
pub enum CompressionMethod {
    None = 0,
    Zstd = 1,           // zstd compression (default)
    Lz4 = 2,            // LZ4 compression (faster, less compression)
    Brotli = 3,         // Brotli compression (slower, better compression)
    Delta = 16,         // Delta coding for positions
    Predictive = 32,    // Predictive coding for scales
    VectorQuant = 64,   // Vector quantization
}
```

### Feature Flags

```rust
Feature Flags (u32 bitmask):
HAS_TEXTURES       = 0x0001  // Per-Gaussian textures
HAS_LOD            = 0x0002  // Level-of-detail bands
HAS_SPATIAL_INDEX  = 0x0004  // Spatial indexing structure
HAS_PROGRESSIVE    = 0x0008  // Progressive loading order
HAS_TILING         = 0x0010  // Spatial tiling
HAS_METADATA       = 0x0020  // Metadata chunk present
IS_HDR             = 0x0040  // HDR color space
HAS_BLUE_NOISE     = 0x0080  // Blue-noise residuals
```

---

## CHUNK-BASED I/O

**File:** `lgi-format/src/chunk.rs`
**Lines:** 195
**Format:** PNG-like chunk structure

### Chunk Structure

```
┌──────────────────────┐
│ Length (4 bytes)     │  u32 big-endian
├──────────────────────┤
│ Type (4 bytes)       │  ASCII fourCC (e.g., "HEAD", "GAUS")
├──────────────────────┤
│ Data (length bytes)  │  Chunk payload
├──────────────────────┤
│ CRC32 (4 bytes)      │  u32 big-endian (of type + data)
└──────────────────────┘
```

### Standard Chunk Types

| Type | FourCC | Purpose | Required |
|------|--------|---------|----------|
| Header | HEAD | File metadata | Yes |
| Gaussian Data | GAUS | Compressed Gaussians | Yes |
| Metadata | meta | JSON metadata | No |
| Index | INDE | Random access index | No |
| End | IEND | End marker | Yes |

### Extended Chunk Types (From container_format.rs)

| Type | FourCC | Purpose | Optional |
|------|--------|---------|----------|
| Tiles | TILE | Spatial tile boundaries | Yes |
| LOD Bands | LODC | Level-of-detail data | Yes |
| Progressive | PRGS | Loading order | Yes |
| ICC Profile | iCCP | Color management | Yes |
| EXIF | eXIf | Camera metadata | Yes |
| Text | tEXt | Comments | Yes |

### Chunk I/O Methods

**Writing:**
- `Chunk::write(writer, type, data)` - Line 82
- Computes CRC32 automatically
- Big-endian byte order

**Reading:**
- `Chunk::read(reader)` - Line 101
- Validates CRC32
- Returns ChunkType and data

**CRC Validation:**
- `compute_crc(type, data)` - Line 132
- Uses crc32fast crate

---

## GAUSSIAN DATA STORAGE

**File:** `lgi-format/src/gaussian_data.rs`
**Lines:** 317
**Purpose:** Multiple storage variants for Gaussian data

### Storage Enum

```rust
pub enum GaussianData {
    /// Raw float32 (lossless)
    Uncompressed(Vec<GaussianVector>),

    /// Quantized with profile
    Quantized {
        profile: QuantizationProfile,
        quantized: Vec<QuantizedGaussian>,
        zstd_compressed: Option<Vec<u8>>,
    },

    /// Vector quantization + optional zstd
    VqCompressed {
        codebook: Vec<GaussianVector>,
        indices: Vec<u8>,
        zstd_compressed: Option<Vec<u8>>,
    },
}
```

### Creation Methods

**Uncompressed:**
- `from_gaussians(gaussians)` - Line 45
- Direct storage as float32

**Quantized:**
- `from_gaussians_quantized(gaussians, profile)` - Line 55
- Applies LGIQ-B/S/H/X profile
- Optional zstd compression

**Vector Quantized:**
- `from_gaussians_vq(gaussians, codebook_size)` - Line 85
- K-means codebook creation
- Index-based representation

**VQ + zstd:**
- `from_gaussians_vq_zstd(gaussians, codebook_size, level)` - Line 104
- Combined VQ and entropy coding

### Conversion & Analysis

**To Gaussians:**
- `to_gaussians()` - Line 125
- Reverses quantization/compression

**Size Analysis:**
- `uncompressed_size()` - Line 192
- `compressed_size()` - Line 198
- `compression_ratio()` - Line 223

Returns actual bytes required for storage.

---

## COMPRESSION CONFIGURATION

**File:** `lgi-format/src/compression.rs`
**Lines:** 239

### CompressionConfig

```rust
pub struct CompressionConfig {
    pub quantization: QuantizationProfile,
    pub enable_vq: bool,
    pub vq_codebook_size: usize,
    pub enable_zstd: bool,
    pub zstd_level: i32,  // 0-22 (9 recommended)
}
```

### Preset Configurations

#### balanced()
Default preset balancing quality and size:
- Quantization: LGIQ-B
- VQ: Enabled (256 codebook)
- zstd: Level 9
- Ratio: 4-10×

#### small()
Maximum compression:
- Quantization: LGIQ-B
- VQ: Enabled (128 codebook)
- zstd: Level 19
- Ratio: Maximum

#### high_quality()
Quality-focused:
- Quantization: LGIQ-H
- VQ: Disabled
- zstd: Level 9

#### lossless()
Bit-exact:
- Quantization: LGIQ-X
- VQ: Disabled
- zstd: Level 19

#### uncompressed()
No compression:
- Quantization: LGIQ-X
- VQ: Disabled
- zstd: Disabled

---

## FILE I/O

### Writer

**File:** `lgi-format/src/writer.rs`
**Lines:** 83
**Purpose:** Write .lgi files

#### Process
1. Write magic number + header
2. Write GAUS chunk (compressed Gaussian data)
3. Write optional chunks (META, etc.)
4. Write IEND chunk

### Reader

**File:** `lgi-format/src/reader.rs`
**Lines:** 199
**Purpose:** Read and validate .lgi files

#### Process
1. Validate magic number
2. Read HEAD chunk
3. Read GAUS chunk
4. Decompress and dequantize Gaussians
5. Read optional chunks
6. Validate structure

### Validation

**File:** `lgi-format/src/validation.rs`
**Lines:** 77
**Purpose:** File structure validation

Checks:
- Magic number correctness
- Chunk CRC integrity
- Required chunks present
- Gaussian parameter ranges

---

## FILE LOCATION INDEX

### lgi-format Crate Structure

```
packages/lgi-rs/lgi-format/src/
├── lib.rs (169 lines) - Public API
├── quantization.rs (535 lines) - LGIQ-B/S/H/X profiles
├── compression.rs (239 lines) - Compression config
├── chunk.rs (195 lines) - PNG-like chunk I/O
├── gaussian_data.rs (317 lines) - Storage variants
├── header.rs (133 lines) - File header
├── reader.rs (199 lines) - File reading
├── writer.rs (83 lines) - File writing
├── validation.rs (77 lines) - Structure validation
├── metadata.rs (97 lines) - JSON metadata
└── error.rs (72 lines) - Error types
```

**Total:** 2,116 lines

### Related Modules

```
packages/lgi-rs/lgi-core/src/
├── container_format.rs - Extended chunk types
├── quantization.rs - LGIQ profiles (may duplicate lgi-format)
├── vector_quantization.rs - VQ implementation
├── compression_utils.rs - Delta coding, zstd helpers
└── rate_distortion.rs - R-D optimization
```

---

## QUANTIZATION PROFILE COMPARISON

| Profile | Bytes | Position | Scale | Rotation | Color | Opacity | PSNR Range |
|---------|-------|----------|-------|----------|-------|---------|------------|
| LGIQ-B | 13 | 16-bit | 12-bit log₂ | 12-bit | 8-bit RGB | 8-bit | 28-32 dB |
| LGIQ-S | 14 | 16-bit | 14-bit | 14-bit | 10-bit RGB | 10-bit | 30-34 dB |
| LGIQ-H | 20 | f16 | f16 | f16 | f16 RGB | f16 | 35-40 dB |
| LGIQ-X | 36 | f32 | f32 | f32 | f32 RGBA | f32 | ∞ (lossless) |

---

## COMPRESSION RATIO ESTIMATES

### Typical Compression (by profile + method)

| Configuration | Bytes/Gaussian | With VQ (256) | With zstd | Total Ratio |
|---------------|----------------|---------------|-----------|-------------|
| LGIQ-B raw | 13 | - | - | 1.0× (baseline) |
| LGIQ-B + VQ | 13 | 2-3× | - | 2-3× |
| LGIQ-B + zstd | 13 | - | 2-4× | 2-4× |
| LGIQ-B + VQ + zstd | 13 | 2-3× | 1.5-2× | 4-10× |
| LGIQ-H + zstd | 20 | - | 2-3× | 2-3× |
| LGIQ-X + zstd | 36 | - | 2-4× | 2-4× |

**Note:** Ratios are vs uncompressed float32 (36 bytes/Gaussian)

---

## SUMMARY STATISTICS

### Module Count
- **Total Modules:** 11 in lgi-format
- **Total Lines:** 2,116
- **Quantization Profiles:** 4 (B/S/H/X)
- **Compression Presets:** 5 (balanced/small/high/lossless/uncompressed)
- **Chunk Types:** 11+ (5 standard + 6 extended)

### Compression Pipeline Options

**Quantization:** 4 choices
**VQ:** 2 choices (enabled/disabled) × codebook sizes
**zstd:** 2 choices × 23 levels
**Total Combinations:** 100+ distinct compression configurations

---

*End of Format & Compression Catalog - 2025-12-07*
