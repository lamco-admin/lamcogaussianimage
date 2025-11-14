# LGI Test Data

**Purpose**: Test images for LGI codec validation and benchmarking
**Total**: 105 images, 252 MB
**Storage**: Git LFS (requires Git LFS for full access)

---

## Quick Start

### For Claude Code Web (No Git LFS)

If you see only 14 images instead of 105, Git LFS is not available:

```bash
cd test-data
./download_test_images.sh
```

This downloads:
- ✅ Kodak dataset (24 images, public domain)
- ✅ Checks synthetic patterns (should be in repo)
- ⚠️ Real photos require Git LFS or local access

**Minimum viable**: Kodak (24) + Synthetic (13-16) = **~37-40 images sufficient for development**

---

### For Local Development (With Git LFS)

```bash
# Install Git LFS
git lfs install

# Pull all images
git lfs pull

# Verify
find . -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l
# Should show 105
```

---

## Dataset Breakdown

### 1. test_images_new_synthetic/ (16 images, 7.9 MB)

**Controlled test patterns** for specific technique validation:

**High-Frequency (Sharp Edges)**:
- `HF_checkerboard_1px.png` - 1-pixel checkerboard (extreme edges)
- `HF_hairlines.png` - Thin lines (edge handling)
- `HF_multi_gratings.png` - Multiple frequency patterns
- `HF_woven_texture.png` - Complex weave pattern
- `HF_fBm_multiscale.png` - Fractal multi-scale texture

**Band-Limited (Smooth)**:
- `BN_blurred_discs.png` - Smooth circular blobs
- `BN_lowfreq_field.png` - Low-frequency variations
- `BN_radial_gradient.png` - Smooth radial gradient
- `BN_gradient_horizontal.png` - Linear gradient
- `BN_gradient_vertical.png` - Linear gradient
- `BN_uniform_gray.png` - Solid color (baseline)

**Masks & Analysis**:
- `MASK_variance_map.png` - Variance visualization
- `MASK_variance_threshold.png` - Thresholded variance

**Purpose**: Controlled testing, technique validation, quick benchmarks
**Usage**: `fast_benchmark.rs` (30 seconds)
**Status**: Should be in repository (not LFS, only 7.9 MB)

---

### 2. kodak-dataset/ (24 images, 56 MB)

**Industry-standard benchmark** for codec comparisons:

**Images**: kodim01.png through kodim24.png
- Resolution: 768×512 each
- Content: Natural scenes, people, objects, textures
- Public domain (http://r0k.us/graphics/kodak/)

**Purpose**: Standard benchmarks, published comparisons
**Usage**: `kodak_benchmark.rs` (10 minutes)
**Status**: Via Git LFS OR downloadable via script

---

### 3. test_images/ (68 images, 188 MB)

**Real-world 4K photos** for validation:

**Content Diversity**:
- Natural scenes: 35 images
- Urban/architecture: 12 images
- Portraits/people: 10 images
- Mixed: 11 images

**Resolution**: Mostly 4K (3840×2160), some 1080p
**Size**: 2-6 MB per image (JPEG compressed)
**Date Range**: 2013-2025 (12 year span)

**Purpose**: Real-world validation, empirical tuning
**Usage**: `real_world_benchmark.rs` (2-4 hours)
**Status**: Via Git LFS (188 MB - requires LFS or local access)

---

## Workarounds for Claude Code Web

### Option 1: Use Minimal Dataset (Recommended)

**Available without LFS**:
- Synthetic patterns: 13-16 images ✅
- Kodak (via download): 24 images ✅
- **Total**: ~37-40 images

**Sufficient for**:
- Algorithm validation ✅
- Technique testing ✅
- Quality measurements ✅
- Performance benchmarks ✅

**Run**:
```bash
./download_test_images.sh  # Gets Kodak dataset
cd ../packages/lgi-rs
cargo run --release --example fast_benchmark
```

---

### Option 2: Download Kodak Only (Fast)

```bash
cd kodak-dataset
for i in $(seq -f "%02g" 1 24); do
    curl -L "http://r0k.us/graphics/kodak/kodak/kodim${i}.png" -o "kodim${i}.png"
done
```

**Result**: 24 industry-standard images (56 MB)
**Time**: ~2-3 minutes
**Usage**: Standard benchmarks, sufficient for validation

---

### Option 3: Generate Synthetic Test Set

If neither works, generate minimal test patterns:

```bash
cd ../packages/lgi-rs/lgi-benchmarks
cargo run --release --bin generate_test_patterns
# Creates synthetic patterns in test-data/generated/
```

**Result**: 10-15 synthetic images
**Usage**: Controlled testing, technique validation

---

## Usage in Benchmarks

### Fast Benchmark (30 seconds)
```bash
cd packages/lgi-rs/lgi-encoder-v2
cargo run --release --example fast_benchmark
```
**Uses**: `test_images_new_synthetic/` (2 images)
**Tests**: Sharp edge + complex pattern
**Purpose**: Quick validation after changes

---

### Kodak Benchmark (10 minutes)
```bash
cd packages/lgi-rs/lgi-benchmarks
cargo run --release --bin kodak_benchmark
```
**Uses**: `kodak-dataset/` (24 images)
**Tests**: Industry-standard images
**Purpose**: Standard comparisons, published results

---

### Real Photo Benchmark (2-4 hours)
```bash
cd packages/lgi-rs/lgi-benchmarks
cargo run --release --bin real_world_benchmark
```
**Uses**: `test_images/` (68 images)
**Tests**: Real 4K photos, diverse content
**Purpose**: Real-world validation, empirical tuning
**Note**: Requires Git LFS or local access

---

## File Formats

**Supported**:
- PNG (lossless, for Kodak and synthetic)
- JPEG (lossy, for real photos)

**Loaded via**:
- `image` crate (Rust)
- Automatic format detection
- RGB color space (converted if needed)

---

## Storage Information

### With Git LFS (Paid GitHub Account)
- Total: 252 MB
- 105 images fully tracked
- Automatic download with `git lfs pull`

### Without Git LFS (Claude Code Web)
- Synthetic: ~7.9 MB (14 images available)
- Kodak: Download via script (56 MB, 24 images)
- Real photos: Not accessible (requires LFS)
- **Minimum viable**: 38 images total

---

## Recommendations

### For Claude Code Web Development

**Use**:
1. Synthetic patterns (available immediately)
2. Kodak dataset (download via script)
3. **Total**: ~40 images, sufficient for:
   - Algorithm validation ✅
   - Technique testing ✅
   - Benchmarking ✅
   - Quality measurement ✅

**Skip** (for now):
- Full 68-photo real-world benchmark
- Can validate locally when needed

### For Local Development

**Use**:
```bash
git lfs install
git lfs pull
# All 105 images available
```

---

## Quick Validation

```bash
# Check what you have
find . -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l

# If < 40: Run download script
./download_test_images.sh

# Verify Kodak
ls kodak-dataset/*.png | wc -l  # Should be 24

# Verify synthetic
ls test_images_new_synthetic/*.png | wc -l  # Should be 13-16
```

---

**Last Updated**: November 14, 2025
**Status**: Workaround available for Claude Code Web (Kodak + Synthetic = ~40 images)
