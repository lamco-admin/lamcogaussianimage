# Important Note for Claude Code Web Users

**Date**: November 14, 2025
**Issue**: Git LFS files not accessible in Claude Code Web environment

---

## The Issue

Claude Code Web does **not** have Git LFS installed, so LFS-tracked files appear as pointer files rather than actual images.

**Impact**:
- 105 test images stored via Git LFS
- Only ~14 images accessible in Claude Code Web
- Blocks full real-world benchmark validation

---

## The Solution

### Immediate Workaround (Available Now)

**Use the minimal dataset** that IS available:

1. **Synthetic patterns** (~14 images, available)
   - Location: `test-data/test_images_new_synthetic/`
   - Use for: Controlled testing, technique validation

2. **Kodak dataset** (24 images, downloadable)
   - Run: `cd test-data && ./download_test_images.sh`
   - Downloads from public domain source
   - Industry-standard benchmarks

**Total Available**: ~38 images (sufficient for development) ✅

---

## What You Can Do

### ✅ CAN DO (With ~38 Images)

**Algorithm validation**:
```bash
cd packages/lgi-rs/lgi-encoder-v2
cargo run --release --example fast_benchmark
# Uses 2 synthetic images, validates techniques
```

**Kodak benchmarking** (after download):
```bash
cd test-data && ./download_test_images.sh
cd ../packages/lgi-rs/lgi-benchmarks
cargo run --release --bin kodak_benchmark
# Tests on 24 industry-standard images
```

**Technique testing**:
- Synthetic patterns for controlled experiments ✅
- Kodak for standard validation ✅
- Quality measurements ✅
- Performance benchmarks ✅

---

### ⚠️ CANNOT DO (Requires Full Dataset)

**Full real-world benchmark**:
```bash
cargo run --release --bin real_world_benchmark
# Needs 68 4K photos (188 MB via LFS)
```

**Empirical R-D curve fitting**:
- Needs diverse photo dataset
- Can start with Kodak (24 images)
- Full tuning requires 60+ images

---

## Recommended Workflow

### For Claude Code Web

**Phase 1: Algorithm Development** (Kodak + Synthetic)
- Use available 38 images ✅
- Validate techniques
- Test optimizations
- Measure quality improvements
- **Sufficient for Track 1 P1 completion**

**Phase 2: Full Validation** (When needed)
- Switch to local development with Git LFS
- Run full 68-photo benchmark
- Empirical R-D curve fitting
- Production validation

---

### Hybrid Approach (Recommended)

**Claude Code Web**: Development and testing (Kodak + Synthetic)
**Local CLI**: Full validation when needed (all 105 images)

**Benefits**:
- ✅ Develop anywhere (Claude Code Web)
- ✅ Validate anywhere (Kodak sufficient)
- ✅ Full benchmarks when ready (local)

---

## Download Script Usage

```bash
cd test-data
./download_test_images.sh
```

**Downloads**:
- Kodak dataset (24 PNG images, ~56 MB)
- From: http://r0k.us/graphics/kodak/ (public domain)
- Time: ~2-3 minutes

**Result**:
- Total images: ~38 (14 synthetic + 24 Kodak)
- Sufficient for: Development, validation, benchmarking
- Missing: 68 real photos (available via Git LFS locally)

---

## Alternative: Keep Synthetic Images in Regular Git

If we want the synthetic images always available (not LFS):

```bash
# Remove from LFS
git lfs untrack "test-data/test_images_new_synthetic/*.png"

# Add back to regular git
git add test-data/test_images_new_synthetic/
git commit -m "Move synthetic images to regular git (7.9 MB, always accessible)"
```

**Benefit**: 16 synthetic images always available (no LFS needed)
**Cost**: 7.9 MB added to repo size (acceptable)

---

## Summary

**For Claude Code Web**:
- ✅ Synthetic patterns: Available (14 images)
- ✅ Kodak dataset: Downloadable (24 images, 2 min)
- ⚠️ Real photos: Requires Git LFS (68 images)
- ✅ **Total workable**: ~38 images (sufficient for development)

**Recommendation**: Use Kodak + Synthetic for Claude Code Web development, run full 68-photo benchmarks locally when needed for final validation.

**No show-stopper**: Can proceed with development using available datasets ✅

---

**Last Updated**: November 14, 2025
**Status**: Workaround documented, development can proceed
