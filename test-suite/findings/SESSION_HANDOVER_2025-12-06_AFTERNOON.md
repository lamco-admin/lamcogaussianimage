# Session Handover - December 6, 2025 (Afternoon)

## Executive Summary

**MAJOR BREAKTHROUGH**: Hybrid pruning approach achieves **+0.69 dB improvement over baseline with 27% fewer Gaussians**. This session identified and fixed two critical bugs that were sabotaging previous densification experiments.

---

## Key Results This Session

### Combined Densification Test (Final Results)

| Approach | kodim03 | Gaussians | vs Baseline |
|----------|---------|-----------|-------------|
| **Baseline** | 21.61 dB | 256 | — |
| Densify (64→256) | 19.60 dB | 273 | **-2.01 dB** |
| Aggressive (64→256) | 19.17 dB | 305 | **-2.44 dB** |
| **Hybrid (256→prune)** | **22.30 dB** | **187** | **+0.69 dB** |

| Approach | kodim05 | Gaussians | vs Baseline |
|----------|---------|-----------|-------------|
| **Baseline** | 17.77 dB | 256 | — |
| Densify (64→256) | 17.96 dB | 314 | +0.19 dB |

### Critical Finding: Growing vs Pruning

- **Growing from sparse (64→256) FAILS**: Adds Gaussians with poor momentum, destabilizes optimization
- **Pruning from dense (256→optimal) SUCCEEDS**: Keeps well-optimized Gaussians, removes redundant ones

---

## Bugs Found and Fixed

### Bug 1: Learning Rates Too Low (2-5×)

**File**: `test_combined_densification.rs`

**Symptom**: Baseline achieved only 19.36 dB instead of expected ~21-22 dB

**Root Cause**: LR values were copied incorrectly from earlier experiments

| Parameter | Wrong Value | Correct Value | Factor |
|-----------|-------------|---------------|--------|
| lr_color | 0.01 | 0.02 | 2× |
| lr_scale | 0.001 | 0.005 | 5× |
| lr_opacity | 0.01 | 0.05 | 5× |

**Impact**: +2 dB improvement from fix

### Bug 2: Gradient Computation Coordinate Space

**File**: `test_combined_densification.rs` function `compute_gradients()`

**Symptom**: PSNR diverged after LR fix (20.27 → 18.70 dB)

**Root Cause**: Gradient computation used **pixel coordinates** instead of **normalized (0-1) coordinates**

```rust
// WRONG - pixel space (values 0-768)
let dx = px as f32 - g.position.x * width;
let dy = py as f32 - g.position.y * height;

// CORRECT - normalized space (values 0-1)
let px_norm = (px as f32 + 0.5) / width;
let py_norm = (py as f32 + 0.5) / height;
let dx = px_norm - g.position.x;
let dy = py_norm - g.position.y;
```

**Impact**: This caused gradient magnitudes to be ~768× too large, overwhelming the optimizer

---

## Insights From This Session

### 1. Why Growing From Sparse Fails

When we add new Gaussians via densification:
- New Gaussians have **fresh Adam state** (m=0, v=0)
- But optimization has already progressed (LR may have decayed)
- The momentum mismatch causes new Gaussians to "fight" the existing optimization

### 2. Why Pruning Works

When we prune from dense:
- All Gaussians are **well-optimized** with good momentum
- Pruning removes **low-opacity** (redundant) Gaussians
- Remaining Gaussians have already found good positions
- Re-optimization converges quickly to better quality

### 3. The Hybrid Approach

```
Phase 1: Start with 256 Gaussians, optimize 100 iterations
Phase 2: Prune Gaussians with opacity < 0.1 → ~187 remain
Phase 3: Re-optimize 100 iterations → 22.30 dB
```

Total: 200 iterations, 218.4 seconds (faster than baseline's 253s!)

### 4. Coordinate Spaces Are Critical

The 2D Gaussian splatting math assumes:
- Positions in **normalized (0-1) space**
- Scales represent **fraction of image dimension**
- Gradients must be computed in **the same space**

Mixing coordinate spaces (pixel vs normalized) causes catastrophic optimization failure.

---

## Updated Performance Comparison

| Configuration | PSNR | Gaussians | Speed | Notes |
|---------------|------|-----------|-------|-------|
| **Hybrid Prune** | **22.30 dB** | **187** | 218s | **NEW BEST** |
| Baseline (256) | 21.61 dB | 256 | 253s | Previous best |
| 3DGS Faithful (prev session) | 20.89 dB | 123 | — | Good but slower |
| Growing Densification | 19.60 dB | 273 | 365s | Worse, slower |

---

## Key Learnings Consolidated

### What We Now Know Works

1. **Jittered grid + Adam optimizer** = solid baseline (21.61 dB)
2. **Hybrid pruning** = +0.69 dB improvement with fewer Gaussians
3. **Per-parameter LR** (position, color, scale, opacity) is critical
4. **Normalized coordinate space** for all gradient computations

### What Doesn't Work

1. **Growing from sparse** - momentum mismatch problem
2. **Closed-form color solver** - destroys quality (-7.84 dB)
3. **Advanced Adam features without densification** - no benefit
4. **Feature-based initialization** (K-Means, GDGS, gradient peaks) - all worse than grid

### What We're Uncertain About

1. **Why kodim05 shows different behavior** - densification helps slightly (+0.19 dB)
2. **Optimal pruning threshold** - 0.1 worked but may need tuning
3. **Whether state inheritance helps hybrid** - untested combination

---

## Code Changes This Session

### Modified: `test_combined_densification.rs`

1. Fixed learning rates:
```rust
lr_color: 0.02,     // was 0.01
lr_scale: 0.005,    // was 0.001
lr_opacity: 0.05,   // was 0.01
```

2. Fixed gradient computation to use normalized coordinates
3. Added hybrid pruning approach with three phases

---

## Files Reference

### Test Files Created/Modified
```
examples/test_combined_densification.rs  # MODIFIED - all fixes applied
examples/test_3dgs_faithful_densification.rs  # Previous session
examples/test_adam_advanced.rs           # Adam feature tests
```

### Log Files
```
test-results/overnight-logs/
├── combined_v2_fixed.log    # Current test (has results)
├── combined_densification_fixed.log
├── adam_advanced_fixed.log
├── 3dgs_faithful.log
```

### Documentation
```
test-results/
├── SESSION_HANDOVER_2025-12-06_AFTERNOON.md  # This file
├── SESSION_HANDOVER_2025-12-06_MORNING.md
├── SESSION_HANDOVER_2025-12-06.md
└── research-state.json  # Machine-readable state
```

---

## Recommended Next Steps

### Immediate (High Priority)

1. **Productionize hybrid pruning**
   - Move pruning logic into a proper module
   - Test on full Kodak dataset (24 images)
   - Tune pruning threshold (try 0.05, 0.15, 0.2)

2. **Update baseline comparison**
   - New baseline should be hybrid (22.30 dB, 187 Gaussians)
   - All future experiments should compare against this

### Short-term (Medium Priority)

3. **Combine hybrid + state inheritance**
   - State inheritance helped +0.48 dB on kodim08
   - May provide additional gains on difficult images

4. **Investigate kodim05 behavior**
   - Why does growing densification help on this image?
   - May reveal image-dependent strategies

### Long-term (Low Priority)

5. **Perceptual loss exploration**
   - MS-SSIM, LPIPS instead of just L2
   - May improve subjective quality

6. **Adaptive pruning threshold**
   - Automatically determine optimal threshold per-image
   - Based on opacity distribution

---

## Summary Statistics

| Metric | Previous Best | This Session | Improvement |
|--------|---------------|--------------|-------------|
| kodim03 PSNR | 21.61 dB | **22.30 dB** | **+0.69 dB** |
| Gaussians | 256 | **187** | **-27%** |
| Encoding time | 253s | **218s** | **-14%** |

**We achieved better quality, with fewer Gaussians, in less time.**

---

## Technical Details for Future Sessions

### Correct Learning Rates (3DGS-calibrated)
```rust
lr_position: 0.0002,  // Small, positions are sensitive
lr_color: 0.02,       // Colors can learn faster
lr_scale: 0.005,      // Scales need moderate LR
lr_opacity: 0.05,     // Opacity needs larger LR
```

### Correct Gradient Computation
```rust
// Always use normalized (0-1) coordinates
let width = target.width as f32;
let height = target.height as f32;
let px_norm = (px as f32 + 0.5) / width;
let py_norm = (py as f32 + 0.5) / height;
let dx = px_norm - gaussian.position.x;
let dy = py_norm - gaussian.position.y;
```

### Hybrid Pruning Pseudocode
```rust
// Phase 1: Initial optimization
for iter in 0..100 { optimize(gaussians) }

// Phase 2: Prune low-opacity
let threshold = 0.1;
gaussians.retain(|g| g.opacity >= threshold);

// Phase 3: Re-optimize
for iter in 0..100 { optimize(gaussians) }
```

---

*End of Session Handover*
