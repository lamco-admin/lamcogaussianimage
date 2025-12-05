# Optimizer Verification Status

**Date**: 2025-12-05
**Test Image**: kodim03.png (768×512)
**Test Config**: 256 Gaussians (16×16 grid), 50 iterations

---

## Summary Table

| Optimizer | Status | Initial PSNR | Final PSNR | Change | Notes |
|-----------|--------|--------------|------------|--------|-------|
| Adam (per-param LR) | **WORKS** | 19.48 dB | 20.66 dB | +1.19 dB | Wrong gradient magnitude but right direction |
| OptimizerV2 | **FIXED** | 19.47 dB | 19.63 dB | +0.15 dB | Position.x sign fixed, converges now |
| OptimizerV2 (Adam LRs) | **BUG FOUND** | 19.47 dB | 14.79 dB | -4.69 dB | (Pre-fix results) |
| OptimizerV3 | NOT TESTED | - | - | - | Requires StructureTensor |
| Hybrid | RUNNING | - | - | - | Adam+L-BFGS |
| L-M | NOT TESTED | - | - | - | Very slow (finite diff) |

---

## Detailed Findings

### Adam Optimizer: WORKS

**File**: `adam_optimizer.rs`

**Result**: Loss decreased monotonically from 0.0113 to 0.0086 (+1.19 dB)

**Config that works**:
```
lr_position: 0.0002 → 0.00002 (exponential decay)
lr_color: 0.02
lr_scale: 0.005
lr_opacity: 0.05
beta1: 0.9, beta2: 0.999
```

**Gradient computation**: SIMPLIFIED
- Uses simple isotropic distance: `dx² + dy²`
- Early cutoff at `dist_sq > 0.1` (normalized coords)
- Ignores rotation in gradient
- Position grad: `error_weighted * weight * dx` (simple)

---

### OptimizerV2: BROKEN

**File**: `optimizer_v2.rs`

**Result**: Loss INCREASED regardless of learning rate!

| LR Config | Initial | Final | Change |
|-----------|---------|-------|--------|
| Default (0.1, 0.6, 0.1) | 19.47 dB | 15.06 dB | -4.42 dB |
| Adam-like (0.0002, 0.02, 0.005) | 19.47 dB | 14.79 dB | -4.69 dB |
| 10× Adam (0.002, 0.2, 0.05) | 19.47 dB | 14.95 dB | -4.52 dB |

**Key observation**: Loss barely changes (~2%) over 50 iterations with any LR.

**Gradient computation**: COMPLEX
- Uses anisotropic rotated Mahalanobis distance
- Computes rotation gradient
- More "mathematically correct" but doesn't work

**ROOT CAUSE CONFIRMED** (see "V2 ROOT CAUSE FOUND" section below):
1. ~~Complex gradient computation may have sign errors~~ **YES - position.x gradient has wrong sign!**
2. ~~Rotation gradient may be interfering with convergence~~ Rotation gradient is correct
3. ~~Different loss function normalization~~ Not the issue

---

### Key Difference: Adam vs V2 Gradient

| Aspect | Adam (WORKS) | V2 (BROKEN) |
|--------|--------------|-------------|
| Distance | `dx² + dy²` | Rotated Mahalanobis |
| Cutoff | `> 0.1` (coords) | `> 12.25` (σ units) |
| Rotation | Ignored | Computed |
| Position grad | Simple `weight * dx` | Complex w/ derivatives |

**Hypothesis**: Adam's simplified gradient (ignoring rotation) is more stable. V2's "correct" gradient may have implementation bugs or numerical instability.

---

## Infrastructure Verified

- **Test framework**: `TestResult` struct and JSON saving work correctly
- **Renderer**: `RendererV2::render()` produces images
- **Loss computation**: MSE/PSNR calculations are consistent
- **Image loading**: Kodak dataset loads correctly

---

## Experiments Saved

```
test-results/2025-12-19/
├── verify_adam_basic_*.json          # Adam PASS
├── verify_optimizer_v2_*.json        # V2 FAIL
├── v2_lr_sensitivity_*.json          # V2 FAIL at all LRs
└── verify_hybrid_*.json              # Pending
```

---

## Recommendations

### Immediate Actions

1. **Use Adam for all experiments** - Only working optimizer
2. **Do NOT use OptimizerV2** - Broken, makes results worse
3. **Investigate V2 gradients** - Compare with finite-difference check
4. **Test L-M on small problem** - Finite diff is slow but correct

### Investigation Needed

1. **V2 gradient bug hunt**: Compare analytical vs finite-diff gradients
2. **Rotation handling**: Does disabling rotation in V2 help?
3. **Scale gradient sign**: Check if scale gradients are inverted

---

## GRADIENT MAGNITUDE BUG FOUND (2025-12-05)

**Test**: `verify_gradients_fd.rs` - Compare analytical vs finite-difference gradients

### Results (64×64 image, 1 Gaussian, ε=0.01)

| Parameter | Finite-Diff Gradient | Analytical Gradient | Ratio | Sign Match |
|-----------|---------------------|---------------------|-------|------------|
| Color R | 0.0 | 0.0 | - | ✓ |
| Color G | 0.0 | 0.0 | - | ✓ |
| Position X | **-0.41** | -0.0018 | **226×** | ✓ |
| Position Y | 0.0 | 0.0 | - | ✓ |
| Scale X | **-0.99** | ~0 | **∞** | ✓ |
| Scale Y | **-0.99** | ~0 | **∞** | ✓ |

### Analysis

**Signs are correct** (optimizer moves in right direction) - explains why Adam eventually converges.

**Magnitudes are WILDLY WRONG**:
- Position gradient underestimated by **226×**
- Scale gradients essentially **ZERO** in analytical but **~1.0** in reality!

### Impact

This explains:
1. Why Adam needs many iterations (magnitude too small)
2. Why scale parameters barely change during optimization
3. Why per-parameter learning rates are critical (to compensate for wrong magnitudes)

### Root Cause Hypothesis

The analytical gradient formula in Adam uses simplified isotropic distance:
```rust
let dist_sq = dx * dx + dy * dy;
let weight = (-0.5 * dist_sq / scale_product).exp();
grad_scale_x += error_weighted * weight * dist_sq / scale_x_sq;
```

But this doesn't match the actual renderer's Gaussian evaluation, leading to massive magnitude errors.

### Recommendation

The optimizer "works" because direction is right, but it's **severely undertrained** because the learning rate multiplied by gradient magnitude is too small. Either:
1. Fix gradient formula to match renderer
2. Increase learning rates by ~200× for position, much more for scale

### Confirmed Bug Location: `adam_optimizer.rs:434-435`

```rust
// INCORRECT formula:
gradients[i].scale_x += error_weighted * weight * dist_sq / scale_x_sq;
```

The correct Gaussian derivative `d(exp(-dist_sq/(2*sx*sy)))/d(sx)` should be:
```rust
// CORRECT formula (approximation):
gradients[i].scale_x += error_weighted * weight * 0.5 * dist_sq / (scale_x.powi(3) * scale_y);
```

**Why Adam still converges despite wrong gradients**: Adam's adaptive learning rate via running second moment `v_hat.sqrt()` acts as automatic rescaling. Over iterations, the division by `sqrt(v)` approximates the true gradient scale, allowing convergence despite wrong analytical formula.

---

## CRITICAL: Renderer vs Gradient Mismatch

### Renderer Forward Pass (renderer_v2.rs:44-59)

```rust
// 1. Rotate to Gaussian's local frame
let dx_rot = dx * cos(θ) + dy * sin(θ);
let dy_rot = -dx * sin(θ) + dy * cos(θ);

// 2. Rotated Mahalanobis distance
let dist_sq = (dx_rot / sx)² + (dy_rot / sy)²;

// 3. Gaussian evaluation
let weight = exp(-0.5 * dist_sq);
```

### Gradient Computation (adam_optimizer.rs:410-415)

```rust
// 1. NO rotation applied!
let dx = px - gaussian.position.x;
let dy = py - gaussian.position.y;

// 2. Simple isotropic distance
let dist_sq = dx * dx + dy * dy;

// 3. Wrong scaling formula
let weight = exp(-0.5 * dist_sq / scale_product);  // ≠ renderer
```

### The Mismatch

| Aspect | Renderer | Gradient |
|--------|----------|----------|
| Rotation | dx_rot, dy_rot with θ | Raw dx, dy (NO θ) |
| Distance | (dx_rot/sx)² + (dy_rot/sy)² | dx² + dy² |
| Exp argument | -0.5 × dist_sq | -0.5 × dist_sq / (sx×sy) |

**This is not a small bug - the formulas are completely different!**

The gradient computation would only match the renderer if:
1. θ = 0 (no rotation) AND
2. sx = sy = 1 (unit scales)

For any real Gaussian with rotation or non-unit scales, gradients are fundamentally wrong.

---

## V2 FIX APPLIED AND VERIFIED (2025-12-05)

**Fix**: Negate position.x gradient only (position.y was already correct)

**Location**: `optimizer_v2.rs:326-327`
```rust
gradients[i].position.x -= error_weighted * grad_weight_x;  // MINUS for x
gradients[i].position.y += error_weighted * grad_weight_y;  // PLUS for y (original)
```

**Verification Result**:
- All gradient signs now match finite-difference
- V2 converges: 19.47 → 19.63 dB (+0.15 dB)
- Still slower than Adam due to gradient magnitude mismatch (226×)

---

## V2 ROOT CAUSE FOUND: Position Gradient Sign Bug

**Test**: `verify_v2_gradient_signs.rs` - Compare V2 gradient direction vs finite-difference

### Results (128×128 image, 1 off-center rotated Gaussian)

| Parameter | FD Gradient | V2 Gradient | Sign Match |
|-----------|-------------|-------------|------------|
| position.x | **+0.224** | **-0.428** | **NO ✗ OPPOSITE!** |
| position.y | -0.018 | -0.000 | YES ✓ |
| scale_x | +1.254 | +0.691 | YES ✓ |
| scale_y | +1.162 | +0.820 | YES ✓ |
| rotation | ~0 | ~0 | YES ✓ |
| color.r | +0.458 | +0.017 | YES ✓ |

### The Bug

V2's position.x gradient has the **WRONG SIGN** compared to finite-difference!

When gradient descent updates: `x -= lr × grad`
- FD says gradient is **positive** → x should **decrease**
- V2 says gradient is **negative** → V2 **increases** x instead!

**This causes V2 to move Gaussians in the wrong direction, increasing loss!**

### Bug Location: `optimizer_v2.rs:314-315`

```rust
// V2's position gradient formula (BUGGY)
let grad_weight_x = weight * (dx_rot * cos_t / (sx * sx) + dy_rot * (-sin_t) / (sy * sy));
let grad_weight_y = weight * (dx_rot * sin_t / (sx * sx) + dy_rot * cos_t / (sy * sy));
```

The issue is the transformation from local (rotated) coordinates back to global coordinates. The formula appears to have a sign error in how `grad_weight_x` combines the rotated components.

---

## Code Quality Observations

- **Adam**: Has defensive checks (`weight.is_nan()`, scale clamping)
- **V2**: Fewer defensive checks, more complex math
- **Hybrid**: Uses Adam internally (should inherit Adam's behavior)
- **L-M**: Uses levenberg-marquardt crate with finite diff (slow but correct)

---

*This document is evidence about tool behavior, not conclusions. Different configurations may yield different results.*
