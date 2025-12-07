# Advanced Adam Optimization Techniques for 2D Gaussian Splatting

This document summarizes key Adam optimizer adaptations from the 3D Gaussian Splatting (3DGS) literature and their application to 2D image encoding.

## Background

Plain Adam in the original 3DGS uses per-parameter-group learning rates and position LR decay (Plenoxels-style), but is otherwise stock Adam on dense tensors. Several papers have improved upon this baseline.

## Key Techniques

### 1. Per-Gaussian Time Counter (BOGausS)

**Problem**: When Gaussians are updated irregularly (e.g., sparse updates, post-densification), the global time counter `t` becomes incorrect for bias correction.

**Solution**: Maintain a separate step counter `t_i` for each Gaussian. Only increment when that Gaussian receives an update.

```rust
// Per-Gaussian time counter
state.t += 1;  // Only incremented when this Gaussian is updated

let bias_correction1 = 1.0 - beta1.powi(state.t as i32);
let bias_correction2 = 1.0 - beta2.powi(state.t as i32);
```

**Benefit**: Correct bias correction for all Gaussians, even after densification or with sparse updates.

**Test**: `test_adam_advanced.rs` - `per_gaussian_time` configuration

---

### 2. State Inheritance After Densification (BOGausS)

**Problem**: When splitting a Gaussian, children start with zero optimizer state (m=0, v=0, t=0), causing:
- Slow warm-up period after each split
- "Newborn Gaussian" instability
- Poor convergence after densification

**Solution**: Inherit parent's optimizer state:
```rust
// Child inherits:
m_child = m_parent / 2   // Split momentum between children
v_child = v_parent       // Keep variance estimate
t_child = t_parent       // Keep time counter
```

**Benefit**: Children immediately have useful gradient statistics, faster convergence post-densification.

**Test**: `test_densification_with_inheritance.rs`

---

### 3. Closed-Form Color Updates

**Insight**: Given fixed positions and shapes, rendering is LINEAR in colors:
```
I_hat(x,y) = sum_i w_i(x,y) * c_i
```

**Solution**: Periodically solve for optimal colors via least-squares:
```
minimize ||I - W*c||^2
```
This has a closed-form solution: `c = (W^T W)^-1 W^T I`

**Implementation**:
1. Use Adam ONLY for positions and shapes
2. Every M iterations, solve for colors in closed form
3. This is like a mini Gauss-Newton step in color space

**Benefit**: Potentially saves many gradient iterations on color parameters.

**Test**: `test_closed_form_colors.rs`

---

### 4. Confidence-Scaled Updates (BOGausS-inspired)

**Problem**: All Gaussians are updated with the same step size, but some are well-constrained (covering important regions) while others are uncertain.

**Solution**: Scale updates by a confidence factor based on:
- Number of pixels the Gaussian covers
- Local error in the Gaussian's footprint
- Gradient energy

```rust
let confidence = compute_confidence(gaussian, error_map);
param -= lr * confidence * m_hat / (v_hat.sqrt() + eps);
```

**Benefit**: Well-constrained Gaussians update boldly; uncertain ones move slowly to avoid instability.

**Test**: `test_adam_advanced.rs` - `confidence_scaling` configuration

---

### 5. Sparse Adam Updates (Taming-3DGS)

**Problem**: Computing gradients for all Gaussians every iteration is wasteful. Many Gaussians don't contribute to high-error regions.

**Solution**: Only update Gaussians that touch high-error pixels:
1. Compute error map: `E = |I - I_hat|`
2. Identify high-error pixels/regions
3. Only update Gaussians overlapping those regions

**Benefit**: ~1.6x training speedup at same quality (reported by Taming-3DGS).

**Test**: `test_adam_advanced.rs` - `sparse_updates` configuration

---

## Per-Parameter Learning Rates (Already Implemented)

Following 3DGS best practices:

| Parameter | Default LR | Notes |
|-----------|------------|-------|
| Position  | 0.0002     | Lowest - positions are "touchiest" |
| Scale     | 0.005      | Medium - shape parameters |
| Color     | 0.02       | Highest - well-behaved, almost linear |
| Opacity   | 0.05       | Medium-high |

Position LR additionally follows exponential decay schedule.

---

## Second-Order Alternatives (Future Work)

Papers like 3DGS-LM and LM-IS replace Adam with Levenberg-Marquardt, reporting:
- 20-30% shorter training time
- Similar or better quality
- Requires efficient GPU Jacobian-vector products

This is a promising direction for future encoder improvements.

---

## Structural Regularization (Future Work)

**DropGaussian**: Randomly drop a fraction of Gaussians during training (like dropout on primitives):
- Avoids overfitting near-camera Gaussians
- Boosts gradients for poorly-seen Gaussians
- Improves sparse-view quality

**Spectral/Shape Regularization**: Penalties on:
- Extreme anisotropy (prevents "needle" Gaussians)
- Large spatial coverage
- Spectral entropy of covariance

---

## References

1. **Original 3DGS**: "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (Kerbl et al., 2023)
2. **BOGausS**: "Better Optimized Gaussian Splatting" (EUSIPCO 2025) - precision-aware Adam, state inheritance
3. **Taming-3DGS**: Sparse/fused Adam for 1.6x speedup
4. **3DGS-LM**: Levenberg-Marquardt optimizer for 3DGS (ICCV 2025)
5. **LM-IS**: LM with importance sampling (Pehlivan et al., 2025)
6. **DropGaussian**: Structural dropout regularization

---

## Test Status

| Test File | Features Tested | Status |
|-----------|-----------------|--------|
| `test_adam_advanced.rs` | Per-Gaussian time, confidence, sparse | Created |
| `test_closed_form_colors.rs` | Closed-form color solver | Created |
| `test_densification_with_inheritance.rs` | State inheritance | Created |
| `test_densification_v2.rs` | Fixed opacity + delayed start | Created |

Results will be saved to `lgi-encoder-v2/test-results/` with TestResult framework.
