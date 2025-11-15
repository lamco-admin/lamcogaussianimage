# Optimization Theory for Gaussian Splatting

**Status**: Phase 1 Research - In Progress
**Last Updated**: November 14, 2025

---

## Overview

This document establishes the mathematical theory for optimizing Gaussian parameters to minimize reconstruction error. Understanding this is critical to debugging our current regression (loss stuck at 0.375001).

---

## 1. The Optimization Problem

### 1.1 Problem Statement

**Given**:
- Target image I_target: H×W×3 (pixel colors)
- Initial Gaussians: {G₁, ..., Gₙ} with parameters θᵢ

**Find**: Parameters θ* that minimize reconstruction error

```
θ* = argmin_θ  L(Render(θ), I_target)
```

Where:
- θ = all Gaussian parameters: {μ, Σ, c, α} for each Gaussian
- Render(θ) = rendered image from Gaussians
- L = loss function (measures error)

### 1.2 Parameter Space

For N Gaussians, we optimize:

**Per Gaussian** (9 parameters):
```
μ = (μₓ, μᵧ) ∈ [0,1]²          // Position (2)
Σ = (σₓ, σᵧ, θ) ∈ ℝ⁺×ℝ⁺×[0,π)  // Shape (3, Euler parameterization)
  or (l₁₁, l₂₁, l₂₂) ∈ ℝ³        // Shape (3, Log-Cholesky)
c = (r, g, b) ∈ [0,1]³         // Color (3)
α ∈ [0,1]                       // Opacity (1)
```

**Total**: 9N parameters

**Constraints**:
- Positions bounded: μ ∈ [0,1]²
- Scales positive: σ > 0
- Colors/opacity bounded: [0,1]
- Covariance must be positive definite (automatically satisfied by parameterization)

---

## 2. Loss Functions

### 2.1 Mean Squared Error (MSE / L2)

**Definition**:
```
L_MSE = (1/P) Σₚ ||I_render(p) - I_target(p)||²
```

Where:
- P = number of pixels (H × W)
- p = pixel position
- ||·||² = squared Euclidean norm (sum of squared RGB differences)

**Gradient**:
```
∂L_MSE/∂I_render(p) = (2/P) × (I_render(p) - I_target(p))
```

**Properties**:
- Simple, differentiable
- Pixel-wise independent
- Can be noisy (sensitive to outliers)
- Doesn't capture perceptual quality well

**PSNR Relationship**:
```
PSNR = 10 × log₁₀(MAX² / MSE)
     = -10 × log₁₀(MSE)  if MAX=1.0

Where MAX = maximum pixel value (1.0 for normalized)
```

### 2.2 L1 Loss (Absolute Error)

**Definition**:
```
L_L1 = (1/P) Σₚ |I_render(p) - I_target(p)|
```

**Gradient**:
```
∂L_L1/∂I_render(p) = (1/P) × sign(I_render(p) - I_target(p))
```

**Properties**:
- More robust to outliers than L2
- Discontinuous gradient at zero (but not a problem in practice)
- Used in 3DGS alongside SSIM

### 2.3 MS-SSIM (Multi-Scale Structural Similarity)

**Definition** (simplified):
```
SSIM(X, Y) = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂))
```

Where:
- μₓ, μᵧ = mean luminance
- σₓ², σᵧ² = variance
- σₓᵧ = covariance
- C₁, C₂ = stability constants

**MS-SSIM**: Computed at multiple scales (downsample image, compute SSIM)

**Loss form**:
```
L_SSIM = 1 - SSIM(I_render, I_target)
```

**Properties**:
- Perceptually motivated (matches human perception better)
- More complex to compute
- Used in 3DGS: L = (1-λ)×L₁ + λ×L_SSIM

**We have**: `lgi-core/src/ms_ssim_loss.rs` (already implemented!)

### 2.4 Combined Loss (3DGS Approach)

From Kerbl et al., 2023:
```
L = (1 - λ) × L₁ + λ × (1 - SSIM)

Where λ = 0.2 typically
```

**Rationale**:
- L₁ for pixel accuracy
- SSIM for perceptual quality
- Balance with λ

**Our Current**: Using pure MSE (L2) in Adam optimizer
- **Potential issue**: L2 is noisier than L1 + SSIM
- **Action**: Consider switching to combined loss

---

## 3. Gradient Computation (Differentiable Rendering)

### 3.1 Chain Rule Through Rendering

To optimize, we need gradients of loss w.r.t. Gaussian parameters:

```
∂L/∂θᵢ = Σₚ (∂L/∂I_render(p)) × (∂I_render(p)/∂θᵢ)
```

**Two parts**:
1. **∂L/∂I_render**: Loss gradient (easy, from loss function)
2. **∂I_render/∂θ**: Rendering gradient (hard, this is differentiable rendering)

### 3.2 Rendering Gradient (Weighted Average Formulation)

Recall our rendering equation:
```
I(p) = (Σᵢ wᵢ(p) × cᵢ) / max(Σᵢ wᵢ(p), ε)
```

Where:
```
wᵢ(p) = αᵢ × exp(-½ dᵢ²(p))
dᵢ²(p) = (p - μᵢ)ᵀ Σᵢ⁻¹ (p - μᵢ)
```

Let W(p) = Σᵢ wᵢ(p) (total weight).

**Simplified** (assuming W ≥ 1 always, so denominator ≈ W):
```
I(p) ≈ (Σᵢ wᵢ(p) × cᵢ) / W(p)
```

#### Gradient w.r.t. Color (cᵢ)

**Easiest case**:
```
∂I(p)/∂cᵢ = wᵢ(p) / W(p)
```

**Chain rule**:
```
∂L/∂cᵢ = Σₚ (∂L/∂I(p)) × (wᵢ(p) / W(p))
```

**Interpretation**: If pixel p has high error and Gaussian i contributes there (wᵢ>0), adjust cᵢ in direction to reduce error.

#### Gradient w.r.t. Position (μᵢ)

**More complex** (involves distance derivative):

```
∂wᵢ/∂μᵢ = wᵢ × ∂(-½ dᵢ²)/∂μᵢ
        = wᵢ × Σᵢ⁻¹ × (p - μᵢ)
```

But this affects both numerator and denominator of I(p).

**Full derivation** (with quotient rule):
```
∂I(p)/∂μᵢ = [W × ∂(wᵢcᵢ)/∂μᵢ - (wᵢcᵢ) × ∂W/∂μᵢ] / W²
          = [(cᵢ - I(p)) × ∂wᵢ/∂μᵢ] / W
```

Where:
```
∂wᵢ/∂μᵢ = wᵢ × Σᵢ⁻¹ × (p - μᵢ)
```

**For our Euler parameterization** (σₓ, σᵧ, θ):

After rotation to local frame:
```
p' = R(-θ) × (p - μ)

∂d²/∂μ = -2 × [R(-θ)ᵀ × [(p'ₓ/σₓ²), (p'ᵧ/σᵧ²)]ᵀ]
```

**Our implementation** (adam_optimizer.rs:182-187):
```rust
let grad_dist_dx_rot = dx_rot / (sx * sx);
let grad_dist_dy_rot = dy_rot / (sy * sy);

gradients[i].position.x += error_weighted * weight * (grad_dist_dx_rot * cos_t - grad_dist_dy_rot * sin_t);
gradients[i].position.y += error_weighted * weight * (grad_dist_dx_rot * sin_t + grad_dist_dy_rot * cos_t);
```

**Key insight**: Rotation matrix appears in chain rule!
- We rotate to local frame for distance
- Must rotate back for position gradient

#### Gradient w.r.t. Scale (σₓ, σᵧ)

```
∂d²/∂σₓ = -2 × (p'ₓ/σₓ)² / σₓ
∂d²/∂σᵧ = -2 × (p'ᵧ/σᵧ)² / σᵧ
```

Where p' = rotated position.

**Our implementation** (adam_optimizer.rs:189-191):
```rust
gradients[i].scale_x += error_weighted * weight * (dx_rot / sx).powi(2) / sx;
gradients[i].scale_y += error_weighted * weight * (dy_rot / sy).powi(2) / sy;
```

**CRITICAL**: Must use rotated coordinates (dx_rot, dy_rot), not unrotated (dx, dy)!

This was the bug we fixed in commit 827d186.

### 3.3 Finite Difference Validation

**To verify our gradients are correct**:

```rust
fn verify_gradients(gaussians, target) {
    let epsilon = 1e-5;

    for param in all_parameters {
        // Analytic gradient
        let grad_analytic = compute_gradient(gaussians, target);

        // Finite difference
        param += epsilon;
        let loss_plus = compute_loss(gaussians, target);
        param -= 2*epsilon;
        let loss_minus = compute_loss(gaussians, target);
        param += epsilon;  // restore

        let grad_fd = (loss_plus - loss_minus) / (2*epsilon);

        let error = |grad_analytic - grad_fd|;
        assert!(error < 1e-3, "Gradient mismatch!");
    }
}
```

**Action item**: We MUST do this to verify our rotation-aware gradients.

---

## 4. Optimization Algorithms

### 4.1 Gradient Descent

**Simplest**:
```
θ_{t+1} = θ_t - η × ∇L(θ_t)
```

Where η = learning rate.

**Problems**:
- Choosing η is hard (too small → slow, too large → diverge)
- Same LR for all parameters (but scales vary wildly!)
- No momentum (can get stuck in ravines)

### 4.2 Adam Optimizer (What We Use)

**Adaptive Moment Estimation**:

```
For each parameter θᵢ:
  m_t = β₁ × m_{t-1} + (1-β₁) × g_t       // First moment (momentum)
  v_t = β₂ × v_{t-1} + (1-β₂) × g_t²      // Second moment (variance)

  m̂_t = m_t / (1 - β₁ᵗ)                    // Bias correction
  v̂_t = v_t / (1 - β₂ᵗ)

  θ_{t+1} = θ_t - η × m̂_t / (√v̂_t + ε)
```

**Parameters** (defaults):
- η = 0.01 (learning rate)
- β₁ = 0.9 (momentum decay)
- β₂ = 0.999 (RMSprop decay)
- ε = 1e-8 (numerical stability)

**Advantages**:
- Adapts per-parameter learning rates
- Handles sparse gradients well
- Generally robust

**Our implementation**: `adam_optimizer.rs`

**Hyperparameters to tune**:
- Learning rate η (currently 0.01 - is this right for all passes?)
- Iterations (currently 100 - enough?)

### 4.3 L-BFGS (Second-Order Method)

**Limited-memory Broyden-Fletcher-Goldfarb-Shanno**:
- Approximates second-order (Hessian) information
- Can be faster than Adam for smooth problems
- More memory intensive

**We have**: `lgi-core/src/lbfgs.rs` (implemented but unused?)

**Consideration**: Try L-BFGS if Adam struggles?

### 4.4 Optimization Schedule

**Question**: Should learning rate decay over iterations?

**3DGS approach** (Kerbl et al.):
- Position learning rate: starts high, exponentially decays
- Scale/color learning rate: constant
- Rationale: position is harder to optimize (more nonlinear)

**Our current**: Fixed LR (0.01) for all parameters, all iterations

**Potential improvement**: Adaptive LR schedule
```
η_position(t) = η₀ × exp(-t / τ)
η_scale(t) = η₀ × 0.1  // scales need smaller LR
η_color(t) = η₀  // colors can use larger LR
```

---

## 5. Convergence Analysis

### 5.1 What "Convergence" Means

**Ideal**:
- Loss decreases monotonically
- Gradients → 0 as minimum approached
- PSNR increases (usually correlates with loss decrease)

**In practice**:
- Loss may plateau (local minimum)
- Loss may oscillate (learning rate too high)
- May early stop (no improvement for N iterations)

### 5.2 Monitoring Convergence

**Metrics to track**:
```
Every iteration t:
  - Loss: L(θ_t)
  - PSNR: 10 log₁₀(1/MSE(θ_t))
  - Gradient norms: ||∇L||₂
  - Parameter changes: ||θ_t - θ_{t-1}||
```

**Healthy convergence**:
```
Iteration    Loss      PSNR     Grad Norm
0           0.850     8.0 dB    0.15
10          0.650     9.5 dB    0.08
20          0.500    11.0 dB    0.05
50          0.350    12.8 dB    0.02
100         0.300    13.5 dB    0.01
```

**Our CURRENT failure** (second pass):
```
Iteration    Loss
10          0.375001
20          0.375001
...
100         0.375001  ← STUCK!
```

**This indicates**:
- Gradients are zero (or numerically negligible)
- Optimizer can't find descent direction
- Possible causes:
  1. Gradients actually zero (local minimum, but PSNR is terrible!)
  2. Gradients incorrect (numerical issues)
  3. Gaussians in invalid state (coverage zero → no gradient signal)
  4. Learning rate too small (not making progress)

### 5.3 Diagnostic: Gradient Magnitude

**Check**:
```rust
let grad_norms: Vec<f32> = gradients.iter()
    .map(|g| {
        let color_norm = (g.color.r.powi(2) + g.color.g.powi(2) + g.color.b.powi(2)).sqrt();
        let pos_norm = (g.position.x.powi(2) + g.position.y.powi(2)).sqrt();
        let scale_norm = (g.scale_x.powi(2) + g.scale_y.powi(2)).sqrt();
        (color_norm, pos_norm, scale_norm)
    })
    .collect();

log::info!("Gradient norms - Color: {:?}, Position: {:?}, Scale: {:?}",
    grad_norms.iter().map(|(c,_,_)| c).collect::<Vec<_>>(),
    grad_norms.iter().map(|(_,p,_)| p).collect::<Vec<_>>(),
    grad_norms.iter().map(|(_,_,s)| s).collect::<Vec<_>>());
```

**Expected**: Non-zero gradients (at least 1e-6)
**If zero**: No gradient signal → investigate why (coverage? NaN?)

---

## 6. Common Failure Modes

### 6.1 Local Minima

**Symptom**: Loss plateaus, but quality is poor

**Cause**: Non-convex optimization landscape (many local minima)

**Solution**:
- Better initialization (structure-aware)
- More Gaussians (more degrees of freedom)
- Restart with perturbation
- Simulated annealing

### 6.2 Exploding/Vanishing Gradients

**Symptom**: Loss diverges or stays constant

**Cause**: Learning rate too high/low, numerical issues

**Solution**:
- Gradient clipping: `g = min(g, max_grad_norm)`
- Adaptive LR (Adam helps with this)
- Check for NaN/Inf

### 6.3 Zero Coverage

**Symptom**: Loss stuck, PSNR terrible (4-5 dB)

**Cause**: Gaussians too small → no contribution to any pixel

**Example**:
```
If all σ < 0.001 in normalized coords:
  → Gaussians cover <1 pixel each
  → Most pixels have zero coverage
  → Gradients are zero (no Gaussian affects those pixels)
  → Optimizer can't improve
```

**Detection**: W_median ≈ 0

**Solution**:
- Enforce minimum scale (σ_min)
- Re-initialize Gaussians
- Check what caused scales to collapse

**This is likely our current issue!**

### 6.4 Scale Collapse

**Symptom**: All Gaussians shrink to tiny sizes

**Cause**:
- Over-aggressive regularization (if we had L2 on scales)
- Clamping too tight (geodesic EDT?)
- Gradient direction wrong (scales being reduced instead of adjusted)

**Prevention**:
- Minimum scale constraint: σ ≥ σ_min
- Check scale gradients (should increase σ in some cases!)
- Balance loss (don't just reward smaller Gaussians)

---

## 7. Debugging Protocol for Stuck Optimization

**When loss doesn't decrease**:

### Step 1: Verify Gradients
```rust
verify_gradients_finite_difference(gaussians, target);
```
If gradients match → implementation correct
If mismatch → bug in gradient computation

### Step 2: Check Gradient Magnitudes
```rust
let max_grad = compute_max_gradient_norm(gradients);
log::info!("Max gradient norm: {}", max_grad);
```
If max_grad < 1e-8 → gradients are effectively zero → investigate why

### Step 3: Check Coverage
```rust
let coverage_stats = compute_coverage_stats(gaussians, width, height);
log::info!("W_median: {}, W_min: {}, W_max: {}",
    coverage_stats.median, coverage_stats.min, coverage_stats.max);
```
If W_median < 0.1 → zero coverage problem

### Step 4: Check Gaussian Scales
```rust
let scale_stats = compute_scale_stats(gaussians);
log::info!("Scale min: {}, median: {}, max: {}",
    scale_stats.min, scale_stats.median, scale_stats.max);
```
If min < 0.001 → scales collapsed → enforce minimum

### Step 5: Visualize
```rust
DiagnosticSuite::render_gaussians_as_ellipses(&image, &gaussians);
DiagnosticSuite::plot_coverage_heatmap(&gaussians, width, height);
```
See WHERE Gaussians are, how large they are

### Step 6: Simplify
- Test with 1 Gaussian, simple image (16×16, white pixel)
- Does optimizer work in this trivial case?
- If no → fundamental gradient bug
- If yes → issue with multi-Gaussian interaction or initialization

---

## 8. Theory vs. Our Implementation

### 8.1 What Matches Theory

- ✅ Weighted average rendering (mathematically sound)
- ✅ MSE loss (standard)
- ✅ Adam optimizer (standard)
- ✅ Gradient computation structure (chain rule)

### 8.2 What We Fixed Recently

- ✅ Rotation in gradients (commit 827d186)
  - Before: Ignored rotation θ
  - After: Correctly transforms gradients through rotation

### 8.3 What's Still Uncertain

- ❓ Are gradients fully correct? (need finite diff verification)
- ❓ Is coverage being maintained? (W_median check needed)
- ❓ Are scales being preserved? (min scale enforcement?)
- ❓ Is learning rate appropriate? (may need tuning per pass)

---

## 9. Next Steps

### 9.1 Immediate Actions

1. **Implement gradient verification**:
   ```rust
   test_gradients_with_finite_differences();
   ```

2. **Add diagnostic logging**:
   ```rust
   log_gradient_norms();
   log_coverage_stats();
   log_scale_stats();
   ```

3. **Run minimal test**:
   - 16×16 image, simple pattern
   - 5 Gaussians
   - 10 iterations
   - Should converge → if not, isolate problem

### 9.2 Research Tasks

1. **Read 3DGS optimization section**
   - How do they handle densification during optimization?
   - What learning rate schedule?
   - Any special tricks?

2. **Study differentiable rendering literature**
   - Verify our gradient derivations
   - Check for missing terms

3. **Experiment with loss functions**
   - Try L1 + SSIM (3DGS approach)
   - Compare convergence to pure MSE

---

## References

- **[Kingma & Ba, 2014]** "Adam: A Method for Stochastic Optimization"
  - Adam optimizer theory and analysis

- **[Kerbl et al., 2023]** Section on optimization and densification
  - Learning rate schedules, adaptive control

- **[Kato et al., 2018]** "Neural 3D Mesh Renderer" (differentiable rendering)
  - Gradient flow through rendering

---

**Status**: Initial theory documented
**TODO**:
- [ ] Implement finite difference gradient verification
- [ ] Add comprehensive diagnostic logging
- [ ] Test on minimal cases (single Gaussian)
- [ ] Integrate 3DGS optimization insights
- [ ] Derive and verify all gradient equations rigorously

**Next Document**: `INITIALIZATION_THEORY.md` - How to set up Gaussians for success
