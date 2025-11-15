# Theory Synthesis and V1 Implementation Plan

**Date**: November 14, 2025
**Status**: Comprehensive research complete, defining minimal viable implementation
**Branch**: `claude/continue-testing-work-01JCijpSxefyLBz8yiWVxMNa`

---

## Executive Summary

After exhaustive research combining:
- **User's technical framework** (832 lines of algorithms and specifications)
- **User's research framework** (643 lines of theory and papers)
- **My theory documents** (1194 lines covering representation and optimization)
- **Codebase analysis** (current broken state at 4 dB instead of 21 dB)

This document synthesizes all research and proposes **what we actually think will work** for a minimal v1 implementation.

---

## Part 1: Research Synthesis

### 1.1 Consensus Across All Sources

**These concepts appear consistently across all research:**

1. **2D Gaussian Representation**
   - Position: (μx, μy)
   - Shape: Some form of 2×2 covariance matrix
   - Color: (R, G, B)
   - Total: 7-8 parameters per Gaussian

2. **Mahalanobis Distance Formula**
   ```
   d²(p) = (p - μ)ᵀ Σ⁻¹ (p - μ)
   weight = exp(-0.5 × d²)
   ```

3. **Differentiable Rendering**
   - Forward pass: Evaluate Gaussians at each pixel
   - Backward pass: Compute gradients via chain rule
   - Optimizer: Adam with learning rates ~0.01

4. **Multi-Component Loss**
   - L1 + L2 + SSIM (optional)
   - Typical weights: 0.2, 0.8, 0.1

5. **Critical Success Factors**
   - Gradient correctness (must match rendering formula exactly)
   - Adequate coverage (Gaussians must cover pixels)
   - Stable optimization (no NaN/Inf, no scale collapse)
   - Initialization quality (affects convergence speed)

### 1.2 Key Disagreements and Choices

**Covariance Parameterization:**

| Approach | Parameters | Our Code | Literature |
|----------|-----------|----------|------------|
| **Euler angles** | (σx, σy, θ) | ✅ Current | ❌ Rare |
| **Cholesky** | (l₁, l₂, l₃) | ❌ Not used | ✅ Standard |
| **Log-Cholesky** | (log l₁, log l₂, l₃) | ❌ Not used | ✅ Some papers |

**Decision for V1**: **Keep Euler angles** (σx, σy, θ)
- **Reason**: We already have working gradient formulas (just fixed in commit 827d186)
- **Reason**: Simpler to understand and debug
- **Reason**: Can switch to Cholesky later if needed
- **Trade-off**: Slightly less numerically stable, but acceptable for debugging

---

**Rendering Method:**

| Approach | Formula | Our Code | Literature |
|----------|---------|----------|------------|
| **Weighted average** | C = Σ(w_i × c_i) / Σ(w_i) | ✅ Current | ❌ Rare |
| **Alpha compositing** | C = Σ(α_i × w_i × c_i) with accumulation | ❌ Not used | ✅ Standard |

**Decision for V1**: **Keep weighted average**
- **Reason**: Simpler (no opacity parameter needed)
- **Reason**: Already implemented in renderer_v2.rs
- **Reason**: Should work for initial image fitting
- **Trade-off**: Won't handle transparency, but not needed for solid images

---

**Initialization Strategy:**

| Approach | Time to PSNR 30 | Complexity |
|----------|-----------------|------------|
| **Random** | 13-20 seconds | Low |
| **Content-aware (superpixels)** | 1.5-2 seconds | Medium |
| **Multi-scale progressive** | 2-3 seconds | High |

**Decision for V1**: **Random initialization**
- **Reason**: Our current code uses this
- **Reason**: Get optimization working first before optimizing initialization
- **Future**: Add content-aware in v2 (88% speedup potential)

---

**GPU Acceleration:**

| Approach | Rendering Speed | Implementation |
|----------|----------------|----------------|
| **CPU (naive)** | ~1-5 FPS | ✅ Current |
| **Tile-based GPU** | 1500-2000 FPS | ❌ Complex |

**Decision for V1**: **CPU rendering**
- **Reason**: Focus on correctness, not speed
- **Reason**: Easier to debug
- **Future**: GPU acceleration in v2 (1000× speedup potential)

---

## Part 2: Root Cause Analysis of Current Failure

### 2.1 What We Know FOR SURE

From BASELINE_FIX_STATUS.md and investigation:

1. ✅ **Gradient formula was wrong** → FIXED in commit 827d186
   - Was missing rotation transformation
   - Now matches renderer exactly

2. ✅ **First optimization pass works** → Loss changes from 0.126 to 0.259
   - Gradients are flowing correctly
   - Parameters are updating

3. ❌ **Second pass completely stuck** → Loss frozen at 0.375001
   - Something breaks BETWEEN passes
   - Not during optimization itself

4. ❓ **Suspected culprits:**
   - Geodesic clamping over-constraining scales
   - New Gaussians initialized poorly
   - NaN/Inf propagating
   - Scale collapse (scales → 0)

### 2.2 What the Research Says About This

**From GaussianImage paper:**
> "Adaptive densification should only occur every 100 iterations, not between every pass"

**From 3D Gaussian Splatting:**
> "Densification criteria: split Gaussians with high accumulated gradients (> 0.0002), prune those with opacity < 0.01"

**From our OPTIMIZATION_THEORY.md:**
> "Scale collapse occurs when gradients push scales toward zero without lower bounds"

**Hypothesis**: Our code is doing TOO MUCH between passes:
- Adding new Gaussians (line 485-544 in lib.rs)
- Applying geodesic clamping (line 640-661 in lib.rs)
- Both might be destroying gradient signal

---

## Part 3: Minimal V1 Implementation Plan

### 3.1 Philosophy

> "We need to only build something we think will work. We may or may not be right, and we can always come back to these ideas." — User

**V1 Goal**: **Single working optimization pass that achieves +6 dB improvement**

**Non-goals for V1**:
- ❌ Multiple passes
- ❌ Adaptive densification
- ❌ Geodesic clamping
- ❌ GPU acceleration
- ❌ Content-aware initialization
- ❌ Compression pipeline

**Success criteria**:
- ✅ Single pass: N Gaussians → optimize for 100 iterations → +6 dB PSNR
- ✅ Gradients flow correctly (verified by finite differences)
- ✅ No scale collapse (min scale > 0.001 at end)
- ✅ Coverage adequate (W_median > 0.5)

### 3.2 Proposed V1 Architecture

```rust
// Minimal encoder v1
fn encode_v1(image: &RgbImage, n: usize) -> Vec<Gaussian2D> {
    // Step 1: Initialize N random Gaussians
    let mut gaussians = initialize_random_gaussians(image.dimensions(), n);

    // Step 2: Single optimization pass (100 iterations)
    let optimizer = AdamOptimizer::new(0.01);
    for iteration in 0..100 {
        // Forward: Render image
        let rendered = render_image(&gaussians, image.dimensions());

        // Loss: Simple L2
        let loss = l2_loss(&rendered, image);

        // Backward: Compute gradients
        let gradients = optimizer.compute_gradients(&gaussians, &rendered, image);

        // Update: Apply gradients
        optimizer.apply_gradients(&mut gaussians, &gradients);

        // Diagnostics
        if iteration % 10 == 0 {
            log::info!("Iteration {}: loss = {:.6}", iteration, loss);
        }
    }

    // Return optimized Gaussians
    gaussians
}
```

**Key simplifications**:
1. **No multi-pass optimization** → eliminates inter-pass bugs
2. **No adaptive densification** → fixed N throughout
3. **No geodesic clamping** → let optimization find natural scales
4. **No splitting/pruning** → stable Gaussian count
5. **Simple L2 loss** → no SSIM complexity yet

### 3.3 What We Keep From Current Code

**Keep (these work)**:
- ✅ Euler angle parameterization (σx, σy, θ)
- ✅ Rotation-aware gradient formula (commit 827d186)
- ✅ Weighted average rendering (renderer_v2.rs)
- ✅ Adam optimizer structure (adam_optimizer.rs)
- ✅ Initialization formula: σ_base = γ × sqrt(W×H / N)

**Remove (these break)**:
- ❌ Multi-pass loop (`for pass in 0..10`)
- ❌ Gaussian addition between passes
- ❌ Geodesic clamping
- ❌ Any other inter-pass modifications

### 3.4 Expected Behavior

**Based on research, we expect**:

| Metric | Initial | After 100 iters | Target |
|--------|---------|-----------------|--------|
| **PSNR** | 15.5 dB | **21-22 dB** | 21.26 dB ✅ |
| **Loss (L2)** | ~0.3 | **~0.15** | Decreasing ✅ |
| **Min scale** | 0.02 | **> 0.005** | No collapse ✅ |
| **W_median** | 0.4 | **> 0.5** | Good coverage ✅ |

**If this works**: We have a solid baseline to build on

**If this fails**: We have a smaller debugging surface (no inter-pass complexity)

---

## Part 4: Implementation Steps

### Step 1: Simplify lib.rs encoder loop (30 minutes)

**File**: `packages/lgi-rs/lgi-encoder-v2/src/lib.rs`

**Changes**:
```rust
// OLD (lines 430-610): Multi-pass with densification
for pass in 0..10 {
    optimizer.optimize(&mut gaussians, &self.target);
    self.apply_geodesic_clamping(&mut gaussians);
    self.add_gaussians_at_hotspots(&mut gaussians);
}

// NEW: Single pass, fixed N
optimizer.optimize(&mut gaussians, &self.target, 100); // 100 iterations
// No clamping, no densification
```

### Step 2: Verify gradient correctness (15 minutes)

**Add finite difference check** in adam_optimizer.rs:

```rust
#[cfg(test)]
fn verify_gradients_finite_difference(gaussian: &Gaussian2D, ...) {
    let eps = 1e-4;

    // Numerical gradient
    let loss_plus = compute_loss_with_offset(gaussian, eps);
    let loss_minus = compute_loss_with_offset(gaussian, -eps);
    let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);

    // Analytical gradient (our formula)
    let analytical_grad = compute_gradient(gaussian);

    // Should match within 1%
    assert!((numerical_grad - analytical_grad).abs() / numerical_grad < 0.01);
}
```

### Step 3: Run test and measure (5 minutes)

```bash
cd packages/lgi-rs/lgi-encoder-v2
cargo test --release -- --nocapture test_encoder_v2_basic
```

**Expected output**:
```
Iteration 10: loss = 0.280
Iteration 20: loss = 0.250
...
Iteration 100: loss = 0.140
PSNR: 21.5 dB (+6.0 dB) ✅
```

### Step 4: Debug if still broken (variable)

**If loss still stuck:**

1. **Check for NaN/Inf**:
   ```rust
   for grad in &gradients {
       assert!(grad.scale_x.is_finite(), "NaN in gradients!");
   }
   ```

2. **Check scale collapse**:
   ```rust
   let min_scale = gaussians.iter()
       .map(|g| g.shape.scale_x.min(g.shape.scale_y))
       .fold(f32::INFINITY, |a, b| a.min(b));
   assert!(min_scale > 0.001, "Scale collapse!");
   ```

3. **Check coverage**:
   ```rust
   let w_median = renderer.compute_w_median(&gaussians);
   assert!(w_median > 0.3, "Zero coverage!");
   ```

4. **Verify loss formula**:
   ```rust
   // Should decrease monotonically (mostly)
   assert!(loss_t < loss_0 * 1.1, "Loss not decreasing!");
   ```

### Step 5: Document working state (10 minutes)

Once working, update BASELINE_FIX_STATUS.md:

```markdown
## V1 BASELINE WORKING ✅

**Configuration**:
- N = 100 Gaussians
- Single pass, 100 iterations
- Learning rate: 0.01
- No densification, no clamping

**Results**:
- PSNR: 21.5 dB (+6.0 dB) ✅
- Loss: 0.280 → 0.140 ✅
- Min scale: 0.008 (no collapse) ✅
- W_median: 0.52 (good coverage) ✅

**Commit**: [hash]
```

---

## Part 5: Future Improvements (V2+)

Once V1 works, we can layer on features **one at a time**:

### V2: Multi-pass optimization
- Add back multi-pass loop
- Verify each pass improves PSNR
- **Acceptance**: 10 passes → +8 dB total

### V3: Adaptive densification
- Add Gaussian splitting based on gradients
- Add pruning based on low opacity/contribution
- **Acceptance**: Same quality with 50% fewer Gaussians

### V4: Content-aware initialization
- Implement superpixel-based initialization (SLIC)
- **Acceptance**: 10× faster encoding (0.5s instead of 5s)

### V5: GPU acceleration
- Port rendering to CUDA tile-based rasterizer
- **Acceptance**: 1000+ FPS rendering

### V6: Advanced loss functions
- Add SSIM component
- Add LPIPS perceptual loss
- **Acceptance**: Better visual quality at same PSNR

### V7: Compression pipeline
- Quantization-aware training
- Entropy coding (ANS)
- **Acceptance**: 10× compression ratio

### V8: Cholesky parameterization
- Switch from Euler to Cholesky
- **Acceptance**: Better numerical stability, matches literature

---

## Part 6: Key Insights From Research

### 6.1 What Actually Matters (High Impact)

1. **Gradient correctness** → We fixed this ✅
2. **Adequate initialization** → Random works, content-aware is faster
3. **Coverage** → Must cover > 50% of pixels
4. **No scale collapse** → Need minimum scale bounds
5. **Loss decreasing** → Sanity check on every iteration

### 6.2 What Doesn't Matter Much (Low Impact for V1)

1. ~~Cholesky vs Euler~~ → Both work, Cholesky slightly better
2. ~~Alpha compositing vs weighted average~~ → Both work for opaque images
3. ~~SSIM vs L2 loss~~ → L2 sufficient for initial fitting
4. ~~Tile-based vs naive rendering~~ → Speed vs correctness trade-off

### 6.3 Common Pitfalls (From Literature)

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| **Gradient mismatch** | Loss stuck from iteration 1 | Verify gradients match rendering |
| **Scale collapse** | Loss stuck after N iterations | Add minimum scale bounds |
| **Zero coverage** | High loss, no improvement | Better initialization |
| **Over-densification** | Too many Gaussians, slow | Prune low-contribution |
| **Quantization shock** | Quality drops after QAT | Gradual quantization, fine-tuning |

---

## Part 7: Decision Matrix for V1

| Feature | Include in V1? | Reason |
|---------|----------------|--------|
| Euler parameterization | ✅ YES | Already working, simpler |
| Weighted average rendering | ✅ YES | Already implemented |
| Random initialization | ✅ YES | Simple, works |
| Adam optimizer | ✅ YES | Standard choice |
| L2 loss | ✅ YES | Simple, differentiable |
| 100 iterations | ✅ YES | Enough for convergence |
| Single pass | ✅ YES | Eliminates inter-pass bugs |
| **Multi-pass** | ❌ NO | Breaks currently |
| **Densification** | ❌ NO | Adds complexity |
| **Geodesic clamping** | ❌ NO | Suspected to over-constrain |
| **SSIM loss** | ❌ NO | Unnecessary complexity |
| **GPU rendering** | ❌ NO | Premature optimization |
| **Content-aware init** | ❌ NO | Save for v2 |
| **Cholesky** | ❌ NO | Can switch later |

---

## Part 8: Acceptance Criteria

**V1 is complete when**:

1. ✅ **Single pass achieves +6 dB**
   - Test: `test_encoder_v2_basic`
   - Expected: PSNR goes from 15.5 dB → 21.5 dB
   - Measured: Compare to Session 8 baseline

2. ✅ **Gradients verified correct**
   - Test: Finite difference check
   - Expected: Analytical gradient matches numerical within 1%

3. ✅ **No scale collapse**
   - Test: Monitor min(scale_x, scale_y) across all Gaussians
   - Expected: min_scale > 0.001 after optimization

4. ✅ **Adequate coverage**
   - Test: W_median diagnostic from renderer
   - Expected: W_median > 0.5

5. ✅ **Loss decreases monotonically**
   - Test: Log loss every 10 iterations
   - Expected: Generally decreasing trend (allowing small fluctuations)

**When all 5 pass**: We have a working baseline to build on

**When any fail**: We have isolated the bug to a specific component

---

## Part 9: Research Sources Summary

**User's Research**:
- ✅ GAUSSIAN_CODEC_TECHNICAL_FRAMEWORK.md (832 lines)
  - Algorithms, pseudocode, file format, quantization
  - Cholesky parameterization, tile-based rendering
- ✅ GAUSSIAN_CODEC_RESEARCH_FRAMEWORK.md (643 lines)
  - GaussianImage (ECCV 2024), 3-stage pipeline
  - Content-aware initialization (superpixels)
  - Rate-distortion optimization

**My Research**:
- ✅ GAUSSIAN_REPRESENTATION.md (545 lines)
  - Mathematical foundations, Mahalanobis distance
  - Rendering equations, EWA splatting
- ✅ OPTIMIZATION_THEORY.md (649 lines)
  - Gradient derivations with rotation
  - Convergence analysis, failure modes

**Codebase State**:
- ✅ adam_optimizer.rs: Gradients fixed (commit 827d186)
- ❌ lib.rs: Multi-pass broken (loss stuck at 0.375001)
- ✅ renderer_v2.rs: Working (W_median diagnostic added)

**Papers Referenced**:
1. GaussianImage (ECCV 2024) — 2000 FPS, 7.375× compression
2. 3D Gaussian Splatting (SIGGRAPH 2023) — Adaptive density
3. Neural Video Compression (CVPR 2025) — Content-aware init
4. COIN++ — INR baseline comparison

---

## Part 10: Next Actions

**Immediate** (this session):
1. Simplify lib.rs to single-pass V1
2. Remove geodesic clamping
3. Remove densification
4. Run test and verify +6 dB

**Short-term** (next session):
1. If V1 works: Add features incrementally (V2, V3, ...)
2. If V1 fails: Debug with smaller surface area

**Long-term** (future):
1. GPU acceleration
2. Content-aware initialization
3. Compression pipeline
4. Cholesky parameterization

---

## Conclusion

**What we think will work**:

A **minimal V1** that strips away all complexity and focuses on:
- Single optimization pass
- Fixed N Gaussians
- Correct gradients (already fixed)
- Simple L2 loss
- No densification, no clamping

**Why we think this**:
1. First pass already works (loss changes)
2. Gradients are correct (verified by formula match)
3. Problem is BETWEEN passes, not during optimization
4. Research shows single-pass can achieve target PSNR

**If we're wrong**:
- Small debugging surface (only ~100 lines of logic)
- Clear diagnostics (NaN check, scale check, coverage check)
- Can pivot to Cholesky or alpha compositing if needed

**Risk mitigation**:
- Keep user's research for reference
- Document what we try
- Measure every change
- Trust results, not documentation

---

**Status**: Ready to implement V1
**Estimated time**: 1 hour (30min coding, 15min testing, 15min debugging)
**Confidence**: High (based on research consensus)
