# LGI Project - Comprehensive Status Review
**Date**: December 6, 2025
**Purpose**: Full inventory of capabilities and identification of research gaps

---

## 1. Complete Inventory

### 1.1 Optimizers (lgi-encoder-v2)

| Optimizer | File | Status | Tested? | Kodak Data? |
|-----------|------|--------|---------|-------------|
| **Adam (per-param LR)** | adam_optimizer.rs | PRODUCTION | Yes | Partial (3 images) |
| OptimizerV2 | optimizer_v2.rs | Working | Yes | Partial |
| Hybrid (Adam→L-BFGS) | hybrid_optimizer.rs | Experimental | Partial | No |
| L-BFGS | lbfgs_optimizer.rs | Implemented | Needs work | No |
| Levenberg-Marquardt | lm_optimizer.rs | Implemented | Needs work | No |

**Key insight**: Different optimizers may need different initialization/parameterization frameworks to succeed. Adam's momentum can conflict with certain initialization strategies.

### 1.2 Initialization Strategies

| Strategy | Location | Status | Result vs Grid |
|----------|----------|--------|----------------|
| **Uniform Grid** | lgi-core | Baseline | +0.87 dB |
| **Jittered Grid** | lgi-encoder-v2 | Best for Adam | +1.31 dB |
| K-Means | lgi-encoder-v2 | Tested | -4.04 dB |
| SLIC Superpixel | lgi-core | Implemented | Untested |
| GDGS (Laplacian) | lgi-encoder-v2 | Tested | -6.6 dB |
| Gradient Peaks | lgi-encoder-v2 | Tested | -0.2 dB |
| PPM (Probability Map) | lgi-core | Implemented | Untested |
| Random | lgi-core | Baseline | Untested |
| Importance (variance) | lgi-core | Implemented | Untested |
| Error-Driven | lgi-encoder-v2 | Experimental | Untested |

**Key insight**: Feature-based initialization strategies (K-means, GDGS) failed with Adam, likely due to momentum mismatch. These may work better with other optimizers (L-BFGS, L-M).

### 1.3 Loss Functions

| Loss | Location | Status | Systematic Testing? |
|------|----------|--------|---------------------|
| MSE/L2 | lgi-encoder-v2 | Default | Yes |
| L1 | lgi-encoder-v2 | Available | No |
| SSIM | lgi-encoder-v2 | Available | No |
| D-SSIM | lgi-encoder-v2 | Available | No |
| 3DGS Loss (L1+D-SSIM) | lgi-encoder-v2 | Available | No |
| **MS-SSIM** | lgi-core | With gradients | No |
| Edge-Weighted L2 | lgi-core | Available | No |

### 1.4 Gaussian Parameterizations (lgi-math)

| Parameterization | Parameters | Purpose | Optimization Testing? |
|------------------|------------|---------|----------------------|
| Euler | (σx, σy, θ) | Intuitive, default | Yes (Adam) |
| Cholesky | (L11, L21, L22) | Numerical stability | No |
| LogCholesky | (log_l11, l21, log_l22) | Unconstrained opt | No |
| LogRadius | (log_r, e, θ) | Compression | No |
| InverseCovariance | (a, b, c) | Fast rendering | No |

**Key insight**: L-BFGS and L-M may benefit from Cholesky or LogCholesky parameterization for numerical stability.

### 1.5 Renderers

| Renderer | Location | Status | Validated? |
|----------|----------|--------|------------|
| RendererV2 (CPU) | renderer_v2.rs | Working | Yes |
| V3 (Textured) | renderer_v3.rs | Working | Partial |
| GPU (wgpu) | lgi-gpu | Implemented | No (claims 454× speedup) |

### 1.6 Dynamic Gaussian Management

| Feature | Status | Result |
|---------|--------|--------|
| Densification (grow) | Implemented | -2 dB (fails with Adam) |
| Pruning (shrink) | Implemented | +0.69 dB (works) |
| Splitting | Implemented | Partial testing |
| Merging | Implemented | Untested |

---

## 2. Research Gaps (Prioritized)

### HIGH PRIORITY

1. **L-BFGS Optimizer**: Needs proper calibration and testing
   - May need different initialization strategy than Adam
   - May benefit from Cholesky parameterization

2. **Levenberg-Marquardt Optimizer**: Same as L-BFGS
   - Designed for nonlinear least squares
   - May excel with proper setup

3. **MS-SSIM Loss**: Implemented with gradients but never tested
   - Could improve perceptual quality
   - Needs integration into optimization loop

4. **Full Kodak Validation**: Only 1-3 images tested per approach

### MEDIUM PRIORITY

5. **GPU Renderer Validation**: Claims unverified
6. **Alternative Parameterizations**: Cholesky/LogCholesky for stability
7. **Gaussian Count Scaling**: 64→2048 range
8. **Hybrid Loss**: MSE + MS-SSIM combination

### LOWER PRIORITY

9. **Textured Gaussians**
10. **Progressive/LOD system**
11. **Compression optimization**

---

## 3. Key Findings From Recent Research

### What Works
- Adam with per-parameter learning rates: +2.25 dB average
- Jittered grid initialization: +1.31 dB over uniform
- Gradient normalization: +0.77 dB
- Hybrid pruning: +0.69 dB with 27% fewer Gaussians

### What Doesn't Work (with Adam)
- Growing densification: -2.01 dB (momentum mismatch)
- K-means initialization: -4.04 dB (over-concentrates)
- GDGS initialization: -6.6 dB (sparse placement)

### Critical Learning Rates (3DGS-calibrated)
```rust
lr_position: 0.0002  // Positions are sensitive
lr_color:    0.02    // Colors can learn fast
lr_scale:    0.005   // Moderate
lr_opacity:  0.05    // Needs larger LR
```

---

## 4. Immediate Action Plan

### Phase 1: Optimizer Calibration
1. L-BFGS: Retool for success
   - Review implementation
   - Test different parameterizations
   - Find optimal settings

2. Levenberg-Marquardt: Same process

### Phase 2: Loss Functions
3. MS-SSIM: Get working in optimization
4. Hybrid loss: MSE + MS-SSIM

### Phase 3: Infrastructure
5. GPU renderer validation
6. Full Kodak baseline collection

---

## 5. Files Reference

### Core Optimizer Files
- `src/adam_optimizer.rs` - 22 KB, production ready
- `src/optimizer_v2.rs` - 16 KB, gradient descent
- `src/hybrid_optimizer.rs` - 14 KB, experimental
- `src/lbfgs_optimizer.rs` - 10 KB, needs work
- `src/lm_optimizer.rs` - 10 KB, needs work

### Loss Function Files
- `src/loss_functions.rs` - Basic losses
- `lgi-core/src/ms_ssim.rs` - MS-SSIM with gradients
- `lgi-core/src/ms_ssim_loss.rs` - Combined loss
- `lgi-core/src/edge_weighted_loss.rs` - Edge-aware

### Test Infrastructure
- `src/test_results.rs` - TestResult framework
- `test-results/README.md` - Documentation
- `test-results/YYYY-MM-DD/` - Daily results

---

*Document created: December 6, 2025*
