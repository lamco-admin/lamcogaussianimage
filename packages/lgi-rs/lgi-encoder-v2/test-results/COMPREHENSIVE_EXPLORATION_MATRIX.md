# LGI Encoder Comprehensive Exploration Matrix
**Created**: 2025-12-07
**Purpose**: Complete inventory of all techniques, their test status, and systematic exploration guide

---

## Executive Summary

**Total Test Files**: 165 examples
**Total Test Results (Dec 5-7)**: 216 JSON files
**Total Modules**: ~59 in lgi-core, ~28 in lgi-encoder-v2, ~13 in lgi-gpu

### Current Best Results (kodim03, 768x512)
| Approach | PSNR | Gaussians | Improvement |
|----------|------|-----------|-------------|
| ErrorDrivenAdam (longer_passes) | 22.64 dB | 274 | +3.53 dB |
| Plateau baseline | 22.82 dB | 304 | +3.71 dB |
| Fine alpha=0.84 (Hybrid Loss) | 22.74 dB | 256 | +2.47 dB |
| Hybrid Pruning (Dec 6) | 22.30 dB | 187 | +0.69 dB (27% fewer G) |

---

## PART 1: OPTIMIZER TECHNIQUES

### 1.1 CPU Optimizers

| Optimizer | File | Status | Best Result | Notes |
|-----------|------|--------|-------------|-------|
| **Adam (per-param LR)** | `adam_optimizer.rs` | TESTED ✓ | +2.25 dB avg | PRODUCTION |
| **OptimizerV2** | `optimizer_v2.rs` | TESTED ✓ | +0.16 dB | Works, slow |
| **OptimizerV2 + Grad Norm** | `optimizer_v2.rs` | TESTED ✓ | +1.17 dB | Significant improvement |
| **Hybrid Adam→L-BFGS** | `hybrid_optimizer.rs` | TESTED ✓ | -0.22 dB | REJECTED |
| **L-M (CPU)** | `lm_optimizer.rs` | TESTED ✓ | Poor | Too slow |
| **L-BFGS (argmin)** | `optimizer_lbfgs.rs` | BROKEN | N/A | Missing dependency |
| **OptimizerV3 Perceptual** | `optimizer_v3_perceptual.rs` | UNTESTED | N/A | Edge-weighted |

### 1.2 GPU Optimizers

| Optimizer | File | Status | Best Result | Notes |
|-----------|------|--------|-------------|-------|
| **GPU Adam** | `lgi-gpu/adam_optimizer.rs` | TESTED ✓ | -0.06 dB parity | 50× faster |
| **GPU L-BFGS** | `lgi-gpu/lbfgs_optimizer.rs` | TESTED ✓ | +0.11 dB | 12× faster |
| **GPU L-M Block-Diag** | `lgi-gpu/lm_optimizer.rs` | TESTED ✓ | +0.05 dB | Works |
| **GPU L-M Full-Dense** | `lgi-gpu/jtj_strategy.rs` | TESTED ✓ | 0.00 dB | Numerically unstable |
| **GPU L-M Sparse JtJ** | `lgi-gpu/jtj_strategy.rs` | CODED | N/A | NOT TESTED |
| **GPU L-M Hierarchical JtJ** | `lgi-gpu/jtj_strategy.rs` | CODED | N/A | NOT TESTED |
| **GPU L-M MatrixFree JtJ** | `lgi-gpu/jtj_strategy.rs` | CODED | N/A | NOT TESTED |

### 1.3 Optimizer Variations NOT TESTED

| Variation | Current Status | Priority |
|-----------|----------------|----------|
| Adam with LR decay schedules | Coded in Adam | HIGH |
| Adam with warmup | Not coded | MEDIUM |
| Adam with gradient clipping | Not coded | MEDIUM |
| RMSprop | Not coded | LOW |
| AdaGrad | Not coded | LOW |
| Nadam | Not coded | LOW |

---

## PART 2: LOSS FUNCTIONS

### 2.1 Implemented Loss Functions

| Loss | File | Status | Result | Notes |
|------|------|--------|--------|-------|
| **L2 (MSE)** | Built-in | TESTED ✓ | Baseline | Standard |
| **Hybrid L2+MS-SSIM (80/20)** | `adam_optimizer.rs` | TESTED ✓ | +0.51 dB | RECOMMENDED |
| **Hybrid L2+MS-SSIM (84/16)** | sweep | TESTED ✓ | +2.47 dB | Best alpha |
| **MS-SSIM only** | `optimizer_v2.rs` | TESTED ✓ | -0.02 dB | Gradient issues |
| **L1 (MAE)** | `loss_functions.rs` | CODED | NOT TESTED | |
| **SSIM** | `loss_functions.rs` | CODED | NOT TESTED | |
| **D-SSIM** | `loss_functions.rs` | CODED | NOT TESTED | |
| **3DGS Loss (L1 + D-SSIM)** | `loss_functions.rs` | CODED | NOT TESTED | HIGH PRIORITY |
| **Combined (λ1×L1 + λ2×L2 + λ3×D-SSIM)** | `loss_functions.rs` | CODED | NOT TESTED | |
| **Edge-weighted** | `optimizer_v3_perceptual.rs` | CODED | NOT TESTED | |
| **Edge-weighted** | `lgi-core/edge_weighted_loss.rs` | CODED | NOT TESTED | |

### 2.2 Loss Variations NOT TESTED

| Variation | Status | Priority |
|-----------|--------|----------|
| L1 loss alone | Coded | HIGH |
| SSIM loss alone | Coded | MEDIUM |
| 3DGS Loss (0.8×L1 + 0.2×D-SSIM) | Coded | HIGH |
| Perceptual loss (LPIPS) | Not coded | MEDIUM |
| VGG feature loss | Not coded | LOW |
| Gradient loss | Not coded | LOW |

---

## PART 3: INITIALIZATION STRATEGIES

### 3.1 Position Initialization

| Strategy | File | Status | Result | Notes |
|----------|------|--------|--------|-------|
| **Uniform Grid** | `initializer.rs` | TESTED ✓ | Baseline | |
| **Jittered Grid** | `initializer.rs` | TESTED ✓ | +1.31 dB | BEST |
| **Halton Sequence** | `initializer.rs` | TESTED ✓ | +0.38 dB | Second best |
| **Golden Spiral** | `initializer.rs` | TESTED ✓ | -0.71 dB | REJECTED |
| **Edge-Aware** | `initializer.rs` | TESTED ✓ | Poor | Needs tuning |
| **K-Means (various)** | `kmeans_init.rs` | TESTED ✓ | -4.04 dB max | REJECTED |
| **GDGS (Laplacian)** | `gdgs_init.rs` | TESTED ✓ | -6.6 dB | REJECTED |
| **Gradient Peaks** | `gradient_peak_init.rs` | TESTED ✓ | -0.2 dB | Marginal |
| **SLIC Superpixel** | `slic_init.rs` | CODED | NOT TESTED | Requires Python |
| **Error-Driven** | `error_driven.rs` | TESTED ✓ | Variable | Works with Adam |
| **Poisson Disk** | Not coded | NOT TESTED | N/A | ChatGPT suggested |
| **Position Probability Map** | `position_probability_map.rs` | CODED | NOT TESTED | |
| **Saliency-based** | `saliency_detection.rs` | CODED | NOT TESTED | |

### 3.2 Color Initialization

| Strategy | File | Status | Result |
|----------|------|--------|--------|
| Single pixel sample | Built-in | TESTED ✓ | Baseline |
| Weighted average | `better_color_init.rs` | TESTED ✓ | Marginal |
| Guided filter color | `guided_filter.rs` | CODED | NOT TESTED |

### 3.3 Shape/Scale Initialization

| Strategy | File | Status | Result |
|----------|------|--------|--------|
| Uniform isotropic | Built-in | TESTED ✓ | Baseline |
| Structure tensor aligned | `structure_tensor.rs` | CODED | PARTIAL |
| Log-Cholesky from tensor | `log_cholesky.rs` | CODED | NOT TESTED |
| Coherence-based anisotropy | `log_cholesky.rs` | CODED | NOT TESTED |

---

## PART 4: DENSIFICATION/PRUNING

### 4.1 Tested Approaches

| Approach | Status | Result | Notes |
|----------|--------|--------|-------|
| **Growing (64→256)** | TESTED ✓ | -2.01 dB | REJECTED (momentum mismatch) |
| **Hybrid Pruning** | TESTED ✓ | +0.69 dB | BEST (27% fewer Gaussians) |
| **3DGS Clone** | TESTED ✓ | Poor | Momentum mismatch |
| **3DGS Split** | TESTED ✓ | Poor | Momentum mismatch |

### 4.2 Untested Densification Variations

| Variation | Status | Priority |
|-----------|--------|----------|
| Prune threshold 0.05 | Identified | HIGH |
| Prune threshold 0.15 | Identified | HIGH |
| Prune threshold 0.20 | Identified | HIGH |
| Multi-stage pruning | Not coded | MEDIUM |
| State inheritance on split | Partial test | MEDIUM |
| Adaptive prune threshold | Not coded | MEDIUM |

---

## PART 5: ADVANCED MODULES (lgi-core)

### 5.1 Rendering

| Module | File | Status | Integration |
|--------|------|--------|-------------|
| Standard renderer | `renderer.rs` | WORKING | Production |
| EWA Splatting v1 | `ewa_splatting.rs` | CODED | NOT INTEGRATED |
| EWA Splatting v2 | `ewa_splatting_v2.rs` | CODED | NOT INTEGRATED |
| Textured Gaussian | `textured_gaussian.rs` | CODED | NOT TESTED |
| Texture Map | `texture_map.rs` | CODED | NOT TESTED |

### 5.2 Image Analysis

| Module | File | Status | Integration |
|--------|------|--------|-------------|
| Structure Tensor | `structure_tensor.rs` | WORKING | Used in init |
| Guided Filter | `guided_filter.rs` | CODED | NOT INTEGRATED |
| Geodesic EDT | `geodesic_edt.rs` | CODED | PROBLEMATIC (regression) |
| Laplacian | `laplacian.rs` | CODED | Used in GDGS |
| Gaussian Filter | `gaussian_filter.rs` | WORKING | Used |
| Content Detection | `content_detection.rs` | CODED | PARTIAL |
| Saliency Detection | `saliency_detection.rs` | CODED | NOT TESTED |
| Specular Detection | `specular_detection.rs` | CODED | NOT TESTED |
| Text Stroke Detection | `text_stroke_detection.rs` | CODED | PARTIAL |

### 5.3 Quality & Metrics

| Module | File | Status | Integration |
|--------|------|--------|-------------|
| MS-SSIM | `ms_ssim.rs` | WORKING | Used |
| MS-SSIM Loss | `ms_ssim_loss.rs` | WORKING | Used |
| MS-SSIM Gradients | `ms_ssim_gradients.rs` | CODED | PARTIAL |
| Error Metrics | `error_metrics.rs` | CODED | PARTIAL |
| Rate-Distortion | `rate_distortion.rs` | CODED | NOT INTEGRATED |
| Entropy | `entropy.rs` | CODED | NOT TESTED |

### 5.4 Progressive/LOD

| Module | File | Status | Integration |
|--------|------|--------|-------------|
| LOD System | `lod_system.rs` | CODED | TESTED (works) |
| Progressive Loading | `progressive_loading.rs` | CODED | NOT TESTED |
| Progressive Render | `progressive.rs` | CODED | PARTIAL |
| Multi-Resolution | `multi_resolution.rs` | CODED | NOT TESTED |
| Gradient Upscaling | `gradient_upscaling.rs` | CODED | NOT TESTED |

### 5.5 Gaussian Operations

| Module | File | Status | Integration |
|--------|------|--------|-------------|
| Gaussian Splitting | `gaussian_splitting.rs` | CODED | Used in densify |
| Gaussian Merging | `gaussian_merging.rs` | CODED | NOT TESTED |
| Adaptive Splitting | `adaptive_splitting.rs` | CODED | PARTIAL |
| Energy-Based Selection | `energy_based_selection.rs` | CODED | NOT TESTED |
| Importance Ordering | `importance_ordering.rs` | CODED | NOT TESTED |

### 5.6 Advanced Techniques

| Module | File | Status | Priority |
|--------|------|--------|----------|
| Blue Noise Residual | `blue_noise_residual.rs` | CODED | HIGH - NOT TESTED |
| Coherence Enhancing Diffusion | `coherence_enhancing_diffusion.rs` | STUB | HIGH - NOT IMPLEMENTED |
| Frangi Vesselness | `frangi_vesselness.rs` | CODED | MEDIUM |
| Poisson Solver | `poisson_solver.rs` | CODED | NOT TESTED |
| Hilbert Ordering | `hilbert_ordering.rs` | CODED | NOT TESTED |
| Vector Quantization | `vector_quantization.rs` | STUB | MEDIUM |
| Predictive Coding | `predictive_coding.rs` | STUB | HIGH - NOT IMPLEMENTED |
| Perceptual Weighting | `perceptual_weighting.rs` | CODED | NOT TESTED |

### 5.7 Triggers & Actions

| Module | File | Status | Integration |
|--------|------|--------|-------------|
| Analytical Triggers | `analytical_triggers.rs` | CODED | PARTIAL |
| Trigger Actions | `trigger_actions.rs` | CODED | NOT TESTED |
| Selective Processing | `selective_processing.rs` | CODED | NOT TESTED |

---

## PART 6: CHATGPT RESEARCH TECHNIQUES

### 6.1 From v2-improvements-research.txt

| Technique | Status | Priority | Codebase Location |
|-----------|--------|----------|-------------------|
| Log-Cholesky Parameterization | IMPLEMENTED | HIGH | `lgi-math/log_cholesky.rs` |
| Geodesic EDT | IMPLEMENTED | HIGH | `lgi-core/geodesic_edt.rs` |
| EWA Splatting | IMPLEMENTED | HIGH | `lgi-core/ewa_splatting_v2.rs` |
| rANS Entropy Coding | NOT IMPLEMENTED | HIGH | N/A |
| Rate-Distortion Budgeting | IMPLEMENTED | HIGH | `lgi-core/rate_distortion.rs` |
| Steerable Gaussian Derivatives | NOT IMPLEMENTED | MEDIUM | N/A |
| Blue-Noise Residuals | IMPLEMENTED | MEDIUM | `lgi-core/blue_noise_residual.rs` |
| Guided Filter Color Model | IMPLEMENTED | HIGH | `lgi-core/guided_filter.rs` |
| Coherence Enhancing Diffusion | STUB ONLY | HIGH | Needs implementation |
| Weighted-Blended OIT | NOT IMPLEMENTED | LOW | N/A |
| Oklab Color Space | NOT IMPLEMENTED | LOW | N/A |

### 6.2 Integration Gaps

| Implemented Module | Integration Status | Action Needed |
|--------------------|-------------------|---------------|
| Log-Cholesky | NOT USED in optimizers | Wire into Adam/rendering |
| Geodesic EDT | PROBLEMATIC | Fix sigma clamping formula |
| EWA Splatting | NOT USED in encoding | Add to render pipeline |
| Rate-Distortion | NOT USED in optimizer | Integrate R-D selection |
| Blue-Noise Residual | NOT USED | Add post-processing step |
| Guided Filter | NOT USED for color init | Wire into initialization |

---

## PART 7: SYSTEMATIC EXPLORATION PLAN

### Phase 1: HIGH PRIORITY TESTS (Immediate)

1. **3DGS Loss Function** (L1 + D-SSIM)
   - File: Create `test_3dgs_loss.rs`
   - Compare: vs L2, vs Hybrid L2+MS-SSIM

2. **L1 Loss Alone**
   - File: Create `test_l1_loss.rs`
   - Expected: Sharper edges

3. **EWA Splatting Rendering**
   - File: Create `test_ewa_rendering.rs`
   - Compare: vs standard renderer at zoom levels

4. **Pruning Threshold Sweep**
   - File: Create `test_pruning_threshold_sweep.rs`
   - Test: 0.05, 0.10, 0.15, 0.20

5. **GPU L-M Sparse JtJ Strategy**
   - File: `compare_lm_strategies.rs` (extend)
   - Test: SparseJtJ vs BlockDiagonal

6. **Blue-Noise Residual Integration**
   - File: Create `test_blue_noise_integration.rs`
   - Add to encoding pipeline

### Phase 2: MEDIUM PRIORITY TESTS

7. **Guided Filter Color Initialization**
8. **Rate-Distortion Gaussian Selection**
9. **Log-Cholesky Optimizer Integration**
10. **OptimizerV3 Perceptual**
11. **LOD System Full Kodak Test**
12. **Coherence Enhancing Diffusion Implementation**

### Phase 3: EXPLORATORY TESTS

13. **SLIC Initialization** (requires Python preprocessing)
14. **Gaussian Merging** (post-optimization consolidation)
15. **Poisson Disk Initialization**
16. **Frangi Vesselness for Medical Images**
17. **Multi-Resolution Encoding**
18. **Textured Gaussians**

---

## PART 8: TEST RESULT FRAMEWORK USAGE

### Required Fields for All Tests
```rust
let mut result = TestResult::new("experiment_name", image_path);
result.optimizer = "Description".to_string();
result.n_gaussians = 256;
result.iterations = 200;
result.initial_psnr = init_psnr;
result.image_dimensions = (target.width, target.height);

// MANDATORY: Set all relevant parameters
result.set_param("lr_position", 0.0002);
result.set_param("lr_color", 0.02);
result.set_param("lr_scale", 0.005);
result.set_param("lr_opacity", 0.05);
result.set_param("beta1", 0.9);
result.set_param("beta2", 0.999);
// Add any technique-specific params

// Run optimization...

result.final_psnr = final_psnr;
result.improvement_db = final_psnr - init_psnr;
result.elapsed_seconds = elapsed;
result.complete();
let _ = result.save_with_image(&rendered, RESULTS_DIR);
```

### Abort Handling
```rust
if psnr < init_psnr - 5.0 {
    result.abort("PSNR diverging", iter, psnr);
    let _ = result.save_partial(RESULTS_DIR);
    return;
}
```

---

## PART 9: KNOWN GOOD CONFIGURATIONS

### Production Config (Adam)
```rust
lr_position: 0.0002
lr_color: 0.02
lr_scale: 0.005
lr_opacity: 0.05
beta1: 0.9
beta2: 0.999
iterations: 200
init: jittered_grid
```

### Best Hybrid Loss Config
```rust
alpha: 0.84  // 84% L2, 16% MS-SSIM
```

### Hybrid Pruning Config
```rust
phase1_iterations: 100
prune_threshold: 0.10
phase2_iterations: 100
```

---

## APPENDIX: File Index

### lgi-encoder-v2/src/
- `adam_optimizer.rs` - Primary optimizer
- `optimizer_v2.rs` - Gradient descent
- `hybrid_optimizer.rs` - Adam→L-BFGS
- `lm_optimizer.rs` - Levenberg-Marquardt
- `loss_functions.rs` - L1, SSIM, D-SSIM, combined
- `kmeans_init.rs` - K-means initialization
- `gdgs_init.rs` - Laplacian peaks
- `gradient_peak_init.rs` - Gradient-based
- `slic_init.rs` - SLIC superpixels
- `encoder_v3_adaptive.rs` - Content-adaptive
- `error_driven.rs` - Error-based placement
- `error_driven_adam.rs` - Error-driven + Adam
- `test_results.rs` - Test framework

### lgi-core/src/
59 modules - see listing above

### lgi-gpu/src/
- `adam_optimizer.rs` - GPU Adam
- `lbfgs_optimizer.rs` - GPU L-BFGS
- `lm_optimizer.rs` - GPU L-M
- `jtj_strategy.rs` - JtJ strategies
- `gradient.rs` - GPU gradients
- `buffer.rs` - GPU buffers

---

*Document auto-generated 2025-12-07. Updated as exploration proceeds.*
