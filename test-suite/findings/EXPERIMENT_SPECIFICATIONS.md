# Experiment Specifications

**Purpose**: Systematic experiments to build an optimization toolkit for LGI Gaussian image codec.

**Principle**: Results are EVIDENCE about tool behavior, NOT conclusions. Changing variables changes results.

---

## Execution Order

```
Phase 1: Optimizer Tuning (MUST complete before Phase 2)
  1.1 Adam tuning
  1.2 OptimizerV2 tuning
  1.3 OptimizerV3 tuning
  1.4 L-BFGS tuning
  1.5 L-M tuning
  1.6 Hybrid tuning

Phase 2: Optimizer Comparison (requires tuned optimizers)
  2.1 Fair comparison on validation set

Phase 3: Placement Strategy (use best-tuned optimizer from Phase 2)
  3.1 Test each placement strategy

Phase 4: Alternative Techniques (independent of Phase 2-3)
  4.1 Truncation radius implementation
  4.2 Separable 1D×1D Gaussians
  4.3 Gabor functions

Phase 5: Full Kodak Baseline (after Phases 1-3)
  5.1 Complete benchmark
```

---

## Standard Configuration

All experiments use these defaults unless specified:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Validation image | `kodim03.png` | 768×512, natural gradients, good baseline |
| N Gaussians | 576 (24×24 grid) | Reasonable density for testing |
| Image dimensions | 768×512 | Standard Kodak |
| Metrics | PSNR, loss, time | Always capture all three |
| Results directory | `test-results/YYYY-MM-DD/` | Date-organized |

---

## Phase 1: Optimizer Tuning Experiments

### 1.1 Adam Optimizer Tuning

**File**: `lgi-encoder-v2/src/adam_optimizer.rs`

**Current Defaults**:
```rust
LearningRates {
    position: 0.0002,
    color: 0.02,
    scale: 0.005,
    opacity: 0.05,
    position_final: 0.00002,  // After decay
}
beta1: 0.9
beta2: 0.999
epsilon: 1e-8
max_iterations: 100
```

**Hyperparameters to Tune**:

| Parameter | Test Values | Methodology |
|-----------|-------------|-------------|
| `lr_position` | [0.0001, 0.0002, 0.0005, 0.001] | Manual sweep |
| `lr_color` | [0.01, 0.02, 0.05, 0.1] | Manual sweep |
| `lr_scale` | [0.002, 0.005, 0.01, 0.02] | Manual sweep |
| `lr_opacity` | [0.02, 0.05, 0.1] | Manual sweep |
| `position_final` | [lr_position/10, lr_position/20] | Relative to initial |
| `max_iterations` | [100, 200, 300] | Convergence study |

**Tuning Protocol**:
1. Start with current defaults (they came from 3DGS research)
2. Sweep ONE parameter at a time, holding others constant
3. Use PSNR improvement as primary metric
4. Record convergence behavior (monotonic vs oscillating)

**"Tuned" Criteria**:
- Monotonic convergence (no oscillation)
- PSNR improvement > 2.0 dB over initial
- No divergence on any test image

**Estimated Time**: 4-6 hours (16 parameter combinations × ~10 min each)

**Results Schema**:
```json
{
  "experiment": "adam_tuning",
  "timestamp": "ISO8601",
  "image": "kodim03.png",
  "n_gaussians": 576,
  "base_config": { "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8 },
  "sweep_parameter": "lr_position",
  "sweep_values": [0.0001, 0.0002, 0.0005, 0.001],
  "results": [
    {
      "value": 0.0001,
      "initial_psnr": 21.5,
      "final_psnr": 24.2,
      "improvement_db": 2.7,
      "convergence": "monotonic|oscillating|divergent",
      "time_seconds": 180,
      "iterations_used": 100
    }
  ],
  "best_value": 0.0002,
  "best_psnr": 24.6
}
```

---

### 1.2 OptimizerV2 Tuning

**File**: `lgi-encoder-v2/src/optimizer_v2.rs`

**Current Defaults**:
```rust
learning_rate_position: 0.10
learning_rate_scale: 0.10
learning_rate_color: 0.6
learning_rate_rotation: 0.02
max_iterations: 100
use_ms_ssim: false
use_edge_weighted: false
```

**Key Feature**: Adaptive LR based on Gaussian density (`density_factor = sqrt(100/n).min(1.0)`)

**Hyperparameters to Tune**:

| Parameter | Test Values | Methodology |
|-----------|-------------|-------------|
| `lr_position` | [0.05, 0.10, 0.15, 0.20] | Manual sweep |
| `lr_color` | [0.3, 0.6, 0.9, 1.2] | Manual sweep |
| `lr_scale` | [0.05, 0.10, 0.15] | Manual sweep |
| `lr_rotation` | [0.01, 0.02, 0.04] | Manual sweep |
| `use_ms_ssim` | [true, false] | A/B test |
| `use_edge_weighted` | [true, false] | A/B test |

**Tuning Protocol**:
1. Test with GPU disabled first (establish baseline)
2. Sweep LRs one at a time
3. A/B test MS-SSIM vs L2 with best LRs
4. A/B test edge-weighted vs uniform with best LRs

**"Tuned" Criteria**:
- Same as Adam: monotonic convergence, >2.0 dB improvement

**Estimated Time**: 4-6 hours

---

### 1.3 OptimizerV3 (Perceptual) Tuning

**File**: `lgi-encoder-v2/src/optimizer_v3_perceptual.rs`

**Current Defaults**:
```rust
learning_rate_position: 0.05
learning_rate_scale: 0.05
learning_rate_color: 0.3
learning_rate_rotation: 0.01
max_iterations: 100
use_ms_ssim: true   // Perceptual by default
use_edge_weighted: true
```

**Note**: This optimizer REQUIRES a `StructureTensorField` for edge weighting.

**Hyperparameters to Tune**:

| Parameter | Test Values | Methodology |
|-----------|-------------|-------------|
| `lr_position` | [0.02, 0.05, 0.10] | Manual sweep |
| `lr_color` | [0.15, 0.3, 0.6] | Manual sweep |
| `lr_scale` | [0.02, 0.05, 0.10] | Manual sweep |
| Loss combo | [MS-SSIM only, edge only, both] | A/B/C test |

**"Tuned" Criteria**:
- MS-SSIM score improvement (not just PSNR)
- Subjective edge quality (visual inspection)

**Estimated Time**: 3-4 hours

---

### 1.4 L-BFGS Tuning

**File**: `lgi-core/src/lbfgs.rs`

**Current Defaults**:
```rust
history_size: 10
c1: 1e-4  // Armijo constant
c2: 0.9   // Wolfe constant
max_line_search_iters: 20
```

**Note**: L-BFGS should NOT be used from scratch. Test only after Adam warm-start.

**Hyperparameters to Tune**:

| Parameter | Test Values | Methodology |
|-----------|-------------|-------------|
| `history_size` | [5, 10, 15, 20] | Manual sweep |
| `c1` (Armijo) | [1e-5, 1e-4, 1e-3] | Manual sweep |
| `tolerance` | [1e-6, 1e-7, 1e-8] | Manual sweep |
| `max_iterations` | [25, 50, 100] | Convergence study |

**Tuning Protocol**:
1. Always use 100 Adam iterations first (warm-start)
2. Then apply L-BFGS with different settings
3. Measure ADDITIONAL improvement from L-BFGS phase

**"Tuned" Criteria**:
- L-BFGS provides >5% additional improvement over Adam alone
- Converges without line search failures

**Estimated Time**: 3-4 hours

---

### 1.5 Levenberg-Marquardt Tuning

**File**: `lgi-encoder-v2/src/lm_optimizer.rs`

**Current Defaults**:
```rust
max_iterations: 100
tolerance: 1e-8
initial_lambda: 0.01
use_finite_diff: true
fd_epsilon: 1e-6
```

**CRITICAL**: L-M uses finite differences for Jacobian, which is O(n_params × n_residuals). VERY SLOW for large problems.

**Hyperparameters to Tune**:

| Parameter | Test Values | Methodology |
|-----------|-------------|-------------|
| `initial_lambda` | [0.001, 0.01, 0.1, 1.0] | Manual sweep |
| `tolerance` | [1e-6, 1e-8, 1e-10] | Convergence study |
| `fd_epsilon` | [1e-5, 1e-6, 1e-7] | Accuracy study |
| `max_iterations` | [25, 50, 100] | Time budget |

**Note**: Test with SMALL Gaussian count first (e.g., 64-144) due to computational cost.

**"Tuned" Criteria**:
- Converges (termination != MaxIterations too often)
- Achieves comparable PSNR to Adam in fewer iterations

**Estimated Time**: 4-6 hours (slow due to Jacobian computation)

---

### 1.6 Hybrid Optimizer Tuning

**File**: `lgi-encoder-v2/src/hybrid_optimizer.rs`

**Current Defaults**:
```rust
adam_iterations: 100
lbfgs_iterations: 50
lbfgs_history: 10
adam_lr: LearningRates::default()
```

**Hyperparameters to Tune**:

| Parameter | Test Values | Methodology |
|-----------|-------------|-------------|
| `adam_iterations` | [50, 100, 150] | Manual sweep |
| `lbfgs_iterations` | [25, 50, 100] | Manual sweep |
| `lbfgs_history` | [5, 10, 15] | Manual sweep |
| Ratio | Adam-heavy vs L-BFGS-heavy | Trade-off study |

**"Tuned" Criteria**:
- Total time comparable to Adam alone
- Final PSNR better than Adam alone
- L-BFGS phase doesn't diverge

**Estimated Time**: 3-4 hours

---

## Phase 2: Optimizer Comparison Experiments

### 2.1 Fair Comparison

**Prerequisites**: All Phase 1 tuning complete

**Standard Configuration**:
```json
{
  "initialization": "uniform_grid",
  "n_gaussians": 576,
  "images": ["kodim03.png", "kodim01.png", "kodim05.png", "kodim08.png", "kodim13.png"],
  "iteration_budget": 200,
  "comparison_mode": "same_iterations"  // or "same_wallclock"
}
```

**Experiments**:

1. **Same iteration budget**: Each optimizer gets 200 iterations
2. **Same wall-clock time**: Each optimizer gets 5 minutes

**Optimizers to Compare**:
- Adam (tuned)
- OptimizerV2 (tuned)
- OptimizerV3 (tuned)
- L-BFGS alone (tuned, just for reference—expected to fail)
- Hybrid Adam→L-BFGS (tuned)
- L-M (tuned)

**Metrics to Capture**:
```json
{
  "experiment": "optimizer_comparison",
  "timestamp": "ISO8601",
  "mode": "same_iterations|same_wallclock",
  "budget": 200,  // iterations or seconds
  "results_per_image": [
    {
      "image": "kodim03.png",
      "optimizers": [
        {
          "name": "Adam (tuned)",
          "config": { ... },
          "initial_psnr": 21.5,
          "final_psnr": 24.8,
          "improvement_db": 3.3,
          "time_seconds": 180,
          "iterations_used": 200,
          "convergence_curve": [21.5, 22.1, 22.8, ...]  // Every 10 iterations
        }
      ]
    }
  ],
  "summary": {
    "best_overall": "Hybrid",
    "best_per_image": { "kodim03": "Adam", "kodim01": "Hybrid" },
    "avg_improvement_by_optimizer": { "Adam": 2.8, "Hybrid": 3.1, ... }
  }
}
```

**Estimated Time**: 6-8 hours (5 images × 6 optimizers × ~10-15 min each)

---

## Phase 3: Placement Strategy Experiments

### 3.1 Placement Comparison

**Prerequisites**: Phase 2 complete (know which optimizer to use)

**Optimizer**: Use BEST optimizer from Phase 2 (tuned settings)

**Strategies to Test**:

| Strategy | File | Description |
|----------|------|-------------|
| Uniform grid | (default) | Evenly spaced N×M grid |
| PPM-weighted | (custom) | Position-weighted by gradient magnitude |
| Laplacian peaks | (custom) | Place at Laplacian response peaks |
| K-means | `kmeans_init.rs` | Cluster pixel colors |
| SLIC superpixel | `slic_init.rs` | Superpixel-based placement |
| Gradient peaks (GDGS) | `gdgs_init.rs` | GDGS-style gradient-driven |
| Gradient peaks | `gradient_peak_init.rs` | Local gradient maxima |

**For EACH strategy**:
- Test with isotropic AND anisotropic Gaussians
- Same Gaussian count (576)
- Same iteration budget (200)

**Why PPM/Laplacian Failed Previously**:
The previous experiment showed -8 dB. This is EVIDENCE that these strategies need different optimizer settings or different Gaussian configurations. Re-test with tuned optimizer.

**Results Schema**:
```json
{
  "experiment": "placement_strategy_comparison",
  "timestamp": "ISO8601",
  "optimizer": "Adam (tuned)",
  "optimizer_config": { ... },
  "n_gaussians": 576,
  "iterations": 200,
  "image": "kodim03.png",
  "results": [
    {
      "strategy": "Uniform (iso)",
      "gaussian_type": "isotropic",
      "initial_psnr": 21.5,
      "final_psnr": 24.8,
      "improvement_db": 3.3,
      "time_seconds": 180,
      "vs_baseline_db": 0.0
    },
    {
      "strategy": "PPM (iso)",
      "gaussian_type": "isotropic",
      "initial_psnr": 18.2,  // Different starting point!
      "final_psnr": 22.1,
      "improvement_db": 3.9,
      "time_seconds": 195,
      "vs_baseline_db": -2.7,
      "notes": "Lower initial PSNR due to clustering"
    }
  ]
}
```

**Key Insight**: Compare FINAL PSNR, not improvement. Different strategies have different starting points.

**Estimated Time**: 8-10 hours (7 strategies × 2 types × ~15 min each)

---

## Phase 4: Alternative Technique Experiments

### 4.1 Truncation Radius

**Implementation Effort**: LOW (1-2 hours)

**What**: Limit Gaussian evaluation to pixels within 3σ or 4σ of center.

**Changes Required**:
1. Modify `renderer_v2.rs` to compute bounding box per Gaussian
2. Skip pixel evaluation if outside bounding box

**Test Configuration**:
```json
{
  "truncation_radii": [3.0, 3.5, 4.0, 5.0],  // In units of max(σ_x, σ_y)
  "baseline": "no_truncation",
  "metrics": ["psnr", "rendering_time", "visual_quality"]
}
```

**Expected**:
- Rendering speedup: 10-100×
- Quality loss: Minimal (<0.1 dB at 3σ)

**Estimated Time**: 2-3 hours (implementation + testing)

---

### 4.2 Separable 1D×1D Gaussians

**Implementation Effort**: MEDIUM (4-8 hours)

**What**: For elongated Gaussians, use product of two 1D Gaussians instead of full 2D.

**Mathematical Basis**:
```
G(x,y) = G_x(x) × G_y(y)
G_x(x) = exp(-x²/(2σ_x²))
G_y(y) = exp(-y²/(2σ_y²))
```

**Implementation**:
1. Add `SeparableGaussian2D` struct
2. Implement separable rendering path
3. Criteria for when to use: aspect ratio > 3:1

**Test Configuration**:
```json
{
  "experiment": "separable_1d_gaussians",
  "aspect_ratio_threshold": [2.0, 3.0, 4.0],
  "test_cases": ["uniform_grid", "edge_heavy_image"],
  "metrics": ["psnr", "rendering_time", "edge_quality"]
}
```

**Expected**:
- Computational speedup: 8× for highly elongated Gaussians
- Quality: Potentially BETTER for edges (more natural representation)

**Estimated Time**: 6-8 hours

---

### 4.3 Gabor Functions

**Implementation Effort**: HIGH (1-2 days)

**What**: Gaussian × sinusoid for texture/edge representation.

**Mathematical Basis**:
```
Gabor(x,y) = Gaussian(x,y) × cos(2πfx' + φ)

Where:
  x' = x*cos(θ) + y*sin(θ)  // Rotated coordinate
  f = spatial frequency
  φ = phase
  θ = orientation
```

**New Parameters per Gabor**:
- Position (x, y)
- Scale (σ_x, σ_y)
- Orientation (θ)
- Frequency (f)
- Phase (φ)
- Color (r, g, b)
- Opacity (α)

**Total: 10 parameters vs 8 for Gaussian** (adds f, φ)

**Implementation**:
1. Create `GaborFunction2D` struct
2. Implement Gabor renderer
3. Modify optimizer to handle frequency/phase gradients
4. Test on edge-heavy images

**Test Configuration**:
```json
{
  "experiment": "gabor_functions",
  "test_images": ["kodim01.png", "kodim08.png"],  // Edge-heavy
  "frequency_range": [0.5, 1.0, 2.0, 4.0],  // cycles per Gaussian width
  "compare_to": "standard_gaussian",
  "metrics": ["psnr", "edge_psnr", "ssim"]
}
```

**Expected**:
- Better edge representation (potentially +2-5 dB on edges)
- More complex optimization (more parameters)

**Estimated Time**: 12-16 hours

---

## Phase 5: Full Kodak Baseline

### 5.1 Complete Benchmark

**Prerequisites**: Phases 1-3 complete

**Configuration**:
```json
{
  "images": "all 24 Kodak images",
  "optimizer": "BEST from Phase 2",
  "placement": "BEST from Phase 3",
  "n_gaussians": 576,
  "iterations": 200
}
```

**Results Schema**:
```json
{
  "experiment": "kodak_baseline",
  "timestamp": "ISO8601",
  "optimizer": "Adam (tuned)",
  "placement": "Uniform",
  "n_gaussians": 576,
  "iterations": 200,
  "results": [
    {
      "image": "kodim01.png",
      "dimensions": [768, 512],
      "content_type": "geometric_edges",
      "initial_psnr": 21.2,
      "final_psnr": 24.1,
      "ssim": 0.87,
      "time_seconds": 185,
      "encoded_size_bytes": 576 * 28,  // 7 params × 4 bytes
      "compression_ratio": "(768*512*3) / encoded_size"
    }
  ],
  "summary": {
    "mean_psnr": 23.8,
    "std_psnr": 1.2,
    "median_psnr": 24.0,
    "min_psnr": { "image": "kodim05.png", "value": 21.5 },
    "max_psnr": { "image": "kodim03.png", "value": 26.2 },
    "mean_ssim": 0.86,
    "mean_time_seconds": 180,
    "mean_compression_ratio": 73.0
  },
  "by_content_type": {
    "natural_gradients": { "images": ["kodim03", "kodim13"], "mean_psnr": 25.1 },
    "geometric_edges": { "images": ["kodim01", "kodim08"], "mean_psnr": 23.5 },
    "organic_textures": { "images": ["kodim05", "kodim15"], "mean_psnr": 22.3 }
  }
}
```

**Estimated Time**: 10-12 hours (24 images × ~25 min each, run sequentially)

---

## Result File Naming Convention

```
test-results/YYYY-MM-DD/
├── adam_tuning_lr_position_HH-MM-SS.json
├── adam_tuning_lr_color_HH-MM-SS.json
├── optimizer_comparison_same_iterations_HH-MM-SS.json
├── placement_comparison_kodim03_HH-MM-SS.json
├── truncation_radius_HH-MM-SS.json
├── separable_1d_HH-MM-SS.json
├── gabor_functions_HH-MM-SS.json
└── kodak_baseline_HH-MM-SS.json
```

---

## Execution Guidelines

1. **ONE experiment at a time** - VM crashes with parallel runs
2. **Save results immediately** - Use `TestResult::save()` after each run
3. **Record ALL parameters** - Full reproducibility
4. **Don't draw conclusions early** - More data = better understanding
5. **Re-run surprising results** - Verify before documenting
6. **Track convergence curves** - Not just final values

---

## Time Estimates Summary

| Phase | Estimated Time | Dependencies |
|-------|----------------|--------------|
| 1.1 Adam tuning | 4-6 hours | None |
| 1.2 V2 tuning | 4-6 hours | None |
| 1.3 V3 tuning | 3-4 hours | None |
| 1.4 L-BFGS tuning | 3-4 hours | None |
| 1.5 L-M tuning | 4-6 hours | None |
| 1.6 Hybrid tuning | 3-4 hours | 1.1, 1.4 |
| 2.1 Comparison | 6-8 hours | All Phase 1 |
| 3.1 Placement | 8-10 hours | Phase 2 |
| 4.1 Truncation | 2-3 hours | None |
| 4.2 Separable | 6-8 hours | None |
| 4.3 Gabor | 12-16 hours | None |
| 5.1 Kodak baseline | 10-12 hours | Phases 1-3 |

**Total**: ~65-90 hours of experiment time

---

## Next Steps

1. Begin with Phase 1.1 (Adam tuning) - it's the most mature optimizer
2. Run Phase 4.1 (truncation) in parallel - independent, quick win
3. After Phase 1 complete, run Phase 2
4. After Phase 2, run Phase 3
5. Phase 4.2-4.3 can run anytime (independent research track)
6. Phase 5 only after Phases 1-3 complete
