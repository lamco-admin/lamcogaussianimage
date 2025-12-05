# Systematic Experiment Framework

## Philosophy

These experiments produce **evidence about tool behavior**, not conclusions about optimal strategies. Different tools work differently on different content. The goal is understanding the parameter space, not finding "the best" approach.

## Available Tools Inventory

### Optimizers (6)
| Optimizer | File | Description | Best Use Case |
|-----------|------|-------------|---------------|
| Adam (per-param) | `adam_optimizer.rs` | Different LRs per parameter type | General purpose, stable |
| L-BFGS | `optimizer_lbfgs.rs` | Quasi-Newton, uses curvature | When gradients are smooth |
| Levenberg-Marquardt | `lm_optimizer.rs` | Nonlinear least-squares | Refinement after warm-start |
| Hybrid | `hybrid_optimizer.rs` | Combines approaches | TBD |
| Optimizer v2 | `optimizer_v2.rs` | Basic gradient descent | Baseline |
| Optimizer v3 | `optimizer_v3_perceptual.rs` | MS-SSIM + edge weighting | Perceptual quality |

### Loss Functions (5)
| Loss | Formula | Notes |
|------|---------|-------|
| L1 | MAE | Simple, robust to outliers |
| L2 | MSE | Differentiable everywhere |
| SSIM | Structural similarity | Perceptual, window-based |
| D-SSIM | 1 - SSIM | Loss form of SSIM |
| Combined | λ₁L1 + λ₂L2 + λ₃(1-SSIM) | Tunable mixture |
| 3DGS | L1 + D-SSIM | From 3D Gaussian Splatting paper |

### Initialization Strategies (5+)
| Strategy | File | Description |
|----------|------|-------------|
| Grid | default | Uniform grid placement |
| GDGS | `gdgs_init.rs` | Gradient-domain guided |
| Gradient Peak | `gradient_peak_init.rs` | Place at gradient maxima |
| K-means | `kmeans_init.rs` | Cluster-based placement |
| SLIC | `slic_init.rs` | Superpixel boundaries |
| Color-aware | `better_color_init.rs` | Improved color sampling |

### Image Content Types (from Kodak)
| Type | Examples | Characteristics |
|------|----------|-----------------|
| Natural gradients | kodim03, kodim04 | Smooth color transitions |
| Geometric edges | kodim01, kodim21 | Sharp man-made structures |
| Organic textures | kodim05, kodim23 | Irregular natural patterns |
| Architectural | kodim08, kodim19 | High-frequency detail |
| Mixed | kodim15, kodim22 | Multiple feature types |

## Experiment Matrix

### Phase 1: Optimizer Comparison (Fixed Init)
**Goal**: Understand how each optimizer behaves on different content types

**Fixed**: Grid init, 576 Gaussians (24×24), 150 iterations
**Variable**: Optimizer, Image

```
Optimizer × Image = 6 × 24 = 144 experiments
```

Output per experiment:
- Initial PSNR, Final PSNR, Improvement (dB)
- Loss curve (every 10 iterations)
- Time (seconds)
- Convergence behavior (monotonic/oscillating/plateau)

### Phase 2: Initialization Strategy Comparison (Fixed Optimizer)
**Goal**: Understand how starting point affects optimization trajectory

**Fixed**: Adam (per-param), 576 Gaussians, 150 iterations
**Variable**: Init strategy, Image

```
Init × Image = 5 × 24 = 120 experiments
```

### Phase 3: N Scaling Study
**Goal**: Understand rate-distortion behavior

**Fixed**: Adam, Grid init, 150 iterations
**Variable**: N (256, 576, 1024, 2048), Image subset (5 representative)

```
N × Image = 4 × 5 = 20 experiments
```

### Phase 4: Loss Function Comparison
**Goal**: Understand perceptual vs pixel-wise optimization

**Fixed**: Adam, Grid init, 576 Gaussians, 150 iterations
**Variable**: Loss function, Image subset

```
Loss × Image = 5 × 5 = 25 experiments
```

### Phase 5: Hybrid Strategies
**Goal**: Test Adam→L-M warm-start, multi-resolution

**Variable**: Strategy combinations

```
~20 experiments (TBD based on Phase 1-4 results)
```

## Total: ~330 experiments (Phase 1-4)

## JSON Result Schema

```json
{
  "experiment_id": "phase1_adam_kodim03_001",
  "phase": 1,
  "timestamp": "2025-12-05T19:00:00",
  "image": {
    "path": "kodim03.png",
    "dimensions": [768, 512],
    "content_type": "natural_gradients"
  },
  "config": {
    "optimizer": "Adam (per-param)",
    "init_strategy": "grid",
    "n_gaussians": 576,
    "iterations": 150,
    "loss_function": "L2"
  },
  "results": {
    "initial_psnr": 18.85,
    "final_psnr": 22.59,
    "improvement_db": 3.75,
    "final_loss": 0.005512,
    "elapsed_seconds": 270.77,
    "convergence": "monotonic"
  },
  "loss_curve": [
    {"iteration": 0, "loss": 0.015},
    {"iteration": 10, "loss": 0.012},
    ...
  ]
}
```

## Analysis Approach

### What We're Looking For (Evidence)
1. **Optimizer affinity**: Which optimizers show most improvement on which content types?
2. **Initialization sensitivity**: How much does starting point affect final quality?
3. **Scaling behavior**: How does PSNR grow with N? Linear? Logarithmic?
4. **Loss landscape**: Do different losses lead to different optima?
5. **Failure modes**: When does each approach break down?

### What We're NOT Doing
- Declaring any approach "best"
- Drawing conclusions from single experiments
- Ignoring variance and edge cases
- Assuming results generalize beyond Kodak

## Implementation Priority

1. **Experiment runner** that saves JSON results (like benchmark_optimizers.rs but systematic)
2. **Phase 1 first** - baseline optimizer comparison
3. **Analysis scripts** to aggregate and visualize results
4. **Incremental execution** - can stop/resume, handles failures gracefully
