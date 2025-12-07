# LGI Encoder Exploration Status
**Last Updated**: 2025-12-06 01:30 UTC
**Purpose**: Complete map of what has been explored, what's being tested, and what remains

---

## MAJOR FINDING (2025-12-06)

**All tested feature-based initialization methods perform WORSE than jittered grid + Adam.**

| Method | Final PSNR | vs Baseline (21.58 dB) |
|--------|------------|------------------------|
| K-Means p=10 | 17.54 dB | **-4.04 dB** |
| GDGS default | 14.99 dB | **-6.59 dB** |
| Gradient peak | 20.85 dB | **-0.73 dB** |
| ErrorDriven (500 G) | 19.90 dB | **-1.68 dB** |
| Weighted color init | 21.12 dB | **-0.46 dB** |

---

## Quick Reference

| Category | Explored | In Testing | Untested |
|----------|----------|------------|----------|
| Optimizers | 4 | 1 | 1 |
| Loss Functions | 3 | 1 | 2 |
| Initialization | 9 | 0 | 1 |
| Densification | 1 | 1 | 0 |
| Encoders | 3 | 0 | 1 |
| Advanced Features | 1 | 0 | 3 |

---

## PART 1: OPTIMIZERS

### Explored & Documented

| Optimizer | Result | File | Notes |
|-----------|--------|------|-------|
| **Adam (per-param LR)** | +2.25 dB avg | `adam_optimizer.rs` | BEST - Production recommended |
| **OptimizerV2** | +0.16 dB | `optimizer_v2.rs` | Works but slow convergence |
| **Hybrid Adam→L-BFGS** | -0.22 dB | `hybrid_optimizer.rs` | L-BFGS hurt quality |
| **L-M (finite diff)** | N/A | `lm_optimizer.rs` | Jacobian too expensive |

### Currently Testing

| Test | Status | Techniques |
|------|--------|------------|
| `test_optimizer_v3_perceptual` | Running (74f922) | MS-SSIM, Edge-weighted loss |

### Not Yet Explored

| Optimizer | File | Blocker |
|-----------|------|---------|
| `optimizer_lbfgs.rs` | argmin-based | Missing dependency - BROKEN |

---

## PART 2: LOSS FUNCTIONS

### Explored & Documented

| Loss | Result | Notes |
|------|--------|-------|
| **L2 (MSE)** | Baseline | Standard, fast |
| **Hybrid L2+MS-SSIM (80/20)** | +0.51 dB vs L2 | Better perceptual quality |
| **MS-SSIM only** | -0.02 dB | Gradient issues |

### Currently Testing

| Test | Status | Techniques |
|------|--------|------------|
| `test_loss_functions` | Running (b5c34a) | L1, D-SSIM, 3DGS combined (SLOW - finite diff) |
| `test_optimizer_v3_perceptual` | Running (74f922) | MS-SSIM loss, Edge-weighted loss |

### Not Yet Explored

| Loss | File | Potential |
|------|------|-----------|
| **LPIPS** | Not implemented | Perceptual similarity |
| **Charbonnier** | Not implemented | Smooth L1 variant |

---

## PART 3: INITIALIZATION STRATEGIES

### Explored & Documented

| Strategy | Result | Notes |
|----------|--------|-------|
| **Jittered Grid** | 21.58 dB | **BEST - Production recommended** |
| **Halton Sequence** | +0.38 dB vs uniform | Second best |
| **Uniform Grid** | 20.99 dB | Baseline |
| **Golden Spiral** | -0.71 dB | Fails on rectangles |
| **K-Means (default)** | Below grid | Tested c=1,p=0.5,g=0.3 |
| **K-Means (p=2)** | 20.14 dB | Closest to grid among K-Means |
| **K-Means (p=10)** | 17.54 dB | **FAILS: -4.04 dB vs baseline** |
| **Gradient Peak** | 20.85 dB (315 G) | **FAILS: -0.73 dB with more Gaussians** |
| **GDGS (Laplacian)** | 14.99 dB (500 G) | **FAILS: -6.59 dB catastrophically worse** |
| **Weighted color init** | 21.12 dB | **FAILS: -0.46 dB vs single-pixel** |
| **Single-pixel color** | 21.58 dB | Standard, works best |

### Key Insight

**Feature-based initialization does not improve quality.** All tested methods (K-Means, GDGS, gradient peaks, weighted color) perform worse than simple jittered grid, even when using more Gaussians.

### Not Yet Explored

| Strategy | File | Notes |
|----------|------|-------|
| **SLIC** | `slic_init.rs` | Requires Python preprocessing (user declined) |

---

## PART 4: DENSIFICATION (3DGS Split/Clone/Prune)

### Explored & Documented

| Configuration | Result | Notes |
|---------------|--------|-------|
| 144→288 Gaussians | +0.54 dB | threshold=0.0001, interval=20 |
| Fair baseline 256 | 20.99 dB | For comparison |

### Currently Testing

| Test | Status | Techniques |
|------|--------|------------|
| `test_densification_variations` | Running (ffe315) | Sparse starts (64, 100), different thresholds |
| `test_adaptive_densification` | Completed | Original single config test |

### Variations to Explore

- Starting from 32/64 Gaussians (more aggressive growth)
- Different prune thresholds
- Longer training after densification

---

## PART 5: ENCODERS

### Explored & Documented

| Encoder | Result | Notes |
|---------|--------|-------|
| **Jittered Grid + Adam** | 21.58 dB | **BEST - Production recommended** |
| **Uniform Grid + Adam** | 20.99 dB | Baseline |
| **EncoderV3Adaptive** | Gap vs Adam | Uses OptimizerV2 internally |
| **ErrorDrivenEncoder** | 19.90 dB (500 G) | **FAILS: -1.68 dB vs baseline with 2× Gaussians** |
| **ErrorDriven+Adam** | 20.44 dB (300 G) | Still -0.55 dB vs baseline |
| **ErrorDriven(64→256)** | 19.78 dB | -1.21 dB vs same-size baseline |

### Key Insight: ErrorDrivenEncoder Does NOT Deliver

The claimed +4.3 dB improvement from adaptive hotspot placement is **false**. ErrorDrivenEncoder actually performs WORSE than uniform grid, even with 2× the Gaussians.

### Not Yet Explored

| Encoder | File | Description |
|---------|------|-------------|
| **CommercialEncoder** | `commercial_encoder.rs` | Full pipeline: textures, residuals, triggers, guided filter |

---

## PART 6: ADVANCED FEATURES

### LOD System (`lgi-core/lod_system.rs`) - TESTED 2025-12-06

**Purpose**: View-dependent rendering, progressive loading

**Test Results** (512 Gaussians on kodim03, Full = 19.41 dB):

| LOD Level | N Gaussians | PSNR | % Full Quality | % of Gaussians |
|-----------|-------------|------|----------------|----------------|
| Coarse | 88 | 17.10 dB | 88% | 17% |
| Coarse+Medium | 196 | 18.11 dB | 93% | 38% |
| Medium | 108 | 16.47 dB | 85% | 21% |
| Fine | 316 | 10.15 dB | 52% | 62% |
| Full | 512 | 19.41 dB | 100% | 100% |

**Key Insight**: LOD system WORKS WELL for progressive streaming:
- Coarse alone provides 88% quality with only 17% of Gaussians
- Coarse+Medium provides 93% quality with 38% of Gaussians
- Enables efficient preview → full quality progressive loading

### Analytical Triggers (`lgi-core/analytical_triggers.rs`)

**Purpose**: Detect when to apply advanced techniques

**7 Metrics**:
1. SED - Spectral Energy Drop (HF loss)
2. ERR - Entropy-Residual Ratio (capacity)
3. LCC - Laplacian Consistency (edge sharpness)
4. AGD - Anisotropy Gradient Divergence (region boundaries)
5. JCI - Jacobian Condition Index (chromatic instability)
6. RCH - Rate-Curvature Heuristic (over-allocation)
7. SPEC - Structure-Perceptual Error Correlation

**Testing Approach**: Integrate with encoder, measure trigger accuracy

### Textured Gaussians (`lgi-core/textured_gaussian.rs`)

**Purpose**: Per-Gaussian texture maps for detail

**Testing Approach**: Compare base vs textured at same Gaussian count

### Blue-Noise Residuals (`lgi-core/blue_noise_residual.rs`)

**Purpose**: Add dithered residual for fine detail

**Testing Approach**: Measure PSNR improvement from residual layer

---

## PART 7: PLACEMENT ITERATIONS SUMMARY

### What We Know

1. **Grid-based methods work well**:
   - Jittered grid > Halton > Uniform > Golden Spiral
   - Simple jitter (+0.4× cell size) gives best results

2. **Feature-based methods need tuning**:
   - K-Means: position-heavy (p=2) closest to grid
   - Gradient Peak: in testing
   - GDGS: in testing

3. **The optimization matters more than init**:
   - Adam optimizer provides +2.25 dB improvement
   - Good init provides +0.5-1.0 dB head start
   - Total: ~+3 dB from uniform grid baseline

### Current Hypothesis Being Tested

| Hypothesis | Test | Expected |
|------------|------|----------|
| Higher K-Means position weight → grid-like | test_kmeans_high_position | p=10 should match grid |
| Sparse start + densification → efficient | test_densification_variations | 64→256 should match 256 direct |
| Gradient peaks → better edge handling | test_gradient_peak_init | +0.5 dB on edge-heavy images |
| GDGS → fewer Gaussians for same quality | test_gdgs_init | 128 GDGS ≈ 256 grid |

---

## PART 8: WHAT'S RUNNING NOW (9 Background Tests)

| ID | Test | Est. Time | Focus |
|----|------|-----------|-------|
| b5c34a | test_loss_functions | Hours | L1, D-SSIM (finite diff) |
| 5e6dc4 | test_kmeans_init | ~30 min | K-Means variations |
| a056f2 | test_gradient_peak_init | ~30 min | Gradient peak init |
| 238886 | test_encoder_v3_with_adam | ~45 min | V3 + Adam |
| 53bb8b | test_kmeans_high_position | ~45 min | K-Means p=5,10 |
| ffe315 | test_densification_variations | ~60 min | Sparse starts |
| 87e8c8 | test_gdgs_init | ~45 min | GDGS Laplacian |
| 74f922 | test_optimizer_v3_perceptual | ~30 min | MS-SSIM, edge-weighted |
| 2cd675 | test_better_color_init | ~30 min | Weighted color |

---

## PART 9: RECOMMENDED NEXT STEPS

### After Current Tests Complete

1. **Analyze results** - Update OPTIMIZER_INVENTORY.md with new observations
2. **Identify winners** - Which init methods improve over jittered grid?
3. **Combine techniques** - Best init + Adam + hybrid loss

### Future Exploration

| Priority | Technique | Reason |
|----------|-----------|--------|
| High | CommercialEncoder | Full feature integration test |
| High | ErrorDrivenEncoder | Adaptive placement claims +4.3 dB |
| Medium | LOD System quality | Measure quality at each LOD level |
| Medium | Analytical Triggers | Validate trigger thresholds |
| Low | Textured Gaussians | Requires renderer_v3_textured |
| Low | Blue-noise Residuals | Final polish step |

---

## File Location Reference

```
lgi-encoder-v2/src/
├── adam_optimizer.rs          # TESTED - Production
├── optimizer_v2.rs            # TESTED - Slower
├── hybrid_optimizer.rs        # TESTED - L-BFGS hurts
├── optimizer_v3_perceptual.rs # TESTING - MS-SSIM
├── loss_functions.rs          # TESTING - L1, D-SSIM
├── adaptive_densification.rs  # TESTING - Split/clone
├── kmeans_init.rs             # TESTING - Clustering
├── gradient_peak_init.rs      # TESTING - Peak placement
├── gdgs_init.rs               # TESTING - Laplacian peaks
├── better_color_init.rs       # TESTING - Weighted color
├── commercial_encoder.rs      # UNTESTED - Full pipeline
├── error_driven.rs            # UNTESTED - Hotspot placement
└── encoder_v3_adaptive.rs     # TESTING - Content-adaptive

lgi-core/src/
├── lod_system.rs              # UNTESTED - Multi-resolution
├── analytical_triggers.rs     # UNTESTED - 7 metrics
├── textured_gaussian.rs       # UNTESTED - Per-splat textures
└── blue_noise_residual.rs     # UNTESTED - Dithered residuals
```

---

*This document tracks exploration progress. Results are configuration-specific observations, not final conclusions.*
