# Session Summary: 2025-12-06 Afternoon

## Experiments Completed

### 1. MS-SSIM Hybrid Loss Alpha Sweep

**Objective**: Test different blend ratios between MSE and MS-SSIM loss functions.

**Formula**: `Loss = α·MSE + (1-α)·MS-SSIM`

**Test Configuration**:
- Image: kodim03.png (768×512)
- Gaussians: 256 (16×16 grid)
- Optimizer: Adam with per-parameter LRs
- Iterations: 200
- Initial PSNR: 20.27 dB

**Results**:

| Alpha | MSE Weight | MS-SSIM Weight | Final PSNR | PSNR Improvement | MS-SSIM Score |
|-------|------------|----------------|------------|------------------|---------------|
| 1.00 | 100% | 0% | 22.33 dB | +2.06 dB | 0.8384 |
| 0.84 | 84% | 16% | 22.74 dB | +2.47 dB | 0.8668 |
| 0.50 | 50% | 50% | 22.55 dB | +2.28 dB | 0.8613 |
| 0.20 | 20% | 80% | 22.24 dB | +1.97 dB | 0.8387 |

**Observations**:
- α=0.84 (3DGS paper recommendation) achieves highest PSNR
- α=0.84 also achieves highest MS-SSIM score
- α=0.20 (heavy MS-SSIM) underperforms even pure MSE
- Performance appears to peak around α=0.84, declining on both sides

---

### 2. L-BFGS Refined Testing

**Objective**: Test L-BFGS with improved constraint handling.

**Test 1a: Colors-only L-BFGS with soft barrier (weight=0.1)**
- Initial: 256 Gaussians, PSNR=20.27 dB
- Final: PSNR=21.18 dB (+0.91 dB)
- OOB colors: 188/256 (73%)
- Time: 703s
- Step size collapsed to α=9.54e-7 after iteration 20

**Test 1b: Colors-only L-BFGS with soft barrier (weight=1.0)**
- Initial: 256 Gaussians, PSNR=20.27 dB
- Final: PSNR=21.19 dB (+0.92 dB)
- OOB colors: 167/256 (65%)
- Time: 324.6s
- Step sizes remained active (α=1.25e-1)

**Test 2: Block-diagonal L-BFGS (separate histories)**
- Initial: 256 Gaussians, PSNR=20.27 dB
- Final: PSNR=20.27 dB (+0.00 dB)
- Step sizes: αc=3.05e-5, αg=3.05e-5 (collapsed)
- Gradient norms constant: ||gc||=2.73e-2, ||gg||=3.10e-1
- No progress after 70 iterations

**Observations**:
- Soft barriers reduce but do not eliminate OOB colors
- Stronger barriers (weight=1.0) slightly better than weaker (weight=0.1)
- Block-diagonal approach with separate histories does not converge
- Line search collapses to tiny step sizes in Tests 1a and 2

---

### 3. Levenberg-Marquardt Test (IN PROGRESS)

**Status**: Running (ID: 197ad9) - L-M phase started

**Configuration**:
- Finite differences for Jacobian computation
- 7 params per Gaussian: x, y, r, g, b, scale_x, scale_y
- Rotation and opacity NOT optimized
- Grid: 16×16 = 256 Gaussians
- Init: `encoder.initialize_gaussians()` → 17.12 dB initial PSNR
- Image: kodim03.png (768×512)

**Partial Results**:

| Phase | Iterations | Start PSNR | End PSNR | Improvement | Time |
|-------|------------|------------|----------|-------------|------|
| Adam-only (Test 1) | 100 | 17.12 dB | 21.37 dB | +4.25 dB | ~80s |
| Adam warm-up (Test 2) | 50 | 17.12 dB | 20.95 dB | +3.83 dB | 39.73s |
| L-M refinement (Test 2) | 20 (max) | 20.95 dB | TBD | TBD | TBD |

**Observations (partial)**:
- Adam achieves +4.25 dB in 100 iterations from 17.12 dB baseline
- Adam best found at iteration 73 (restored from checkpoint)
- L-M phase started, awaiting completion

---

## Raw Data Observations

### MS-SSIM Hybrid Loss Behavior
- Peak PSNR at α=0.84: +2.47 dB
- Pure MSE (α=1.0): +2.06 dB
- Equal blend (α=0.5): +2.28 dB
- MS-SSIM heavy (α=0.2): +1.97 dB
- Pattern: Inverted U-shape with peak around α=0.84

### L-BFGS Behavior
- Barrier weight 0.1 → 1.0: OOB dropped from 73% to 65%
- Similar final PSNR (~21.19 dB) regardless of barrier weight
- Block-diagonal: Complete failure to improve
- Step size collapse correlates with lack of progress

### Parameter Scale Comparison
| Parameter | Adam LR | L-BFGS observed step | Ratio |
|-----------|---------|---------------------|-------|
| Colors | 0.02 | ~0.01 | ~2× |
| Position | 0.0002 | collapsed | N/A |
| Scale | 0.005 | collapsed | N/A |

---

## Files Modified/Created

- `test-results/EXPERIMENT_FINDINGS_2025-12-05.md` - Updated with MS-SSIM and L-BFGS data
- `test-results/SESSION_2025-12-06_AFTERNOON.md` - This document

## Tests Still Running

1. **L-M Optimizer** (197ad9) - Collecting finite-difference Jacobian data
2. **L-BFGS Refined** (e58573) - Test 2 may still be running

---

## Math Verification

### PSNR Formula Consistency Check

**L-M test formula** (test_lm_optimizer.rs:144):
```rust
20.0 * (1.0 / mse.sqrt()).log10()
```
Expands to: `20 * log10(MSE^(-0.5)) = -10 * log10(MSE)` ✓

**MS-SSIM test formula** (test_adam_msssim_hybrid.rs:412):
```rust
10.0 * (1.0 / mse).log10()
```
Expands to: `10 * log10(MSE^(-1)) = -10 * log10(MSE)` ✓

**Result**: Both formulas are mathematically equivalent.

### Initialization Methods

| Test | Init Function | Initial PSNR |
|------|---------------|--------------|
| L-M | `encoder.initialize_gaussians()` | 17.12 dB |
| MS-SSIM | `create_jittered_grid()` | 20.27 dB |

**Observation**: Different initialization methods produce ~3 dB difference in starting quality. This must be accounted for when comparing results across tests.

---

## Next Steps

- Await L-M results
- Consider additional α values (0.9, 0.8, 0.7) for finer resolution around peak
- Consider L-BFGS with different initial Hessian approximations
- Consider L-BFGS-B (bounded variant) instead of soft barriers

## Notes on Initialization

Each test uses its native initialization method. When comparing results:
- L-M test: `encoder.initialize_gaussians()` → 17.12 dB baseline
- MS-SSIM test: `create_jittered_grid()` → 20.27 dB baseline

Compare improvements (Δ dB) rather than absolute final PSNR when evaluating optimizer effectiveness across different initialization methods.
