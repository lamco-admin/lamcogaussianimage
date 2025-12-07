# Session Handover: 2025-12-06 Evening

## Executive Summary

This session focused on **optimizer exploration and adaptive placement strategies** for the LGI Gaussian image encoder. The work is intentionally **exploratory**—gathering data across multiple approaches without declaring winners. Key activity areas:

1. Created `ErrorDrivenAdamEncoder` - Adam-based variant of error-driven placement
2. Collected baseline comparisons across uniform vs adaptive placement
3. MS-SSIM hybrid loss alpha sweep completed
4. L-BFGS constraint handling experiments
5. Levenberg-Marquardt test still running (very slow)

---

## Active Background Processes

Several tests are still running from previous sessions. Check status with `BashOutput`:

| Bash ID | Test | Notes |
|---------|------|-------|
| `197ad9` | L-M Optimizer | Finite-diff Jacobians, extremely slow |
| `136a11` | 3DGS Faithful Densification | |
| `640b04` | Adam Advanced | |
| `9c07fb` | Combined Densification | |
| `bc0380` | L-BFGS Standalone | |
| `e58573` | L-BFGS Refined | |
| `79ef23` | Adam MS-SSIM Hybrid | |

---

## New Files Created This Session

### 1. `src/error_driven_adam.rs` (NEW - 380 lines)

Adam-based error-driven encoder addressing identified limitations of the OptimizerV2 variant:

**Key Design Decisions:**
- Fresh Adam optimizer state per pass (old momentum invalid when Gaussians added)
- Spatial clustering for hotspot selection (prevents redundant adjacent placements)
- Early stopping when PSNR improvement falls below threshold
- Configurable iterations per pass (default: 50)

**Configuration:**
```rust
ErrorDrivenAdamConfig {
    target_error: 0.001,           // Target MSE threshold
    initial_gaussians: 64,         // Starting grid
    max_gaussians: 512,            // Upper limit
    split_percentile: 0.05,        // Top 5% error pixels
    iterations_per_pass: 50,       // Adam iters between additions
    min_improvement_db: 0.05,      // Early stopping threshold
    max_passes: 8,                 // Maximum refinement cycles
    min_gaussian_spacing: 0.02,    // ~15px at 768px width
}
```

### 2. `examples/test_error_driven_adam.rs` (NEW - 314 lines)

Test comparing 4 configurations:
1. Uniform+Adam (256 fixed)
2. ErrorDriven (OptimizerV2, original)
3. ErrorDrivenAdam (64→256)
4. ErrorDrivenAdam (64→512, larger budget)

---

## Data Collected

### ErrorDrivenAdam Test Results (kodim03.png, 768×512)

| Configuration | Initial N | Final N | Final PSNR | Time | Notes |
|---------------|-----------|---------|------------|------|-------|
| Uniform+Adam | 256 | 256 | 21.05 dB | 95.5s | Fixed grid baseline |
| ErrorDrivenAdam | 64 | 256 | 22.45 dB | 192.3s | Adaptive placement |
| ErrorDrivenAdam (large) | 64 | 424 | 23.79 dB | 390.7s | Hit stall, not max |

**Observations (not conclusions):**
- ErrorDrivenAdam at 256 Gaussians: 22.45 dB vs Uniform+Adam: 21.05 dB (+1.4 dB)
- Larger budget (512 max) reached 424 Gaussians before improvement stalled
- The improvement curve flattens—diminishing returns with more Gaussians

### ErrorDrivenEncoder (OptimizerV2) Baseline (from earlier test)

| Configuration | Final N | Final PSNR | Time |
|---------------|---------|------------|------|
| ErrorDriven (V2), 500 max | 500 | 19.90 dB | 171.3s |
| ErrorDriven + Adam polish | 300 | 20.44 dB | 100.2s |
| ErrorDriven (64→256) + Adam | 256 | 19.78 dB | 71.1s |

**Observations:**
- OptimizerV2-based ErrorDriven underperformed uniform+Adam
- Post-hoc Adam polish improved results by ~0.5 dB
- Suggests OptimizerV2 optimizer itself is the bottleneck, not the placement strategy

### MS-SSIM Hybrid Loss Alpha Sweep

| Alpha | MSE Weight | MS-SSIM Weight | Final PSNR | MS-SSIM Score |
|-------|------------|----------------|------------|---------------|
| 1.00 | 100% | 0% | 22.33 dB | 0.8384 |
| 0.84 | 84% | 16% | 22.74 dB | 0.8668 |
| 0.50 | 50% | 50% | 22.55 dB | 0.8613 |
| 0.20 | 20% | 80% | 22.24 dB | 0.8387 |

**Observations:**
- α=0.84 (3DGS paper recommendation) shows peak in both metrics
- Pattern suggests inverted U-shape centered around α≈0.84
- Finer α sweep (0.9, 0.8, 0.7) would help map the curve precisely

### L-BFGS Experiments

| Test | Method | Final PSNR | OOB Colors | Notes |
|------|--------|------------|------------|-------|
| Soft barrier (w=0.1) | Colors-only | 21.18 dB | 73% | Step collapse after iter 20 |
| Soft barrier (w=1.0) | Colors-only | 21.19 dB | 65% | Better step stability |
| Block-diagonal | Separate histories | 20.27 dB | N/A | Complete failure |

**Observations:**
- L-BFGS struggles with multi-scale Gaussian parameters
- Colors-only optimization shows some success
- Block-diagonal approach needs reconsideration
- Bounded L-BFGS (L-BFGS-B) not yet tested

---

## Questions For Exploration

These emerged during the session but remain open:

1. **Why does ErrorDrivenAdam plateau at 424/512 Gaussians?**
   - Is min_gaussian_spacing too restrictive?
   - Are all high-error regions already covered?
   - Would adaptive sigma sizing help?

2. **MS-SSIM α sweet spot:**
   - Is 0.84 truly optimal or just sampled near the peak?
   - Does optimal α depend on image content?
   - Should α be adaptive during optimization?

3. **L-BFGS parameter grouping:**
   - What if colors and geometry used completely separate optimizers?
   - Would alternating optimization (freeze one, optimize other) work?

4. **Levenberg-Marquardt feasibility:**
   - Current finite-diff Jacobian is prohibitively slow
   - Analytical Jacobian derivation for Gaussian splatting?
   - Sparse approximations?

---

## Files Modified

- `src/lib.rs` - Added `pub mod error_driven_adam;`

---

## Pending Tasks (from todo list)

| Task | Status |
|------|--------|
| GPU renderer validation | Pending |
| Full Kodak baseline collection | Pending |
| L-M optimizer results | Running (197ad9) |

---

## Technical Notes

### Initialization Matters

Different tests use different initialization methods, affecting baselines:

| Method | Typical Initial PSNR |
|--------|---------------------|
| `encoder.initialize_gaussians()` | ~17 dB |
| `create_jittered_grid()` | ~20 dB |
| `create_uniform_grid()` | ~20 dB |

**When comparing:** Use improvement (Δ dB) rather than absolute PSNR across tests with different initialization.

### PSNR Formula Verification

All tests use equivalent formulas:
- `10.0 * (1.0 / mse).log10()` = `-10 * log10(MSE)` ✓

### TestResult Framework

All experiments save JSON results via `TestResult`:
```
lgi-encoder-v2/test-results/YYYY-MM-DD/
├── experiment_name_HH-MM-SS.json
└── ...
```

---

## Session Documents

| Document | Purpose |
|----------|---------|
| `SESSION_HANDOVER_2025-12-06_MORNING.md` | Previous handover |
| `SESSION_2025-12-06_AFTERNOON.md` | Afternoon progress |
| `EXPERIMENT_FINDINGS_2025-12-05.md` | Cumulative findings |
| `EXPLORATION_STATUS.md` | Optimizer inventory |

---

## Continuation Prompt

Use the following prompt to continue this work in a new session:

---

**PROMPT FOR NEXT SESSION:**

```
I'm continuing optimizer exploration for the LGI Gaussian image encoder. Please read the following files to get context:

1. `/home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs/lgi-encoder-v2/test-results/SESSION_HANDOVER_2025-12-06_EVENING.md` - Latest handover with full context

2. `/home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs/lgi-encoder-v2/test-results/EXPERIMENT_FINDINGS_2025-12-05.md` - Cumulative experimental data

3. `/home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs/lgi-encoder-v2/src/error_driven_adam.rs` - New ErrorDrivenAdamEncoder implementation

4. Check background test status:
   - BashOutput for 197ad9 (L-M optimizer)
   - Look for new JSON results in test-results/2025-12-06/

Current exploratory focus areas:
- Optimizer variants (Adam, L-BFGS, L-M, OptimizerV2)
- Adaptive Gaussian placement strategies (error-driven, densification)
- Loss functions (MSE, MS-SSIM hybrid, perceptual)
- Initialization methods

Spirit: This is data-gathering exploration. We're mapping the landscape of what works, not declaring winners. Collect empirical results, note observations, flag questions for further investigation.

Questions to address:
1. Did L-M finish? What were the results?
2. What new data can we collect?
3. Any background tests complete with interesting findings?
```

---

## Raw Metrics Summary

For quick reference, key numbers from this session:

```
ErrorDrivenAdam (256 Gaussians): 22.45 dB
Uniform+Adam (256 Gaussians):    21.05 dB
ErrorDriven V2 (500 Gaussians):  19.90 dB

MS-SSIM α=0.84: 22.74 dB (best sampled)
MS-SSIM α=1.0:  22.33 dB (pure MSE)

L-BFGS colors-only: 21.19 dB (65% OOB)
```

---

*Document created: 2025-12-06 Evening*
*Context window continuation marker: exploration continues*
