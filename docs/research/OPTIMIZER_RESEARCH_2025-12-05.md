# Optimizer Research Summary - December 5, 2025

**Purpose**: Document validated optimization approaches for the LGI Gaussian image codec.

---

## EXPERIMENTAL RESULTS (This Session)

### Experiment 1: Per-Parameter Learning Rates vs Single LR

**Test**: kodim03.png (768×512), 1024 Gaussians (32×32 grid), 200 iterations

| Configuration | Final PSNR | Loss | Trend |
|---------------|------------|------|-------|
| Single LR (0.01) | 19.77 dB | 0.0058 | **Chaotic oscillation** |
| Per-Param LRs | **24.56 dB** | 0.0035 | **Monotonic decrease** |
| Initial | 21.55 dB | - | - |

**Per-Param LRs used:**
- Position: 0.0002 → 0.00002 (exponential decay)
- Color: 0.02
- Scale: 0.005
- Opacity: 0.05

**RESULT: Per-param LRs WIN by +4.79 dB**

**Key observation**: Single LR caused loss to oscillate wildly (divergence), while per-param showed stable monotonic convergence. This validates the 3DGS paper's approach.

### Experiment 2: Adam→L-BFGS Hybrid (PENDING)

Test in progress...

---

## Key Findings

### 1. We Have a Working L-BFGS Already!

**Location**: `packages/lgi-rs/lgi-core/src/lbfgs.rs` (291 lines)
- Status: COMPLETE, TESTED, PRODUCTION-READY
- Tests pass (verified)
- Example runs: `cargo run --example lbfgs_test`
- Uses analytical gradients (user-provided closures)
- No external dependencies

**NOT to use**: `lgi-encoder-v2/src/optimizer_lbfgs.rs` (broken, missing argmin dependency, incomplete gradients)

### 2. Critical Missing Feature: Per-Parameter Learning Rates

**Current Adam implementation**: Single learning rate (0.01) for all parameters

**3DGS best practice** (from original Kerbl paper):
| Parameter | Initial LR | Final LR | Decay |
|-----------|------------|----------|-------|
| Position | 0.00016 | 0.0000016 | Exponential over 30K iter |
| Color/SH | 0.0025 | - | Constant |
| Opacity | 0.05 | - | Constant |
| Scale | 0.005 | - | Constant |
| Rotation | 0.001 | - | Constant |

**Impact**: Position should be ~100× smaller than color learning rate. This is LOW-HANGING FRUIT for improvement.

### 3. Validated Optimization Strategies (from published papers)

#### Strategy A: Adam → L-BFGS Hybrid (3DGS-LM, ICCV 2025)
- **Phase 1**: Adam + densification (6K-8K iterations) for initialization
- **Phase 2**: L-BFGS for refinement
- **Speedup**: 20-30% faster than Adam-only
- **Critical**: Starting with L-BFGS from scratch provides NO benefit; hybrid is essential

#### Strategy B: Levenberg-Marquardt (3DGS-LM)
- **Why**: Our problem IS nonlinear least-squares: `||rendered - target||²`
- **Speedup**: 20-30% faster than Adam
- **Memory**: High (Jacobian scales with Gaussians × pixels)
- **Best for**: Refinement phase after rough initialization

#### Strategy C: Second-Order with Local Optimization (3DGS², Jan 2025)
- **Insight**: Each Gaussian's attributes contribute independently to loss
- **Method**: Per-Gaussian local optimization (not global Hessian)
- **Speedup**: 10× faster than standard Adam
- **Claim**: One Newton iteration > 100 gradient descent iterations

#### Strategy D: SGLD + Quasi-Newton (Opt3DGS, Nov 2025) - STATE OF THE ART
- **Phase 1**: Stochastic Gradient Langevin Dynamics (exploration, escape local optima)
- **Phase 2**: Local Quasi-Newton (L-BFGS for positions only)
- **Result**: State-of-the-art quality by optimization alone (no representation changes)
- **Improvement**: +0.5 dB PSNR over baseline

### 4. What DOESN'T Work

- **Plain SGD**: Consistently underperforms, "had to be truncated to fit in same plot"
- **Full second-order without sparsity**: Too expensive
- **L-BFGS from scratch**: No benefit without Adam warm-start

### 5. Levenberg-Marquardt Implementation Path

**Rust crate**: `levenberg-marquardt` (crates.io, 304K downloads)
- Port of MINPACK (classic Fortran)
- Uses nalgebra for linear algebra
- Floating-point identical to MINPACK

**Key requirement**: Efficient Jacobian computation
- Each Gaussian only affects nearby pixels (sparsity!)
- Don't build full J matrix - compute J^T J and J^T r products
- Cache intermediate gradients during rendering

**Damping strategy**: Delayed Gratification
```rust
if loss_improved {
    λ = λ / 3.0;  // More Gauss-Newton (fast)
} else {
    λ = λ * 2.0;  // More gradient descent (safe)
}
```

---

## Implementation Status

### COMPLETED

1. **Per-parameter learning rates for Adam** ✅
   - File: `adam_optimizer.rs`
   - Added `LearningRates` struct with position, color, scale, opacity
   - Exponential decay for position LR
   - **Validated: +4.79 dB improvement on kodim03**

2. **Restore best parameters** ✅
   - Adam now tracks best_loss and best_gaussians
   - Restores best state at end (imperative, not optional)
   - Prevents returning degraded state from oscillation

3. **Adam → L-BFGS hybrid** ✅ (code complete, testing)
   - File: `hybrid_optimizer.rs`
   - Uses working `lgi-core/src/lbfgs.rs`
   - Default: 100 Adam + 50 L-BFGS iterations
   - Currently uses finite differences (expensive but correct)

4. **Levenberg-Marquardt** ✅ (code complete, testing)
   - File: `lm_optimizer.rs`
   - Uses `levenberg-marquardt` crate (v0.14, MINPACK port)
   - Implements `LeastSquaresProblem` trait for Gaussian optimization
   - Residuals: per-pixel RGB differences (n_pixels × 3 residuals)
   - Currently uses finite differences for Jacobian (very expensive)
   - Best used after Adam warm-start (NOT from scratch)

### Research Track (Future)

5. **Per-Gaussian local optimization**
   - Inspired by 3DGS²
   - Each Gaussian optimized independently
   - Could enable massive parallelization

---

## Code References

- Working L-BFGS: `lgi-core/src/lbfgs.rs`
- Adam optimizer: `lgi-encoder-v2/src/adam_optimizer.rs`
- Hybrid optimizer: `lgi-encoder-v2/src/hybrid_optimizer.rs`
- L-M optimizer: `lgi-encoder-v2/src/lm_optimizer.rs`
- Broken argmin version: `lgi-encoder-v2/src/optimizer_lbfgs.rs` (DO NOT USE)

---

## Sources

- 3D Gaussian Splatting (Kerbl et al., 2023)
- 3DGS-LM: Levenberg-Marquardt Optimization (ICCV 2025)
- 3DGS²: Near Second-order Converging (Jan 2025)
- Opt3DGS: Quasi-Newton with Curvature-Aware Exploitation (Nov 2025)
- Second-order Optimization with Importance Sampling (April 2025)
- Gavin's LM Tutorial (Duke University)
