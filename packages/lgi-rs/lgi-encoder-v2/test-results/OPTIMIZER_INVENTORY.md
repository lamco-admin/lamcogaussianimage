# Optimizer Techniques Inventory
**Last Updated**: 2025-12-07
**Purpose**: Complete status of all optimizer techniques - tested, coded, and planned

---

## Executive Summary

| Category | Tested | Working | Rejected | Untested |
|----------|--------|---------|----------|----------|
| **Optimizers** | 4 | 1 (Adam) | 3 | 2 |
| **Loss Functions** | 3 | 2 | 1 | 3 (1 in progress) |
| **Initialization** | 6 | 2 | 3 | 2 |
| **Densification** | 1 | 0 | 0 | 0 |
| **Encoders** | 2 | 1 | 1 | 1 |

**Key Finding**: Many techniques have been tested and REJECTED. Adam optimizer with jittered grid and hybrid loss remains the best configuration.

---

## GPU OPTIMIZER LANDSCAPE (NEW: 2025-12-07)

The `lgi-gpu` crate provides GPU-accelerated implementations of the main optimizers using wgpu compute shaders. This section documents the current status of each GPU optimizer.

### GPU Implementation Summary

| Optimizer | GPU Status | CPU Parity | Speed | Recommendation |
|-----------|------------|------------|-------|----------------|
| **GPU Adam** | ✅ WORKING | -0.06 dB | ~50× faster | USE FOR PRODUCTION |
| **GPU L-BFGS** | ✅ CONSISTENT | +0.11 dB | ~12× faster | VERIFIED (10-param test) |
| **GPU L-M** | ✅ WORKING | N/A (strategies differ) | ~10× faster | EXPERIMENTAL |

### GPU Adam (`lgi-gpu/src/adam_optimizer.rs`)

**Status**: PRODUCTION READY

**Implementation**:
- Single compute shader (`shaders/adam_update.wgsl`)
- 64 threads per workgroup, 10 parameters per Gaussian
- Struct-aligned `GpuGaussian` for 16-byte boundaries
- All computations in f32 (no fixed-point)

**Parity Testing** (2025-12-07):
- GPU: 20.93 dB → within -0.06 dB of CPU (20.99 dB)
- Difference likely due to floating-point ordering

**Key Files**:
```
lgi-gpu/src/adam_optimizer.rs   # GpuAdamOptimizer struct
lgi-gpu/src/shaders/adam_update.wgsl  # Compute shader
```

### GPU L-BFGS (`lgi-gpu/src/lbfgs_optimizer.rs`)

**Status**: ✅ VERIFIED CONSISTENT WITH CPU (0.11 dB parity)

**Bugs Fixed** (2025-12-07):
1. ✅ Line search target: Was targeting old render, now targets new
2. ✅ Armijo condition: Was using stale `initial_loss`, now uses current
3. ✅ History initialization: Was skipping first (s,y) pair because `prev_params=None` on iter 1
4. ✅ Non-descent handling: Was calling `reset()`, now returns tiny step (1e-8) like CPU
5. ✅ CPU safeguards: Added is_finite() guards and γ safeguard to CPU L-BFGS for consistency

**Consistency Fix Details**:
- GPU has `if rho_i.is_finite()` guards in two-loop recursion to skip degenerate history
- GPU has `if yy > 1e-10 { sy / yy } else { 1.0 }` safeguard for γ calculation
- These are defensive programming (edge case handling), not algorithmic differences
- Added same safeguards to CPU (`lgi-core/src/lbfgs.rs`) for consistency

**Consistency Test Results** (2025-12-07):
| Metric | CPU L-BFGS | GPU L-BFGS | Difference |
|--------|------------|------------|------------|
| Initial PSNR | 10.04 dB | 10.04 dB | 0.00 dB |
| Final PSNR | 13.21 dB | 13.32 dB | **+0.11 dB** |
| Improvement | +3.17 dB | +3.28 dB | +0.11 dB |
| Time | 88.16s | 7.36s | **12× speedup** |

**Comparison with Other Optimizers**:
| Optimizer | Consistency Type | GPU/CPU Parity |
|-----------|-----------------|----------------|
| Adam | Pure formula (no edge cases) | -0.06 dB ✓ |
| L-BFGS | Defensive guards (edge cases) | +0.11 dB ✓ |
| L-M | Algorithmic choice (JtJ strategy) | N/A (intentionally different) |

**Key Files**:
```
lgi-gpu/src/lbfgs_optimizer.rs   # GpuLBFGSOptimizer
lgi-core/src/lbfgs.rs            # CPU LBFGS (updated with safeguards)
lgi-gpu/src/shaders/lbfgs_*.wgsl # 4 compute shaders
```

### GPU Levenberg-Marquardt (`lgi-gpu/src/lm_optimizer.rs`)

**Status**: WORKING WITH PLUGGABLE STRATEGIES

**Architecture** (2025-12-07):
- Strategy pattern via `JtJStrategy` trait
- Pluggable J^T J computation methods
- Uses fixed-point atomics (FIXED_SCALE = 1,000,000.0)

**J^T J Strategies** (`lgi-gpu/src/jtj_strategy.rs`):

| Strategy | Complexity | Captures Interactions | Status |
|----------|------------|----------------------|--------|
| `BlockDiagonalJtJ` | O(G) per pixel | ❌ No | ✅ DEFAULT, BEST |
| `FullDenseJtJ` | O(G²) per pixel | ✅ Yes (67%) | ⚠️ Numerically unstable |
| `SparseJtJ` | O(G×k) per pixel | Partial | Not implemented |
| `HierarchicalJtJ` | Multi-scale | Partial | Not implemented |

**Convergence Test Results** (2025-12-07, kodim03, 16 Gaussians, 20 iterations):

| Strategy | Improvement | Time | λ Final | Notes |
|----------|-------------|------|---------|-------|
| **Block-Diagonal** | +0.05 dB | 0.39s | 1e5 | Steps accepted |
| **Full-Dense** | +0.00 dB | 2.82s | 1e10 | All steps rejected! |

**Matrix Norm Analysis** (explains why full-dense fails):

| Metric | Block-Diagonal | Full-Dense | Ratio |
|--------|----------------|------------|-------|
| ||J^T J||_F | 15,850 | 27,766 | 1.75× |
| max(J^T J) | 2,121 | 2,125 | 1.00× |
| ||J^T r|| | 7,101 | 7,100 | 1.00× |
| Energy (diagonal) | 100% | 32.6% | - |
| Energy (off-diag) | 0% | 67.4% | - |

**Key Insight**: Despite 67% of Hessian energy being in off-diagonal blocks, the full-dense strategy performs **worse** because:
1. Off-diagonal blocks introduce near-linear dependencies → matrix nearly singular
2. Condition number explodes → requires λ→1e10 for regularization
3. With λ=1e10, step size ≈ 1/λ ≈ 1e-10 → no meaningful progress
4. Gradient J^T r is identical (7100 vs 7101), so the problem is matrix conditioning, not gradient

**Fixes Applied** (2025-12-07):
- ✅ Early exit checks: Skip Gaussian pairs with negligible contribution
- ✅ Higher thresholds: `MIN_CONTRIBUTION=1e-4`, `MIN_CROSS_CONTRIBUTION=1e-6`
- ✅ Matrix symmetrization: `A = (A + A^T) / 2` before solving

**Result**: Full-dense is now 7× slower (down from 80×), but still numerically unstable.

**Recommendation**: Use `BlockDiagonalJtJ` (default). Full-dense captures more information but worse conditioning negates the benefit.

**Usage**:
```rust
use lgi_gpu::{GpuLMOptimizer, BlockDiagonalJtJ, FullDenseJtJ};

// Default (block-diagonal)
let optimizer = GpuLMOptimizer::new(&device)?;

// Explicit strategy
let strategy = FullDenseJtJ::new(&device)?;
let optimizer = GpuLMOptimizer::with_strategy(&device, Box::new(strategy))?;

// Change strategy at runtime
optimizer.set_strategy(Box::new(BlockDiagonalJtJ::new(&device)?));
```

**Fixed-Point Conversion** (critical for WGSL atomics):
```rust
// Encode: f32 → u32 (biased to avoid negative atomics)
let raw_bits = (value * FIXED_SCALE) as i32;
let encoded = (raw_bits as u32).wrapping_add(0x80000000u32);

// Decode: u32 → f32
let raw_bits = stored.wrapping_sub(0x80000000u32) as i32;
let value = (raw_bits as f32) / FIXED_SCALE;
```

### GPU Optimizer Shaders Summary

```
lgi-gpu/src/shaders/
├── adam_update.wgsl        # Adam parameter update
├── lbfgs_two_loop.wgsl     # L-BFGS two-loop recursion
├── lbfgs_search.wgsl       # L-BFGS line search direction
├── lbfgs_line_search.wgsl  # Armijo backtracking
├── lbfgs_update.wgsl       # History update
├── lm_jacobian.wgsl        # Block-diagonal J^T J
├── lm_full_dense_jtj.wgsl  # Full-dense J^T J (O(G²))
├── lm_solve.wgsl           # Cholesky solve
├── lm_update.wgsl          # Parameter update
└── lm_render.wgsl          # Forward render
```

---

## PART 1: CPU OPTIMIZERS

### TESTED AND WORKING

| Optimizer | File | Improvement | Status | Notes |
|-----------|------|-------------|--------|-------|
| **Adam (per-param LR)** | `adam_optimizer.rs` | **+2.25 dB avg** | PRODUCTION | Best general-purpose optimizer |
| **OptimizerV2** | `optimizer_v2.rs` | +0.16 dB | SLOW | Works but gradient magnitudes too small |
| **Hybrid (Adam→L-BFGS)** | `hybrid_optimizer.rs` | -0.22 dB | REJECTED | L-BFGS refinement hurts quality |
| **L-M (finite diff)** | `lm_optimizer.rs` | Untested | NEEDS GPU | Jacobian computation benefits from GPU parallelism |

### CODED BUT UNTESTED

| Optimizer | File | Description | Blocker |
|-----------|------|-------------|---------|
| **optimizer_lbfgs.rs** | `optimizer_lbfgs.rs` | argmin-based L-BFGS | Missing argmin dependency - BROKEN |
| **optimizer_v3_perceptual** | `optimizer_v3_perceptual.rs` | Edge-weighted optimization | Needs testing |

### CONFIGURATION: Adam Production Settings

```rust
// Best-practice learning rates (validated +4.79 dB improvement)
LearningRates {
    position: 0.0002,    // 100× smaller than color
    color: 0.02,
    scale: 0.005,
    opacity: 0.05,
    position_decay: 0.995,  // Exponential decay
}
```

---

## PART 2: LOSS FUNCTIONS

### TESTED AND WORKING

| Loss | File | Improvement | Status |
|------|------|-------------|--------|
| **L2 (MSE)** | Built-in | Baseline | PRODUCTION |
| **Hybrid L2+MS-SSIM (80/20)** | `adam_optimizer.rs` | **+0.51 dB** | PRODUCTION RECOMMENDED |
| **MS-SSIM only** | `optimizer_v2.rs` | -0.02 dB | REJECTED (gradient issues) |

### CODED BUT UNTESTED

| Loss | File | Description |
|------|------|-------------|
| **L1 (MAE)** | `loss_functions.rs:13-19` | Mean Absolute Error |
| **SSIM** | `loss_functions.rs:79-125` | Structural Similarity (8×8 window) |
| **D-SSIM** | `loss_functions.rs:129-131` | 1 - SSIM |
| **Combined (λ1×L1 + λ2×L2 + λ3×D-SSIM)** | `loss_functions.rs:135-147` | Configurable weighted loss |
| **3DGS Loss (L1 + D-SSIM)** | `loss_functions.rs:150-154` | Standard 3DGS loss function |
| **Edge-weighted** | `optimizer_v3_perceptual.rs` | Higher weight in edge regions |

### TEST PRIORITY

1. **3DGS Loss (L1 + D-SSIM)** - Most promising, used in all 3DGS papers
2. **Combined loss with tuned lambdas** - Fine control over trade-offs
3. **L1 loss alone** - Often better for sharpness

---

## PART 3: INITIALIZATION STRATEGIES

### TESTED AND WORKING

| Strategy | Improvement | Status |
|----------|-------------|--------|
| **Jittered Grid** | **+0.44 dB vs uniform** | BEST |
| **Halton Sequence** | +0.38 dB | Second best |
| **Uniform Grid** | Baseline | Works |
| **Golden Spiral** | -0.71 dB | REJECTED (fails on rectangles) |
| **Edge-Aware** | Poor init PSNR | Needs tuning |

### TESTED - NEEDS FURTHER EXPLORATION

| Strategy | Init PSNR | Final PSNR | Notes |
|----------|-----------|------------|-------|
| **K-Means (default c=1,p=0.5,g=0.3)** | 14.22 dB | 18.62 dB | Below grid in this config |
| **K-Means (color-heavy c=2,p=0.3,g=0.1)** | 12.42 dB | 15.35 dB | Lowest in this test |
| **K-Means (position-heavy c=0.5,p=2,g=0.3)** | 15.50 dB | 20.14 dB | Closest to grid methods |

**Observations**: Tested configurations showed lower final PSNR than grid methods. Position-heavy weighting performed best among K-Means variants.

**Variations to explore**: Higher position weights, more K-Means iterations, hybrid grid+K-Means approaches.

### CODED BUT UNTESTED

| Strategy | File | Description |
|----------|------|-------------|
| **SLIC** | `slic_init.rs` | Superpixel-based (requires Python preprocessing) |
| **Gradient Peak** | `gradient_peak_init.rs` | Place Gaussians at gradient maxima |
| **GDGS** | `gdgs_init.rs` | Laplacian peak detection |
| **Better Color Init** | `better_color_init.rs` | Improved initial color assignment |

### TEST PRIORITY

1. **Gradient Peak** - Should work well with our structure tensor
2. **SLIC** - Industry standard but needs Python

---

## PART 4: ADAPTIVE DENSIFICATION

### TESTED - ONE CONFIGURATION

| Method | N_init | N_final | PSNR | Notes |
|--------|--------|---------|------|-------|
| Baseline (144 Gaussians) | 144 | 144 | 20.32 dB | No densification |
| **With Densification** | 144 | 288 | 20.86 dB | +0.54 dB improvement |
| Fair Baseline (256 Gaussians) | 256 | 256 | 20.99 dB | For comparison |

**Observations**: In this configuration (threshold=0.0001, interval=20), densification improved quality but the final count (288) with slightly lower PSNR than starting with 256.

**Configuration tested:**
```rust
densify_grad_threshold: 0.0001,  // Lower threshold tested
densify_interval: 20,            // Every 20 iterations
prune_interval: 50,              // Every 50 iterations
```

**Variations to explore:**
- Different gradient thresholds
- Starting from fewer Gaussians (64, 100)
- More iterations after densification
- Different split/prune intervals

---

## PART 5: ENCODERS

### TESTED AND WORKING

| Encoder | Function | Description |
|---------|----------|-------------|
| **Error-Driven Adam** | `encode_error_driven_adam()` | Standard production encoder |

### TESTED - NEEDS FURTHER EXPLORATION

| Encoder | Adam Baseline | V3 Result | V3 Gaussians | Notes |
|---------|---------------|-----------|--------------|-------|
| **V3 Adaptive (kodim03)** | 20.99 dB | 9.11 dB | 246 | Large gap in this config |
| **V3 Adaptive (kodim08)** | 12.79 dB | 12.83 dB | 1790 | Similar PSNR, more Gaussians |

**Configuration:**
- Uses `OptimizerV2::default()` internally
- Content detection: Both detected as "HighFrequency"
- Initial scale: 0.01

**Variations to explore:**
- Swap OptimizerV2 for Adam optimizer
- Different initial scale values
- Different gamma configurations
- More iterations
- Test each component independently (content detection, LOD, triggers)

### CODED BUT UNTESTED

| Encoder | File | Description |
|---------|------|-------------|
| **Commercial** | `commercial_encoder.rs` | Full production encoder |
| **Error-Driven** | `error_driven.rs` | Alternative error-based placement |

---

## PART 6: SPECIAL TECHNIQUES

### TESTED

| Technique | Result | Status |
|-----------|--------|--------|
| **Gradient Normalization** | +1.17 dB for V2 | WORKS - bridges 78% of gap to Adam |
| **Isotropic Edges** | +1.87 dB | VALIDATED via quantum research |

### NOT TESTED

| Technique | File | Description |
|-----------|------|-------------|
| **LOD System** | `lgi-core/lod_system.rs` | Multi-resolution Gaussian bands |
| **Analytical Triggers** | `lgi-core/analytical_triggers.rs` | Quality-based refinement triggers |
| **Truncation Radius** | Not implemented | 100× speedup claimed in literature |

---

## PART 7: TEST INFRASTRUCTURE

### Existing Test Framework

**Location**: `lgi-encoder-v2/test-results/`

**TestResult API**:
```rust
let mut result = TestResult::new("experiment_name", "/path/to/image.png");
result.optimizer = "Adam".to_string();
result.initial_psnr = 20.0;
result.final_psnr = 22.0;
result.save(RESULTS_DIR)?;
```

### Existing Test Examples

| Example | Purpose |
|---------|---------|
| `verify_all_optimizers.rs` | Quick sanity check for all optimizers |
| `verify_hybrid_optimizer.rs` | Adam vs Hybrid comparison |
| `test_multi_kodak.rs` | Multi-image Adam benchmark |
| `test_placement_strategies.rs` | Initialization comparison |
| `test_perceptual_loss.rs` | L2 vs Hybrid loss |

---

## PART 8: RECOMMENDED TEST SEQUENCE

### Priority 1: Adaptive Densification (HIGH IMPACT)

**Why**: Core 3DGS technique, could significantly improve quality
**Expected**: +1-3 dB (if working correctly)

```bash
# Create test_adaptive_densification.rs
cargo run --release --example test_adaptive_densification -p lgi-encoder-v2
```

### Priority 2: 3DGS Combined Loss (L1 + D-SSIM)

**Why**: Standard in all 3DGS papers, different from our L2/MS-SSIM
**Expected**: Potentially sharper edges

### Priority 3: K-Means Initialization

**Why**: Well-proven technique, uses existing structure tensor
**Expected**: Better initial placement = faster convergence

### Priority 4: Content-Adaptive Encoder (V3)

**Why**: Combines LOD + triggers + content detection
**Expected**: Better for diverse images

---

## APPENDIX: File Locations

### GPU Source Files (lgi-gpu)

```
lgi-gpu/src/
├── lib.rs                  # Public exports
├── adam_optimizer.rs       # GPU Adam - PRODUCTION READY
├── lbfgs_optimizer.rs      # GPU L-BFGS - Needs investigation
├── lm_optimizer.rs         # GPU Levenberg-Marquardt
├── jtj_strategy.rs         # JtJ strategy trait & implementations
├── shaders/
│   ├── adam_update.wgsl        # Adam parameter update
│   ├── lbfgs_two_loop.wgsl     # L-BFGS two-loop recursion
│   ├── lbfgs_search.wgsl       # Line search direction
│   ├── lbfgs_line_search.wgsl  # Armijo backtracking
│   ├── lbfgs_update.wgsl       # History update
│   ├── lm_jacobian.wgsl        # Block-diagonal J^T J
│   ├── lm_full_dense_jtj.wgsl  # Full-dense J^T J (O(G²))
│   ├── lm_solve.wgsl           # Cholesky solve
│   ├── lm_update.wgsl          # Parameter update
│   └── lm_render.wgsl          # Forward render
```

### CPU Source Files (lgi-encoder-v2)

```
lgi-encoder-v2/src/
├── adam_optimizer.rs       # PRIMARY - Production optimizer
├── optimizer_v2.rs         # Secondary - Gradient descent
├── hybrid_optimizer.rs     # Adam→L-BFGS hybrid
├── lm_optimizer.rs         # Levenberg-Marquardt (not practical)
├── optimizer_lbfgs.rs      # BROKEN - missing argmin
├── optimizer_v3_perceptual.rs # Edge-weighted
├── loss_functions.rs       # L1, SSIM, D-SSIM, combined
├── adaptive_densification.rs # Split/clone/prune
├── kmeans_init.rs          # K-means initialization
├── slic_init.rs            # SLIC superpixels
├── gradient_peak_init.rs   # Gradient-based init
├── gdgs_init.rs            # Laplacian peaks
├── encoder_v3_adaptive.rs  # Content-adaptive encoder
├── commercial_encoder.rs   # Production encoder
└── error_driven.rs         # Error-driven placement
```

### Test Results

```
lgi-encoder-v2/test-results/
├── EXPERIMENT_FINDINGS_2025-12-05.md   # Comprehensive results
├── OPTIMIZER_INVENTORY.md              # This file
├── 2025-12-05/                         # Daily results (JSON)
└── rendered-images/                    # Visual outputs
```

---

## Summary: Testing Priorities

| Rank | Technique | Potential | Effort | Status |
|------|-----------|-----------|--------|--------|
| 1 | 3DGS Loss (L1+D-SSIM) | Medium | Low | **IN PROGRESS** |
| 2 | Gradient Peak Init | Medium | Low | Untested |
| 3 | V3Adaptive with Adam | Medium | Medium | Variation to test |
| 4 | K-Means (higher position weight) | Medium | Low | Variation to test |
| 5 | Densification (from 64 Gaussians) | Medium | Low | Variation to test |

## Currently Best-Performing Configurations

| Technique | Result | Configuration |
|-----------|--------|---------------|
| **Adam optimizer** | +2.25 dB avg | Per-param LRs, 100 iterations |
| **Jittered grid init** | +0.44 dB vs uniform | Small random offsets |
| **Hybrid L2+MS-SSIM (80/20)** | +0.51 dB vs L2 | 80% L2, 20% MS-SSIM |
| **Gradient normalization** | +1.17 dB for V2 | Unit-length gradients + small LR |

## Tested Configurations Needing Exploration

| Technique | Observation | Variations to try |
|-----------|-------------|-------------------|
| K-Means init | Position-heavy closest to grid | Higher position weight, hybrid approaches |
| Adaptive densification | Small gap vs fair baseline | Start from fewer Gaussians, different thresholds |
| EncoderV3Adaptive | Gap when using OptimizerV2 | Swap to Adam optimizer |
| Hybrid Adam→L-BFGS | L-BFGS increased loss | Different L-BFGS configs, line search params |

---

*This inventory tracks experimental observations. Results are configuration-specific.*
*Last updated: 2025-12-06*
