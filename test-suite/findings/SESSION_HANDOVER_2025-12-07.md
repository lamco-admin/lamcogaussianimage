# Session Handover: 2025-12-07 L-M JtJ Strategies & Optimizer Toolkit

## Session Context

This is a research and toolkit development project. We are NOT in decision-making mode about "best" optimizers. The goal is to build a complete toolkit with multiple optimization approaches, document their characteristics, and collect empirical data.

---

## Session Accomplishments (2025-12-07)

### 1. Created `compare_lm_strategies.rs` Test Harness

A comprehensive comparison test that runs all optimizers on the same image with identical initialization:
- CPU Adam
- GPU Adam
- L-M Block-Diagonal JtJ
- L-M Sparse JtJ (3σ overlap)
- L-M Hierarchical JtJ (multi-scale)

**Location**: `lgi-encoder-v2/examples/compare_lm_strategies.rs`

### 2. Fixed API Compatibility Issues

Several fixes were required to get the test compiling:
- `GpuAdamOptimizer::new()` takes only `device` (not learning rates)
- CPU `AdamOptimizer` uses `.optimize()` method, not `.step()`
- `LMConfig` field names: `lambda_up` not `lambda_up_factor`, `lambda_min` not `min_lambda`

### 3. Implemented Additional JtJ Strategies

Two new L-M JtJ computation strategies added to `jtj_strategy.rs`:

| Strategy | Line | Description |
|----------|------|-------------|
| `SparseJtJ` | 815 | Only computes cross-Gaussian J^T J for overlapping pairs (configurable σ threshold) |
| `HierarchicalJtJ` | 1149 | Multi-scale approach using multiple sigma levels (6.0, 3.0, 1.5) |

**Shader files created**:
- `lm_jacobian_sparse.wgsl` - Overlap detection via `dist² < (r_i + r_j)²`
- `lm_jacobian_dense.wgsl` - Full O(G²) computation

### 4. Test Results Collected

**Test**: L-M Strategy Comparison (kodim03, 64 Gaussians)

| # | Optimizer | Init PSNR | Final PSNR | Improvement | Time |
|---|-----------|-----------|------------|-------------|------|
| 0 | CPU Adam | 19.19 dB | 19.66 dB | +0.48 dB | 15.99s |
| 1 | GPU Adam | 19.19 dB | 20.19 dB | +1.01 dB | 7.85s |
| 2 | L-M Block-Diag | 19.19 dB | 19.19 dB | +0.00 dB | 1.85s |
| 3 | L-M Sparse | 19.19 dB | 19.07 dB | -0.12 dB | 20.14s |
| 4 | L-M Hierarchical | 19.19 dB | 19.05 dB | -0.14 dB | 54.48s |

**Observations** (data only, no recommendations):
- L-M Block-Diagonal shows 0% improvement (all steps rejected, λ saturated at 1e7)
- L-M Sparse and Hierarchical show slight negative PSNR change
- This data indicates L-M optimizers may need parameter preconditioning for this problem domain

---

## Current Optimizer Toolkit Inventory

### CPU Optimizers (lgi-encoder-v2)

| Optimizer | File | Status |
|-----------|------|--------|
| Adam (per-param LR) | `adam_optimizer.rs` | Tested |
| OptimizerV2 (gradient descent) | `optimizer_v2.rs` | Tested |
| Hybrid (Adam→L-BFGS) | `hybrid_optimizer.rs` | Tested |
| L-M (finite diff) | `lm_optimizer.rs` | Needs GPU |
| L-BFGS (argmin) | `optimizer_lbfgs.rs` | BROKEN (missing dependency) |

### GPU Optimizers (lgi-gpu)

| Optimizer | File | Status |
|-----------|------|--------|
| GPU Adam | `adam_optimizer.rs` | Tested, -0.06 dB vs CPU |
| GPU L-BFGS | `lbfgs_optimizer.rs` | Tested, +0.11 dB vs CPU, 12× speedup |
| GPU L-M | `lm_optimizer.rs` | Tested with JtJ strategies |

### JtJ Strategies for L-M (lgi-gpu/src/jtj_strategy.rs)

| Strategy | Line | Complexity | Status |
|----------|------|------------|--------|
| `BlockDiagonalJtJ` | 161 | O(G) per pixel | Tested |
| `FullDenseJtJ` | 501 | O(G²) per pixel | Tested (numerically unstable) |
| `SparseJtJ` | 815 | O(G×k) per pixel | Tested |
| `HierarchicalJtJ` | 1149 | Multi-scale | Tested |
| `MatrixFreeJtJ` | 1274 | CG-based | Implemented, untested |

---

## Background Task Status

These background processes were started in previous sessions:

| ID | Command | Status |
|----|---------|--------|
| `2df442` | bench_gpu_vs_cpu_adam | Completed (log processes running) |
| `e5bbf2` | bench_gpu_vs_cpu_lbfgs | Was killed |
| `4ecefd` | 6 overnight tests batch | Completed |

### Log Files Available

```
lgi-encoder-v2/test-results/
├── gpu_lbfgs_benchmark.log     (90 KB)
├── gradient_norm.log           (95 KB)
├── multi_kodak.log             (96 KB)
├── overnight_commercial.log    (125 KB)
├── overnight_densification.log (502 KB)
└── overnight_hybrid_loss.log   (101 KB)
```

---

## Test Results Directory Structure

```
lgi-encoder-v2/test-results/
├── 2025-12-05/   # Earlier experiments
├── 2025-12-06/   # Large batch of tests
├── 2025-12-07/   # Today's results
│   ├── lm_strategy_comparison_12-48-36_partial.json
│   ├── lbfgs_gpu_cpu_consistency_12-19-25.json
│   ├── multi_kodak_*.json       (5 files)
│   ├── overnight_commercial_*.json  (many configs)
│   ├── overnight_densify_*.json     (many configs)
│   └── overnight_hybrid_*.json      (6 files)
├── 2025-12-19/   # Previous session results
├── 2025-12-20/   # Previous session results
├── OPTIMIZER_INVENTORY.md
├── COMPREHENSIVE_FINDINGS_2025-12-06.md
├── DENSIFICATION_RESEARCH_SUMMARY.md
└── SESSION_HANDOVER_2025-12-06_*.md (4 files)
```

---

## Key File Reference for Continuation

### GPU Code

```
lgi-gpu/src/
├── lib.rs                    # Public exports (JtJStrategy, optimizers)
├── adam_optimizer.rs         # GpuAdamOptimizer
├── lbfgs_optimizer.rs        # GpuLbfgsOptimizer
├── lm_optimizer.rs           # GpuLMOptimizer
├── jtj_strategy.rs           # JtJ trait and 5 implementations
└── shaders/
    ├── adam_update.wgsl
    ├── lbfgs_*.wgsl          # (4 files)
    ├── lm_jacobian.wgsl      # Block-diagonal
    ├── lm_jacobian_dense.wgsl
    ├── lm_jacobian_sparse.wgsl
    └── lm_update.wgsl
```

### CPU Code

```
lgi-encoder-v2/src/
├── adam_optimizer.rs
├── optimizer_v2.rs
├── hybrid_optimizer.rs
├── loss_functions.rs
├── adaptive_densification.rs
├── kmeans_init.rs
├── encoder_v3_adaptive.rs
└── test_results.rs           # TestResult framework
```

### Test Examples

```
lgi-encoder-v2/examples/
├── compare_lm_strategies.rs      # NEW - comprehensive optimizer comparison
├── test_jtj_strategies.rs        # JtJ sparsity testing
├── bench_gpu_vs_cpu_adam.rs
├── bench_gpu_vs_cpu_lbfgs.rs
├── test_multi_kodak.rs
├── test_placement_strategies.rs
├── test_overnight_densification.rs
├── test_overnight_hybrid_loss.rs
└── test_overnight_commercial.rs
```

---

## Documentation Files to Read

| File | Content |
|------|---------|
| `OPTIMIZER_INVENTORY.md` | Complete status of all techniques (CPU/GPU) |
| `COMPREHENSIVE_FINDINGS_2025-12-06.md` | Research findings summary |
| `SESSION_HANDOVER_2025-12-06_GPU.md` | GPU development context |
| `CLAUDE.md` (project root) | Build commands, testing requirements |

---

## Continuation Commands

### Check Current State
```bash
cd /home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs

# Verify build
cargo build --release -p lgi-gpu -p lgi-encoder-v2

# List today's results
ls -la lgi-encoder-v2/test-results/2025-12-07/

# View most recent handover
cat lgi-encoder-v2/test-results/SESSION_HANDOVER_2025-12-07.md
```

### Run Optimizer Comparisons
```bash
# L-M strategies comparison
cargo run --release --example compare_lm_strategies -p lgi-encoder-v2

# JtJ sparsity analysis
cargo run --release --example test_jtj_strategies -p lgi-encoder-v2

# GPU vs CPU benchmarks
cargo run --release --example bench_gpu_vs_cpu_adam -p lgi-encoder-v2
cargo run --release --example bench_gpu_vs_cpu_lbfgs -p lgi-encoder-v2
```

---

## Areas for Future Exploration

These are documented observations, not prioritized recommendations:

1. **L-M Parameter Preconditioning**: L-M strategies show λ saturation at 1e7 with step rejection. Literature suggests parameter preconditioning or trust region methods.

2. **Sparse JtJ Threshold Tuning**: Current 3σ threshold may not capture enough interactions. 6σ also tested in `test_jtj_strategies.rs`.

3. **MatrixFreeJtJ**: Implemented but untested. Uses conjugate gradient instead of direct solve.

4. **Loss Functions**: Tested L2, MS-SSIM hybrid. Other loss functions in `loss_functions.rs` remain untested.

5. **Densification**: Overnight tests ran densification experiments. Results in `overnight_densification.log` and 2025-12-07 JSON files.

---

## Git State

```
Branch: main
Commit: 45f773d
Dirty: true (modified files, untracked examples)
```

Significant untracked files:
- `examples/compare_lm_strategies.rs`
- `examples/test_jtj_strategies.rs`
- Various overnight test examples
- `lgi-gpu/src/shaders/lm_jacobian_sparse.wgsl`

---

*Document created: 2025-12-07*
*Purpose: Session handover for toolkit evaluation*
