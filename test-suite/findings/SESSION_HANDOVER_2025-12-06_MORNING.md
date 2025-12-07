# Session Handover - December 6, 2025 (Morning)

## Executive Summary

We ran extensive overnight tests on advanced Adam optimization techniques from 3DGS literature. **Densification still underperforms baseline by 0.5-1.3 dB** despite many parameter variations. Deep research into faithful 3DGS densification implementation is the recommended next step.

---

## Current Best Baseline

| Metric | Value |
|--------|-------|
| Configuration | Jittered Grid + Adam (per-param LR) |
| Gaussians | 256 |
| Iterations | 100-200 |
| PSNR (kodim03) | **21.58-21.68 dB** |
| PSNR (kodim05) | ~17.0-17.3 dB |
| PSNR (kodim08) | ~12.3-12.6 dB |

---

## Overnight Test Results Summary

### 1. Densification V2 (Fixed Opacity + Delayed Start)
**File**: `examples/test_densification_v2.rs`
**Log**: `test-results/overnight-logs/densification_v2.log`

| Config | Init→Final | PSNR | vs Baseline |
|--------|------------|------|-------------|
| baseline_256 | 256→256 | 21.68 dB | — |
| v2_100_delayed | 100→256 | 21.13 dB | **-0.55 dB** |
| v2_128_to_256 | 128→256 | 20.74 dB | -0.94 dB |
| v2_64_conservative | 64→256 | 20.64 dB | -1.04 dB |

**Conclusion**: Delayed start + full opacity helped but didn't solve the gap.

### 2. Closed-Form Color Solver
**File**: `examples/test_closed_form_colors.rs`
**Log**: `test-results/overnight-logs/closed_form_colors.log`

| Config | PSNR | vs Adam |
|--------|------|---------|
| Adam Baseline | 21.91 dB | — |
| solve_every_10 | 14.07 dB | **-7.84 dB** |
| solve_every_25 | 14.11 dB | -7.80 dB |
| solve_every_50 | 15.28 dB | -6.63 dB |

**Conclusion**: FAILED. Linear solver destroys quality. Either bug or wrong approach.

### 3. State Inheritance (BOGausS Technique)
**File**: `examples/test_densification_with_inheritance.rs`
**Log**: `test-results/overnight-logs/densification_inheritance.log`

| Image | no_inheritance | full_inheritance | Δ |
|-------|----------------|------------------|---|
| kodim03 | **22.37 dB** | 21.89 dB | -0.48 |
| kodim05 | **17.96 dB** | 17.70 dB | -0.26 |
| kodim08 | 12.45 dB | **12.93 dB** | **+0.48** |

**Conclusion**: Inheritance HELPS on difficult images (+0.48 dB on kodim08). Image-dependent benefit.

### 4. Advanced Adam Features
**File**: `examples/test_adam_advanced.rs`
**Log**: `test-results/overnight-logs/adam_advanced.log`

| Feature | kodim03 | kodim05 | kodim08 |
|---------|---------|---------|---------|
| baseline_global_time | 22.54 dB | 17.67 dB | 12.59 dB |
| per_gaussian_time | 22.13 dB | 17.65 dB | 12.54 dB |
| confidence_scaling | 22.30 dB | 17.65 dB | 12.55 dB |
| sparse_updates | 22.13 dB | 17.65 dB | 12.54 dB |
| all_features | 22.19 dB | 17.67 dB | 12.55 dB |

**Conclusion**: No significant improvement. Features may need densification context.

---

## The Core Problem: Why Densification Fails

We've tried many parameter variations (opacity, delay, thresholds, intervals, starting points) and ALL hit the same ceiling: **0.5-1.3 dB below baseline**.

### What We're Doing vs What 3DGS Does

| Aspect | Our Approach | 3DGS Paper |
|--------|--------------|------------|
| Trigger | Local error at Gaussian center | **Accumulated position gradients** over N iterations |
| Split Logic | Error-based, fixed offset | **Clone** (under-reconstruction) vs **Split** (over-reconstruction) |
| Pruning | Not implemented | **Prune α < 0.005** AND large screen-space Gaussians |
| Gradient Type | Spatial error | **View-space position gradient magnitude** |
| Opacity Reset | Not implemented | **Periodic reset to small value** |

---

## Recommended Next Step: 3DGS-Faithful Densification

Create `test_3dgs_faithful_densification.rs` implementing:

1. **Gradient accumulation**: Track `∇_x L` per Gaussian over ~100 iterations
2. **Clone vs Split distinction**:
   - Clone: Gaussian too small → duplicate and shift
   - Split: Gaussian covers too much variation → replace with 2 smaller
3. **Pruning**: Remove low-opacity (<0.005) and oversized Gaussians
4. **Periodic opacity reset**: Reset all opacities to small value every N iterations
5. **State inheritance**: Use for difficult images (conditional on image complexity)

This is research-grade implementation, not parameter tuning.

---

## Key Files Reference

### Test Files Created This Session
```
examples/test_adam_advanced.rs          # Per-Gaussian time, confidence, sparse
examples/test_closed_form_colors.rs     # Linear color solver (FAILED)
examples/test_densification_with_inheritance.rs  # State inheritance
examples/test_densification_v2.rs       # Fixed opacity + delayed start
```

### Documentation Created
```
test-results/ADAM_OPTIMIZATION_TECHNIQUES.md   # 3DGS Adam best practices
test-results/SESSION_HANDOVER_2025-12-06_MORNING.md  # This file
```

### Key Existing Files
```
src/adam_optimizer.rs           # Current Adam implementation
src/adaptive_densification.rs   # Current densification (needs replacement)
src/renderer_v2.rs              # Rendering pipeline
src/test_results.rs             # TestResult framework (ALWAYS USE)
```

### Overnight Logs
```
test-results/overnight-logs/
├── adam_advanced.log
├── closed_form_colors.log
├── densification_inheritance.log
├── densification_v2.log
├── densification.log
├── commercial.log
├── hybrid_loss.log
└── init_recalibration.log
```

---

## Testing Requirements (from CLAUDE.md)

**ALWAYS use TestResult framework:**
```rust
use lgi_encoder_v2::test_results::TestResult;

let mut result = TestResult::new("experiment_name", image_path);
result.optimizer = "Description".to_string();
result.initial_psnr = init_psnr;
result.final_psnr = final_psnr;
result.improvement_db = final_psnr - init_psnr;
let _ = result.save("lgi-encoder-v2/test-results");
```

**ALWAYS show iteration progress:**
```rust
if iter % 10 == 0 {
    let direction = if psnr > last_psnr { "↑" } else { "↓" };
    println!("    [{:3}/{:3}] PSNR={:.2}dB {}", iter, max, psnr, direction);
}
```

**Test on multiple images**: kodim03, kodim05, kodim08 minimum.

---

## Build Commands

```bash
# Build specific example
cargo build --release --example test_3dgs_faithful_densification -p lgi-encoder-v2

# Run test
cargo run --release --example test_3dgs_faithful_densification -p lgi-encoder-v2

# Run in background for overnight
nohup cargo run --release --example test_3dgs_faithful_densification -p lgi-encoder-v2 \
  > lgi-encoder-v2/test-results/overnight-logs/3dgs_faithful.log 2>&1 &
```

---

## What NOT to Do

1. **Don't run Python preprocessing** (no SLIC, no external tools)
2. **Don't iterate on parameters** - we've exhausted that approach
3. **Don't skip TestResult framework** - all experiments must be tracked
4. **Don't test on single image** - always use at least 3 images

---

## Priority for Next Session

1. **Implement 3DGS-faithful densification** with:
   - Gradient accumulation (position gradients over N iterations)
   - Clone vs Split distinction
   - Pruning (low opacity + oversized)
   - Periodic opacity reset

2. **Test state inheritance conditionally** - use on difficult images only

3. **Debug closed-form color solver** if time permits (low priority)

---

## Context for AI Assistant

When continuing this work:
- Working directory: `/home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs`
- Current baseline: 21.58-21.68 dB on kodim03 with 256 Gaussians
- The gap to close: 0.5-1.3 dB between densification and baseline
- Key insight: Inheritance helps on difficult images (+0.48 dB on kodim08)
- Read the 3DGS paper densification section carefully before implementing
