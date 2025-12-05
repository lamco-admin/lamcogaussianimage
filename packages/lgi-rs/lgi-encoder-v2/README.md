# LGI Encoder V2 - Research & Development

## Overview

This module contains the V2 encoder for Learned Gaussian Image compression, along with comprehensive testing infrastructure for optimizer research.

## Testing Infrastructure Guidelines

**IMPORTANT FOR ALL CONTRIBUTORS (including Claude):**

### 1. All Experiments MUST Use TestResult Framework

Every experiment must save results using the `TestResult` struct:

```rust
use lgi_encoder_v2::test_results::{TestResult, save_rendered_image};

// After running experiment:
let mut result = TestResult::new("experiment_name", image_path);
result.optimizer = "Optimizer description".to_string();
result.image_dimensions = (width, height);
result.n_gaussians = n_gaussians;
result.iterations = iterations;
result.initial_psnr = init_psnr;
result.final_psnr = final_psnr;
result.improvement_db = final_psnr - init_psnr;
result.final_loss = final_loss;
result.elapsed_seconds = elapsed.as_secs_f32();
result.extra_metrics = Some(json!({
    "custom_field": value,
}));
let _ = result.save("lgi-encoder-v2/test-results");
```

Results are saved as timestamped JSON files in `test-results/YYYY-MM-DD/`.

### 2. All Tests MUST Show Verifiable Progress

Users should NEVER have to guess if an experiment is working. Every test must:

```rust
// Show progress every N iterations with clear indicators
if iter % 10 == 0 {
    let direction = if psnr > last_psnr { "↑" } else if psnr < last_psnr { "↓" } else { "→" };
    let delta = psnr - last_psnr;
    println!("    [{:3}/{:3}] loss={:.6} PSNR={:.2}dB {} ({:+.3}dB)",
             iter, max_iterations, loss, psnr, direction, delta);
    last_psnr = psnr;
}

// Clear early stop messages
if patience >= threshold {
    println!("    [EARLY STOP] No improvement for {} iterations at iter {}", threshold, iter);
    break;
}
```

### 3. Test File Structure

All experiment files should follow this pattern:

```rust
//! Test description and hypothesis
//!
//! Run: cargo run --release --example test_name -p lgi-encoder-v2

use lgi_encoder_v2::test_results::TestResult;
// ... other imports

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    // 1. Banner describing the experiment
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   EXPERIMENT NAME                                            ║");
    println!("║   Hypothesis being tested                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // 2. Configuration output
    println!("Config: param1={}, param2={}, ...\n", p1, p2);

    // 3. Per-iteration progress with direction indicators

    // 4. Summary results

    // 5. Save using TestResult framework

    // 6. Hypothesis conclusion (CONFIRMED/REJECTED/PARTIAL)
}
```

---

## Research Priorities

### Priority 1: Initial Gaussian Placement Strategies

Mathematical variants to explore:

1. **Grid-based placement**
   - Uniform grid (current default)
   - Jittered grid (random offset within cell)
   - Golden ratio spiral
   - Blue noise distribution

2. **Content-aware placement**
   - Edge-detector guided (place more Gaussians near edges)
   - Gradient magnitude weighted
   - Superpixel centroid placement
   - K-means clustering on colors

3. **Scale-aware initialization**
   - Structure tensor eigenvalue-based scale
   - Local variance-based scale
   - Multi-scale initialization (hierarchical)

4. **Mathematical distributions**
   - Poisson disk sampling
   - Halton sequence
   - Sobol sequence
   - Fibonacci lattice

### Priority 2: Perceptual Loss Functions

Loss functions to implement and test:

1. **MS-SSIM (Multi-Scale Structural Similarity)**
   - Already implemented but needs gradient fixes
   - Weight tuning for Gaussian optimization

2. **LPIPS (Learned Perceptual Image Patch Similarity)**
   - VGG-based perceptual distance
   - Requires pre-trained weights

3. **Hybrid Loss Functions**
   - α·L2 + β·MS-SSIM
   - Frequency-domain losses (DCT/FFT)
   - Edge-preserving losses (total variation)

4. **Adaptive Loss Weighting**
   - Region-based loss scaling
   - Importance map generation
   - Attention-weighted loss

### Priority 3: Optimizer Improvements

Based on current findings:

| Method | Improvement | Status |
|--------|-------------|--------|
| Adam (per-param LR) | +2.25 dB avg | Baseline |
| V2 Default | +0.16 dB | Fixed gradient signs |
| V2 + Gradient Norm | +0.77 dB avg | 34% of gap closed |
| Hybrid (Adam→L-BFGS) | TBD | Needs investigation |

Key findings:
- Gradient normalization helps but doesn't fully close the gap
- Per-parameter learning rate tuning is critical
- Adaptive LR methods (Adam's momentum/RMSprop) provide significant benefits

---

## Running Experiments

### Quick Start

```bash
cd packages/lgi-rs

# Run specific experiment
cargo run --release --example test_v2_gradient_norm -p lgi-encoder-v2

# Run multi-image test
cargo run --release --example test_multi_kodak -p lgi-encoder-v2
```

### Available Experiments

| Example | Description |
|---------|-------------|
| `comprehensive_optimizer_comparison` | Compare all optimizer variants |
| `test_v2_gradient_norm` | Gradient normalization experiment |
| `test_v2_with_adam_lrs` | V2 with Adam's learning rates |
| `test_v2_with_adam_moments` | V2 with Adam's momentum |
| `test_v2_with_adam_distance` | V2 with Adam's distance formula |
| `test_v2_scaled_lr` | V2 with scaled learning rates |
| `test_multi_kodak` | Adam optimizer on multiple images |
| `test_multi_kodak_gradient_norm` | Gradient norm on multiple images |
| `verify_optimizer_v2` | Verify V2 convergence |
| `verify_v2_gradient_signs` | Verify gradient sign correctness |

---

## Current Status

See `test-results/EXPERIMENT_FINDINGS_2025-12-05.md` for comprehensive experimental findings.

**Latest breakthrough:** Gradient normalization with Adam-like LRs achieves +1.17 dB on kodim03, bridging 78% of the gap to Adam. However, this generalizes to +0.77 dB average across multiple images.

---

## Code Organization

```
lgi-encoder-v2/
├── src/
│   ├── lib.rs           # Module exports
│   ├── optimizer_v2.rs  # Main V2 optimizer (gradient descent)
│   ├── adam_optimizer.rs # Adam optimizer (baseline)
│   ├── hybrid_optimizer.rs # Hybrid Adam→L-BFGS
│   ├── lm_optimizer.rs  # Levenberg-Marquardt
│   ├── renderer_v2.rs   # Gaussian renderer
│   └── test_results.rs  # TestResult framework
├── examples/            # All experiments
└── test-results/        # JSON results and findings
```
