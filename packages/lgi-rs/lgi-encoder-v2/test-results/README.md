# Test Results Corpus

Persistent storage for optimizer experiment results. **Evidence-gathering, not conclusion-drawing.**

## Purpose

Build a corpus of experimental data to understand tool behavior across different:
- Optimizers (Adam, L-BFGS, L-M, hybrid)
- Initialization strategies (grid, PPM, Laplacian, k-means, SLIC)
- Gaussian configurations (isotropic vs anisotropic, count, scale)
- Image content types (gradients, edges, textures)

## Directory Structure

```
test-results/
├── YYYY-MM-DD/                    # Daily results (JSON metrics)
│   ├── experiment_name_HH-MM-SS.json
│   └── ...
├── rendered-images/               # Visual outputs with timestamps
│   ├── experiment_suffix_YYYY-MM-DD_HH-MM-SS.png
│   └── ...
├── README.md                      # This file
└── EXPERIMENT_FRAMEWORK.md        # Experiment design (needs update)
```

## Infrastructure (Rust)

### TestResult struct (`src/test_results.rs`)

```rust
pub struct TestResult {
    pub experiment_name: String,
    pub timestamp: String,           // ISO format: YYYY-MM-DDTHH:MM:SS
    pub image_path: String,
    pub image_dimensions: (u32, u32),
    pub n_gaussians: usize,
    pub iterations: usize,
    pub optimizer: String,
    pub initial_psnr: f32,
    pub final_psnr: f32,
    pub improvement_db: f32,
    pub final_loss: f32,
    pub elapsed_seconds: f32,
    pub notes: String,
    pub extra_metrics: Option<serde_json::Value>,  // Flexible extension
}
```

### Key Functions

```rust
// Save metrics to JSON
result.save(RESULTS_DIR) -> Result<String>

// Save rendered image with timestamp
save_rendered_image(&image, RESULTS_DIR, "experiment_name", "suffix") -> Result<String>

// Load all results from a date
load_results(RESULTS_DIR, "2025-12-05") -> Vec<TestResult>
```

### Usage in Examples

```rust
use lgi_encoder_v2::test_results::{TestResult, save_rendered_image};

const RESULTS_DIR: &str = ".../test-results";

let mut result = TestResult::new("my_experiment", "/path/to/image.png");
result.optimizer = "Adam (per-param LRs)".to_string();
result.n_gaussians = 576;
// ... run experiment, fill in metrics ...
result.save(RESULTS_DIR)?;
save_rendered_image(&final_image, RESULTS_DIR, "my_experiment", "final")?;
```

## Completed Experiments (2025-12-05)

### 1. per_param_lr_comparison.json
**Finding**: Per-parameter LRs beat single LR by **+4.79 dB**
- Single LR (0.01): Chaotic oscillation, -1.78 dB regression
- Per-param LR: Monotonic convergence, +3.01 dB improvement
- **Key insight**: Position LR must be ~100× smaller than color (0.0002 vs 0.02)

### 2. multi_kodak_per_param_lr.json
**Finding**: Average +2.25 dB across 5 image types, but **huge variance**
| Image | Type | Improvement |
|-------|------|-------------|
| kodim03 | Natural gradients | +3.75 dB (best) |
| kodim01 | Geometric edges | +2.01 dB |
| kodim08 | Architectural texture | +1.99 dB |
| kodim13 | Smooth regions | +1.83 dB |
| kodim05 | Organic textures | +1.66 dB (worst) |

### 3. placement_strategy_comparison.json
**Finding**: "Smart" placement performed **8 dB worse** than uniform grid!
| Strategy | PSNR | vs Baseline |
|----------|------|-------------|
| Uniform (iso) | 23.02 dB | baseline |
| Uniform (aniso) | 23.01 dB | -0.01 dB |
| PPM (iso) | 14.22 dB | **-8.80 dB** |
| PPM (aniso) | 14.33 dB | **-8.70 dB** |
| Laplacian | 14.84 dB | **-8.19 dB** |

**Note**: This is evidence about how these tools behave with current optimizer settings, not a conclusion that PPM is bad.

### 4. hypotheses_feature_placement.json
Draft hypotheses about image-type to placement mapping. **Treat as working notes, not conclusions.**

## Available Optimizers (to be tested systematically)

| Optimizer | File | Status |
|-----------|------|--------|
| Adam (per-param LR) | `adam_optimizer.rs` | Partially tested |
| L-BFGS | `optimizer_lbfgs.rs` | Not tested |
| Levenberg-Marquardt | `lm_optimizer.rs` | Not tested |
| Hybrid | `hybrid_optimizer.rs` | Not tested |
| Optimizer v2 (basic GD) | `optimizer_v2.rs` | Not tested |
| Optimizer v3 (perceptual) | `optimizer_v3_perceptual.rs` | Not tested |

## Available Initialization Strategies (to be tested)

| Strategy | File | Status |
|----------|------|--------|
| Grid (uniform) | default | Tested |
| GDGS | `gdgs_init.rs` | Not tested |
| Gradient Peak | `gradient_peak_init.rs` | Not tested |
| K-means | `kmeans_init.rs` | Not tested |
| SLIC superpixel | `slic_init.rs` | Not tested |

## Alternative Techniques (from EXISTING_TECHNIQUES_RESEARCH.md)

These documented alternatives have NOT been implemented/tested:
1. **Separable 1D×1D Gaussians** - Product of 1D Gaussians for edges (8× speedup claimed)
2. **Gabor functions** - Gaussian × sinusoid (may be better for edges/textures)
3. **Truncation radius** - Standard optimization, claimed 100× speedup

## Next Experiments Needed

1. **Optimizer tuning**: Figure out optimal settings for EACH optimizer before comparison
2. **Systematic optimizer comparison**: Same init, same images, different optimizers
3. **Full Kodak dataset**: All 24 images, after optimizer tuning
4. **Alternative techniques**: Gabor, separable 1D×1D, truncation radius

## Principles

1. **Evidence, not conclusions**: Results show tool behavior under specific conditions
2. **Changing variables changes results**: Different optimizer = different findings
3. **Document everything**: Every experiment saved with full parameters
4. **Reproducible**: JSON contains all info needed to reproduce
