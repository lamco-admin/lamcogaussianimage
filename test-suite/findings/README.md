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
├── YYYY-MM-DD/                    # Daily results
│   ├── experiment_name_HH-MM-SS.json    # Metrics + parameters
│   ├── experiment_name_HH-MM-SS.png     # Rendered result (matching timestamp)
│   ├── experiment_name_HH-MM-SS_aborted.json  # Aborted experiments
│   └── ...
├── overnight-logs/                # Long-running test output logs
│   └── *.log
├── rendered-images/               # Legacy location (deprecated)
├── README.md                      # This file
└── EXPERIMENT_FRAMEWORK.md        # Experiment design
```

---

## TestResult Framework (Enhanced Dec 6, 2025)

### Core Types

```rust
use lgi_encoder_v2::test_results::{TestResult, TestStatus, OptimizerParams, CodeVersion};

/// Test result with full tracking
pub struct TestResult {
    pub experiment_name: String,
    pub timestamp: String,           // ISO: YYYY-MM-DDTHH:MM:SS
    pub status: TestStatus,          // Running, Completed, Aborted, Failed
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
    pub params: OptimizerParams,     // All optimizer parameters
    pub code_version: CodeVersion,   // Git hash for reproducibility
    pub rendered_image_path: Option<String>,
    pub extra_metrics: Option<serde_json::Value>,
}

/// Git version info (auto-captured)
pub struct CodeVersion {
    pub git_hash: Option<String>,    // Short commit hash
    pub dirty: Option<bool>,         // Uncommitted changes?
    pub branch: Option<String>,      // Current branch
}

/// Structured optimizer parameters
pub struct OptimizerParams {
    pub lr_position: Option<f32>,
    pub lr_color: Option<f32>,
    pub lr_scale: Option<f32>,
    pub lr_opacity: Option<f32>,
    pub lr_rotation: Option<f32>,
    pub beta1: Option<f32>,
    pub beta2: Option<f32>,
    pub densify_grad_threshold: Option<f32>,
    pub prune_opacity_threshold: Option<f32>,
    pub densify_interval: Option<usize>,
    pub start_gaussians: Option<usize>,
    pub target_gaussians: Option<usize>,
    pub custom: HashMap<String, serde_json::Value>,  // Additional params
}
```

### Test Status

```rust
pub enum TestStatus {
    Completed,  // Test finished successfully
    Aborted,    // Test manually stopped (e.g., diverging)
    Failed,     // Test errored
    Running,    // Test still in progress
}
```

---

## Usage Examples

### Basic Usage (with image saving)

```rust
use lgi_encoder_v2::test_results::TestResult;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

// Create result at start of test
let mut result = TestResult::new("my_experiment", image_path);
result.optimizer = "Adam (per-param LRs)".to_string();
result.n_gaussians = 256;
result.iterations = 200;
result.initial_psnr = init_psnr;
result.image_dimensions = (target.width, target.height);

// Record optimizer parameters
result.set_param("lr_position", 0.0002);
result.set_param("lr_color", 0.02);
result.set_param("lr_scale", 0.005);
result.set_param("lr_opacity", 0.05);
result.set_param("beta1", 0.9);
result.set_param("beta2", 0.999);

// ... run experiment ...

// Complete and save with rendered image
result.final_psnr = final_psnr;
result.improvement_db = final_psnr - init_psnr;
result.elapsed_seconds = elapsed;
result.complete();
let _ = result.save_with_image(&rendered, RESULTS_DIR);
```

### Handling Aborted Tests

```rust
// If test is diverging or going wrong
if psnr < init_psnr - 5.0 {
    result.abort(
        "PSNR diverging - dropped 5dB below initial",
        current_iteration,
        current_psnr
    );
    let _ = result.save_partial(RESULTS_DIR);  // Saves as *_aborted.json
    return;
}
```

### Custom Parameters

```rust
// For densification experiments
result.set_param("densify_grad_threshold", 0.0002);
result.set_param("prune_opacity_threshold", 0.1);
result.set_param("densify_interval", 30);
result.set_param("start_gaussians", 64);
result.set_param("target_gaussians", 256);

// Custom parameters (any JSON-serializable value)
result.set_param("custom_metric", some_value);
result.set_param("phase1_iters", 100_usize);
```

---

## Output Files

### JSON Result (experiment_name_HH-MM-SS.json)

```json
{
  "experiment_name": "combined_hybrid_prune",
  "timestamp": "2025-12-20T15:30:45",
  "status": "Completed",
  "image_path": "/path/to/kodim03.png",
  "image_dimensions": [768, 512],
  "n_gaussians": 187,
  "iterations": 200,
  "optimizer": "Hybrid: Dense→Prune→Optimize",
  "initial_psnr": 20.27,
  "final_psnr": 22.30,
  "improvement_db": 2.03,
  "final_loss": 0.0,
  "elapsed_seconds": 218.4,
  "notes": "",
  "params": {
    "lr_position": 0.0002,
    "lr_color": 0.02,
    "lr_scale": 0.005,
    "lr_opacity": 0.05,
    "prune_opacity_threshold": 0.1,
    "start_gaussians": 256,
    "final_gaussians": 187
  },
  "code_version": {
    "git_hash": "af677c0",
    "dirty": true,
    "branch": "main"
  },
  "rendered_image_path": "test-results/2025-12-20/combined_hybrid_prune_15-30-45.png"
}
```

### PNG Image (same timestamp as JSON)

The rendered output image with matching filename/timestamp for easy correlation.

---

## Versioning Strategy

### Hybrid Approach (Recommended)

1. **Automatic**: TestResult captures git hash + dirty status on creation
2. **Manual checkpoints**: Commit before running important experiments

```bash
# Before running experiment
git add -A && git commit -m "Checkpoint before hybrid pruning test"

# Run experiment
cargo run --release --example test_combined_densification -p lgi-encoder-v2

# Results now linked to clean commit
```

### Reproducing Results

1. Check the `code_version.git_hash` in the JSON
2. If `dirty: false`, checkout that commit
3. If `dirty: true`, the exact code state wasn't captured (check session notes)

---

## Principles

1. **Evidence, not conclusions**: Results show tool behavior under specific conditions
2. **Save everything**: Parameters, images, git version, aborted tests
3. **Reproducible**: JSON + git hash = can reproduce experiment
4. **Document failures**: Aborted tests are valuable data points
5. **Visual inspection**: Always save rendered images for human review

---

## Key Files

| File | Purpose |
|------|---------|
| `src/test_results.rs` | TestResult framework implementation |
| `examples/test_combined_densification.rs` | Example using enhanced framework |
| `test-results/README.md` | This documentation |

---

*Last updated: December 6, 2025*
