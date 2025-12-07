# LGI Test Suite
**Centralized Testing Infrastructure for Gaussian Image Codec**

**Created:** 2025-12-07
**Purpose:** Independent test suite for systematic codec evaluation

---

## 📁 DIRECTORY STRUCTURE

```
test-suite/
├── scripts/              [212 test implementations - FLAT directory]
│   ├── isolated_structure_tensor_test.rs
│   ├── comprehensive_benchmark.rs
│   ├── test_adam_beta_sweep.rs
│   └── ... (all 212 .rs files)
│   # Organized by SCRIPTS_INDEX.md, not subdirectories
│
├── data/                 [ALL TEST RESULTS - 678 JSON + 193 PNG]
│   ├── *.json            [Test results with metadata]
│   ├── *.png             [Rendered outputs (same names as JSON)]
│   ├── *.log             [Execution logs]
│   └── */                [Subdirectories for organized data]
│
├── findings/             [23 research findings documents]
│   ├── TOOLKIT_FINDINGS_*.md
│   ├── OPTIMIZER_COMPARISON_*.md
│   ├── SESSION_HANDOVER_*.md
│   └── ...
│
├── pipelines/            [Test workflow definitions]
│   ├── toolkit_evaluation.toml
│   ├── optimizer_comparison.toml
│   └── README.md
│
└── analysis/             [Query and analysis tools]
    ├── query_results.py
    ├── compare_optimizers.py
    └── README.md
```

---

## 🚀 RUNNING TESTS

### Running Individual Tests

From project root:
```bash
cargo run --release --example <test_name> -p lgi-encoder-v2
```

Examples:
```bash
# Toolkit test
cargo run --release --example isolated_structure_tensor_test -p lgi-encoder-v2

# Benchmark
cargo run --release --example comprehensive_benchmark -p lgi-encoder-v2

# Sweep
cargo run --release --example test_adam_beta_sweep -p lgi-encoder-v2
```

**Note:** All test scripts are in `test-suite/scripts/` (flat).
Cargo finds them via symlink: `lgi-encoder-v2/examples -> test-suite/scripts`

### Running Pipelines (Future)

```bash
python test-suite/analysis/run_pipeline.py test-suite/pipelines/toolkit_evaluation.toml
```

---

## 🔍 QUERYING RESULTS

### Using jq (JSON Query Tool)

**Find all toolkit results:**
```bash
cd test-suite/data
jq 'select(.module_under_test | contains("toolkit"))' *.json
```

**Compare optimizers:**
```bash
jq 'select(.optimizer == "Adam") | {name: .test_name, psnr: .final_psnr}' *.json
```

**Find results above threshold:**
```bash
jq 'select(.final_psnr > 30.0) | {test: .test_name, psnr: .final_psnr, optimizer: .optimizer}' *.json
```

**Group by image:**
```bash
jq -s 'group_by(.image_path) | map({image: .[0].image_path, count: length, avg_psnr: (map(.final_psnr) | add/length)})' *.json
```

### Using grep (Quick Searches)

**Find all tests on specific image:**
```bash
grep -l "kodim03" *.json
```

**Find all Adam tests:**
```bash
grep -l '"optimizer":"Adam"' *.json
```

**Find tests with specific module:**
```bash
grep -l '"module_under_test":"structure_tensor"' *.json
```

### Python Analysis Scripts (Future)

```bash
# Query with filters
python analysis/query_results.py --optimizer Adam --metric final_psnr --min 30.0

# Compare optimizers
python analysis/compare_optimizers.py Adam L-BFGS "Hybrid"

# Analyze toolkit
python analysis/analyze_toolkit.py --generate-report

# Aggregate by category
python analysis/aggregate_findings.py --category toolkit --output findings/TOOLKIT_SUMMARY.md
```

---

## 📊 DATA FORMAT

### JSON Result Files

Every test produces a JSON file with metadata:

```json
{
  "test_name": "isolated_structure_tensor_control_kodim03",
  "image_path": "test-data/kodak-dataset/kodim03.png",
  "timestamp": "2025-12-07T15:30:45",
  "optimizer": "Adam",
  "n_gaussians": 256,
  "iterations": 200,
  "initial_psnr": 22.15,
  "final_psnr": 28.42,
  "improvement_db": 6.27,
  "elapsed_seconds": 45.2,
  "image_dimensions": [768, 512],
  "module_under_test": "structure_tensor",
  "parameters": {
    "lr_position": 0.0002,
    "lr_color": 0.02,
    ...
  },
  "code_version": {
    "git_hash": "45f773d",
    "dirty": false
  }
}
```

**Key fields for queries:**
- `test_name`: Test identifier
- `optimizer`: Which optimizer used
- `module_under_test`: Which component tested
- `final_psnr`: Quality metric
- `parameters`: All hyperparameters
- `code_version`: Reproducibility info

### PNG Image Files

**Naming convention:** `{test_name}_{timestamp}.png`
**Matches:** JSON file with same name
**Location:** Same directory as JSON

Example pair:
- `isolated_structure_tensor_kodim03_15-30-45.json`
- `isolated_structure_tensor_kodim03_15-30-45.png`

---

## 📋 TEST CATEGORIES

### Toolkit Tests (8 - ACTIVE)
**Purpose:** Measure isolated contribution of each toolkit module

Tests:
- isolated_structure_tensor_test
- isolated_geodesic_edt_test
- isolated_saliency_test
- isolated_frangi_vesselness_test
- isolated_text_stroke_test
- isolated_perceptual_weight_test
- constraint_geodesic_edt_test (optimization-time)
- weighted_loss_perceptual_test (optimization-time)

**Research Question:** Does module X improve quality? By how much?

### Benchmarks (7 - VALIDATION)
**Purpose:** Validate codec quality across datasets

Tests:
- comprehensive_benchmark (full test suite)
- kodak_benchmark (24 standard images)
- real_world_benchmark (67 photos)
- fast_benchmark (quick validation)
- bench_gpu_vs_cpu_adam
- bench_gpu_vs_cpu_lbfgs
- benchmark_optimizers

**Research Question:** What quality do we achieve? How fast?

### Optimizers (~27 - RESEARCH)
**Purpose:** Compare optimization algorithms

Categories:
- Adam variants (beta sweep, epsilon sweep, LR scheduling)
- L-BFGS tests (history sweep, consistency, standalone)
- L-M tests (strategies, convergence, JtJ methods)
- Hybrid optimizer tests
- Cross-optimizer comparisons

**Research Question:** Which optimizer for which problem?

### Initialization (~18 - RESEARCH)
**Purpose:** Compare Gaussian placement strategies

Tests:
- All init strategies comparison
- K-means, GDGS, SLIC, gradient peaks
- Structure tensor-aware initialization
- Color initialization methods
- Placement strategy sweeps

**Research Question:** Where to place Gaussians?

### Sweeps (~11 - TUNING)
**Purpose:** Find optimal hyperparameters

Tests:
- Adam hyperparameters (beta, epsilon)
- Gradient clipping sweeps
- JtJ strategy sweep
- L-BFGS history sweep
- Loss function sweep

**Research Question:** What are the best hyperparameters?

---

## 📖 PIPELINES (Optional Workflows)

### Concept

Pipelines define test sequences without rigid folder structure:

```toml
# pipelines/toolkit_evaluation.toml
[pipeline]
name = "Toolkit Component Evaluation"
description = "Measure each toolkit module's contribution"

[[tests]]
script = "toolkit/isolated_structure_tensor_test"
images = ["kodim03", "kodim01", "kodim19", "kodim21"]

[[tests]]
script = "toolkit/isolated_geodesic_edt_test"
images = ["kodim03", "kodim01", "kodim19", "kodim21"]

[analysis]
compare_metric = "final_psnr"
generate_report = "findings/TOOLKIT_SUMMARY_2025-12.md"
```

### Creating Pipelines

1. Define test sequence in TOML
2. Specify images to test
3. Set analysis parameters
4. Run with: `python analysis/run_pipeline.py pipeline_file.toml`

---

## 📈 ANALYSIS WORKFLOWS

### Workflow 1: Compare Optimizers

```bash
# Find all optimizer test results
cd test-suite/data
jq 'select(.test_name | contains("optimizer")) | {test: .test_name, opt: .optimizer, psnr: .final_psnr}' *.json

# Group by optimizer
jq -s 'group_by(.optimizer) | map({optimizer: .[0].optimizer, avg_psnr: (map(.final_psnr) | add/length), count: length})' *.json
```

### Workflow 2: Analyze Toolkit Modules

```bash
# All toolkit results
jq 'select(.module_under_test == "structure_tensor" or .module_under_test == "geodesic_edt")' *.json

# Treatment vs control
jq 'select(.test_name | contains("treatment")) | {module: .module_under_test, psnr: .final_psnr}' *.json
jq 'select(.test_name | contains("control")) | {module: .module_under_test, psnr: .final_psnr}' *.json
```

### Workflow 3: Image-Specific Analysis

```bash
# All tests on kodim03
grep -l "kodim03" *.json | xargs jq '{test: .test_name, psnr: .final_psnr}'

# Best result for each image
jq -s 'group_by(.image_path) | map({image: .[0].image_path, best_psnr: (map(.final_psnr) | max), best_test: (sort_by(.final_psnr) | reverse | .[0].test_name)})' *.json
```

---

## 🛠️ STATISTICS

### Current State

**Test Scripts:** 212 files organized
- Toolkit: 8
- Benchmarks: 7
- Baselines: 4
- Debug: 6
- Optimizers: ~27
- Initialization: ~18
- Sweeps: ~11
- Archive: ~100

**Test Data:** 871 files
- JSON results: 678
- PNG images: 193

**Findings Docs:** 28 documents

---

## ⚠️ REQUIRED CODE UPDATES

### 1. TestResult Framework

**File:** `packages/lgi-rs/lgi-encoder-v2/src/test_results.rs`

**Current behavior:**
```rust
pub fn save_with_image(&self, rendered: &ImageBuffer, base_dir: &str) {
    // Creates: base_dir/YYYY-MM-DD/test_name_HH-MM-SS.json
}
```

**Required change:**
```rust
pub fn save_with_image(&self, rendered: &ImageBuffer, data_dir: &str) {
    // Create: data_dir/test_name_HH-MM-SS.json (flat, no date folder)
}
```

### 2. Test Script Paths

All 212 test scripts now reference:
```rust
const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";
```

**Required change:**
```rust
const RESULTS_DIR: &str = "test-suite/data";
```

Or use environment variable/config.

### 3. Cargo.toml Example Paths

**File:** `packages/lgi-rs/lgi-encoder-v2/Cargo.toml`

May need to update if examples path changed from default.

---

*This test suite provides centralized, queryable test infrastructure independent of implementation code.*
