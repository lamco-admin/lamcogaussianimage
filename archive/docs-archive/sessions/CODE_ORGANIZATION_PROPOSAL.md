# Code & Test Organization Proposal
**Date:** 2025-12-07
**Purpose:** Reorganize 212 test examples and test results into logical structure

---

## CURRENT STATE ANALYSIS

### Test Examples (212 files)

**Current:** Flat directory `lgi-encoder-v2/examples/`
**Problem:** Hard to navigate, unclear which tests are active vs obsolete

**Identified patterns:**
- Toolkit tests: 8 files (isolated_*, constraint_*, weighted_*)
- Parameter sweeps: 11 files (test_*_sweep)
- Benchmarks: 7 files (bench_*, *benchmark*)
- Experiments: 10+ files (exp_*)
- Initialization tests: 10+ files (test_*init*, test_*strategy*)
- Optimizer tests: 15+ files (test_*optimizer*, test_*adam*, test_*lbfgs*, test_*lm*)
- Debug/visualization: 6 files (debug_*, visual*)
- Baselines: 4 files (baseline_*, quality_gate_*)
- Overnight: 6+ files (test_overnight_*, overnight_*)
- Comparison: 10+ files (compare_*, *_comparison)
- Integration: Many general test_* files

### Test Results

**Current:** `lgi-encoder-v2/test-results/`
- 5 date folders (2025-12-05, 06, 07, 19, 20)
- 678 JSON result files
- Many markdown findings docs
- Session handovers mixed in
- Subdirectories: analysis_driven_results, feature_classifier_results, metrics, overnight-logs, rendered-images

**Problem:** Documentation mixed with data, no clear separation

---

## PROPOSED ORGANIZATION

### Option A: Categorize Examples by Purpose

```
lgi-encoder-v2/
├── examples/
│   ├── toolkit/              [8 files - ACTIVE RESEARCH]
│   │   ├── isolated_structure_tensor_test.rs
│   │   ├── isolated_geodesic_edt_test.rs
│   │   ├── isolated_saliency_test.rs
│   │   ├── isolated_frangi_vesselness_test.rs
│   │   ├── isolated_text_stroke_test.rs
│   │   ├── isolated_perceptual_weight_test.rs
│   │   ├── constraint_geodesic_edt_test.rs
│   │   └── weighted_loss_perceptual_test.rs
│   │
│   ├── benchmarks/           [7 files - VALIDATION]
│   │   ├── comprehensive_benchmark.rs
│   │   ├── kodak_benchmark.rs
│   │   ├── real_world_benchmark.rs
│   │   ├── fast_benchmark.rs
│   │   ├── bench_gpu_vs_cpu_adam.rs
│   │   ├── bench_gpu_vs_cpu_lbfgs.rs
│   │   └── benchmark_optimizers.rs
│   │
│   ├── sweeps/               [11+ files - HYPERPARAMETER TUNING]
│   │   ├── test_adam_beta_sweep.rs
│   │   ├── test_adam_epsilon_sweep.rs
│   │   ├── test_gradient_clipping_sweep.rs
│   │   ├── test_jtj_strategy_sweep.rs
│   │   ├── test_lbfgs_history_sweep.rs
│   │   ├── test_loss_function_sweep.rs
│   │   ├── gamma_sweep_test.rs
│   │   └── ...
│   │
│   ├── optimizers/           [15+ files - OPTIMIZER RESEARCH]
│   │   ├── test_gpu_optimizer_comparison.rs
│   │   ├── comprehensive_optimizer_comparison.rs
│   │   ├── test_hybrid_optimizer.rs
│   │   ├── test_error_driven_adam.rs
│   │   ├── compare_lm_strategies.rs
│   │   └── ...
│   │
│   ├── initialization/       [10+ files - PLACEMENT RESEARCH]
│   │   ├── test_all_init_strategies.rs
│   │   ├── test_kmeans_init.rs
│   │   ├── test_gdgs_init.rs
│   │   ├── test_structure_tensor_init.rs
│   │   └── ...
│   │
│   ├── debug/                [6 files - VISUALIZATION TOOLS]
│   │   ├── debug_initialization.rs
│   │   ├── debug_text_detection.rs
│   │   ├── visual_debug_demo.rs
│   │   ├── visualize_structure_tensor.rs
│   │   └── ...
│   │
│   ├── baselines/            [4 files - REFERENCE IMPLEMENTATIONS]
│   │   ├── baseline_1_gaussianimage.rs
│   │   ├── baseline_2_video_codec.rs
│   │   ├── baseline_3_3dgs.rs
│   │   └── quality_gate_1.rs
│   │
│   ├── experiments/          [Exploratory tests - ARCHIVE CANDIDATES]
│   │   ├── exp_*.rs
│   │   └── test_overnight_*.rs
│   │
│   └── archive/              [Obsolete/completed tests]
│       └── ... (tests no longer needed)
```

### Option B: Keep Flat, Add Naming Convention

Prefix-based organization:
- `toolkit__*` - Toolkit tests
- `bench__*` - Benchmarks
- `sweep__*` - Parameter sweeps
- `opt__*` - Optimizer tests
- `init__*` - Initialization tests
- etc.

**Easier migration, less file moving**

---

## PROPOSED TEST-RESULTS ORGANIZATION

### Current Issues

1. Markdown docs mixed with JSON data
2. Session handovers in same directory as results
3. Date folders created by TestResult framework (not useful to you)
4. Findings documents scattered

### Proposed Structure

```
lgi-encoder-v2/
├── test-results/
│   ├── data/                     [Raw test result files]
│   │   ├── toolkit/              [Toolkit test JSONs + images]
│   │   ├── benchmarks/           [Benchmark result JSONs]
│   │   ├── sweeps/               [Parameter sweep data]
│   │   └── experiments/          [Exploratory test data]
│   │
│   ├── findings/                 [Research findings documents]
│   │   ├── TOOLKIT_FINDINGS_2025-12.md
│   │   ├── OPTIMIZER_COMPARISON_2025-12.md
│   │   ├── DENSIFICATION_RESEARCH_SUMMARY.md
│   │   ├── COMPREHENSIVE_FINDINGS_2025-12-06.md
│   │   └── ...
│   │
│   ├── sessions/                 [Session handovers - ACTIVE ONLY]
│   │   ├── 2025-12-07_TOOLKIT.md
│   │   ├── 2025-12-07_API_FIXES.md
│   │   └── ... (Dec 4-7 only)
│   │
│   └── archive/                  [Old test data]
│       └── 2025-12-05/          [Old date folders if kept]
│
└── test-results-old/             [OR: rename current dir and start fresh]
```

**Alternative:** Modify TestResult framework to not create date folders

---

## PYTHON CODE ORGANIZATION

### Current State

**Python scattered across:**
- `/quantum_research/` (635MB, 40+ scripts)
- `/phase_*.py` files (6 files, parent directory)
- `/experiments/gaussian_primitives_sprint/` (Python infrastructure)
- `/packages/lgi-legacy/image-gs/` (Old Python+CUDA codec)

### Proposed Structure

```
python/
├── quantum/                      [Quantum research - SEPARATE]
│   ├── q1_channel_discovery/
│   │   ├── Q1_production_real_data.py
│   │   ├── Q1_enhanced_features.py
│   │   └── ...
│   ├── q2_algorithm_discovery/
│   │   └── run_q2_experiments.py
│   ├── data/
│   │   ├── kodak_gaussians_quantum_ready.pkl
│   │   ├── kodak_gaussians_quantum_ready_enhanced.pkl
│   │   └── comprehensive_dataset.pkl
│   ├── results/
│   │   ├── channel_1_substructure_analysis.json
│   │   ├── gaussian_fidelity_clustering_results.json
│   │   └── ...
│   └── README.md                 [Quantum research overview]
│
├── preprocessing/                [Image preprocessing tools]
│   ├── preprocess_image_v2.py   [8-map pipeline]
│   └── ... (if still relevant)
│
├── experiments/                  [Historical experiments]
│   ├── phase_0_experiments/
│   │   ├── phase_0_corrected_sweeps.py
│   │   ├── phase_0_edge_function_discovery.py
│   │   └── results/
│   ├── phase_0.5_experiments/
│   │   └── ...
│   └── gaussian_primitives_sprint/
│       └── ...
│
└── legacy/                       [Old Python+CUDA codec]
    └── image-gs/                 [From packages/lgi-legacy/]
```

**Move from:**
- `/quantum_research/` → `/python/quantum/`
- `/phase_*.py` + `/phase_*_results/` → `/python/experiments/`
- `/experiments/` → `/python/experiments/`
- `/packages/lgi-legacy/image-gs/` → `/python/legacy/`

---

## DETAILED BREAKDOWN: 212 TEST EXAMPLES

### ACTIVE (Keep in main examples/)

**Toolkit (8)** - Current research focus:
```
toolkit/
├── isolated_structure_tensor_test.rs
├── isolated_geodesic_edt_test.rs
├── isolated_saliency_test.rs
├── isolated_frangi_vesselness_test.rs
├── isolated_text_stroke_test.rs
├── isolated_perceptual_weight_test.rs
├── constraint_geodesic_edt_test.rs
└── weighted_loss_perceptual_test.rs
```

**Benchmarks (7)** - Validation suite:
```
benchmarks/
├── comprehensive_benchmark.rs
├── kodak_benchmark.rs
├── real_world_benchmark.rs
├── fast_benchmark.rs
├── bench_gpu_vs_cpu_adam.rs
├── bench_gpu_vs_cpu_lbfgs.rs
└── benchmark_optimizers.rs
```

**Baselines (4)** - Reference implementations:
```
baselines/
├── baseline_1_gaussianimage.rs
├── baseline_2_video_codec.rs
├── baseline_3_3dgs.rs
└── quality_gate_1.rs
```

**Debug Tools (6)** - Development utilities:
```
debug/
├── debug_initialization.rs
├── debug_text_detection.rs
├── debug_gradient_comparison.rs
├── visual_debug_demo.rs
├── visualize_structure_tensor.rs
└── visual_strategy_comparison.rs
```

### RESEARCH (Organize by topic)

**Optimizer Studies (~20 files):**
- test_*optimizer*, test_*adam*, test_*lbfgs*, test_*lm*
- test_hybrid*, compare_*optimizer*
- verify_hybrid_optimizer.rs

**Initialization Studies (~15 files):**
- test_*init*, test_*placement*, test_*strategy*
- test_kmeans*, test_gdgs*, test_slic*

**Parameter Sweeps (~15 files):**
- test_*_sweep (beta, epsilon, lr, clipping, etc.)
- gamma_sweep*, simple_n_sweep, n_sweep_test

**GPU Tests (~10 files):**
- gpu_*, compare_gpu_cpu_*, test_coarse_to_fine_gpu

**Densification (~8 files):**
- test_*densification*
- test_adaptive_*, test_combined_*

**Feature/Content (~8 files):**
- test_feature_*, test_content_*
- test_analysis_driven_*

### ARCHIVE CANDIDATES (~100 files)

**Experiments (exp_*)** - Exploratory, possibly complete:
- exp_plateau_investigation
- exp_error_driven_hybrid
- exp_fine_alpha_sweep
- ... (10 total)

**Overnight tests** - Long-running explorations:
- test_overnight_* (6+ files)

**One-off tests** - Completed investigations:
- premature_optimization_demo.rs
- verify_fix.rs
- test_t0_single_gaussian.rs
- ... (many test_*.rs files)

---

## RECOMMENDED ORGANIZATION PLAN

### Phase 1: Categorize Without Moving

Create **index document** categorizing all 212 tests:

`lgi-encoder-v2/examples/TEST_INDEX.md`:
```markdown
# Test Examples Index

## Toolkit Tests (8) - ACTIVE
- isolated_structure_tensor_test.rs - Line in code
- ... [with descriptions]

## Benchmarks (7) - VALIDATION
- comprehensive_benchmark.rs - Full suite
- ... [with purposes]

## [Continue for all categories]
```

**Benefit:** Understand what exists before moving anything

### Phase 2: Move Active Tests (After review)

Only move **clearly active** categories:
```bash
mkdir examples/toolkit
mv examples/isolated_* examples/constraint_* examples/weighted_* examples/toolkit/

mkdir examples/benchmarks
mv examples/*benchmark* examples/bench_* examples/benchmarks/
```

### Phase 3: Archive Completed Tests (After review)

```bash
mkdir examples/archive
mv examples/exp_* examples/test_overnight_* examples/archive/
```

---

## TEST RESULTS REORGANIZATION

### Issue: TestResult Framework Creates Date Folders

**Current behavior** (from code):
```rust
result.save_with_image(&rendered, "lgi-encoder-v2/test-results");
// Creates: test-results/YYYY-MM-DD/test_name_HH-MM-SS.json
```

**Your preference:** No date folders

### Solutions

**Option 1: Modify TestResult framework**
Change to save flat or by test category:
```
test-results/
├── toolkit_results/
├── benchmark_results/
└── sweep_results/
```

**Option 2: Post-process organization**
Let tests create date folders, then reorganize later:
```bash
# Move all toolkit results to one place
find test-results/2025-*/isolated_*.json -exec mv {} test-results/toolkit_data/ \;
```

**Option 3: Keep date folders, add category subdirs**
```
test-results/
├── toolkit/
│   ├── isolated_structure_tensor_*.json
│   └── ...
├── benchmarks/
└── sweeps/
```

---

## PYTHON ORGANIZATION PROPOSAL

### Separate Quantum from General Python

```
/python/                          [NEW - all Python code]
├── quantum/                      [Quantum research - INDEPENDENT]
│   ├── README.md                 [Quantum overview, Q1/Q2/Q3/Q4 status]
│   ├── q1_channel_discovery/
│   │   ├── Q1_production_real_data.py
│   │   ├── Q1_enhanced_features.py
│   │   ├── Q1_moderate_500_samples.py
│   │   └── results/
│   │       ├── gaussian_channels_kodak_quantum.json
│   │       └── ...
│   ├── q2_algorithm_discovery/
│   │   ├── run_q2_experiments.py
│   │   └── results/
│   ├── data/                     [Shared quantum datasets]
│   │   ├── kodak_gaussians_quantum_ready.pkl
│   │   ├── kodak_gaussians_quantum_ready_enhanced.pkl
│   │   ├── comprehensive_dataset.pkl
│   │   └── kodak_gaussian_data/ (682K configs)
│   ├── analysis/
│   │   ├── classical_baselines.py
│   │   ├── analyze_channel_1_substructure.py
│   │   └── xanadu_gaussian_fidelity.py
│   └── docs/
│       ├── QUANTUM_RESULTS_ANALYSIS.md
│       └── ISOTROPIC_EDGE_VALIDATION_PLAN.md
│
├── preprocessing/                [Image preprocessing tools - IF STILL USED]
│   └── preprocess_image_v2.py   [8-map pipeline]
│
├── experiments-archive/          [Historical Python experiments]
│   ├── phase_0/
│   │   ├── phase_0_corrected_sweeps.py
│   │   ├── phase_0_edge_function_discovery.py
│   │   ├── phase_0_sweep_5_verification.py
│   │   └── results/
│   ├── phase_0.5/
│   │   ├── phase_0.5_edge_region_isolated.py
│   │   └── results/
│   ├── phase_1/
│   │   ├── phase_1_multi_primitive_composition.py
│   │   └── results/
│   └── gaussian_primitives_sprint/
│       └── infrastructure/
│
└── legacy/                       [Old Python+CUDA codec]
    └── image-gs/                 [From packages/lgi-legacy/]
        ├── model.py
        ├── main.py
        ├── gsplat/ (CUDA kernels)
        └── utils/
```

**Moves required:**
1. `/quantum_research/` → `/python/quantum/`
2. `/phase_*.py` + `phase_*_results/` → `/python/experiments-archive/`
3. `/experiments/` → `/python/experiments-archive/`
4. `/packages/lgi-legacy/image-gs/` → `/python/legacy/`

**Clean:**
- Delete `/quantum_env/` (864MB venv - recreatable)

---

## QUESTIONS FOR YOU

### Test Examples

1. **Should I create TEST_INDEX.md first** (categorize without moving)?
2. **Or directly create subdirectories** and move files?
3. **Which tests are obsolete** and should be archived immediately?

### Test Results

4. **Should I modify TestResult framework** to avoid date folders?
5. **Or reorganize existing results** post-hoc?
6. **Keep date folders** but add category organization?

### Python

7. **Is preprocessing/ still actively used** or archive it?
8. **Quantum stays completely independent** in /python/quantum/, correct?
9. **Delete quantum_env/** venv (864MB) - yes/no?

---

**My recommendation:** Start with TEST_INDEX.md (no file moves yet) so you can review categorization before restructuring.