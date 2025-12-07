# Test Scripts Index
**Total:** 212 test implementations
**Location:** `test-suite/scripts/` (flat directory)
**Organization:** Categorized by purpose below, queryable by JSON metadata

---

## ORGANIZATION PHILOSOPHY

**Flat directory + metadata-driven categorization**
- All 212 .rs files in `scripts/`
- Categories defined here (documentation)
- JSON metadata is source of truth for querying
- No rigid folder structure to duplicate information

---

## ACTIVE RESEARCH (25 tests)

### Toolkit Tests (8) - Current Research Focus

**Purpose:** Measure isolated contribution of each toolkit module

| Test | Module | Type |
|------|--------|------|
| isolated_structure_tensor_test | structure_tensor | Initialization-based |
| isolated_geodesic_edt_test | geodesic_edt | Initialization-based |
| isolated_saliency_test | saliency_detection | Initialization-based |
| isolated_frangi_vesselness_test | frangi_vesselness | Initialization-based |
| isolated_text_stroke_test | text_stroke_detection | Initialization-based |
| isolated_perceptual_weight_test | perceptual_weighting | Initialization-based |
| constraint_geodesic_edt_test | geodesic_edt | Optimization-time constraints |
| weighted_loss_perceptual_test | perceptual_weighting | Optimization-time gradient weighting |

**Status:** Results in test-suite/data/ (90 JSON files)
**Findings:** test-suite/findings/SESSION_HANDOVER_2025-12-07_TOOLKIT.md

---

### Benchmarks (7) - Validation Suite

| Test | Purpose |
|------|---------|
| comprehensive_benchmark | Full quality validation |
| kodak_benchmark | 24 standard Kodak images |
| real_world_benchmark | 67 real photos |
| fast_benchmark | Quick validation |
| bench_gpu_vs_cpu_adam | GPU vs CPU Adam comparison |
| bench_gpu_vs_cpu_lbfgs | GPU vs CPU L-BFGS comparison |
| benchmark_optimizers | Multi-optimizer comparison |

---

### Baselines (4) - Reference Implementations

| Test | Reference |
|------|-----------|
| baseline_1_gaussianimage | GaussianImage ECCV 2024 |
| baseline_2_video_codec | Video codec comparison |
| baseline_3_3dgs | 3D Gaussian splatting reference |
| quality_gate_1 | Quality validation gate |

---

### Debug Tools (6) - Visualization

| Test | Purpose |
|------|---------|
| debug_initialization | Visualize Gaussian placement |
| debug_text_detection | Text stroke detection visualization |
| debug_gradient_comparison | Gradient computation validation |
| visual_debug_demo | Interactive debugging |
| visualize_structure_tensor | Structure tensor field visualization |
| visual_strategy_comparison | Compare strategies visually |

---

## RESEARCH BY TOPIC (~80 tests)

### Optimizer Studies (~27 tests)

**Adam variants:**
- test_adam_beta_sweep
- test_adam_epsilon_sweep
- test_adam_lr_scheduling
- test_adam_msssim_hybrid
- test_adam_advanced
- test_encoder_v3_with_adam
- test_error_driven_adam

**L-BFGS variants:**
- test_lbfgs_consistency
- test_lbfgs_history_sweep
- test_lbfgs_refined
- test_lbfgs_scaled
- test_lbfgs_standalone
- test_lbfgs_unconstrained

**L-M variants:**
- test_lm_convergence
- test_lm_optimizer
- test_lm_strategies
- test_gpu_lm
- compare_lm_strategies
- test_jtj_strategy_sweep

**Hybrid/Comparison:**
- test_hybrid_optimizer
- verify_hybrid_optimizer
- test_gpu_optimizer_comparison
- comprehensive_optimizer_comparison
- adam_vs_sgd_test

**V2/V3 Optimizers:**
- test_optimizer_v3_perceptual
- test_v2_with_adam_distance
- test_v2_with_adam_lrs
- test_v2_with_adam_moments

---

### Initialization Studies (~18 tests)

**Strategy comparisons:**
- test_all_init_strategies
- test_placement_strategies
- test_init_optimizer_interaction

**Specific strategies:**
- test_kmeans_init
- test_kmeans_high_position
- test_gdgs_init
- test_slic_init (if exists)
- test_gradient_peak_init
- test_gradient_density_placement
- test_ppm_placement
- test_structure_tensor_init
- test_feature_aware_init

**Parameter studies:**
- test_initial_scales
- test_opacity_init_sweep
- test_rotation_init_sweep
- test_better_color_init
- test_color_sampling_strategies
- test_closed_form_colors
- test_closed_form_color_optimization

---

### Parameter Sweeps (~15 tests)

**Hyperparameter tuning:**
- test_adam_beta_sweep
- test_adam_epsilon_sweep
- test_gradient_clipping_sweep
- test_weight_decay_sweep
- test_lbfgs_history_sweep
- test_optimizer_scale_sweep
- test_jtj_strategy_sweep

**Gaussian parameters:**
- test_gaussian_count_sweep
- test_opacity_init_sweep
- test_rotation_init_sweep

**Loss functions:**
- test_loss_function_sweep
- test_loss_functions

**Coverage/scaling:**
- gamma_sweep_test (if exists)
- simple_n_sweep
- n_sweep_test

---

### GPU Tests (~10 tests)

- gpu_optimizer_test
- gpu_realistic_test
- gpu_speedup_test
- gpu_minimal_debug
- test_gpu_lm
- test_gpu_optimizer_comparison
- test_coarse_to_fine_gpu
- test_geodesic_clamping_gpu
- test_gradient_density_gpu
- compare_gpu_cpu_render
- cpu_vs_gpu_same_image
- bench_gpu_vs_cpu_adam
- bench_gpu_vs_cpu_lbfgs

---

### Densification Studies (~10 tests)

- test_densification_v2
- test_densification_variations
- test_densification_with_inheritance
- test_adaptive_densification
- test_feature_guided_densification
- test_combined_densification
- test_overnight_densification

---

## ARCHIVED/COMPLETED (~100 tests)

**Experiments (exp_*):**
- exp_plateau_investigation
- exp_error_driven_hybrid
- exp_fine_alpha_sweep
- exp_content_types
- exp_edge_systematic
- exp_gamma_autotune
- exp_high_density_edges
- exp_beta_sweep
- exp_learning_rate_sweep
- exp_complex_photo_synthetic

**Overnight tests:**
- test_overnight_commercial
- test_overnight_hybrid_loss
- test_overnight_init_recalibration

**Misc completed:**
- premature_optimization_demo
- test_t0_single_gaussian
- tuning_max_iterations
- parameter_calibration
- ... (~80 more)

**Status:** Moved to scripts/ but candidates for archiving

---

## HOW TO USE THIS INDEX

### Finding Tests

**By category:** Search this document for category name
**By name:** `ls test-suite/scripts | grep keyword`
**By metadata:** Query JSON files in test-suite/data/

### Running Tests

```bash
cargo run --release --example <test_name> -p lgi-encoder-v2
```

Example:
```bash
cargo run --release --example isolated_structure_tensor_test -p lgi-encoder-v2
```

### Finding Results

All results in `test-suite/data/`:
```bash
# Find all results for a test
ls test-suite/data/isolated_structure_tensor_*.json

# Query by metadata
jq 'select(.test_name | contains("structure_tensor"))' test-suite/data/*.json
```

---

## CATEGORIZATION METADATA

For automated categorization, tests include metadata:

```json
{
  "test_name": "isolated_structure_tensor_control_kodim03",
  "module_under_test": "structure_tensor",
  "optimizer": "Adam",
  "category": "toolkit"
}
```

Use `test-suite/analysis/query_results.py` to query by these fields.

---

*212 tests organized by purpose, not folder structure. Metadata is single source of truth.*
