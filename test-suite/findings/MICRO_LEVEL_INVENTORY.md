# Micro-Level Module Inventory - 2025-12-07

## Overview

This document catalogs all the micro-level building blocks in lgi-core that haven't been systematically tested. The focus is on **image interpretation** and **Gaussian parameter derivation** techniques.

## Image Interpretation Modules

### 1. Structure Analysis

| Module | Purpose | Status | Best With |
|--------|---------|--------|-----------|
| `structure_tensor.rs` | Edge orientation, coherence | Testing | L-M, direct |
| `laplacian.rs` | Second derivative, peak detection | Testing | Any |
| `frangi_vesselness.rs` | Tubular structure detection | **Untested** | Anisotropic |
| `coherence_enhancing_diffusion.rs` | Anisotropic diffusion (Weickert) | **Untested** | Pre-processing |

### 2. Saliency & Importance

| Module | Purpose | Status | Best With |
|--------|---------|--------|-----------|
| `saliency_detection.rs` | Gradient-based importance | **Untested** | PPM |
| `entropy.rs` | Tile-based complexity | Testing | Adaptive N |
| `position_probability_map.rs` | Combined entropy+gradient+saliency | Testing | Importance sampling |
| `perceptual_weighting.rs` | 3× weight at edges | **Untested** | Weighted loss |

### 3. Content Detection

| Module | Purpose | Status | Best With |
|--------|---------|--------|-----------|
| `content_detection.rs` | Smooth/Sharp/Mixed/Photo/HighFreq | **Untested** | Auto-tuning |
| `specular_detection.rs` | Log-luminance highlights | **Untested** | Special handling |
| `text_stroke_detection.rs` | Thin strokes via structure tensor | **Untested** | Direct placement |

### 4. Distance & Geometry

| Module | Purpose | Status | Best With |
|--------|---------|--------|-----------|
| `geodesic_edt.rs` | Edge-respecting distance field | **Untested** | Sigma clamping |
| `gaussian_splitting.rs` | Split large Gaussians | Partially tested | Densification |
| `gaussian_merging.rs` | Merge overlapping Gaussians | **Untested** | Compression |
| `adaptive_splitting.rs` | Content-aware splitting | **Untested** | Refinement |

## Initialization Strategies

| Strategy | Implementation | Status | Notes |
|----------|---------------|--------|-------|
| Random | `InitStrategy::Random` | Testing | Baseline |
| Grid | `InitStrategy::Grid` | Testing | Simple |
| Gradient | `InitStrategy::Gradient` | Testing | Edge-aware |
| Importance | `InitStrategy::Importance` | Testing | Entropy-based |
| SLIC | `InitStrategy::SLIC` | Testing | Superpixel, anisotropic |
| Structure Tensor | Custom | Testing | Eigenanalysis shapes |
| PPM Sampling | Custom | Testing | Importance sampling |
| Laplacian Peaks | Custom | Testing | Feature points |

## Gaussian Parameterizations

| Type | Parameters | Status | Best For |
|------|-----------|--------|----------|
| Euler | (scale_x, scale_y, rotation) | Used | Intuitive |
| Cholesky | L where Σ=LLᵀ | **Untested** | Numerical stability |
| LogRadius | (log_r, eccentricity, θ) | **Untested** | Compression |
| InverseCovariance | Direct Σ⁻¹ | **Untested** | Fast rendering |

## Scale & Shape Parameters

| Parameter | Default | Tested Range | Status |
|-----------|---------|--------------|--------|
| Base scale | 0.02 | - | Testing 0.005-0.1 |
| Anisotropy ratio | 1.0 | - | Testing 1.0-5.0 |
| Coverage-based scale | N/A | - | Testing 0.7/√N |

## Key Insights for Optimizer Compatibility

### Adam-Compatible
- Standard grid/random init with isotropic Gaussians
- Any placement with gradient-descent-friendly starting points

### Potentially Non-Adam
- Structure tensor-derived shapes (orientation/scale from eigenanalysis)
- Geodesic EDT sigma clamping (constraint, not optimization target)
- Text stroke detection (creates specific shape presets)
- Closed-form color optimization

### Better with L-M or Direct
- Analytically derived anisotropic shapes
- Pre-computed orientation from image analysis
- Constrained scale from geodesic distance

## Tests Running Now

1. `test_all_init_strategies` - All 5 InitStrategy variants
2. `test_structure_tensor_init` - Eigenanalysis-based shapes
3. `test_ppm_placement` - Entropy+gradient importance sampling
4. `test_initial_scales` - Base scale and anisotropy sweep
5. `test_ewa_vs_standard` - EWA splatting comparison
6. `test_densification_variations` - Pruning threshold sweep
7. `test_guided_filter_impact` - Edge-preserving color init

## Priority Tests to Create

1. **GeodesicEDT sigma clamping** - Prevent color bleeding
2. **Frangi vesselness** - Tubular structure detection
3. **ContentAnalyzer auto-tuning** - Content-adaptive parameters
4. **Cholesky/LogRadius parameterizations** - Alternative representations
5. **Gaussian merging** - Post-optimization compression

## Hypothesis: The Missing Link

Current approach: Initialize → Optimize with Adam → Done

Potential improvement: **Image interpretation → Closed-form shape derivation → Optional Adam refinement**

The structure tensor, geodesic EDT, and content detection modules suggest a pipeline where:
1. Analyze image to determine optimal Gaussian shapes
2. Derive shapes analytically (not through gradient descent)
3. Only use Adam for fine-tuning colors and minor position adjustments

This could:
- Reduce iterations needed
- Improve convergence on edge-heavy images
- Provide better starting points for second-order methods
