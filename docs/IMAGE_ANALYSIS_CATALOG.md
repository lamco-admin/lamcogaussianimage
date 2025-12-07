# IMAGE ANALYSIS CATALOG - Gaussian Image Codec
## Complete Image Processing & Feature Detection Inventory

**Document Version:** 2.0
**Date:** 2025-12-07
**Repository:** lgi-project
**Source:** `packages/lgi-rs/lgi-core/src/`
**Total Modules:** 62 analysis modules

---

## TABLE OF CONTENTS

1. [Overview](#overview)
2. [Toolkit Modules (Active Research)](#toolkit-modules)
3. [Edge & Structure Detection](#edge--structure-detection)
4. [Feature Classification](#feature-classification)
5. [Quality Metrics](#quality-metrics)
6. [Spatial Processing](#spatial-processing)
7. [Advanced Processing](#advanced-processing)
8. [Compression & Quantization](#compression--quantization)
9. [Complete Module Index](#complete-module-index)

---

## OVERVIEW

The lgi-core crate contains 62 image analysis and processing modules that support Gaussian encoding. These range from fundamental operations (structure tensor, gradients) to advanced techniques (artifact detection, iterative refinement).

### Module Categories

- **Toolkit Modules** (8) - Active research focus for Dec 2025
- **Edge/Structure Detection** (10+) - Edge analysis and orientation
- **Feature Classification** (5+) - Content-type detection
- **Quality Metrics** (8+) - PSNR, SSIM, MS-SSIM, perceptual
- **Spatial Processing** (10+) - Filtering, transforms, ordering
- **Advanced** (15+) - Artifact detection, refinement, specialized
- **Compression** (6+) - Quantization, encoding, rate-distortion

---

## TOOLKIT MODULES

**Research Focus:** Dec 2025 isolated toolkit tests
**Purpose:** Composable tools tested independently for contribution measurement

### 1. STRUCTURE TENSOR

**File:** `structure_tensor.rs`
**Lines:** (from file count)
**Purpose:** Edge orientation and coherence analysis

#### Description
Computes local image structure via eigendecomposition of gradient outer product. Provides:
- Edge tangent and normal directions
- Coherence (edge strength): (λ₁-λ₂)/(λ₁+λ₂)
- Principal variances (λ₁, λ₂)

#### Theory (from code comments)
```
J = [[Ix²,   Ix·Iy],  ⊗ G_σs
     [Ix·Iy, Iy²  ]]
```

Where:
- Ix, Iy: Image gradients
- G_σs: Smoothing Gaussian kernel
- ⊗: Convolution

#### Key Methods
- `StructureTensorField::compute(image, σ_gradient, σ_smooth)`
- Returns field with tensor at each pixel

#### Parameters
- σ_gradient: Gradient computation scale (default: 1.2)
- σ_smooth: Tensor smoothing scale (default: 2.5)

#### Output Fields
- eigenvalue_major, eigenvalue_minor
- eigenvector_major, eigenvector_minor
- coherence

#### Research Status
✓ Tested in `isolated_structure_tensor_test.rs`

---

### 2. GEODESIC EDT

**File:** `geodesic_edt.rs`
**Lines:** (from file count)
**Purpose:** Edge-respecting distance transform

#### Description
Computes distance fields that cannot cross strong edges. Unlike Euclidean distance, geodesic paths are penalized when crossing structure.

#### Theory (from code comments)
Geodesic distance D(p,q) is shortest path where cost is weighted by edge strength:
```
D(p,q) = min_γ ∫_γ C(x,y) ds
```

Where C(x,y) is high at edges.

#### Algorithm
Jump Flooding Algorithm (JFA) - GPU-friendly parallel O(log N) computation.

#### Key Methods
- `GeodesicEDT::compute(image, tensor_field, coherence_threshold, edge_penalty)`
- `clamp_sigma(x, y, sigma)` - Limit Gaussian scale to edge distance

#### Benefits for Gaussian Encoding (from code comments)
- Prevents color bleeding across boundaries
- Enables edge-aware Gaussian size clamping
- Respects semantic structure
- Critical for text, faces, segmented content

#### Research Status
✓ Tested in `isolated_geodesic_edt_test.rs` (initialization)
✓ Tested in `constraint_geodesic_edt_test.rs` (optimization-time constraints)

#### References (from code)
- Jump Flooding Algorithm (Rong & Tan, 2006)
- FastGeodis (Python/CUDA implementation)

---

### 3. SALIENCY DETECTION

**File:** `saliency_detection.rs`
**Lines:** (from file count)
**Purpose:** Detect visually important regions

#### Description
Gradient-based saliency computation to identify attention-worthy regions.

#### Implementation
Simple gradient magnitude approach:
- Compute horizontal and vertical gradients
- Combine gradient magnitudes
- Threshold at configurable level (default: 0.5)

#### Key Methods
- `SaliencyDetector::compute_saliency_map(image)`
- Returns Vec<f32> with per-pixel saliency scores

#### Configuration
- `threshold`: Saliency threshold (default: 0.5)

#### Research Status
✓ Tested in `isolated_saliency_test.rs`

---

### 4. FRANGI VESSELNESS

**File:** `frangi_vesselness.rs`
**Lines:** (from file count)
**Purpose:** Tubular structure detection (vessels, hair, thin lines)

#### Description
Detects tube-like structures using eigenvalue analysis of Hessian matrix. Originally for medical imaging (blood vessels).

#### Algorithm (Frangi et al. 1998)
Based on eigenvalues λ₁, λ₂ of structure tensor:
- Tubular: λ₁ >> λ₂ (high anisotropy)
- Blob: λ₁ ≈ λ₂
- Background: both small

#### Key Methods
- `FrangiFilter::compute_vesselness(tensor_field)`
- Returns Vec<f32> with per-pixel vesselness scores

#### Parameters
- alpha: Plate-like structure sensitivity (default: 0.5)
- beta: Background suppression (default: 15.0)
- c: Second-order structureness (default: 500.0)

#### Research Status
✓ Tested in `isolated_frangi_vesselness_test.rs`

---

### 5. TEXT STROKE DETECTION

**File:** `text_stroke_detection.rs`
**Lines:** (from file count)
**Purpose:** Detect thin linear strokes (text characters, line art)

#### Description
Identifies thin, high-coherence structures characteristic of text strokes.

#### Detection Criteria
- High coherence (> 0.8) - linear structure
- Thin width (< 2 pixels) - stroke-like

#### Key Methods
- `TextStrokeDetector::detect_strokes(image, tensor_field)`
- Returns Vec<bool> mask of stroke pixels

#### Configuration
- `coherence_threshold`: Default 0.8
- `width_max`: Default 2.0 pixels

#### Research Status
✓ Tested in `isolated_text_stroke_test.rs`

---

### 6. PERCEPTUAL WEIGHTING

**File:** `perceptual_weighting.rs`
**Lines:** (from file count)
**Purpose:** Compute importance weights for loss functions

#### Description
Combines multiple perceptual factors (edges, saliency, MS-SSIM) to create weight maps that emphasize important regions during optimization.

#### Key Methods
- `PerceptualWeighter::compute_weight_map(image, edge_map, saliency_map)`
- Returns Vec<f32> with per-pixel importance weights

#### Configuration
- `use_ms_ssim`: Include MS-SSIM weighting (default: true)
- `use_edge_weight`: 3× weight at edges (default: true)
- `use_saliency`: Weight by saliency (default: false)

#### Typical Output
- Uniform regions: weight = 1.0
- Edge regions: weight = 3.0
- Salient edges: weight = 3.0+

#### Research Status
✓ Tested in `isolated_perceptual_weight_test.rs` (initialization)
✓ Tested in `weighted_loss_perceptual_test.rs` (optimization-time gradient weighting)

---

### 7. FEATURE CLASSIFIER

**File:** `feature_classifier.rs`
**Lines:** (from file count)
**Purpose:** Per-pixel feature type classification

#### Description
Combines structure tensor, Laplacian, and luminance to classify each pixel into actionable feature types.

#### Feature Types (from code)
```rust
pub enum FeatureType {
    Smooth,          // Low energy, low texture → large Gaussians
    SmoothGradient,  // Low energy, varying color → medium Gaussians
    Edge,            // High coherence, high energy → anisotropic aligned
    Corner,          // Low coherence, high energy → small isotropic
    TextStroke,      // Very high coherence, thin → tiny elongated
    Texture,         // High local variance → medium, potentially anisotropic
    Specular,        // Very high luminance → small, high opacity
}
```

#### Key Methods
- `FeatureClassifier::classify(image)` - Returns Vec<FeatureType>

#### Integration
Each feature type suggests:
- Gaussian size
- Isotropy vs anisotropy
- Density requirements

#### Research Status
✓ Implemented
Tested in research experiments

---

### 8. ARTIFACT DETECTOR

**File:** `artifact_detector.rs`
**Lines:** (from file count)
**Purpose:** Detect encoding artifacts to guide refinement

#### Artifact Types Detected (from code table)

| Artifact | Cause | Detection Method | Suggested Fix |
|----------|-------|------------------|---------------|
| Color Bleeding | Gaussians crossing edges | Cross-edge color correlation | Geodesic clamping |
| Blur | Insufficient Gaussians | High-freq loss vs target | Add smaller Gaussians |
| Ringing | Oscillating overlaps | Sign-alternating error | Adjust positions/scales |
| Banding | Quantization in gradients | Step-like transitions | Add gradient Gaussians |
| Ghosting | Misaligned overlaps | Faint duplicates | Prune/merge similar |

#### Key Methods
- `ArtifactDetector::analyze(rendered, target)` - Returns ArtifactReport

#### Report Fields
- bleeding_score
- blur_regions (Vec of high-freq loss areas)
- ringing_regions
- banding_regions
- ghosting_candidates

#### Research Status
✓ Implemented
Testing status unknown

---

## EDGE & STRUCTURE DETECTION

### Core Gradient & Edge Detection

#### LAPLACIAN
**File:** `laplacian.rs`
**Purpose:** Second derivative for edge/corner detection

Key function: `compute_laplacian(image)` - Returns Laplacian magnitude field

#### GAUSSIAN FILTER
**File:** `gaussian_filter.rs`
**Purpose:** Gaussian smoothing kernels

Used by structure tensor and multi-scale processing.

---

## QUALITY METRICS

### 1. ERROR METRICS (PSNR, MSE, MAE)

**File:** `error_metrics.rs`
**Purpose:** Standard reconstruction quality metrics

#### Functions
- `compute_psnr(target, rendered)` - Peak Signal-to-Noise Ratio (dB)
- `compute_mse(target, rendered)` - Mean Squared Error
- `compute_mae(target, rendered)` - Mean Absolute Error

### 2. MS-SSIM (Multi-Scale Structural Similarity)

**File:** `ms_ssim.rs`
**Purpose:** Perceptual quality metric

Multi-scale pyramid (5 levels default) measuring luminance, contrast, structure similarity.

**File:** `ms_ssim_loss.rs`
**Purpose:** MS-SSIM as loss function (differentiable)

**File:** `ms_ssim_gradients.rs`
**Purpose:** Analytical gradients of MS-SSIM for optimization

### 3. EDGE-WEIGHTED LOSS

**File:** `edge_weighted_loss.rs`
**Purpose:** Emphasize edges in loss computation

Weight formula: `w(x,y) = 1 + 2·coherence(x,y)`
Effect: 3× gradient at edges (coherence=1)

---

## SPATIAL PROCESSING

### Ordering & Organization

#### ORDERING
**File:** `ordering.rs`
**Purpose:** Gaussian ordering strategies

#### IMPORTANCE ORDERING
**File:** `importance_ordering.rs`
**Purpose:** Order by importance for progressive rendering

#### HILBERT ORDERING
**File:** `hilbert_ordering.rs`
**Purpose:** Space-filling curve ordering for locality

#### SPATIAL INDEX
**File:** `spatial_index.rs`
**Purpose:** BVH + grid acceleration for fast queries

### LOD & Progressive

#### LOD SYSTEM
**File:** `lod_system.rs`
**Purpose:** Level-of-detail bands for adaptive rendering

Classification: Coarse/Medium/Fine based on det(Σ)

#### PROGRESSIVE LOADING
**File:** `progressive_loading.rs`
**Purpose:** Stream-decode most important Gaussians first

---

## ADVANCED PROCESSING

### Iterative Refinement

#### ITERATIVE_REFINEMENT
**File:** `iterative_refinement.rs`
**Purpose:** Multi-pass encoding with analysis-driven refinement

Workflow:
1. Encode initial
2. Analyze artifacts
3. Apply targeted fixes
4. Re-encode
5. Repeat until quality threshold

#### Research Status
✓ Implemented as toolkit module
Currently being tested

### Gaussian Manipulation

#### ADAPTIVE_SPLITTING
**File:** `adaptive_splitting.rs`
**Purpose:** Split Gaussians in high-error regions

#### GAUSSIAN_SPLITTING
**File:** `gaussian_splitting.rs`
**Purpose:** Divide Gaussian along major axis

#### GAUSSIAN_MERGING
**File:** `gaussian_merging.rs`
**Purpose:** Combine similar overlapping Gaussians

### Image-Specific

#### SPECULAR_DETECTION
**File:** `specular_detection.rs`
**Purpose:** Detect specular highlights

#### CONTENT_DETECTION
**File:** `content_detection.rs`
**Purpose:** Classify image type (smooth, sharp, mixed, high-frequency)

Output: ContentType enum
- Smooth (γ=0.6-0.8)
- Sharp (γ=0.5-0.6)
- Mixed (γ=0.6-0.7)
- HighFrequency (γ=0.4-0.5)

Used for auto-parameter selection.

---

## COMPLETE MODULE INDEX

### Analysis & Detection (23 modules)

| Module | File | Purpose |
|--------|------|---------|
| Structure Tensor | `structure_tensor.rs` | Edge orientation, coherence |
| Geodesic EDT | `geodesic_edt.rs` | Edge-respecting distance |
| Saliency Detection | `saliency_detection.rs` | Visual importance |
| Frangi Vesselness | `frangi_vesselness.rs` | Tubular structures |
| Text Stroke Detection | `text_stroke_detection.rs` | Thin linear strokes |
| Perceptual Weighting | `perceptual_weighting.rs` | Loss function weights |
| Feature Classifier | `feature_classifier.rs` | Per-pixel feature types |
| Artifact Detector | `artifact_detector.rs` | Encoding artifacts |
| Content Detection | `content_detection.rs` | Image type classification |
| Specular Detection | `specular_detection.rs` | Highlights |
| Image Analysis | `image_analysis.rs` | Comprehensive analysis |
| Laplacian | `laplacian.rs` | Edge/corner detection |
| Iterative Refinement | `iterative_refinement.rs` | Multi-pass encoding |
| Analytical Triggers | `analytical_triggers.rs` | Adaptive processing triggers |
| Entropy | `entropy.rs` | Complexity analysis |
| Gaussian Filter | `gaussian_filter.rs` | Smoothing kernels |
| Coherence Enhancing Diffusion | `coherence_enhancing_diffusion.rs` | Structure-aware filtering |
| Blue Noise Residual | `blue_noise_residual.rs` | High-freq detail encoding |
| Guided Filter | `guided_filter.rs` | Edge-preserving smoothing |
| Gradient Upscaling | `gradient_upscaling.rs` | High-res gradient maps |
| Poisson Solver | `poisson_solver.rs` | Boundary value problems |
| Position Probability Map | `position_probability_map.rs` | Multi-signal Gaussian placement |
| Predictive Coding | `predictive_coding.rs` | Residual prediction |

### Quality Metrics (8 modules)

| Module | File | Purpose |
|--------|------|---------|
| Error Metrics | `error_metrics.rs` | PSNR, MSE, MAE |
| MS-SSIM | `ms_ssim.rs` | Multi-scale structural similarity |
| MS-SSIM Loss | `ms_ssim_loss.rs` | Differentiable MS-SSIM |
| MS-SSIM Gradients | `ms_ssim_gradients.rs` | Analytical MS-SSIM derivatives |
| Edge-Weighted Loss | `edge_weighted_loss.rs` | Edge-emphasized loss |
| Rate-Distortion | `rate_distortion.rs` | R-D optimization |
| Loss Functions | (in encoder-v2) | L1, L2, SSIM variants |
| Entropy | `entropy.rs` | Complexity metrics |

### Rendering (6 modules)

| Module | File | Purpose |
|--------|------|---------|
| Renderer | `renderer.rs` | Alpha compositing, accumulated sum |
| EWA Splatting | `ewa_splatting.rs` | Elliptical weighted average |
| EWA Splatting V2 | `ewa_splatting_v2.rs` | Enhanced EWA with zoom |
| Textured Gaussian | `textured_gaussian.rs` | Per-primitive textures |
| Texture Map | `texture_map.rs` | Texture handling |
| LOD System | `lod_system.rs` | Level-of-detail |

### Gaussian Operations (9 modules)

| Module | File | Purpose |
|--------|------|---------|
| Adaptive Splitting | `adaptive_splitting.rs` | Error-driven splits |
| Gaussian Splitting | `gaussian_splitting.rs` | Divide along axis |
| Gaussian Merging | `gaussian_merging.rs` | Combine similar |
| Ordering | `ordering.rs` | Sort strategies |
| Importance Ordering | `importance_ordering.rs` | Progressive rendering order |
| Hilbert Ordering | `hilbert_ordering.rs` | Space-filling curve |
| Spatial Index | `spatial_index.rs` | BVH acceleration |
| Energy Based Selection | `energy_based_selection.rs` | Energy computation |
| Adaptive Densification | (in encoder-v2) | Split/clone/prune from 3D-GS |

### Compression & Format (7 modules)

| Module | File | Purpose |
|--------|------|---------|
| Quantization | `quantization.rs` | LGIQ profiles (B/S/H/X) |
| Vector Quantization | `vector_quantization.rs` | K-means codebook |
| Compression Utils | `compression_utils.rs` | Delta coding, zstd |
| Container Format | `container_format.rs` | File format specification |
| Progressive | `progressive.rs` | Progressive loading |
| Multi-Resolution | `multi_resolution.rs` | Pyramid representation |
| Tile Cache | `tile_cache.rs` | Spatial tiling for streaming |

### Spatial Processing (9 modules)

| Module | File | Purpose |
|--------|------|---------|
| Tiling | `tiling.rs` | Spatial tile operations |
| Viewport Culling | `viewport_culling.rs` | View frustum culling |
| Screen Space Culling | `screen_space_culling.rs` | Screen-space optimization |
| Selective Processing | `selective_processing.rs` | Region-based processing |
| Trigger Actions | `trigger_actions.rs` | Adaptive action triggers |

---

## RESEARCH-CRITICAL MODULES

These modules are central to the Dec 2025 toolkit research:

### Primary Toolkit (Isolated Tests Running)
1. structure_tensor - Edge analysis
2. geodesic_edt - Anti-bleeding constraints
3. saliency_detection - Importance weighting
4. frangi_vesselness - Tubular structures
5. text_stroke_detection - Text handling
6. perceptual_weighting - Gradient weighting

### Supporting Analysis
7. feature_classifier - Adaptive strategy selection
8. artifact_detector - Quality improvement
9. image_analysis - Comprehensive characterization
10. iterative_refinement - Multi-pass encoding

### Future Integration
11. content_detection - Auto-parameter selection
12. analytical_triggers - When to apply techniques

---

## SUMMARY STATISTICS

### Module Count by Category
- **Analysis & Detection:** 23 modules
- **Quality Metrics:** 8 modules
- **Rendering:** 6 modules
- **Gaussian Operations:** 9 modules
- **Compression & Format:** 7 modules
- **Spatial Processing:** 9 modules

**Total:** 62 modules in lgi-core

### Toolkit Research Status (Dec 2025)

| Module | Isolated Test | Optimization-Time Test | Status |
|--------|--------------|----------------------|--------|
| structure_tensor | ✓ Running | - | Active |
| geodesic_edt | ✓ Running | ✓ Running | Active |
| saliency_detection | ✓ Running | - | Active |
| frangi_vesselness | ✓ Running | - | Active |
| text_stroke_detection | ✓ Running | - | Active |
| perceptual_weighting | ✓ Running | ✓ Running | Active |

---

*End of Image Analysis Catalog - 2025-12-07*
