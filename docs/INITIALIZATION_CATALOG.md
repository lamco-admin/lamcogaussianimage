# INITIALIZATION CATALOG - Gaussian Image Codec
## Complete Gaussian Placement Strategies Inventory

**Document Version:** 2.0
**Date:** 2025-12-07
**Repository:** lgi-project
**Total Lines:** 1,700 lines of initialization code

---

## TABLE OF CONTENTS

1. [Overview](#overview)
2. [Core Strategies (lgi-core)](#core-strategies)
3. [Enhanced Strategies (lgi-encoder-v2)](#enhanced-strategies)
4. [Encoder-Integrated Methods](#encoder-integrated-methods)
5. [File Location Index](#file-location-index)

---

## OVERVIEW

Initialization strategies determine **where to place Gaussians** before optimization begins. This is distinct from optimization (how to refine parameters).

### Why Initialization Matters

From RESEARCH_PHILOSOPHY.md:
- 960,000+ possible configurations
- Initialization is one axis of this space
- Different images may need different strategies
- Building toolkit, not finding single optimal

### Strategy Categories

1. **Uniform Coverage** - Grid, random (baseline)
2. **Feature-Based** - Gradient peaks, SLIC, k-means
3. **Sparse** - GDGS (Laplacian peaks)
4. **Content-Adaptive** - Preprocessing maps, hybrid counts

---

## CORE STRATEGIES

**File:** `lgi-core/src/initializer.rs`
**Lines:** 656
**Purpose:** Base initialization methods

### 1. RANDOM

**Function:** Part of Initializer enum
**Strategy:** Uniform random placement

#### Description
Places Gaussians at uniformly random (x, y) positions.

#### Use Case
Baseline comparison

#### Characteristics
- No structure awareness
- Stochastic (different each run unless seeded)
- Fastest to compute

---

### 2. GRID

**Function:** Part of Initializer enum
**Strategy:** Regular √N × √N grid

#### Description
Places Gaussians on regular grid with uniform spacing.

#### Characteristics
- Deterministic
- Uniform coverage
- Good baseline for optimization

#### Grid Computation
For N Gaussians:
- cols = ⌈√N⌉
- rows = ⌈N / cols⌉
- spacing_x = width / cols
- spacing_y = height / rows

---

### 3. GRADIENT

**Function:** Part of Initializer enum
**Strategy:** Edge-aware placement (Sobel-based)

#### Description
Places Gaussians preferentially at high-gradient (edge) regions.

#### Use Case
Sharp images, text, boundaries

#### Characteristics
- Emphasizes edges
- May under-represent smooth regions
- Structure-aware

---

### 4. IMPORTANCE

**Function:** Part of Initializer enum
**Strategy:** Variance-based sampling

#### Description
Samples based on pixel variance within local windows.

#### Use Case
Complex content with varying detail levels

#### Characteristics
- Adapts to local complexity
- Balanced coverage

---

### 5. SLIC

**Function:** Part of Initializer enum
**Strategy:** Superpixel segmentation

#### Description
Uses SLIC (Simple Linear Iterative Clustering) to create superpixels, places Gaussian at each superpixel center.

**Also:** `lgi-encoder-v2/src/slic_init.rs` (139 lines)
Enhanced implementation with additional features.

#### Use Case
Natural photos with coherent regions

#### Characteristics
- Content-adaptive boundaries
- Color + spatial coherence
- Good for segmented content

---

## ENHANCED STRATEGIES

**Location:** `lgi-encoder-v2/src/`
**Total Lines:** 1,044 (5 modules)

### 6. K-MEANS CLUSTERING

**File:** `kmeans_init.rs`
**Lines:** 312
**Strategy Code:** D (from research classification)

#### Description
Clusters pixels by 6D features: (R, G, B, x, y, gradient_mag)
Places Gaussian at each cluster centroid.

#### Configuration

```rust
pub struct KMeansConfig {
    pub k: usize,                  // Number of clusters
    pub max_iterations: usize,     // Default: 50
    pub weight_color: f32,         // Default: 1.0
    pub weight_position: f32,      // Default: 0.5
    pub weight_gradient: f32,      // Default: 0.3
}
```

#### Feature Vector
6 dimensions:
- RGB color (3)
- XY position (2)
- Gradient magnitude (1)

#### Use Case
Classic CV approach, well-proven

---

### 7. GDGS (Gradient Domain)

**File:** `gdgs_init.rs`
**Lines:** 206
**Strategy Code:** F (from research classification)

#### Description
Places Gaussians ONLY at Laplacian peaks (edges, corners).

#### Algorithm
1. Compute Laplacian (∇²I)
2. Find local maxima (peaks)
3. Place Gaussian at each peak
4. Orient along gradient direction
5. (Optional) Poisson reconstruction

#### Configuration

```rust
pub struct GDGSConfig {
    pub lap_threshold: f32,        // Peak detection (default: 0.01)
    pub peak_window_size: usize,   // Local maxima window (default: 5)
    pub max_gaussians: usize,      // Budget limit (default: 500)
    pub base_sigma: f32,           // Base scale (default: 0.02)
}
```

#### Research Claim (from code comments)
"10-100× fewer Gaussians for same quality"

#### Paper Reference
arXiv:2405.05446v1 "Gradient Domain Gaussian Splatting"

---

### 8. GRADIENT PEAK

**File:** `gradient_peak_init.rs`
**Lines:** 283
**Strategy Code:** H (from research classification)

#### Description
Hybrid approach combining gradient peaks with sparse background grid.

#### Algorithm
- 80% Gaussians at gradient maxima (edges, details)
- 20% Gaussians on background grid (smooth regions coverage)

#### Advantage over GDGS (from code comments)
"Better than GDGS (doesn't neglect smooth regions)"

#### Configuration

```rust
pub struct GradientPeakConfig {
    pub target_peak_gaussians: usize,  // 80% of total
    pub background_grid_size: u32,     // √(20% of total)
    // ... additional config
}
```

---

### 9. SLIC SUPERPIXEL (Enhanced)

**File:** `slic_init.rs`
**Lines:** 139
**Strategy Code:** E (from research classification)

#### Description
Enhanced SLIC implementation with additional Gaussian placement features.

#### Features
- Superpixel boundary detection
- Per-superpixel statistics
- Color and spatial coherence

---

### 10. BETTER COLOR INITIALIZATION

**File:** `better_color_init.rs`
**Lines:** 104
**Strategy Code:** O (from research classification)

#### Description
Improved color sampling using Gaussian-weighted averaging rather than nearest-pixel sampling.

#### Method
For each Gaussian position:
- Compute weighted average of nearby pixels
- Weight by Gaussian kernel
- Results in smoother color initialization

---

## ENCODER-INTEGRATED METHODS

These methods are part of EncoderV2 (lib.rs) and use preprocessing:

### 11. STRUCTURE-TENSOR-AWARE GRID

**Function:** `EncoderV2::initialize_gaussians(grid_size)`
**Line:** lib.rs:85-86
**Purpose:** Grid initialization with structure tensor shapes

#### Description
Regular grid placement but with:
- Anisotropic shapes aligned to structure tensor
- Scales modulated by coherence
- Geodesic EDT clamping

#### Shape Selection
- Low coherence (<0.2): Isotropic
- High coherence: Anisotropic aligned to edge (σ∥ = 4·σ⊥)

Used by most error-driven encoders.

---

### 12. GUIDED FILTER COLORS

**Function:** `EncoderV2::initialize_gaussians_guided(grid_size)`
**Line:** lib.rs:90-92
**Purpose:** Grid with guided filter color initialization

Similar to #11 but uses guided filter for better color sampling.

---

### 13. AUTO/HYBRID COUNT SELECTION

**Functions:**
- `EncoderV2::auto_gaussian_count()` - Line 98-102
- `EncoderV2::hybrid_gaussian_count()` - Line 116-151

#### Auto (Entropy-Based)
Uses `lgi_core::entropy::adaptive_gaussian_count()` to select N based on image complexity.

#### Hybrid (60% Entropy + 40% Gradient)
Combines:
- Base: Entropy-driven N
- Adjustment: Mean gradient magnitude
- Formula: `N = N_entropy × (0.6 + 0.4 × gradient_factor)`
- Range: 50 to 50,000 Gaussians

---

## PLACEMENT STRATEGIES TABLE

| Strategy | File | Lines | Basis | Coverage Type |
|----------|------|-------|-------|---------------|
| Random | `initializer.rs` | (part of 656) | Uniform random | Stochastic |
| Grid | `initializer.rs` | (part of 656) | Regular grid | Uniform |
| Gradient | `initializer.rs` | (part of 656) | Sobel edges | Edge-biased |
| Importance | `initializer.rs` | (part of 656) | Pixel variance | Complexity-adaptive |
| SLIC | `initializer.rs` + `slic_init.rs` | 656 + 139 | Superpixels | Content-adaptive |
| K-Means | `kmeans_init.rs` | 312 | 6D clustering | Feature-based |
| GDGS | `gdgs_init.rs` | 206 | Laplacian peaks | Sparse |
| Gradient Peak | `gradient_peak_init.rs` | 283 | Peaks + background | Hybrid |
| ST-aware Grid | `lib.rs` (EncoderV2) | (part of 1,778) | Grid + structure tensor | Structured uniform |
| Guided Colors | `lib.rs` (EncoderV2) | (part of 1,778) | Grid + guided filter | Structured uniform |
| Better Colors | `better_color_init.rs` | 104 | Gaussian-weighted colors | Color-improved |

---

## GAUSSIAN COUNT SELECTION

### Entropy-Based (Auto)
**Function:** `lgi_core::entropy::adaptive_gaussian_count(image)`
**Basis:** Shannon entropy of image regions

Analysis: Tiles image into blocks, computes entropy, maps to N

### Hybrid (Entropy + Gradient)
**Function:** `EncoderV2::hybrid_gaussian_count()`
**Lines:** 116-151

Formula:
```
N_hybrid = N_entropy × (0.6 + 0.4 × gradient_factor)
```

Where:
- N_entropy: From entropy analysis
- gradient_factor: Mean gradient magnitude normalized [0,1]
- Weights: 60% complexity, 40% edge detail

---

## PREPROCESSING-BASED STRATEGIES

### 14. PREPROCESSING LOADER

**File:** `preprocessing_loader.rs`
**Lines:** (in encoder-v2)
**Purpose:** Load Python-generated placement maps

#### Maps Loaded (8 total)
From Python preprocessing pipeline:
1. Entropy map
2. Gradient map
3. Texture map
4. SLIC segments
5. Saliency map
6. Distance transform
7. Skeleton
8. Position probability map

#### Integration
Weighted combination into final probability map for Gaussian placement.

---

## SUMMARY STATISTICS

### Implementation Count
- **Core Strategies:** 5 (in initializer.rs)
- **Enhanced Strategies:** 6 (encoder-v2 modules)
- **Encoder Methods:** 3 (lib.rs integrated)
- **Total:** 14 distinct initialization approaches

### Line Count
- **Core (lgi-core):** 656 lines
- **Enhanced (encoder-v2):** 1,044 lines
- **Total:** 1,700 lines

### Feature Dimensions Used

| Strategy | Dimensions | Features |
|----------|------------|----------|
| Random | 0 | None (uniform) |
| Grid | 0 | None (uniform) |
| Gradient | 1 | Gradient magnitude |
| Importance | 1 | Local variance |
| SLIC | 5 | Color (3) + position (2) |
| K-Means | 6 | Color (3) + position (2) + gradient (1) |
| GDGS | 1 | Laplacian magnitude |
| Gradient Peak | 1 | Gradient magnitude |
| ST-aware Grid | 4 | Coherence + eigenvalues + eigenvectors |
| Preprocessing | 8 | All preprocessing maps |

---

## FILE LOCATION INDEX

### Core Module
```
packages/lgi-rs/lgi-core/src/
└── initializer.rs (656 lines)
    ├── InitStrategy enum
    ├── Random placement
    ├── Grid placement
    ├── Gradient placement
    ├── Importance sampling
    └── SLIC placement
```

### Enhanced Modules
```
packages/lgi-rs/lgi-encoder-v2/src/
├── kmeans_init.rs (312 lines) - K-means clustering
├── gdgs_init.rs (206 lines) - GDGS Laplacian peaks
├── gradient_peak_init.rs (283 lines) - Peaks + background
├── slic_init.rs (139 lines) - Enhanced SLIC
├── better_color_init.rs (104 lines) - Improved color sampling
└── preprocessing_loader.rs - Python map integration
```

### Encoder Integration
```
packages/lgi-rs/lgi-encoder-v2/src/
└── lib.rs (1,778 lines)
    ├── initialize_gaussians() - ST-aware grid
    ├── initialize_gaussians_guided() - Grid + guided filter
    ├── initialize_gdgs() - Wrapper for GDGS
    ├── initialize_gradient_peaks() - Wrapper for gradient peaks
    ├── auto_gaussian_count() - Entropy-based N
    └── hybrid_gaussian_count() - Entropy + gradient N
```

---

*End of Initialization Catalog - 2025-12-07*
