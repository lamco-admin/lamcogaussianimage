# THEORETICAL FOUNDATIONS - Gaussian Image Codec
## Core Concepts, Definitions, and Mathematical Framework

**Document Version:** 2.0
**Date:** 2025-12-07
**Extracted From:**
- GAUSSIAN_GEOMETRY_DETAILS.md (Nov 15)
- LGI_spec_and_experiments.md (Nov 15)
- GROUND_TRUTH.md (Dec 5)

---

## TABLE OF CONTENTS

1. [Gaussian Definition](#gaussian-definition)
2. [Parameter Space](#parameter-space)
3. [Parameterizations](#parameterizations)
4. [Layered Representation Concept](#layered-representation-concept)
5. [Coverage and Density](#coverage-and-density)
6. [Fundamental Principles](#fundamental-principles)

---

## GAUSSIAN DEFINITION

### 2D Elliptical Gaussian

Each Gaussian is defined by:

```
G(x, y) = opacity × exp(-0.5 × d^T × Σ⁻¹ × d) × color
```

Where:
- `d = [x - μ_x, y - μ_y]` = distance from center position
- `Σ` = 2×2 covariance matrix (defines ellipse shape and orientation)
- `opacity` ∈ [0, 1] = transparency for alpha blending
- `color` = (R, G, B) ∈ [0, 1]³

### Image Formation

Multiple Gaussians combine to form an image:

```
I(x, y) = Σᵢ (colorᵢ × αᵢ × G(x,y | μᵢ, Σᵢ))
```

Two blending modes:

**Alpha Compositing (physically-based):**
```
for each Gaussian (back to front):
    α_acc ← α_acc + (1 - α_acc) × αᵢ × Gᵢ(x,y)
    C_acc ← C_acc + (1 - α_acc) × cᵢ × αᵢ × Gᵢ(x,y)
```

**Accumulated Sum (GaussianImage ECCV 2024):**
```
C(x,y) = Σ(cᵢ × αᵢ × Gᵢ(x,y)) / Σ(αᵢ × Gᵢ(x,y))
```

---

## PARAMETER SPACE

### Intrinsic Parameters (9 per Gaussian)

| Parameter | Dimensions | Range | Type |
|-----------|------------|-------|------|
| **Position** | 2 (x, y) | [0, 1] normalized | Spatial |
| **Scale** | 2 (σ_x, σ_y) | > 0 | Geometric |
| **Rotation** | 1 (θ) | radians | Geometric |
| **Color** | 3 (R, G, B) | [0, 1] | Appearance |
| **Opacity** | 1 (α) | [0, 1] | Appearance |

**Total: 9 degrees of freedom**

### Optional Parameters

| Parameter | Type | Purpose |
|-----------|------|---------|
| Weight | f32 | Progressive rendering priority |
| Texture | Image | Per-primitive detail maps |
| LOD Band | u8 | Level-of-detail classification |

---

## PARAMETERIZATIONS

Four equivalent ways to represent the covariance matrix Σ:

### 1. Euler (Current Default)

**Parameters:** (σ_x, σ_y, θ)
- σ_x: Scale along major axis
- σ_y: Scale along minor axis
- θ: Rotation angle (radians)

**Advantages:**
- Most intuitive (can see ellipse dimensions)
- Easy to constrain (clamp scales independently)
- Simple initialization (isotropic = σ_x = σ_y)

**Code:** `lgi-math/src/parameterization.rs` - Euler struct

---

### 2. Cholesky

**Parameters:** L where Σ = L·L^T (lower triangular)
- L₁₁, L₂₁, L₂₂ (3 values)

**Advantages:**
- Numerically stable
- Guarantees positive definite Σ
- Standard in research literature (log-Cholesky)

**Code:** `lgi-math/src/parameterization.rs` - Cholesky struct

---

### 3. LogRadius

**Parameters:** (log r, eccentricity, θ)
- log r: Logarithm of geometric mean radius
- eccentricity: Ratio of axes
- θ: Rotation

**Advantages:**
- Better dynamic range for quantization
- Logarithmic scale encoding natural

**Code:** `lgi-math/src/parameterization.rs` - LogRadius struct

**Status:** Specified in format, not heavily used yet

---

### 4. InverseCovariance

**Parameters:** Σ⁻¹ elements directly (3 values for 2D)

**Advantages:**
- No matrix inversion at render time
- Fastest for GPU shaders

**Code:** `lgi-math/src/parameterization.rs` - InverseCovariance struct

**Usage:** GPU shader format

---

## LAYERED REPRESENTATION CONCEPT

**Source:** LGI_spec_and_experiments.md
**Concept:** Semantic feature decomposition

### Feature Types (Hypothetical Layers)

| Layer | Symbol | Feature Type | Gaussian Characteristics |
|-------|--------|--------------|-------------------------|
| Macro | M | Global low-frequency field | Large, sparse, overlapping |
| Edges | E | Curves, boundaries | Anisotropic, aligned, dense along curves |
| Junctions | J | Corners, intersections | Small clusters, isotropic |
| Regions | R | Homogeneous interiors | Medium, uniform coverage |
| Blobs | B | Spots, keypoints | Small, isotropic |
| Texture | T | Fine microstructure | Very small or procedural |

### Layered Hypothesis (H5 from spec)

> "Given the same total Gaussian budget K_total, a **layered residual model**:
>  M → E → J → R → B → T
>  will outperform a **single-layer Gaussian model** in PSNR/SSIM"

### Relation to Toolkit Approach

The toolkit philosophy aligns with layered representation:
- Each toolkit module (structure tensor, text stroke, etc.) detects a feature type
- Modules can be composed to build layered strategies
- **Constraint:** No geometric tiling - semantic layers only

**Status:** Theoretical framework, not yet fully implemented

---

## COVERAGE AND DENSITY

### Gaussian Coverage

For an image of dimensions W × H with N Gaussians:

**Pixels per Gaussian:**
```
pixels_per_gaussian = (W × H) / N
```

**Gaussian size for coverage:**
```
σ = γ × √(pixels_per_gaussian)
  = γ × √((W × H) / N)
```

Where γ is the coverage parameter (typical: 0.5-0.8)

### Coverage Parameter γ

| γ | Coverage | Overlap | Use Case |
|---|----------|---------|----------|
| 0.3 | Sparse | Minimal | High-frequency detail |
| 0.5 | Standard | Moderate | Balanced |
| 0.7 | Dense | High | Smooth regions |
| 1.0 | Full | Very high | Background fill |

**Formula in code:** (from lib.rs)
```rust
let sigma_base_px = gamma * ((width_px * height_px) / n as f32).sqrt();
```

### Adaptive Gamma

```rust
fn adaptive_gamma(n: usize) -> f32 {
    // More Gaussians → need smaller coverage
    // Empirically tuned
    (0.8 - 0.2 * (n as f32 / 1000.0).min(1.0)).max(0.4)
}
```

Range: 0.8 (N=0) → 0.4 (N≥1000)

---

## FUNDAMENTAL PRINCIPLES

**From GROUND_TRUTH.md and RESEARCH_PHILOSOPHY.md**

### Principle 1: Layers, Never Tiles

**CRITICAL CONSTRAINT:**
- NO tiling of image geometry
- NO patch-based processing
- NO 2D spatial segmentation

**Rationale:**
- Tiling introduces boundary artifacts
- Loses global coherence
- Gaussians are global primitives

**Acceptable:**
- Semantic layers (edges, textures, smooth)
- Feature-based separation
- LOD bands (not spatial tiles)

### Principle 2: Toolkit, Not Single Optimum

**Research Approach:**
- Build composable tools (structure tensor, geodesic EDT, etc.)
- Test each in isolation
- Understand contributions separately
- Compose based on image content

**Configuration Space:**
```
Initialization:   14+ strategies
Optimizers:       9 algorithms
Renderers:        7 implementations
Loss functions:   3+ options
Parameterizations: 4 options
───────────────────────────────
Combinations:     960,000+
```

**Goal:** NOT finding "optimal configuration"
**Goal:** Building toolkit for adaptive encoding

### Principle 3: Optimization Toolkit

From GROUND_TRUTH.md:

> "Adam is what we've learned to use, NOT the mathematical answer. We need a toolkit of well-understood optimization methods."

Each optimizer (Adam, L-BFGS, L-M) needs:
- Its own hyperparameter tuning
- Its own initialization strategy
- Understanding of when it excels

**Status:** Active research

---

## GAUSSIAN PARAMETER DETAILS

**From GAUSSIAN_GEOMETRY_DETAILS.md**

### Position (2 parameters)
- **Range:** [0, 1] normalized coordinates
- **Initialization:** Grid points or feature-based sampling
- **Constraints:** Clamped to [0, 1]
- **Update:** Continuous within bounds

### Shape (3 parameters)
- **scale_x, scale_y:** Half-widths along principal axes
- **rotation:** Orientation in radians
- **Initialization:**
  - Isotropic: scale_x = scale_y = 0.01-0.1
  - rotation = 0.0 (axis-aligned)
- **Constraints:**
  - Scales: [0.001, 0.5] (prevents collapse and oversizing)
  - Rotation: Unconstrained

**Note:** Gaussians start isotropic (circles) and become elliptical through optimization

### Color (3 parameters)
- **Range:** [0, 1] per channel (linear RGB)
- **Initialization:** Sampled from target image or random
- **Constraints:** Clamped to [0, 1]

### Opacity (1 parameter)
- **Range:** [0, 1]
- **Initialization:** Typically 1.0 (fully opaque)
- **Optimization:** Often fixed or rarely updated

---

## COVERAGE ANALYSIS

**From GAUSSIAN_GEOMETRY_DETAILS.md**

### N-to-Pixels Ratios

For 64×64 = 4,096 pixel image:

| N | Pixels/Gaussian | Coverage % | Gaussian Diameter |
|---|-----------------|------------|-------------------|
| 20 | 204.8 | 0.488% | ~5.0 px |
| 50 | 81.9 | 1.221% | ~3.1 px |
| 100 | 41.0 | 2.441% | ~2.2 px |
| 256 | 16.0 | 6.25% | ~1.4 px |
| 512 | 8.0 | 12.5% | ~1.0 px |

**Interpretation:**
- Sparse (N=20): Each Gaussian covers ~205 pixels
- Dense (N=512): Each Gaussian covers ~8 pixels
- Very dense (N>1000): Sub-pixel Gaussians possible

### Scaling Laws

For image dimensions W × H:

```
Pixel coverage = (W × H) / N
Gaussian diameter ≈ γ × √(pixel_coverage)
```

With typical γ = 0.5:
- 256×256 image, N=256: diameter ≈ 8 pixels
- 512×512 image, N=1024: diameter ≈ 11 pixels
- 768×512 image (Kodak), N=256: diameter ≈ 27 pixels

---

## SUMMARY

### Core Definitions Established

- **9 parameters per Gaussian** (position, scale, rotation, color, opacity)
- **4 equivalent parameterizations** (Euler, Cholesky, LogRadius, InvCov)
- **2 blending modes** (AlphaComposite, AccumulatedSum)
- **Coverage formula:** σ = γ√(area/N)

### Fundamental Constraints

1. **No geometric tiling** (layers never tiles)
2. **Toolkit approach** (not optimal configuration)
3. **Optimization diversity** (multiple methods, each with strengths)

### Theoretical Framework

- Layered semantic decomposition (M/E/J/R/B/T)
- Feature-specific Gaussian strategies
- Systematic micro-level testing
- 960,000+ configuration space

**Status:** Framework established, implementation ongoing

---

*End of Theoretical Foundations - 2025-12-07*
