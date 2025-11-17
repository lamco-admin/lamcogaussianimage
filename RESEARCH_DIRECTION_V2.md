# Research Direction v2: Function-Based Gaussian Representation

**Date:** 2025-11-17
**Status:** Active Research - Phase 0
**Approach:** Understanding primitives through empirical observation

---

## Core Concept Shift

### NOT: Primitive Types (M/E/J/R/B/T)

**Problem with type-based approach:**
- Hard classification (is this Edge or Region?)
- Boundary artifacts (discontinuities at transitions)
- Tiling/segmentation incompatible with Gaussian overlap
- Chicken-and-egg (need to classify before representing)

### YES: Initialization Functions

**Function-based approach:**
- Each Gaussian's parameters determined by local image properties
- No discrete categories, continuous adaptation
- Compatible with Gaussian overlap (smooth transitions)
- No spatial partitioning required

---

## The Core Question

**Not:** "What are the primitive types?"

**But:** "Given local image properties, what Gaussian parameters should we use?"

### Example: Edge Function

```
f_edge: (image_properties) → (gaussian_parameters)

Inputs:
  - position on edge curve
  - local curvature κ(s)
  - edge blur σ_edge(s)
  - contrast ΔI(s)

Outputs:
  - σ_parallel (along-curve spread)
  - σ_perp (across-curve spread)
  - spacing (distance between Gaussians)
  - alpha (opacity)
  - orientation θ (= tangent)
```

**Research goal: Discover what f_edge should be through empirical observation.**

---

## Why This Avoids Tiling Problems

### Traditional Tiling:
- Hard spatial boundaries (this tile = Edge, that tile = Region)
- Discontinuous decisions at boundaries
- Artifacts where tiles meet
- **Incompatible with Gaussian overlap**

### Function-Based:
- Every Gaussian independently samples local properties
- Parameters change continuously across space
- Natural transitions (gradual parameter shifts)
- **Compatible with overlap** (neighboring Gaussians have similar parameters)

### Example Transition (Edge → Flat Region):

```
At edge:      anisotropy = 10  → σ_parallel = 10px, σ_perp = 1px
Near edge:    anisotropy = 3   → σ_parallel = 6px,  σ_perp = 2px
Far from edge: anisotropy = 1.1 → σ_parallel = 5px,  σ_perp = 4.5px
```

**No hard boundary. Smooth transition.**

---

## Reframing "Layers"

### Layers = Parameter Regimes, Not Spatial Regions

**Traditional view (problematic):**
- Layer E = spatial region containing edges
- Layer R = spatial region containing flat areas
- Must partition image into non-overlapping regions

**Function-based view (better):**
- "Layer E" = all Gaussians initialized by f_edge
- "Layer R" = all Gaussians initialized by f_region
- **Layers can spatially overlap** (same (x,y) can have both edge and region Gaussians)

**Key insight:** Layer membership is about WHICH FUNCTION initialized the Gaussian, not WHERE it is.

---

## Content-Adaptive Without Classification

### How to Apply Different Techniques Without Tiling:

Instead of classifying regions, use **continuous local properties:**

#### At edges:
- High gradient magnitude → place Gaussians
- Gradient direction → orientation θ
- Structure tensor eigenvalues → elongated shape (σ_parallel >> σ_perp)
- **Function:** f_edge determines parameters

#### In flat regions:
- Low gradient magnitude → sparse placement
- Isotropic structure → round shape (σ_parallel ≈ σ_perp)
- Large σ (fewer, bigger Gaussians)
- **Function:** f_region determines parameters

#### Transition zones:
- Medium gradient → moderate density
- Partial anisotropy → elliptical (not elongated, not round)
- **Natural interpolation between functions**

**No classification boundary. Properties vary smoothly.**

---

## Current Research Phase: Phase 0

### Goal: Understand ONE function deeply

**Focus:** Edge function f_edge

**Not yet concerned with:**
- How do multiple functions compose?
- What about optimization?
- What about layers vs monolithic?
- What about other functions (region, blob, texture)?

**Just:** Given an edge with known properties, what Gaussian parameters represent it well?

---

## Methodology: Empirical Discovery

### Not: Optimization-based learning

We're NOT trying to:
- Fit Gaussians with gradient descent
- Compare convergence rates
- Measure PSNR improvements

### Yes: Manual exploration and observation

We ARE trying to:
- Place Gaussians with explicit parameters
- Render and visually inspect
- Identify patterns (what works, what doesn't)
- Formulate simple rules

**Think: physics experiment, not machine learning**

---

## Success Criteria

### Phase 0 succeeds if we can write:

```
# Edge Gaussian Rules v1.0

For straight edges with blur σ_edge and contrast ΔI:

σ_perp = k1 × σ_edge        # where k1 ≈ ___
σ_parallel = k2 × σ_edge    # where k2 ≈ ___ (or constant?)
spacing = k3 × σ_parallel   # where k3 ≈ ___
alpha = k4 × ΔI             # where k4 ≈ ___

For curved edges, add:
spacing_adjustment = 1 + k5 × |κ|  # where k5 ≈ ___

Conditions where these rules work:
- σ_edge range: [0.5, 10] pixels
- ΔI range: [0.1, 1.0]
- Curvature |κ| < 0.5 (1/radius > 2 pixels)

Conditions where they fail:
- Very soft edges (σ_edge > 10px)
- Very high curvature (corners)
- Low contrast (ΔI < 0.1)
```

**This is UNDERSTANDING, not just performance numbers.**

---

## Long-Term Vision (Deferred)

Eventually, we want multiple functions:

### f_edge: For edge-like structures
- Input: curvature, blur, contrast
- Output: elongated, oriented Gaussians

### f_region: For flat areas
- Input: area, variance, gradient magnitude
- Output: large, isotropic Gaussians

### f_junction: For corner points
- Input: arm angles, sharpness
- Output: small cluster of Gaussians

### f_texture: For high-frequency detail
- Input: frequency spectrum, orientation
- Output: dense micro-Gaussians or parametric model

**But Phase 0 focuses only on f_edge.**

---

## Why Functions > Types

### Advantages:

1. **No hard classification** (avoid chicken-and-egg)
2. **Continuous transitions** (no boundary artifacts)
3. **Compatible with Gaussian overlap** (natural)
4. **Content-adaptive** (parameters from local properties)
5. **Interpretable** (can inspect and understand)
6. **Composable** (functions can coexist in same spatial location)

### Harmonization Question (Future):

When multiple functions suggest placing Gaussians at the same location:
- Do we place both? (additive)
- Does one take priority?
- Do they interfere?

**This is a future concern. Phase 0 tests only ONE function in isolation.**

---

## Relationship to Original LGI Spec

The `LGI_spec_and_experiments.md` document is:
- **Conceptual framework** (useful for big picture)
- **Hypothesis bank** (questions to eventually answer)
- **NOT a rigid specification** (will evolve based on findings)

The M/E/J/R/B/T primitive types were:
- **Initial attempt** at content adaptation
- **Borrowed from computer vision** (semantic features)
- **Problematic** (hard boundaries, classification issues)

Function-based approach addresses these problems while keeping the core insight:
**Different image structures need different Gaussian configurations.**

---

## Current Status

**Phase:** 0 (Fundamental understanding)
**Focus:** Edge function f_edge
**Method:** Manual parameter exploration
**Goal:** Empirical rules for edge Gaussian placement

**Next phases (future):**
- Phase 0.5: Test edge function generalization (curved, soft, etc.)
- Phase 1: Add region function, test composition
- Phase 2: Optimization as refinement
- Phase 3: Multiple functions, harmonization

**Timeline:** Phase 0 must succeed before moving forward. No rushing.

---

**Document Status:** Living document, will evolve based on experimental findings.
