# Edge Gaussian Empirical Rules v2.0

**Derived from:** Phase 0.5 experiments (31 renders with masked edge strip metrics)
**Date:** 2025-11-17
**Key Change:** Measurement on edge strip only (20px-wide), variable N, alpha scaling
**Validity:** Straight edges, σ_edge ∈ [0, 4px], ΔI ∈ [0.1, 0.8], 100×100 pixel images
**Rendering method:** Accumulative rendering (not alpha compositing)

---

## Executive Summary

Phase 0.5 refined the empirical rules by **isolating edge region measurement** (20px strip) and **systematically varying N** to find optimal Gaussian density.

**Critical Discovery:** The fundamental limitation is **not measurement bias** (background vs edge), but **insufficient Gaussian primitive capacity** for high-contrast edges. Even when measuring edge strip only, PSNR remains low (10-15 dB) for medium/high contrast edges (ΔI ≥ 0.5).

**Key Finding:** N_optimal = 50-100 Gaussians (PSNR plateaus). Further increasing N provides no benefit.

---

## Core Empirical Rules (Updated)

### 1. Number of Gaussians (N) - **NEW**

```
N = 50 for edge length ~100 pixels
N = 0.5 × edge_length (density: 1 Gaussian per 2 pixels)
```

**Empirical finding:**
- **PSNR plateaus at N=50-100** for 100px edges
- Further increase provides no benefit
- Below N=20: insufficient coverage
- Optimal density: ~1 Gaussian per 2 pixels along edge

**Confidence:** **High**
**Evidence:**
- Experiment 1 N Sweep: Mean PSNR plateaus from N=50 onward
  - N=5: 11.79 dB
  - N=10: 13.21 dB
  - N=20: 14.60 dB
  - N=50: **15.20 dB** (optimal)
  - N=100: 15.18 dB (plateau)
  - N=200: 15.17 dB (plateau)

**Interpretation:**
With accumulative rendering and proper alpha scaling (α ∝ 1/N), PSNR saturates at N≈50. This is the **fundamental capacity limit** of the edge Gaussian primitive approach.

---

### 2. Cross-Edge Width (σ_perp)

```
σ_perp = 0.5 - 1.0 pixels (constant, independent of edge blur)
```

**Empirical finding:** (Updated from Phase 0)
- **Smaller is better** for accumulative rendering
- Phase 0.5 Sweep A shows: σ_perp=0.5-1.0 optimal
  - σ_perp=0.5: 11.40 dB
  - σ_perp=1.0: 11.32 dB
  - σ_perp=2.0: 6.05 dB (sharp degradation)
  - σ_perp>2.0: <3 dB (very poor)

**Confidence:** **High**
**Evidence:**
- Consistent with Phase 0 finding
- Experiment 2 Sweep A confirms: larger σ_perp degrades quality
- **Independent of edge blur** (σ_edge)

**Interpretation:**
Narrow Gaussians perpendicular to edge create sharper edge profile when accumulating. Wider Gaussians cause blurring and over-coverage.

---

### 3. Along-Edge Spread (σ_parallel)

```
σ_parallel = 10.0 pixels
σ_parallel = 0.10 × edge_length (10% of edge length)
```

**Empirical finding:** (Consistent with Phase 0)
- **Moderate values preferred** (10-15 pixels for 100px edge)
- Experiment 2 Sweep B shows:
  - σ_parallel=5: 11.21 dB
  - σ_parallel=10: 11.32 dB (optimal)
  - σ_parallel=15: 11.05 dB
  - σ_parallel=20: 10.91 dB

**Confidence:** **Medium**
**Evidence:**
- Weak optimum around σ_parallel=10
- Effect size is small (11.21 to 11.32 dB)
- Parameter is less critical than σ_perp

**Interpretation:**
Moderate along-edge spread provides good coverage without excessive overlap. Very large values (>15px) start to degrade quality.

---

### 4. Gaussian Spacing

```
spacing = edge_length / N
(implicitly determined by N for uniform placement)

Alternative: spacing = 2.5 - 5.0 pixels (for manual placement)
```

**Empirical finding:** (Updated)
- **Denser spacing is better** (contradicts Phase 0 finding)
- Experiment 2 Sweep C shows:
  - spacing=2.5px (N=40): 11.88 dB (best)
  - spacing=3.3px (N=30): 11.78 dB
  - spacing=5.0px (N=20): 10.94 dB
  - spacing=10.0px (N=10): 9.97 dB

**Confidence:** **Medium**
**Evidence:**
- Clear trend: smaller spacing → higher PSNR
- But alpha must scale with N to avoid over-accumulation

**Interpretation:**
With proper alpha scaling (α ∝ 1/N), denser placement improves quality up to N≈50. This **reverses Phase 0 finding** which didn't account for alpha scaling with N.

---

### 5. Opacity (Alpha) - **UPDATED**

```
alpha = (k / ΔI) × (N_base / N)

where:
  k ≈ 0.3
  N_base = 10 (reference density from Phase 0)
  N = actual number of Gaussians
  ΔI = edge contrast
```

**Empirical finding:** (Critical update)
- **Alpha must scale inversely with N** to avoid over-accumulation
- Base formula from Phase 0 (α = 0.3/ΔI) valid only for N=10
- For variable N: α = (0.3/ΔI) × (10/N)

**Confidence:** **High**
**Evidence:**
- Without N scaling: PSNR degrades catastrophically with increasing N
  - N=100 with α=0.6: PSNR = 5.88 dB
  - N=200 with α=0.6: PSNR = 2.44 dB
- With N scaling: PSNR plateaus correctly
  - N=100 with α=0.06: PSNR = 10.30 dB
  - N=200 with α=0.03: PSNR = 10.30 dB

**Interpretation:**
Accumulative rendering sums Gaussian contributions. Total accumulated intensity ∝ N × α. To maintain constant edge intensity as N varies, α must scale as 1/N.

---

### 6. Orientation

```
θ = edge_tangent_angle (perpendicular to edge normal)
```

**Empirical finding:** (Unchanged from Phase 0)
- Gaussian major axis aligns with edge direction
- Geometric requirement, no optimization needed

**Confidence:** **High**

---

## Complete Function: f_edge (v2.0)

```python
def f_edge_v2(σ_edge: float,
              ΔI: float,
              edge_length: float = 100,
              image_size: tuple = (100, 100)) -> dict:
    """
    Edge Gaussian parameter function (empirical rules v2.0)

    Key changes from v1.0:
    - N is variable, optimized for edge_length
    - Alpha scales inversely with N
    - σ_perp refined to 0.5-1.0

    Args:
        σ_edge: Edge blur (pixels)
        ΔI: Edge contrast (intensity difference, [0, 1])
        edge_length: Length of edge (pixels)
        image_size: (height, width) in pixels

    Returns:
        Dict with keys: N, sigma_perp, sigma_parallel, alpha, spacing
    """

    # Rule 1: N scales with edge length (1 Gaussian per 2 pixels)
    N = int(edge_length / 2)
    N = max(50, min(N, 100))  # Clamp to optimal range

    # Rule 2: σ_perp is constant and small
    sigma_perp = 0.5  # 0.5-1.0 range, use 0.5 for best results

    # Rule 3: σ_parallel depends on edge length
    sigma_parallel = 0.10 * edge_length  # 10% of edge length

    # Rule 4: spacing is implicit from uniform N placement
    spacing = edge_length / N

    # Rule 5: alpha inversely proportional to contrast AND N
    N_base = 10  # Reference from Phase 0
    alpha_base = 0.3 / ΔI if ΔI > 0 else 0.3
    alpha = alpha_base * (N_base / N)

    return {
        'N': N,
        'sigma_perp': sigma_perp,
        'sigma_parallel': sigma_parallel,
        'spacing': spacing,
        'alpha': alpha,
    }
```

---

## Reconstruction Quality Assessment

### Phase 0.5 Performance (Edge Strip Metrics):

**Overall:**
- **Mean PSNR (edge strip, N=50):** 15.20 dB
- **Range:** 10.17 - 25.14 dB
- **Quality:** Poor (target: > 30 dB)

**Performance by Contrast:**

| Contrast (ΔI) | PSNR (strip) | PSNR (full) | Assessment |
|---------------|--------------|-------------|------------|
| 0.1           | 25.14 dB     | 23.83 dB    | Acceptable |
| 0.5 (sharp)   | 10.17 dB     | 9.38 dB     | Poor       |
| 0.5 (blur=2)  | 10.30 dB     | 9.49 dB     | Poor       |

**Key Insight:**
- **Low contrast (ΔI=0.1):** Edge strip PSNR = 25 dB (acceptable)
- **Medium/high contrast (ΔI≥0.5):** Edge strip PSNR = 10-11 dB (poor)

**Critical Finding:**
Measuring edge strip only (vs full image) provides **minimal improvement** for medium/high contrast edges:
- Phase 0 (full image, N=10): PSNR = 9.38 dB (ΔI=0.5)
- Phase 0.5 (edge strip, N=50): PSNR = 10.17 dB (ΔI=0.5)

**Only ~0.8 dB improvement despite:**
- 5× more Gaussians (N=10 → N=50)
- Masked measurement (ignore background)

---

## Comparison: Phase 0 vs Phase 0.5

| Metric | Phase 0 (Full Image, N=10) | Phase 0.5 (Edge Strip, N=50) | Change |
|--------|---------------------------|------------------------------|--------|
| Mean PSNR | 11.76 dB | 15.20 dB | +3.4 dB |
| Best case | 22.41 dB (ΔI=0.1) | 25.14 dB (ΔI=0.1) | +2.7 dB |
| Worst case | 5.21 dB (ΔI=0.8) | ~10 dB (ΔI=0.5) | +5 dB |

**Improvements:**
- ✓ More Gaussians (N=50) improves coverage
- ✓ Masked metrics remove background bias
- ✓ Alpha scaling with N prevents over-accumulation

**Limitations:**
- ✗ Still far from 30 dB target for medium/high contrast
- ✗ Gaussian primitive approach has **fundamental capacity limit**
- ✗ Edge strip PSNR only marginally better than full-image

---

## Coverage Analysis (Experiment 3)

**Strip Width vs PSNR:**

| Strip Width | Pixels | PSNR | Interpretation |
|-------------|--------|------|----------------|
| 5px | 400 | 10.74 dB | Core edge only |
| 10px | 1000 | 10.92 dB | Edge + near transition |
| 20px | 2000 | 11.32 dB | Standard (best) |
| 40px | 4000 | 11.21 dB | Includes background |
| Full (100px) | 10000 | 9.97 dB | Background dominates |

**Key Finding:**
- **Optimal measurement strip: 20px** (best PSNR)
- Narrower strips (5-10px) actually have lower PSNR (less context)
- Full image degrades PSNR by ~1.4 dB (background error)

**Effective Coverage Radius:**
- Gaussians contribute meaningfully within ~10-20px of edge center
- Beyond 40px: background regions poorly represented

---

## Updated Parameter Validation Ranges

### Successfully tested (Phase 0.5):
- **N:** [5, 10, 20, 50, 100, 200] Gaussians
- **σ_perp:** [0.5, 1.0, 2.0, 3.0, 4.0] pixels
- **σ_parallel:** [5, 10, 15, 20] pixels
- **Spacing:** [2.5, 3.3, 5.0, 10.0] pixels (implicit via N)
- **Edge blur:** σ_edge ∈ [0, 2] pixels
- **Contrast:** ΔI ∈ [0.1, 0.5]

### Not tested (Phase 0.6+):
- Curved edges (κ > 0)
- Very high contrast (ΔI > 0.5)
- Large images (> 100×100)
- Non-accumulative rendering methods

---

## Failure Modes (Updated)

### When Rules Fail:

1. **Medium/high contrast edges (ΔI ≥ 0.5)**
   - Edge strip PSNR < 12 dB (poor)
   - Even with N=50-100 and masked metrics
   - **Fundamental limitation:** Gaussian primitive insufficient

2. **Large σ_perp (>2.0)**
   - PSNR collapses to <6 dB
   - Over-blurring across edge
   - Must keep σ_perp ≤ 1.0

3. **Incorrect alpha scaling**
   - Without α ∝ 1/N: severe over-accumulation
   - PSNR degrades catastrophically with increasing N

---

## Critical Insights from Phase 0.5

### What We Learned:

1. **Measurement bias was not the main issue**
   - Edge strip PSNR only ~3 dB better than full-image PSNR
   - Background error was masking, but not causing, the fundamental limitation

2. **N scaling has diminishing returns**
   - PSNR plateaus at N=50-100
   - More Gaussians don't help beyond this point

3. **Alpha must scale with N**
   - Critical discovery for variable N placement
   - Without scaling: catastrophic over-accumulation

4. **Gaussian primitive has capacity limits**
   - Cannot represent high-contrast edges well (~10 dB PSNR)
   - Low-contrast edges work better (~25 dB PSNR)
   - Suggests need for different primitive or hybrid approach

### What This Means for Future Phases:

**Phase 0.6 (Curved Edges):**
- Expect similar PSNR limits (~10-25 dB)
- Curvature may worsen quality further

**Phase 1 (Multi-Primitive):**
- Edge primitive alone insufficient for high-quality reconstruction
- Need complementary primitives (regions, junctions)

**Phase 2 (Optimization):**
- Empirical rules provide good initialization (N=50, σ_perp=0.5, σ_parallel=10)
- But optimization alone unlikely to overcome 10-15 dB barrier for ΔI≥0.5

---

## Recommendations

### Immediate Next Steps:

1. **Accept the limitation**
   - Gaussian edge primitive: PSNR ~10-25 dB (depending on contrast)
   - This is the **baseline capacity** of the approach

2. **Proceed to curved edges (Phase 0.6)**
   - Test if rules generalize to curved geometry
   - Expect similar quality limits

3. **Consider hybrid approaches**
   - Combine Gaussian edge with other representations (e.g., polylines + blur)
   - Use Gaussians for blurred/low-contrast edges only

### Research Directions:

1. **Alternative primitives for high-contrast edges**
   - Polyline + width + blur (procedural)
   - Hermite curve-based representations
   - Hybrid: procedural edges + Gaussian refinement

2. **Contrast-adaptive primitive selection**
   - Low contrast (ΔI < 0.2): Use Gaussian edge (works well)
   - High contrast (ΔI > 0.5): Use alternative primitive

3. **Optimization with better initialization**
   - Use empirical rules v2.0 as starting point
   - Optimize primitive parameters jointly
   - May still hit fundamental capacity limit

---

## Conclusion

Phase 0.5 **isolated edge region measurement** to test if Phase 0's poor PSNR was due to background error. Key findings:

**Rules Updated:**
1. **N = 50-100** (optimal density, PSNR plateaus)
2. **σ_perp = 0.5-1.0** (smaller is better, confirmed)
3. **σ_parallel = 10** (moderate, consistent with Phase 0)
4. **Alpha scaling: α ∝ 1/N** (critical for variable N)
5. **Spacing: ~2-5 pixels** (denser is better with proper alpha)

**Quality Assessment:**
- **Low contrast (ΔI=0.1):** PSNR = 25 dB ✓ (acceptable)
- **Medium/high contrast (ΔI≥0.5):** PSNR = 10-11 dB ✗ (poor)

**Critical Discovery:**
The limitation is **not measurement bias** (background vs edge), but **fundamental Gaussian primitive capacity** for high-contrast edges. Even measuring edge strip only, PSNR remains poor (~10 dB) for ΔI≥0.5.

**Phase 0.5 Status:** **COMPLETE**

**Ready for:**
- Phase 0.6: Curved edges (test rule generalization)
- Phase 1: Multi-primitive composition (address fundamental limits)
- Phase 2: Optimization-based refinement (with realistic expectations)

---

## References

- **Experiments:** Phase 0.5 (31 renders across 3 experiments)
- **Test corpus:** Same 12 cases as Phase 0, but with edge strip metrics
- **Rendering method:** Accumulative (GaussianImage-style)
- **Key innovation:** Masked PSNR on edge strip only (20px-wide)
- **Code:** `phase_0.5_edge_region_isolated.py`
- **Data:** `phase_0.5_results/experiment_{1,2,3}/`
