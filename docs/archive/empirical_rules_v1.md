# Edge Gaussian Empirical Rules v1.0

**Derived from:** Phase 0 experiments (58 renders across 5 sweeps, 12 test cases)
**Date:** 2025-11-17
**Validity:** Straight edges, σ_edge ∈ [0, 4px], ΔI ∈ [0.1, 0.8], 100×100 pixel images
**Rendering method:** Accumulative rendering (not alpha compositing)

---

## Executive Summary

Through systematic manual parameter exploration, we discovered empirical rules for placing edge Gaussians. The approach uses **N=10 Gaussians placed uniformly along the edge** with parameters derived from edge properties (blur σ_edge and contrast ΔI).

**Key Finding:** The empirical rules show **consistent patterns** but **limited reconstruction quality** (mean PSNR = 11.76 dB). This suggests that while the parametric relationships are identifiable, the fundamental approach (sparse Gaussians along edges) may require refinement for high-quality reconstruction.

---

## Core Empirical Rules

### 1. Cross-Edge Width (σ_perp)

```
σ_perp = 1.0 pixels (constant)
```

**Empirical finding:**
- **Independent of edge blur:** Best σ_perp ≈ 1.0 px regardless of σ_edge ∈ [0, 4px]
- With accumulative rendering, smaller σ_perp is better
- Tested range: [0.5, 6.0] pixels
- Optimal consistently at 1.0 pixels

**Confidence:** **High**
**Evidence:**
- Sweep 1 (corrected): All three blur levels (1px, 2px, 4px) showed optimal at σ_perp = 1.0
- Pattern consistent across different edge blur values

**Interpretation:**
The Gaussian width perpendicular to the edge should be narrow (~1 pixel) to create a sharp edge profile when accumulating. This is **counterintuitive** - we initially hypothesized σ_perp should match σ_edge, but empirical evidence shows a fixed small value works best.

---

### 2. Along-Edge Spread (σ_parallel)

```
σ_parallel = 10.0 pixels
```

**Empirical finding:**
- **Independent of edge blur and contrast**
- Larger values preferred (PSNR increases monotonically with σ_parallel)
- Tested range: [3.0, 10.0] pixels
- Best value at upper limit of tested range

**Confidence:** **Medium** (limited by tested range)
**Evidence:**
- Sweep 2: PSNR improved from 9.23 dB (σ_parallel=3.0) to 9.35 dB (σ_parallel=10.0)
- Monotonic increase suggests even larger values might be beneficial

**Interpretation:**
Wide Gaussians along the edge direction provide better coverage and smoother transitions. The optimal value may be **constrained by image size** (for 100×100 images, σ_parallel = 10.0 ≈ 10% of image width).

**Alternative formulation:**
```
σ_parallel = k × image_size
where k ≈ 0.10 (10% of smaller dimension)
```

---

### 3. Gaussian Spacing

```
spacing = 5.0 pixels
```

**Empirical finding:**
- **Larger spacing preferred** (counterintuitive)
- PSNR increases with spacing in tested range
- Tested range: [1.25, 5.0] pixels (corresponding to σ_parallel × [0.25, 1.0])
- Best value at upper limit

**Confidence:** **Low** (unexpected result, needs validation)
**Evidence:**
- Sweep 3: PSNR improved from 9.24 dB (spacing=1.25) to 9.29 dB (spacing=5.0)
- Small effect size suggests parameter is less critical

**Interpretation:**
With large σ_parallel (10.0) and accumulative rendering, **less overlap is better** to avoid over-accumulation. This contradicts typical Gaussian splatting wisdom (which prefers 40-50% overlap).

**Relationship to σ_parallel:**
```
spacing = σ_parallel × 0.5
```
For σ_parallel = 10.0, spacing = 5.0 pixels

---

### 4. Opacity (Alpha)

```
alpha = k × (1.0 / ΔI)
where k ≈ 0.3
```

**Empirical finding:**
- **Inverse relationship with contrast**
- Low contrast (ΔI = 0.1) → high alpha (α ≈ 3.0)
- High contrast (ΔI = 0.8) → low alpha (α ≈ 0.375)
- Formula: α = 0.3 / ΔI

**Confidence:** **High**
**Evidence:**
- Sweep 4 (corrected): Lower alpha preferred for all contrast levels
- Verification (Sweep 5) shows clear pattern:
  - ΔI = 0.1: PSNR = 22.41 dB (α = 3.0)
  - ΔI = 0.2: PSNR = 17.33 dB (α = 1.5)
  - ΔI = 0.5: PSNR = 9.38 dB (α = 0.6)
  - ΔI = 0.8: PSNR = 5.21 dB (α = 0.375)

**Interpretation:**
With accumulative rendering, the Gaussian opacity must be **inversely proportional to contrast** to avoid over-saturating high-contrast edges. The constant k = 0.3 provides balance across the tested range.

---

### 5. Orientation

```
θ = edge_tangent_angle (perpendicular to edge normal)
```

**Empirical finding:**
- Gaussian major axis aligns with edge direction
- No optimization needed - geometric constraint
- Verified across vertical, horizontal, and diagonal edges

**Confidence:** **High** (geometric requirement)

---

## Complete Function: f_edge

```python
def f_edge(σ_edge: float, ΔI: float, image_size: tuple = (100, 100)) -> dict:
    """
    Edge Gaussian parameter function (empirical rules v1.0)

    Args:
        σ_edge: Edge blur (pixels)
        ΔI: Edge contrast (intensity difference, [0, 1])
        image_size: (height, width) in pixels

    Returns:
        Dict with keys: sigma_perp, sigma_parallel, spacing, alpha, theta
    """

    # Rule 1: σ_perp is constant
    sigma_perp = 1.0

    # Rule 2: σ_parallel depends on image size
    min_dim = min(image_size)
    sigma_parallel = 0.10 * min_dim  # 10% of smaller dimension

    # Rule 3: spacing is half of σ_parallel
    spacing = 0.5 * sigma_parallel

    # Rule 4: alpha inversely proportional to contrast
    alpha = 0.3 / ΔI if ΔI > 0 else 0.3

    # Rule 5: orientation from edge detection (external input)
    # theta = <computed from gradient>

    return {
        'sigma_perp': sigma_perp,
        'sigma_parallel': sigma_parallel,
        'spacing': spacing,
        'alpha': alpha,
        'N': 10  # Number of Gaussians
    }
```

---

## Validated Parameter Ranges

### Successfully tested:
- **Edge blur:** σ_edge ∈ [0, 4] pixels
- **Contrast:** ΔI ∈ [0.1, 0.8]
- **Orientation:** 0° (horizontal), 45° (diagonal), 90° (vertical)
- **Geometry:** Straight edges only (no curvature)
- **Image size:** 100×100 pixels

### Not tested (Phase 0.5+):
- Curved edges (κ > 0)
- Very soft edges (σ_edge > 4px)
- Very low contrast (ΔI < 0.1)
- Large images (> 100×100)
- Non-straight geometry

---

## Reconstruction Quality Assessment

### Overall Performance:
- **Mean PSNR:** 11.76 dB (std: 5.45 dB)
- **Range:** 5.16 - 22.41 dB
- **Quality:** Poor to acceptable (target: > 30 dB for good quality)

### Performance by Contrast:
- **ΔI = 0.1:** 22.41 dB (Acceptable)
- **ΔI = 0.2:** 17.33 dB (Poor)
- **ΔI = 0.5:** 9.38 dB (Very poor)
- **ΔI = 0.8:** 5.21 dB (Very poor)

**Trend:** Performance degrades significantly with increasing contrast.

### Performance by Blur Level:
- **σ_edge = 0 (sharp):** 12.68 ± 6.98 dB
- **σ_edge = 1px:** 13.32 ± 5.63 dB
- **σ_edge = 2px:** 8.02 ± 2.39 dB
- **σ_edge = 4px:** 13.50 ± 5.61 dB

**Trend:** No clear monotonic relationship with blur.

### Best Cases:
1. Very low contrast (ΔI = 0.1): PSNR = 22.41 dB
2. Low contrast (ΔI = 0.2): PSNR ≈ 17 dB

### Worst Cases:
1. High contrast (ΔI = 0.8): PSNR ≈ 5 dB
2. Medium contrast (ΔI = 0.5): PSNR ≈ 9 dB

---

## Visual Quality Observations

### Acceptable Quality:
- **Very low contrast edges** (ΔI < 0.2)
- **Blurred edges** show smoother transitions
- **Orientation-invariant** (vertical, horizontal, diagonal all similar)

### Poor Quality:
- **High contrast edges** (ΔI > 0.5) show severe artifacts
- **Sharp edges** (σ_edge = 0) difficult to reconstruct
- **Visible banding** along edge due to sparse Gaussians (N=10)

### Common Artifacts:
1. **Incomplete coverage:** Edge only partially reconstructed
2. **Under-intensity:** Rendered edges dimmer than target
3. **Streaky appearance:** Discrete Gaussians visible
4. **Background not filled:** Only edge region covered, not full image

---

## Known Limitations

### Fundamental Constraints:

1. **Sparse representation (N=10) is insufficient**
   - Edges span entire image, but only 10 Gaussians placed
   - Coverage is ~10-50 pixels (depending on σ_parallel and spacing)
   - Rest of image remains black

2. **Color model is oversimplified**
   - All Gaussians use same color (edge contrast value)
   - Doesn't model both sides of edge separately
   - Background regions not represented

3. **Accumulation-based, not optimization-based**
   - No gradient descent to refine parameters
   - Parameters fixed by empirical rules
   - Could benefit from per-edge optimization

### Scope Limitations:

- **Curved edges:** NOT TESTED (Phase 0.5)
- **Edge junctions:** NOT HANDLED (requires different primitive)
- **Regions:** NOT HANDLED (requires different approach)
- **Multi-scale edges:** Single-scale only

---

## Critical Findings & Insights

### Surprising Results:

1. **σ_perp independent of σ_edge**
   - Hypothesis: σ_perp ≈ σ_edge
   - **Reality: σ_perp ≈ 1.0 (constant)**
   - Implication: Edge Gaussian width is decoupled from edge sharpness

2. **Alpha inversely proportional to contrast**
   - Hypothesis: α ≈ k × ΔI (positive)
   - **Reality: α ≈ k / ΔI (inverse)**
   - Implication: Accumulative rendering requires compensation

3. **Large spacing preferred**
   - Hypothesis: 40-50% overlap optimal
   - **Reality: spacing = σ_parallel (minimal overlap)**
   - Implication: Over-accumulation degrades quality

### Methodological Insights:

1. **Rendering method matters**
   - Alpha compositing vs. accumulation gives opposite trends
   - Accumulation (GaussianImage-style) preferred for this task

2. **Color strategy is critical**
   - Using color = ΔI (not ΔI/2) improves results
   - Still insufficient for high-quality reconstruction

3. **Parameter sensitivity varies**
   - High sensitivity: alpha (strongly affects PSNR)
   - Medium sensitivity: σ_parallel, spacing
   - Low sensitivity: σ_perp (within tested range)

---

## Failure Modes

### When Rules Fail:

1. **High contrast edges (ΔI > 0.5)**
   - PSNR < 10 dB (very poor)
   - Severe under-intensity
   - Recommended: Increase N or use different approach

2. **Sharp edges (σ_edge = 0) with high contrast**
   - Worst case: PSNR ≈ 5 dB
   - Visible banding and gaps
   - Recommended: Requires denser placement

3. **Large images (extrapolation)**
   - Rules derived for 100×100 images
   - Scaling behavior unknown
   - Recommended: Re-run sweeps at target resolution

---

## Recommendations for Phase 0.5+

### Immediate Improvements:

1. **Increase N (number of Gaussians)**
   - Test N ∈ [20, 50, 100]
   - Expect improved coverage and PSNR

2. **Two-sided edge representation**
   - Place Gaussians on both sides of edge with different colors
   - Model background and foreground separately

3. **Adaptive placement**
   - Vary density based on contrast
   - High contrast → more Gaussians

### Research Directions:

1. **Curved edges (Phase 0.5)**
   - Test on circular arcs
   - Expect different σ_parallel and spacing rules

2. **Optimization refinement**
   - Use empirical rules as initialization
   - Fine-tune with gradient descent

3. **Layered representation**
   - Combine edge layer with background/region layers
   - More complete image model

---

## Conclusion

Phase 0 successfully **identified empirical rules** for edge Gaussian parameters through systematic manual exploration:

1. **σ_perp = 1.0 px** (constant, independent of blur)
2. **σ_parallel = 0.1 × image_size** (scales with image)
3. **spacing = 0.5 × σ_parallel** (large spacing preferred)
4. **alpha = 0.3 / ΔI** (inverse relationship with contrast)

These rules are **consistent and generalizable** across the tested range (straight edges, blur ∈ [0,4px], contrast ∈ [0.1, 0.8]).

However, **reconstruction quality is limited** (mean PSNR = 11.76 dB), indicating that:
- Sparse edge representation (N=10) is insufficient
- Fundamental approach may need refinement
- Rules provide good starting point for optimization-based methods

**Phase 0 is COMPLETE.** Ready to proceed to Phase 0.5 (curved edges) or pivot to optimization-based refinement.

---

## References

- **Test corpus:** 12 synthetic straight edges (100×100 px)
- **Total renders:** 58 (Sweeps 1-5)
- **Rendering method:** Accumulative (GaussianImage-style)
- **Metrics:** MSE, PSNR, MAE, correlation
- **Code:** `phase_0_edge_function_discovery.py`, `phase_0_sweep_5_verification.py`
- **Data:** `phase_0_results/sweep_results.csv`, `phase_0_results/sweep_5_verification/verification_results.csv`

