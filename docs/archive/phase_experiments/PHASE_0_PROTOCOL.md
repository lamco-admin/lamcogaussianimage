# Phase 0: Edge Gaussian Function Discovery

**Goal:** Understand how to map edge properties to optimal Gaussian parameters
**Method:** Manual parameter exploration, visual inspection, empirical rule synthesis
**Scope:** Straight edges only (curved edges deferred to Phase 0.5)
**No optimization** - just rendering with explicit parameters

---

## Objective

Define the function f_edge:

```
f_edge(σ_edge, ΔI) → (σ_perp, σ_parallel, spacing, alpha)
```

**Discover empirical rules through systematic parameter sweeps.**

---

## Test Image Corpus

### Simple, Synthetic, Known Ground Truth

**12 test cases total:**

#### Set A: Straight Sharp Edges (4 cases)

```python
1. edge_straight_sharp_contrast_low
   - Orientation: vertical (90°)
   - Blur: σ_edge = 0 (sharp)
   - Contrast: ΔI = 0.2
   - Background: gray(0.4), foreground: gray(0.6)
   - Size: 100×100 pixels

2. edge_straight_sharp_contrast_medium
   - Same but ΔI = 0.5 (gray 0.25 → 0.75)

3. edge_straight_sharp_contrast_high
   - Same but ΔI = 0.8 (gray 0.1 → 0.9)

4. edge_straight_sharp_diagonal
   - 45° diagonal
   - ΔI = 0.5
```

#### Set B: Straight Blurred Edges (4 cases)

```python
5. edge_straight_blur1_contrast_medium
   - Vertical
   - Blur: σ_edge = 1.0 pixel
   - Contrast: ΔI = 0.5

6. edge_straight_blur2_contrast_medium
   - Vertical
   - Blur: σ_edge = 2.0 pixels
   - Contrast: ΔI = 0.5

7. edge_straight_blur4_contrast_medium
   - Vertical
   - Blur: σ_edge = 4.0 pixels
   - Contrast: ΔI = 0.5

8. edge_straight_blur2_contrast_high
   - Vertical
   - Blur: σ_edge = 2.0 pixels
   - Contrast: ΔI = 0.8
```

#### Set C: Variations (4 cases)

```python
9. edge_horizontal_blur2
   - Horizontal orientation (0°)
   - Blur: σ_edge = 2.0 pixels
   - Contrast: ΔI = 0.5

10. edge_straight_blur1_contrast_low
    - Vertical
    - Blur: σ_edge = 1.0 pixel
    - Contrast: ΔI = 0.2

11. edge_straight_blur4_contrast_low
    - Vertical
    - Blur: σ_edge = 4.0 pixels
    - Contrast: ΔI = 0.2

12. edge_straight_sharp_contrast_verylow
    - Vertical, sharp
    - Contrast: ΔI = 0.1 (challenging low contrast)
```

### Why These Test Cases?

- **Simple:** Straight edges only (no curvature complications)
- **Small:** 100×100 pixels (fast rendering, easy to visualize)
- **Systematic:** Vary one property at a time (blur, contrast, orientation)
- **Known ground truth:** Exact edge properties by construction
- **Representative:** Cover range of real-world edges (sharp to soft, low to high contrast)

### Image Generation Specification

```python
def generate_straight_edge(
    size=(100, 100),
    orientation='vertical',  # or angle in degrees
    blur_sigma=0,            # 0 = sharp, >0 = Gaussian blur
    contrast=0.5,            # ΔI in [0, 1]
    position=0.5             # relative position (0.5 = center)
) -> np.ndarray:
    """
    Returns: Grayscale image in [0, 1] with edge
    """
```

**Edge position:** Always at image center (simplifies Gaussian placement)

**Edge profile:** Step function convolved with Gaussian (blur)

**Pixel values:** Floating point [0, 1] (not uint8)

---

## Parameter Sweep Design

### Fixed Parameters (Per Test Case)

- **N:** 10 Gaussians along edge (fixed for all sweeps)
- **Orientation θ:** Perpendicular to edge (derived from image)
- **Position:** Uniformly spaced along edge, centered on edge location
- **Color:** Average of both sides of edge (gray value)

### Sweep Parameters

#### Sweep 1: σ_perp (Cross-Edge Width)

**Hypothesis:** σ_perp should approximately match edge blur σ_edge

**Test on:** Cases 5, 6, 7 (blur = 1, 2, 4 pixels)

**Sweep range:** σ_perp = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] pixels

**Fixed:** σ_parallel = 5.0 pixels, spacing = 2.5 pixels (σ_parallel / 2)

**Total:** 3 test cases × 7 σ_perp values = **21 renders**

**Expected outcome:** For each blur level, identify optimal σ_perp

#### Sweep 2: σ_parallel (Along-Edge Spread)

**Hypothesis:** σ_parallel controls smoothness along edge

**Test on:** Case 6 (blur = 2px, contrast = 0.5) - representative case

**Sweep range:** σ_parallel = [3.0, 4.0, 5.0, 7.0, 10.0] pixels

**Fixed:** σ_perp = 2.0 pixels (from Sweep 1), spacing = σ_parallel / 2

**Total:** 1 test case × 5 σ_parallel values = **5 renders**

**Expected outcome:** Identify good σ_parallel value (independent of blur?)

#### Sweep 3: Spacing (Gap Between Gaussians)

**Hypothesis:** Spacing should be fraction of σ_parallel for smooth coverage

**Test on:** Case 6 (blur = 2px, contrast = 0.5)

**Sweep range:** spacing = σ_parallel × [1/4, 1/3, 1/2, 2/3, 1.0]

**Fixed:** σ_perp = 2.0 pixels, σ_parallel = 5.0 pixels

**Total:** 1 test case × 5 spacing values = **5 renders**

**Expected outcome:** Identify optimal overlap (probably 40-60%)

#### Sweep 4: Alpha (Opacity)

**Hypothesis:** Alpha should relate to contrast ΔI

**Test on:** Cases 2, 3, 10 (contrast = 0.5, 0.8, 0.2)

**Sweep range:** alpha = ΔI × [0.5, 0.75, 1.0, 1.25, 1.5]

**Fixed:** σ_perp = 1.0, σ_parallel = 5.0, spacing = 2.5

**Total:** 3 test cases × 5 alpha values = **15 renders**

**Expected outcome:** Direct relationship alpha = k × ΔI where k ≈ ?

#### Sweep 5: Verification on All Cases

**Using best parameters from Sweeps 1-4:**

**Test on:** All 12 test cases

**Parameters:** Use empirical rules derived from sweeps 1-4

**Total:** 12 renders

**Expected outcome:** Verify rules generalize across test corpus

---

## Total Experimental Load

**Sweep 1:** 21 renders
**Sweep 2:** 5 renders
**Sweep 3:** 5 renders
**Sweep 4:** 15 renders
**Sweep 5:** 12 renders

**Total:** 58 renders

**Expected time:** ~2-4 hours (depending on rendering speed)

---

## Required Outputs Per Render

### 1. Rendered Image
- Filename: `{testcase}_{param}_{value}.png`
- Example: `blur2_sigmaperp_2.0.png`

### 2. Comparison Image
- Side-by-side: target | rendered | residual (abs difference)
- Filename: `{testcase}_{param}_{value}_comparison.png`

### 3. Metrics
- MSE (mean squared error)
- PSNR (peak signal-to-noise ratio)
- Max error (worst pixel)
- Logged to CSV: `sweep_results.csv`

### 4. Visual Quality Notes
- Halos present? (yes/no)
- Gaps visible? (yes/no)
- Over-smoothing? (yes/no)
- Overall quality: (poor/acceptable/good/excellent)

---

## Analysis Requirements

### After Each Sweep:

**1. Generate Sweep Plot**
- X-axis: Parameter value (e.g., σ_perp)
- Y-axis: PSNR (dB)
- One line per test case
- Identify optimal parameter value

**2. Create Visual Atlas**
- Grid layout: rows = test cases, columns = parameter values
- Small thumbnail images for quick comparison
- Highlight best result per row

**3. Write Findings Summary**
```markdown
## Sweep 1: σ_perp Results

### Optimal Values:
- σ_edge = 1.0 → σ_perp = 1.0 (PSNR: 42.3 dB)
- σ_edge = 2.0 → σ_perp = 2.0 (PSNR: 43.1 dB)
- σ_edge = 4.0 → σ_perp = 4.0 (PSNR: 41.8 dB)

### Observed Pattern:
**σ_perp ≈ σ_edge × 1.0** (direct correspondence)

### Visual Observations:
- σ_perp < σ_edge: Visible gaps, high-frequency residual
- σ_perp > σ_edge: Halos, over-blurring
- σ_perp = σ_edge: Clean reconstruction

### Failure Cases:
None observed in this range.
```

---

## Final Deliverable: Empirical Rules Document

### Edge Gaussian Rules v1.0

```markdown
# Edge Gaussian Function: Empirical Rules

**Derived from:** Phase 0 experiments (58 renders, 12 test cases)
**Date:** [completion date]
**Validity:** Straight edges, σ_edge ∈ [0, 4px], ΔI ∈ [0.1, 0.8]

## Core Rules

### 1. Cross-Edge Width (σ_perp)
σ_perp = σ_edge × k1

**Empirical constant:** k1 = ___
**Confidence:** High/Medium/Low
**Range tested:** σ_edge ∈ [0, 4] pixels

### 2. Along-Edge Spread (σ_parallel)
σ_parallel = k2

**Empirical constant:** k2 = ___ pixels (appears independent of blur)
**Confidence:** High/Medium/Low
**Alternative:** σ_parallel = σ_edge × k2b (if blur-dependent)

### 3. Gaussian Spacing
spacing = σ_parallel × k3

**Empirical constant:** k3 = ___ (overlap factor)
**Typical:** k3 ≈ 0.4-0.5 (40-50% overlap)
**Confidence:** High/Medium/Low

### 4. Opacity
alpha = ΔI × k4

**Empirical constant:** k4 = ___
**Confidence:** High/Medium/Low
**Range tested:** ΔI ∈ [0.1, 0.8]

### 5. Orientation
θ = edge_tangent_angle

**Derived from:** Image gradient direction
**Fixed constraint:** Not optimized

## Complete Function

f_edge(σ_edge, ΔI) = {
    σ_perp: σ_edge × k1,
    σ_parallel: k2,
    spacing: k2 × k3,
    alpha: ΔI × k4,
    θ: edge_tangent
}

## Validated Range

- Edge blur: σ_edge ∈ [0, 4] pixels
- Contrast: ΔI ∈ [0.1, 0.8]
- Orientation: Any (0-360°)
- Geometry: Straight edges only

## Known Limitations

- Curved edges: NOT TESTED (Phase 0.5)
- Very soft edges (σ_edge > 4px): NOT TESTED
- Very low contrast (ΔI < 0.1): Marginal quality
- Junctions: NOT HANDLED (different function needed)

## Visual Quality Assessment

- **Excellent:** [list conditions]
- **Good:** [list conditions]
- **Acceptable:** [list conditions]
- **Poor:** [list conditions]

## Failure Modes

[Document cases where rules fail]

## Next Steps

- Phase 0.5: Test on curved edges
- Refine rules if needed
- Extend to other edge types (soft, textured)
```

---

## Implementation Requirements

### Code Structure

```python
# edge_function_discovery.py

def generate_test_corpus():
    """Create 12 synthetic edge images"""
    pass

def place_edge_gaussians(edge_image, sigma_perp, sigma_parallel, spacing, alpha):
    """Manually place N=10 Gaussians with specified parameters"""
    pass

def render_gaussians(gaussians, image_size):
    """Render using EWA splatting"""
    pass

def compute_metrics(target, rendered):
    """MSE, PSNR, max error"""
    pass

def run_sweep(sweep_name, test_cases, parameter_range, fixed_params):
    """Execute one parameter sweep"""
    pass

def generate_atlas(sweep_results):
    """Create visual grid of results"""
    pass

def plot_sweep_results(sweep_results):
    """Plot PSNR vs parameter value"""
    pass

def synthesize_findings(all_sweep_results):
    """Generate empirical rules document"""
    pass

# Main execution
if __name__ == "__main__":
    corpus = generate_test_corpus()

    results_sweep1 = run_sweep("sigma_perp", ...)
    results_sweep2 = run_sweep("sigma_parallel", ...)
    results_sweep3 = run_sweep("spacing", ...)
    results_sweep4 = run_sweep("alpha", ...)
    results_sweep5 = run_sweep("verification", ...)

    synthesize_findings([results_sweep1, ..., results_sweep5])
```

### Tools to Use

**From existing codebase:**
- EWA splatting renderer (lgi-core): Use if accessible
- Basic metrics (PSNR, MSE): Implement if needed (simple)

**Build fresh:**
- Synthetic edge generator (simple: NumPy)
- Manual Gaussian placement (simple: list of dicts)
- Plotting/visualization (matplotlib)
- Analysis framework (pandas + matplotlib)

**Do NOT use:**
- Optimizers (not needed)
- Feature detectors (synthetic data has known properties)
- Complex metrics (MS-SSIM, perceptual)
- GPU rendering (CPU is fine for 100×100 images)

---

## Success Criteria

### Minimum Success:
- ✓ All 58 renders completed
- ✓ Metrics logged for all cases
- ✓ Visual atlas generated
- ✓ At least 3 clear empirical rules identified

### Target Success:
- ✓ All of above
- ✓ Rules have clear patterns (e.g., σ_perp = 1.0 × σ_edge)
- ✓ Rules generalize across test corpus (Sweep 5 validates)
- ✓ Visual quality is good for most cases

### Stretch Success:
- ✓ All of above
- ✓ Identified edge cases where rules fail
- ✓ Proposed refinements for failures
- ✓ Ready to proceed to Phase 0.5 (curved edges)

---

## Failure Conditions

### Stop and Report if:
- ✗ Rendering takes >10 seconds per image (too slow)
- ✗ No clear pattern emerges after Sweep 1-2 (σ_perp, σ_parallel)
- ✗ All parameter values give similar PSNR (parameters don't matter?)
- ✗ Visual quality is poor for all configurations (approach may be wrong)

### Request Guidance if:
- ? Rules seem to contradict each other
- ? Optimal parameters are at edge of tested range (need to expand)
- ? High variability across test cases (no consistent pattern)

---

## Timeline Estimate

**Hour 0-2:** Build infrastructure
- Synthetic edge generator
- Gaussian placement function
- Rendering integration
- Metrics computation

**Hour 2-4:** Execute sweeps
- Sweep 1: σ_perp (21 renders)
- Sweep 2: σ_parallel (5 renders)
- Sweep 3: spacing (5 renders)
- Sweep 4: alpha (15 renders)

**Hour 4-5:** Analysis
- Generate plots and atlases
- Identify patterns
- Write findings summaries

**Hour 5-6:** Verification and documentation
- Sweep 5: verification (12 renders)
- Synthesize empirical rules document
- Generate final report

**Total: ~6-8 hours**

---

## Deliverables Checklist

Before marking Phase 0 complete:

- [ ] 12 synthetic test images generated and saved
- [ ] 58 renders completed (all parameter combinations)
- [ ] sweep_results.csv with all metrics
- [ ] Visual atlases for each sweep (5 atlases)
- [ ] PSNR plots for each sweep (5 plots)
- [ ] Findings summary for each sweep (5 summaries)
- [ ] Empirical rules document (Edge Gaussian Rules v1.0)
- [ ] Visual quality assessment completed
- [ ] Failure modes documented
- [ ] Phase 0 completion report

---

## IMPORTANT: No Scope Creep

**DO NOT:**
- ✗ Add optimization (Phase 0 is manual only)
- ✗ Test curved edges (that's Phase 0.5)
- ✗ Test other primitives (regions, blobs)
- ✗ Compare layered vs monolithic (much later)
- ✗ Add more test cases beyond the 12 specified
- ✗ Expand parameter ranges without reason

**FOCUS:**
- ✓ Execute the protocol as specified
- ✓ Document findings clearly
- ✓ Identify empirical rules
- ✓ Stop at checkpoint

**Phase 0 is complete when you have empirical rules for straight edges. Nothing more.**
