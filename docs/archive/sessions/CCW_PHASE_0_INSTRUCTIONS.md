# Claude Code Web: Phase 0 - Edge Function Discovery

**Mission:** Discover empirical rules for edge Gaussian parameters through systematic manual exploration

**Duration:** ~6-8 hours
**Scope:** Tightly constrained - NO scope creep
**Approach:** Manual parameter sweeps, NOT optimization

---

## READ THESE FIRST

**Essential documents (in order):**

1. **RESEARCH_DIRECTION_V2.md** - Why function-based approach, not primitive types
2. **PHASE_0_PROTOCOL.md** - Complete experimental protocol (THIS IS YOUR BIBLE)

**Context documents (optional):**
3. LGI_spec_and_experiments.md - Background (but not binding)

---

## Your Mission

### Goal:
Define the function f_edge that maps edge properties to Gaussian parameters:

```
f_edge(σ_edge, ΔI) → (σ_perp, σ_parallel, spacing, alpha)
```

### Method:
- Create 12 simple synthetic edge images (straight edges only, 100×100 pixels)
- Place Gaussians manually with explicit parameters (N=10, fixed positions)
- Sweep 4 parameters systematically (σ_perp, σ_parallel, spacing, alpha)
- Render each configuration, measure quality (MSE, PSNR)
- Identify patterns, formulate empirical rules

### NOT doing:
- ✗ Optimization (no gradient descent, no Adam, no fitting)
- ✗ Curved edges (Phase 0.5)
- ✗ Other primitives (regions, blobs, junctions)
- ✗ Comparison experiments (layered vs monolithic)
- ✗ Large images (stay at 100×100)

---

## Test Images (12 total)

**All images: 100×100 pixels, grayscale, straight edges**

### Set A: Sharp edges, varying contrast (4 images)
1. Vertical, sharp (σ=0), ΔI=0.2
2. Vertical, sharp, ΔI=0.5
3. Vertical, sharp, ΔI=0.8
4. Diagonal (45°), sharp, ΔI=0.5

### Set B: Varying blur, fixed contrast (4 images)
5. Vertical, blur σ=1px, ΔI=0.5
6. Vertical, blur σ=2px, ΔI=0.5
7. Vertical, blur σ=4px, ΔI=0.5
8. Vertical, blur σ=2px, ΔI=0.8

### Set C: Additional variations (4 images)
9. Horizontal, blur σ=2px, ΔI=0.5
10. Vertical, blur σ=1px, ΔI=0.2
11. Vertical, blur σ=4px, ΔI=0.2
12. Vertical, sharp, ΔI=0.1 (challenging)

**Edge location:** Always centered (x=50 or y=50)
**Pixel format:** Float [0, 1], not uint8

---

## Parameter Sweeps (5 sweeps, 58 renders total)

### Sweep 1: σ_perp (cross-edge width)
**Test on:** Cases 5, 6, 7 (blur = 1, 2, 4px)
**Vary:** σ_perp = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
**Fixed:** σ_parallel=5px, spacing=2.5px, alpha=0.5
**Total:** 3 × 7 = 21 renders
**Question:** Does σ_perp ≈ σ_edge?

### Sweep 2: σ_parallel (along-edge spread)
**Test on:** Case 6 (blur=2px, ΔI=0.5)
**Vary:** σ_parallel = [3.0, 4.0, 5.0, 7.0, 10.0]
**Fixed:** σ_perp=2px, spacing=σ_parallel/2, alpha=0.5
**Total:** 1 × 5 = 5 renders
**Question:** What's optimal σ_parallel? Constant or blur-dependent?

### Sweep 3: Spacing
**Test on:** Case 6
**Vary:** spacing = [1.25, 1.67, 2.5, 3.33, 5.0] (= σ_parallel × [1/4, 1/3, 1/2, 2/3, 1])
**Fixed:** σ_perp=2px, σ_parallel=5px, alpha=0.5
**Total:** 1 × 5 = 5 renders
**Question:** How much overlap needed?

### Sweep 4: Alpha (opacity)
**Test on:** Cases 2, 3, 10 (ΔI = 0.5, 0.8, 0.2)
**Vary:** alpha = [0.1, 0.15, 0.2, 0.25, 0.3] (= ΔI × [0.5, 0.75, 1.0, 1.25, 1.5])
**Fixed:** σ_perp=1px, σ_parallel=5px, spacing=2.5px
**Total:** 3 × 5 = 15 renders
**Question:** alpha = k × ΔI where k = ?

### Sweep 5: Verification
**Test on:** All 12 cases
**Use:** Best parameters from Sweeps 1-4
**Total:** 12 renders
**Question:** Do rules generalize?

---

## Implementation Steps

### Step 1: Build Infrastructure (2 hours max)

**1.1 Synthetic edge generator:**
```python
def generate_straight_edge(
    size=(100, 100),
    orientation='vertical',  # or angle in degrees
    blur_sigma=0,
    contrast=0.5,
    position=0.5
) -> np.ndarray  # float64, [0, 1]
```

**1.2 Gaussian placement:**
```python
def place_edge_gaussians(
    edge_position,
    orientation_angle,
    N=10,
    sigma_perp=1.0,
    sigma_parallel=5.0,
    spacing=2.5,
    alpha=0.5,
    color_left=0.25,
    color_right=0.75
) -> List[Dict]  # List of Gaussian parameters
```

Each Gaussian: `{x, y, sigma_x, sigma_y, theta, color, alpha}`

**1.3 Renderer:**
- **If Rust EWA splatting accessible:** Use it
- **If not:** Implement simple splatting in Python (NumPy)
```python
def render_gaussians(gaussians, size=(100, 100)) -> np.ndarray
```

**1.4 Metrics:**
```python
def compute_metrics(target, rendered):
    mse = np.mean((target - rendered)**2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    max_error = np.max(np.abs(target - rendered))
    return {'mse': mse, 'psnr': psnr, 'max_error': max_error}
```

**Test infrastructure:** Run on 1-2 test cases, verify it works before sweeps.

---

### Step 2: Execute Sweeps (2-3 hours)

**For each sweep:**

1. Generate test images
2. For each parameter combination:
   - Place Gaussians with specified parameters
   - Render
   - Compute metrics
   - Save images (target, rendered, residual)
   - Log to CSV
3. Generate atlas (grid of thumbnails)
4. Plot PSNR vs parameter
5. Write findings summary

**Output directory structure:**
```
phase_0_results/
  test_images/
    case_01_sharp_contrast_low.png
    case_02_sharp_contrast_medium.png
    ...
  sweep_1_sigma_perp/
    blur1_sigmaperp_0.5.png
    blur1_sigmaperp_1.0.png
    ...
    atlas.png
    psnr_plot.png
    findings.md
  sweep_2_sigma_parallel/
    ...
  sweep_3_spacing/
    ...
  sweep_4_alpha/
    ...
  sweep_5_verification/
    ...
  sweep_results.csv  # All metrics
```

---

### Step 3: Analysis (1-2 hours)

**For each sweep, answer:**
- What is the optimal parameter value?
- Is there a clear pattern? (e.g., σ_perp = 1.0 × σ_edge)
- Does it hold across test cases?
- Where does it fail?

**Visual inspection:**
- Look at atlases - what causes halos? gaps? over-smoothing?
- Is PSNR a good indicator or do some high-PSNR renders look bad?

**Synthesize findings:**

Write: `empirical_rules_v1.md`

```markdown
# Edge Gaussian Empirical Rules v1.0

## Derived Parameters

σ_perp = ___ × σ_edge
σ_parallel = ___ (constant) OR ___ × σ_edge (if blur-dependent)
spacing = ___ × σ_parallel
alpha = ___ × ΔI

## Confidence
- σ_perp rule: High/Medium/Low
- σ_parallel rule: High/Medium/Low
- spacing rule: High/Medium/Low
- alpha rule: High/Medium/Low

## Validated Range
- σ_edge: [0, 4] pixels
- ΔI: [0.1, 0.8]
- Geometry: Straight edges only

## Known Failures
[List cases where rules don't work]

## Visual Quality
[Describe quality of renders using these rules]

## Next Steps
[Phase 0.5: curved edges, etc.]
```

---

### Step 4: Final Report (1 hour)

**Generate:** `PHASE_0_REPORT.md`

```markdown
# Phase 0 Completion Report

## Objective
Discover empirical rules for f_edge

## Execution
- Test images: 12 generated
- Parameter sweeps: 5 completed
- Total renders: 58
- Duration: ___ hours

## Key Findings

### Empirical Rules
[Copy from empirical_rules_v1.md]

### Patterns Observed
- σ_perp: [pattern]
- σ_parallel: [pattern]
- spacing: [pattern]
- alpha: [pattern]

### Surprises
[Unexpected results]

### Failure Modes
[Where rules don't work]

## Visual Quality Assessment
- Excellent: [conditions]
- Good: [conditions]
- Acceptable: [conditions]
- Poor: [conditions]

## Deliverables
- [x] 12 test images
- [x] 58 renders
- [x] sweep_results.csv
- [x] 5 atlases
- [x] 5 PSNR plots
- [x] 5 findings summaries
- [x] empirical_rules_v1.md
- [x] Phase 0 report

## Success Criteria Met
- [x] All renders completed
- [x] Clear patterns identified: Yes/Partial/No
- [x] Rules generalize: Yes/Partial/No
- [x] Visual quality: Excellent/Good/Acceptable/Poor

## Recommendation
- [ ] Proceed to Phase 0.5 (curved edges)
- [ ] Refine Phase 0 (patterns unclear)
- [ ] Pivot approach (fundamental issue found)

## Next Steps
[Recommendations for next phase]
```

---

## Critical Constraints

### YOU MUST:
- ✓ Follow PHASE_0_PROTOCOL.md exactly
- ✓ Use 12 specified test cases (100×100 pixels)
- ✓ Complete all 5 sweeps (58 renders)
- ✓ Generate all required outputs (images, plots, atlases, summaries)
- ✓ Document findings clearly

### YOU MUST NOT:
- ✗ Add optimization (stay manual)
- ✗ Test curved edges (Phase 0.5)
- ✗ Add more test cases (scope creep)
- ✗ Use large images (>100×100)
- ✗ Test other primitives (regions, blobs)
- ✗ Skip sweeps or reduce renders to save time

### STOP CONDITIONS:

**Stop and report if:**
- Rendering is too slow (>10 seconds per 100×100 image)
- No clear patterns after first 2 sweeps
- Infrastructure issues blocking progress

**Do NOT proceed to Phase 0.5 without human approval.**

---

## Performance Considerations

**If rendering is slow:**
- 100×100 images should render in <1 second on CPU
- If slower, simplify renderer (fewer Gaussian evaluations)
- Consider rasterizing to grid first, then sample
- Do NOT switch to GPU (adds complexity)

**If gradient computation is mentioned in errors:**
- You should NOT be computing gradients (no optimization!)
- Check: Are you accidentally calling an optimizer?
- Phase 0 is rendering only, no backprop

---

## Tools to Use

**From codebase (if accessible):**
- EWA splatting renderer (lgi-core)
- Basic metrics

**Build fresh (recommended):**
- Edge generator (simple: NumPy step + Gaussian blur)
- Gaussian placement (simple: arithmetic)
- Renderer (simple: NumPy Gaussian evaluation)
- Visualization (matplotlib)

**Do NOT use:**
- Optimizers (Adam, L-BFGS, etc.)
- Complex feature detectors
- GPU rendering
- Advanced metrics (MS-SSIM, LPIPS)

---

## Timeline

**Hour 0-2:** Infrastructure + testing
**Hour 2-4:** Sweeps 1-3
**Hour 4-5:** Sweep 4 + analysis
**Hour 5-6:** Sweep 5 + verification
**Hour 6-8:** Documentation + report

**Total: 6-8 hours**

---

## Deliverables Checklist

Before marking complete:

- [ ] Infrastructure tested and working
- [ ] 12 test images generated
- [ ] Sweep 1 completed (21 renders)
- [ ] Sweep 2 completed (5 renders)
- [ ] Sweep 3 completed (5 renders)
- [ ] Sweep 4 completed (15 renders)
- [ ] Sweep 5 completed (12 renders)
- [ ] All metrics logged to CSV
- [ ] 5 atlases generated
- [ ] 5 PSNR plots generated
- [ ] 5 findings summaries written
- [ ] empirical_rules_v1.md written
- [ ] PHASE_0_REPORT.md written
- [ ] Visual quality assessment completed

---

## Final Note

**Phase 0 is about UNDERSTANDING, not performance.**

The goal is to be able to say:
> "For a straight edge with blur σ_edge and contrast ΔI, use σ_perp = k1 × σ_edge, σ_parallel = k2, spacing = k3 × σ_parallel, alpha = k4 × ΔI, where k1≈___, k2≈___, k3≈___, k4≈___"

With confidence about when these rules work and when they fail.

**If you can write that, Phase 0 succeeds.**

---

**BEGIN WHEN READY**

Read PHASE_0_PROTOCOL.md thoroughly. Follow it precisely. Report findings clearly.
