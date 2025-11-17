# Claude Code Web: Phase 0.5 - Edge Region Isolated Testing

**Mission:** Fix Phase 0 by measuring edge quality ONLY on edge region (ignore background)

**Duration:** ~1-2 hours
**Key Change:** Masked PSNR (edge strip only), variable N

---

## What Changed from Phase 0

### Phase 0 Problem:
- Measured PSNR on full 100×100 image (10,000 pixels)
- Edge Gaussians only cover ~50px strip
- Background error dominated measurement
- Couldn't isolate edge quality

### Phase 0.5 Solution:
- Measure PSNR only on 20px-wide edge strip (~2,000 pixels)
- Ignore background completely
- Focus entirely on edge representation quality
- Sweep N to find sufficient Gaussian density

---

## Key Insight

**We're testing how to draw a LINE with Gaussians, not how to fill an image.**

Background coverage is irrelevant. That's a different primitive (region fill) for a different experiment.

In a layered representation, **layers don't have full coverage** - each layer represents its feature only.

---

## Your Mission

### Three Experiments (31 renders total, ~1-2 hours)

**Experiment 1:** N Sweep (find sufficient density)
**Experiment 2:** Parameter refinement at optimal N
**Experiment 3:** Coverage analysis (edge strip width)

**All measurements:** PSNR computed on edge strip only (masked)

---

## Experiment 1: N Sweep

**Goal:** How many Gaussians needed for good edge representation?

### Test Cases (3 representative)
- case_02: Sharp, contrast=0.5
- case_06: Blur=2px, contrast=0.5
- case_12: Sharp, contrast=0.1 (best from Phase 0)

### Fixed Parameters
- σ_perp = 1.0 pixels
- σ_parallel = 10.0 pixels
- spacing = 5.0 pixels (uniform along edge)
- alpha = 0.3 / ΔI

### Sweep N
- Values: [5, 10, 20, 50, 100, 200]
- For N Gaussians, space uniformly along 100px edge
- spacing_actual = 100 / N

### Measurement
- Create edge mask: 20px-wide strip centered on edge
- Compute PSNR only on masked region
- Ignore all pixels outside edge strip

**Total:** 3 test cases × 6 N values = **18 renders**

### Expected Results
- N=5: Poor PSNR (gaps visible)
- N=10: Low PSNR (Phase 0 baseline)
- N=20: Better PSNR
- N=50-100: PSNR plateaus (sufficient density)
- N=200: Diminishing returns

### Decision
**Identify N_optimal:** Minimum N where PSNR > 30 dB on edge strip

---

## Experiment 2: Parameter Refinement

**Goal:** Re-sweep parameters with sufficient N

**Use:** N_optimal from Experiment 1 (e.g., N=50 or N=100)

**Test case:** case_06 (blur=2px, contrast=0.5)

### Sweep A: σ_perp
- Values: [0.5, 1.0, 2.0, 3.0, 4.0]
- Fixed: N=N_optimal, σ_parallel=10, alpha=0.6
- **5 renders**

### Sweep B: σ_parallel
- Values: [5, 10, 15, 20]
- Fixed: N=N_optimal, σ_perp=1.0, alpha=0.6
- Adjust spacing = σ_parallel / 2
- **4 renders**

### Sweep C: Spacing
- Values: [σ_parallel/4, σ_parallel/3, σ_parallel/2, σ_parallel]
- Fixed: N=N_optimal, σ_perp=1.0, σ_parallel=10, alpha=0.6
- **4 renders**

**Total:** 5 + 4 + 4 = **13 renders**

**Measurement:** PSNR on 20px edge strip only

### Expected Results

**With sufficient N, parameters should matter more:**
- σ_perp: Expect optimum near σ_edge (2.0 for this case)
- σ_parallel: Expect plateau (larger is better until coverage overlap)
- Spacing: Expect optimum at 40-50% overlap

**If patterns are clear:** Update empirical rules v2.0

**If patterns still unclear:** N may still be insufficient OR parameters genuinely don't matter much

---

## Experiment 3: Edge Strip Width Analysis

**Goal:** Understand coverage radius

**Fixed:**
- Use best configuration from Experiment 2
- Single test case: case_06
- Same render, different measurement masks

### Vary Strip Width
- 5px, 10px, 20px, 40px, full image (100px)

**Total:** **5 measurements** (0 new renders, just re-analyze existing render)

### Expected Results
- 5px strip: Highest PSNR (core edge only)
- 10px strip: High PSNR (edge + near transition)
- 20px strip: Medium PSNR (edge + transition region)
- 40px strip: Lower PSNR (includes some background)
- Full image: Lowest PSNR (background dominates)

**Tells us:** How far from edge center do Gaussians effectively contribute?

---

## Implementation

### Required Code Additions

**1. Edge Mask Function:**
```python
def create_edge_mask(image_size, edge_position, edge_orientation, strip_width=20):
    """Create binary mask for edge region"""
    h, w = image_size
    mask = np.zeros((h, w), dtype=bool)
    half_width = strip_width // 2

    if edge_orientation == 'vertical':
        x_center = edge_position
        mask[:, max(0, x_center-half_width):min(w, x_center+half_width)] = True
    elif edge_orientation == 'horizontal':
        y_center = edge_position
        mask[max(0, y_center-half_width):min(h, y_center+half_width), :] = True
    elif 'diagonal' in edge_orientation:
        # For diagonal, create diagonal strip (approximate)
        # Can implement as rotated rectangle if needed
        pass

    return mask


def compute_metrics_masked(target, rendered, mask):
    """Compute metrics only on masked pixels"""
    target_masked = target[mask]
    rendered_masked = rendered[mask]

    mse = np.mean((target_masked - rendered_masked) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    mae = np.mean(np.abs(target_masked - rendered_masked))

    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae,
        'num_pixels': np.sum(mask)
    }
```

**2. Variable N Placement:**
```python
def place_edge_gaussians_variable_N(edge_descriptor, N, sigma_perp, sigma_parallel, alpha):
    """Place N Gaussians uniformly along edge"""
    gaussians = []

    # Uniform spacing based on N
    edge_length = 100  # pixels
    spacing = edge_length / N

    positions = np.linspace(5, 95, N)  # Avoid exact edges

    for i, pos in enumerate(positions):
        # Position along edge
        if edge_descriptor['orientation'] == 'vertical':
            x, y = edge_descriptor['position'], pos
            theta = 0  # horizontal Gaussians
        else:
            x, y = pos, edge_descriptor['position']
            theta = np.pi/2

        g = Gaussian2D(
            x=x, y=y,
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
            theta=theta,
            color=np.array([edge_descriptor['contrast']]),  # or adjust
            alpha=alpha,
            layer_type='E'
        )
        gaussians.append(g)

    return gaussians
```

---

## Execution Steps

### Step 1: Implement Masked Metrics (~15 min)
- Add create_edge_mask function
- Add compute_metrics_masked function
- Test on one example

### Step 2: Run Experiment 1 (N Sweep) (~30 min)
- Generate 3 test cases
- Sweep N = [5, 10, 20, 50, 100, 200]
- Log results with masked PSNR
- Plot PSNR vs N
- Identify N_optimal

### Step 3: Run Experiment 2 (Parameter Refinement) (~30 min)
- Use N_optimal from Experiment 1
- Sweep σ_perp, σ_parallel, spacing
- Log results
- Generate plots

### Step 4: Run Experiment 3 (Coverage Analysis) (~15 min)
- Take best render from Experiment 2
- Re-measure with different strip widths
- Plot PSNR vs strip_width

### Step 5: Analysis & Documentation (~30 min)
- Synthesize findings
- Update empirical_rules_v2.md
- Generate PHASE_0.5_REPORT.md
- Create comparison: Phase 0 vs Phase 0.5 results

---

## Success Criteria

### Minimum Success:
- ✓ Find N where edge strip PSNR > 25 dB
- ✓ Identify if parameters matter more with sufficient N
- ✓ Document edge coverage radius

### Target Success:
- ✓ Edge strip PSNR > 30 dB at some N
- ✓ Clear parameter patterns (σ_perp optimum, spacing optimum)
- ✓ Updated rules with confidence levels

### Stretch Success:
- ✓ Edge strip PSNR > 35 dB
- ✓ Rules generalize across all test cases
- ✓ Ready for curved edges (Phase 0.6)

---

## Key Constraints

**DO:**
- ✓ Use masked PSNR (edge strip only)
- ✓ Sweep N systematically
- ✓ Reuse Phase 0 infrastructure (Gaussian2D, GaussianRenderer, etc.)
- ✓ Visual inspection of edge region specifically

**DON'T:**
- ✗ Measure quality on full image (background irrelevant)
- ✗ Worry about coverage (this is edge primitive only)
- ✗ Use optimization (still manual)
- ✗ Test curved edges yet (Phase 0.6)

---

## Deliverables Checklist

- [ ] Masked metrics implementation
- [ ] Experiment 1: N sweep (18 renders)
- [ ] N_optimal identified
- [ ] Experiment 2: Parameter refinement (13 renders)
- [ ] Experiment 3: Coverage analysis (5 measurements)
- [ ] PSNR plots (N sweep, parameter sweeps, coverage)
- [ ] Visual atlas showing edge region quality
- [ ] empirical_rules_v2.md (updated rules)
- [ ] PHASE_0.5_REPORT.md (findings and comparison to Phase 0)

---

## Expected Timeline

**Hour 0.0-0.25:** Implement masked metrics
**Hour 0.25-0.75:** Experiment 1 (N sweep)
**Hour 0.75-1.25:** Experiment 2 (parameter refinement)
**Hour 1.25-1.5:** Experiment 3 (coverage analysis)
**Hour 1.5-2.0:** Analysis and documentation

**Total: ~2 hours max**

---

## Critical Success Factor

**Edge strip PSNR must be significantly higher than Phase 0 full-image PSNR.**

**Phase 0:** PSNR ~9-22 dB (full image)
**Phase 0.5:** Expect PSNR ~25-35 dB (edge strip only)

**If edge strip PSNR is still low (<20 dB) even with N=100:**
→ Gaussian primitive may not be good for edges
→ Need fundamental rethink

**If edge strip PSNR is high (>30 dB) with N=50-100:**
→ Validates approach
→ Shows Phase 0 was measuring wrong thing
→ Proceed to curved edges (Phase 0.6)

---

**BEGIN Phase 0.5**

Read PHASE_0.5_PROTOCOL.md, implement masked metrics, run experiments.

Focus: Edge region only. Background is irrelevant.
