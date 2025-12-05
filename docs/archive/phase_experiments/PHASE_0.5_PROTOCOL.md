# Phase 0.5: Edge Primitive - Isolated Testing

**Goal:** Understand edge Gaussian parameters by testing ONLY the edge region
**Key Change:** Measure quality on edge strip only, ignore background
**Approach:** Increase N systematically, isolate edge representation from full-image coverage

---

## Lessons from Phase 0

### What We Learned
1. ✓ Rendering infrastructure works
2. ✓ Parameter sweeps are systematic
3. ✓ Alpha-contrast relationship discovered (α = 0.3/ΔI for accumulative rendering)
4. ✓ Larger σ_parallel preferred (monotonic improvement)

### What Was Misleading
- ✗ PSNR on full image dominated by background error
- ✗ N=10 too sparse to represent 100px edge
- ✗ Couldn't isolate edge quality from coverage issue

### New Approach
**Focus on edge region only:**
- Create target that's JUST the edge strip (e.g., 10-20px wide band across image)
- Place Gaussians only in that strip
- Measure quality only in that strip
- Background irrelevant for this experiment

---

## Core Concept: Edge Strip Testing

### What Is an "Edge Strip"?

**Target image:**
```
100×100 pixel image, but we only care about central 20px-wide vertical strip
```

**Visual:**
```
|<-40px->|<-20px->|<-40px->|
[ignore ][  EDGE  ][ignore ]
```

**Measurement:**
- PSNR computed ONLY on the 20px-wide strip
- Background pixels masked out
- Focus entirely on edge representation quality

---

## Test Image Corpus (12 cases - same as Phase 0)

**Keep same 12 test cases BUT change evaluation:**

1-12. Same edge types (sharp/blurred, varying contrast)

**Difference:**
- Still render full 100×100 image
- But compute metrics on **edge-region mask only**
- Edge region = 20px-wide strip centered on edge

---

## Experimental Design: Three Key Experiments

### Experiment 1: N Sweep (How Many Gaussians?)

**Goal:** Find minimum N for acceptable edge quality

**Fixed parameters:**
- σ_perp = 1.0 (from Phase 0)
- σ_parallel = 10.0 (from Phase 0)
- spacing = σ_parallel / 2 = 5.0
- alpha = 0.3 / ΔI (from Phase 0)

**Sweep N:**
- Test cases: 3 representative (case_02, case_06, case_12)
- N values: [5, 10, 20, 50, 100, 200]
- Total: 3 × 6 = **18 renders**

**Measurement:**
- PSNR on edge strip only (20px-wide mask)
- Visual inspection of edge region
- Identify N where PSNR > 30 dB

**Expected outcome:** Find that N=50-100 needed for good edge

**Decision criterion:**
- If PSNR plateaus at some N → use that N for subsequent experiments
- If PSNR keeps improving → edge primitive may need very dense sampling

---

### Experiment 2: Parameter Refinement at Optimal N

**Goal:** Re-sweep parameters with sufficient N

**Use N from Experiment 1** (let's say N=50 for example)

**Sweep A: σ_perp (refined)**
- Test case: case_06 (blur=2px, contrast=0.5)
- σ_perp: [0.5, 1.0, 2.0, 3.0, 4.0]
- Fixed: N=50, σ_parallel=10, spacing=5, alpha=0.6

**Sweep B: σ_parallel (refined)**
- σ_parallel: [5, 10, 15, 20]
- Adjust spacing = σ_parallel / 2

**Sweep C: Spacing (refined)**
- spacing: [σ_parallel/4, σ_parallel/3, σ_parallel/2, σ_parallel]

**Total:** 5 + 4 + 4 = **13 renders**

**Measurement:** PSNR on edge strip only

**Expected outcome:**
- With sufficient N, parameters should show clearer signal
- Identify optimal values

---

### Experiment 3: Edge Strip Width Analysis

**Goal:** Understand how far from edge center Gaussians contribute

**Approach:**
- Fix all parameters (use best from Experiments 1-2)
- Vary edge strip width used for PSNR measurement: [5px, 10px, 20px, 40px, full]
- Same renders, just different measurement masks

**Total:** 1 test case × 5 masks = **5 measurements** (no new renders)

**Expected outcome:**
- See how quality degrades as you include more background
- Understand effective "coverage radius" of edge Gaussians

---

## Implementation Changes

### Key Addition: Edge Region Mask

```python
def create_edge_mask(image_size, edge_position, edge_orientation, strip_width=20):
    """
    Create binary mask for edge region

    Args:
        image_size: (height, width)
        edge_position: center position of edge (pixel coordinate)
        edge_orientation: 'vertical' or 'horizontal'
        strip_width: width of strip to evaluate (pixels)

    Returns:
        Binary mask (True = edge region, False = ignore)
    """
    h, w = image_size
    mask = np.zeros((h, w), dtype=bool)

    half_width = strip_width // 2

    if edge_orientation == 'vertical':
        # Vertical edge: mask is horizontal strip around edge
        x_center = edge_position
        mask[:, max(0, x_center-half_width):min(w, x_center+half_width)] = True
    elif edge_orientation == 'horizontal':
        # Horizontal edge: mask is vertical strip
        y_center = edge_position
        mask[max(0, y_center-half_width):min(h, y_center+half_width), :] = True

    return mask


def compute_metrics_masked(target, rendered, mask):
    """
    Compute metrics only on masked region

    Args:
        target: Target image
        rendered: Rendered image
        mask: Binary mask (True = compute, False = ignore)

    Returns:
        Dict with mse, psnr, mae on masked region only
    """
    # Extract pixels in mask
    target_masked = target[mask]
    rendered_masked = rendered[mask]

    # Compute metrics
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

### Usage in Sweeps

```python
# Generate test case
target, descriptor = generator.generate_straight_edge(...)

# Create edge mask (20px-wide strip)
mask = create_edge_mask(
    image_size=(100, 100),
    edge_position=50,  # center
    edge_orientation='vertical',
    strip_width=20
)

# Place Gaussians and render (same as Phase 0)
gaussians = place_edge_gaussians(...)
rendered = renderer.render(gaussians, 100, 100)

# Compute metrics on edge strip ONLY
metrics = compute_metrics_masked(target, rendered, mask)

print(f"PSNR on edge strip: {metrics['psnr']:.2f} dB")
```

---

## Expected Outcomes

### Experiment 1: N Sweep

**Hypothesis:**
- N=10: PSNR ~15-20 dB on edge strip (still poor, even isolated)
- N=50: PSNR ~30-35 dB on edge strip (acceptable)
- N=100: PSNR ~35-40 dB on edge strip (good)
- N=200: PSNR plateaus (diminishing returns)

**This tells us:** Minimum Gaussian density for edge primitive

### Experiment 2: Parameter Refinement

**With sufficient N (say 50-100):**
- σ_perp sweep should show clearer optimum (maybe σ_perp = σ_edge now visible?)
- σ_parallel sweep shows plateau or optimum
- Spacing shows 40-50% overlap optimum (classic Gaussian splatting)

**Rules derived will be meaningful** (enough samples to see patterns)

### Experiment 3: Coverage Analysis

**Shows:**
- At strip_width=5px: Very high PSNR (just edge core)
- At strip_width=20px: Lower PSNR (includes transition)
- At strip_width=40px: Much lower (includes background)

**Tells us:** Effective radius of edge Gaussian influence

---

## Success Criteria

### Phase 0.5 succeeds if:

1. ✓ We find N where edge strip PSNR > 30 dB
2. ✓ Parameter sweeps at that N show clear patterns
3. ✓ Rules are consistent across test cases
4. ✓ Visual inspection confirms edge looks clean (ignoring background)

### Phase 0.5 provides foundation for:
- Curved edge testing (Phase 0.6)
- Multi-primitive composition (Phase 1)
- Optimization-based refinement (Phase 2)

---

## Timeline

**Experiment 1:** 18 renders × ~0.5s = ~10 seconds + analysis (30 min total)
**Experiment 2:** 13 renders × ~0.5s = ~7 seconds + analysis (30 min total)
**Experiment 3:** 5 measurements (no renders) + analysis (15 min)

**Total: ~1-2 hours**

---

## Deliverables

1. N sweep results with plot (PSNR vs N on edge strip)
2. Optimal N determination
3. Refined parameter rules at optimal N
4. Coverage analysis report
5. Updated empirical_rules_v2.md

---

**This is Option 2: Fix the experiment by isolating edge region.**

Ready to implement?
