# Practical Gaussian Image Toolkit Guide

**Purpose:** Understand the basic concepts we've explored and when to apply them
**Audience:** Personal reference, cut the marketing BS
**Date:** 2025-11-17

---

## Core Concept: What Are We Actually Doing?

**Representing images as collections of 2D Gaussians.**

Each Gaussian = blob of color with position, size, orientation, opacity.

**The challenge:** Where to put them and what parameters to use so the image looks good with fewest Gaussians.

---

## Part 1: Placement Strategies (Where to Put Gaussians)

### The Basic Question
Given an image and a budget of N Gaussians, where do you place them?

### Core Approaches (5 meaningful categories)

#### 1. **Uniform/Agnostic**
**Concept:** Ignore image content, place systematically.

**Variations:**
- Grid: Regular √N × √N grid
- Random: Uniform random positions
- Poisson disk: Random but with minimum spacing

**When to use:**
- Baseline comparison
- When you have NO image analysis
- Extremely simple images

**Pros:** Fast, no analysis needed
**Cons:** Wastes Gaussians in flat areas, under-samples complex areas

---

#### 2. **Gradient-Based**
**Concept:** Place more Gaussians where image has high gradient (edges, transitions).

**How it works:**
- Compute gradient magnitude |∇I| at every pixel
- Sample positions with probability ∝ |∇I|
- More Gaussians near edges, fewer in flat regions

**Variations:**
- Sobel/Canny edge detection → place along edges
- Structure tensor → place along strong gradient directions
- Laplacian peaks → place at detail points

**When to use:**
- Images with clear edges (diagrams, text, buildings)
- When edges matter more than smooth regions

**Pros:** Efficient for edge-heavy content
**Cons:** May under-represent smooth gradients (sky, faces)

---

#### 3. **Variance/Entropy-Based**
**Concept:** Place more Gaussians where local complexity is high.

**How it works:**
- Compute local variance or entropy in small patches
- High variance = complex detail → more Gaussians
- Low variance = flat region → fewer Gaussians

**Variations:**
- Tile-based entropy (16×16 tiles)
- Sliding window variance
- Frequency domain energy

**When to use:**
- Natural photos with mixed content
- When both edges AND textures matter
- Balanced coverage needed

**Pros:** Adapts to overall complexity
**Cons:** Doesn't distinguish edge from texture (treats differently but both get Gaussians)

---

#### 4. **Feature-Based**
**Concept:** Detect semantic features (edges, corners, regions) and place specialized Gaussians.

**How it works:**
- Detect edges → place elongated Gaussians along curves
- Detect corners/junctions → place clusters
- Segment regions → place fill Gaussians inside
- Detect keypoints → place blobs

**Variations:**
- SLIC superpixels + centroid placement
- Harris corners + Gaussian per corner
- Edge chains + uniform spacing along curve

**When to use:**
- When you want structured, interpretable representation
- Content-adaptive strategies (different methods per feature)
- Research/analysis (understand what Gaussians represent)

**Pros:** Interpretable, efficient for structured content
**Cons:** Requires feature detection (complex), brittle if detection fails

---

#### 5. **Error-Driven/Iterative**
**Concept:** Start with some Gaussians, measure error, add more where error is high.

**How it works:**
1. Place initial Gaussians (any method)
2. Render, compute residual (error map)
3. Place new Gaussians where residual is highest
4. Repeat until budget exhausted or quality sufficient

**Variations:**
- Adaptive densification (split/clone high-error Gaussians)
- Greedy refinement (add one at max-error location)
- Split-based (divide large Gaussians in high-error regions)

**When to use:**
- When you have time for iterative refinement
- Highest quality needed
- Unknown content (adaptive approach)

**Pros:** Optimal for given budget (error-driven = efficient)
**Cons:** Slow (iterative), requires multiple render passes

---

### Practical Recommendation

**Start simple, add complexity only if needed:**

1. **For research/understanding:** Uniform grid (establishes baseline)
2. **For edges/diagrams:** Gradient-based (80% of benefit, 20% complexity)
3. **For photos:** Entropy-based (balanced coverage)
4. **For quality:** Error-driven (best results, more computation)
5. **For interpretability:** Feature-based (understand structure)

**The 80/20 rule:** Gradient-based placement gets you most of the way there.

---

## Part 2: Optimization Methods (How to Refine Parameters)

### The Basic Question
Given Gaussians at some positions, how do you adjust their parameters (position, size, color, opacity) to improve quality?

### Core Approaches (3 that matter)

#### 1. **Adam (Adaptive Moment Estimation)**
**Concept:** Gradient descent with momentum and adaptive learning rates.

**What it does:**
- Computes gradient of loss w.r.t. each parameter
- Uses moving averages (momentum) to smooth updates
- Adapts learning rate per parameter

**Parameters:**
- Learning rate: 0.001-0.01 typical
- β₁=0.9, β₂=0.999 (standard, rarely change)

**When to use:**
- Default choice for most cases
- Works well for 10-1000 Gaussians
- Robust to poor initialization

**Pros:** Reliable, well-tested, handles most cases
**Cons:** Can be slow for very large N (>1000)

**Implementation notes:**
- Need differentiable renderer (backprop through splatting)
- 500-2000 iterations typical
- Watch for divergence (reduce LR if loss increases)

---

#### 2. **L-BFGS (Limited-memory BFGS)**
**Concept:** Second-order quasi-Newton method (uses curvature information).

**What it does:**
- Approximates Hessian (second derivatives)
- Typically converges in fewer iterations than Adam
- Uses more memory per iteration

**When to use:**
- Small problems (N < 100 Gaussians)
- When you want fast convergence (100-300 iters)
- Smooth loss landscape (not many local minima)

**Pros:** Fast convergence, fewer iterations
**Cons:** Doesn't scale to large N, sensitive to initialization

**Practical note:**
- Good for "final polish" after Adam gets close
- Or for isolated primitives (single edge, 10-20 Gaussians)

---

#### 3. **No Optimization (Manual/Analytic)**
**Concept:** Set parameters based on rules, don't optimize.

**When to use:**
- When you understand the mapping (feature → parameters)
- Fast encoding needed (no iteration)
- Research Phase 0 (understand primitives first)

**Examples:**
- Edge with blur σ → σ_perp = σ (derived rule)
- Blob at location → Gaussian at location (trivial)
- Macro field → few large Gaussians (heuristic)

**Pros:** Instant (no iteration), interpretable
**Cons:** Quality limited by rules, doesn't adapt to surprises

---

### Practical Recommendation

**Default: Adam** (unless you have reason to do otherwise)

**Use L-BFGS:** For small, isolated problems (N < 50)
**Skip optimization:** When you have good initialization rules (Phase 0 goal)

**Learning rate guidance:**
- Start: 0.01
- If diverging: reduce to 0.001
- If converging slowly: increase to 0.02-0.05
- Position: may need higher LR (0.1)
- Color/opacity: may need lower LR (0.001)

---

## Part 3: Loss Functions (What to Optimize For)

### The Basic Question
How do you measure quality? What should optimizer minimize?

### Core Options (4 meaningful choices)

#### 1. **MSE (Mean Squared Error)**
**Formula:** `(1/N) Σ(target - rendered)²`

**Pros:** Simple, fast, well-behaved gradients
**Cons:** Treats all pixels equally (edges vs flat), doesn't match perception

**When to use:** Default, research, baseline

---

#### 2. **MS-SSIM (Multi-Scale Structural Similarity)**
**Formula:** Compares structure/luminance/contrast at multiple scales

**Pros:** Better perceptual correlation than MSE
**Cons:** Slower to compute, more complex gradients

**When to use:**
- Perceptual quality matters
- Willing to spend 2-3× compute time
- Natural photos (structure matters)

---

#### 3. **Edge-Weighted MSE**
**Formula:** `MSE × (1 + k × edge_strength)`

**Concept:** Penalize errors near edges more than in flat regions.

**Pros:** Prioritizes edges (often most visible errors)
**Cons:** Somewhat ad-hoc (k is tunable)

**When to use:**
- Diagrams, text, line art
- When edges are perceptually critical

---

#### 4. **Rate-Distortion (R-D)**
**Formula:** `D + λR` where D=distortion, R=rate (bits), λ=tradeoff

**Concept:** Jointly optimize quality and compression.

**Pros:** Codec-appropriate (balances quality and size)
**Cons:** Complex (need bitrate estimation), requires λ tuning

**When to use:**
- Building actual codec
- Compression ratio matters
- Not for pure research

---

### Practical Recommendation

**Start with MSE** (simple, works, interpretable)

**Upgrade to MS-SSIM** if perceptual quality matters and you have compute budget.

**Edge-weighted** if image is edge-dominated (text, diagrams).

**R-D** only if building compression system.

---

## Part 4: Rendering (How to Generate Image from Gaussians)

### The Basic Question
Given Gaussians, how do you create pixel values?

### Core Concept: Splatting
Each Gaussian contributes to nearby pixels based on distance.

**Formula:** `weight(x,y) = alpha × exp(-0.5 × [(x-μ)ᵀ Σ⁻¹ (x-μ)])`

Where Σ is covariance matrix (captures σ_parallel, σ_perp, θ).

### Practical Variations (2 that matter)

#### 1. **Alpha Compositing (Over Operator)**
**Concept:** Layer Gaussians front-to-back, blend with opacity.

**Formula:** `I_final = Σ alpha_i × color_i × (1 - alpha_accumulated)`

**When to use:**
- Standard approach
- Natural occlusion behavior
- Order-dependent (sort by depth if 3D)

**Pros:** Realistic blending
**Cons:** Order matters (need sorting), slightly slower

---

#### 2. **Additive Accumulation**
**Concept:** Sum all Gaussian contributions.

**Formula:** `I_final = Σ alpha_i × color_i × weight_i`

**When to use:**
- Order-independent (no sorting needed)
- Research (simpler gradients)
- GaussianImage ECCV 2024 approach

**Pros:** Simple, order-independent, fast
**Cons:** Can over-saturate (need to normalize alpha)

---

### CPU vs GPU

**CPU:**
- Simple implementation (NumPy)
- 100×100 image: <1s
- 1000×1000 image: ~10s
- Good for research

**GPU:**
- 10-100× faster
- Needed for interactive (>1MP images)
- More complex setup

**Recommendation:** Start CPU, move to GPU only if speed matters.

---

## Part 5: Analysis Tools (How to Evaluate Quality)

### Core Metrics (4 essential)

#### 1. **PSNR (Peak Signal-to-Noise Ratio)**
**Formula:** `10 × log10(1 / MSE)`

**Range:** 20-50 dB typical (higher = better)

**Interpretation:**
- <25 dB: Poor quality
- 25-30 dB: Acceptable
- 30-35 dB: Good
- 35-40 dB: Very good
- >40 dB: Excellent/near-lossless

**Pros:** Standard, comparable across studies
**Cons:** Doesn't match perception well

**When to use:** Always report (baseline metric)

---

#### 2. **SSIM (Structural Similarity)**
**Range:** 0-1 (1 = perfect)

**Interpretation:**
- <0.8: Poor
- 0.8-0.9: Acceptable
- 0.9-0.95: Good
- >0.95: Excellent

**Pros:** Better perceptual correlation than PSNR
**Cons:** Slower to compute

**When to use:** Perceptual quality assessment

---

#### 3. **Visual Inspection**
**Tools:** Save rendered images, compare side-by-side

**What to look for:**
- Halos (over-blurring)
- Gaps (under-coverage)
- Color banding
- Texture loss
- Edge artifacts

**When to use:** Always (numbers lie, eyes don't)

---

#### 4. **Gaussian Efficiency**
**Metric:** PSNR per Gaussian

**Example:** 32 dB with 100 Gaussians = 0.32 dB/Gaussian

**When to use:**
- Comparing placement strategies
- Rate-distortion analysis
- Understanding efficiency

---

## Part 6: Putting It Together (Practical Workflow)

### Typical Pipeline

```
1. Image Analysis
   ↓
   [Compute gradients, entropy, features]
   ↓
2. Placement Strategy
   ↓
   [Choose method: gradient/entropy/feature-based]
   ↓
3. Initialize Gaussians
   ↓
   [Position + rough parameter guesses]
   ↓
4. Optimization (optional)
   ↓
   [Adam/L-BFGS for 500-2000 iterations]
   ↓
5. Rendering
   ↓
   [Splat Gaussians → pixel values]
   ↓
6. Evaluation
   ↓
   [PSNR, SSIM, visual inspection]
```

### Example: Edge-Heavy Image (Diagram)

**Best choices:**
- **Placement:** Gradient-based (Sobel + threshold)
- **Initialization:** Elongated Gaussians along edges (σ_parallel >> σ_perp)
- **Optimization:** Adam, 1000 iters, edge-weighted loss
- **Rendering:** Alpha compositing
- **Evaluation:** PSNR + visual (check edge sharpness)

### Example: Natural Photo

**Best choices:**
- **Placement:** Entropy-based (balanced coverage)
- **Initialization:** Isotropic Gaussians, size ∝ local frequency
- **Optimization:** Adam, 2000 iters, MS-SSIM loss
- **Rendering:** Alpha compositing
- **Evaluation:** SSIM + visual (check texture, faces)

---

## Part 7: What Actually Matters (The Short Version)

### If You Remember Only 3 Things:

**1. Placement matters more than optimization**
- Good initialization (gradient-based) with no optimization > random initialization with 2000 iterations
- Get placement right first

**2. Start simple, add complexity only if needed**
- Gradient-based placement + Adam + MSE gets you 80% of the way
- MS-SSIM, feature-based, error-driven refinement = last 20%

**3. Visual inspection beats metrics**
- PSNR can be high while image looks bad (over-smoothing)
- Always look at the rendered result

---

## Part 8: Common Mistakes to Avoid

**1. Too many Gaussians in flat regions**
→ Use content-adaptive placement (gradient/entropy)

**2. Under-sampled edges**
→ Use gradient-based or feature-based placement

**3. Wrong Gaussian sizes**
→ Match Gaussian σ to local feature scale (σ_perp ≈ edge blur)

**4. Over-optimization**
→ 500-1000 iterations often enough, diminishing returns after

**5. Ignoring visual quality**
→ Metrics are guidelines, not gospel

---

## Part 9: Research Directions We've Explored

### What Worked Well
- ✓ Gradient-based placement (simple, effective)
- ✓ Adam optimization (robust, reliable)
- ✓ Content-adaptive parameter selection (N from entropy)
- ✓ Error-driven refinement (+4 dB improvement documented)

### What's Still Open Questions
- ? Optimal Gaussian size rules (Phase 0 goal!)
- ? Feature-specific strategies (edges vs regions)
- ? Layered vs monolithic (need empirical test)
- ? Texture handling (dense Gaussians vs parametric)

### What Didn't Work
- ✗ Uniform/random placement (wastes Gaussians)
- ✗ Very large σ (over-smoothing)
- ✗ Too few iterations (<100, under-optimized)

---

## Part 10: Quick Reference

### Placement Decision Tree

```
Image Type?
├─ Edges/diagrams → Gradient-based
├─ Natural photos → Entropy-based
├─ Mixed content → Feature-based
└─ Unknown → Error-driven (iterative)
```

### Optimization Decision Tree

```
Problem Size?
├─ N < 50 → L-BFGS (fast convergence)
├─ N = 50-1000 → Adam (default)
├─ N > 1000 → Adam with batching
└─ Rules known → No optimization (analytic)
```

### Loss Function Decision Tree

```
Content Type?
├─ Research/baseline → MSE
├─ Natural photos → MS-SSIM
├─ Edges/diagrams → Edge-weighted MSE
└─ Codec → Rate-distortion
```

---

## Conclusion: Your Practical Toolkit

**Core Tools (use these):**
- Gradient-based placement
- Adam optimizer
- MSE or MS-SSIM loss
- Alpha compositing renderer
- PSNR + visual inspection

**Advanced Tools (use when needed):**
- Feature-based placement (interpretability)
- Error-driven refinement (quality)
- Edge-weighted loss (edge-heavy images)
- L-BFGS (small problems)

**Research Tools (Phase 0):**
- Manual placement + parameter sweeps
- Understand primitives before optimizing
- Derive empirical rules

**Remember:** The catalog has ~40-50 distinct concepts. Most are variations on these core themes. Start with the basics, add complexity only when justified by results.

---

**This guide is your map. The catalog is the detailed atlas. Use the map until you need the atlas.**
