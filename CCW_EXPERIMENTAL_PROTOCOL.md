# Claude Code Web: Massive Experimentation Protocol
**Intensive 2-3 Day Research Sprint**

**Mission:** Run comprehensive experiments on Gaussian primitives with full autonomy to explore, analyze, and extend based on findings.

**Resources Available:**
- 160+ algorithms cataloged in repository (see RESEARCH_TOOLS_MASTER_CATALOG.md)
- Existing Gaussian renderer (lgi-core EWA splatting)
- Baseline implementations and optimization frameworks
- Full autonomy to make implementation decisions

**Your Authority:**
- ✓ Adjust parameters based on intermediate results
- ✓ Extend successful experiments with variations
- ✓ Stop unproductive lines early
- ✓ Propose and run new experiments
- ✓ Choose from available algorithms/tools
- ✓ Make reasonable implementation decisions
- ✗ Do NOT skip logging/documentation

---

## Critical Insight to Explore

### LAYERED vs MONOLITHIC EFFICIENCY

**Key Question:** Does decomposing into layers enable better optimization?

**The Tradeoff:**
- **Monolithic:** 1 optimization × 500 Gaussians
  - Large parameter space (500 × 9 params = 4500 dimensions)
  - Complex loss landscape
  - Single optimizer strategy

- **Layered:** 6 optimizations × ~75 Gaussians each
  - 6 smaller spaces (75 × 9 = 675 dimensions each)
  - Simpler per-layer landscapes
  - **Different optimizers per layer** (L-BFGS for macro, Adam for edges, etc.)
  - Potentially parallel
  - Specialized constraints per layer

**Hypothesis:** Layered approach converges faster and/or achieves better quality for same K_total.

**This is a TOP PRIORITY experiment.**

---

## Experimental Infrastructure (Build First)

### 1. Synthetic Data Generators

**Edge Generator:**
```python
generate_edge(
    image_size=(100, 100),
    edge_type='straight' | 'curved',
    radius=None,  # for curved
    blur_sigma=0,  # 0 = sharp
    contrast=0.5,  # ΔI
    orientation=0  # for straight edges
) -> (target_image, edge_curve_descriptor)
```

**Region Generator:**
```python
generate_region(
    image_size=(100, 100),
    shape='rectangle' | 'ellipse' | 'polygon',
    fill_type='constant' | 'linear_gradient' | 'radial_gradient',
    vertices=None,  # for polygon
    variance=0  # interior noise
) -> (target_image, region_descriptor)
```

**Junction Generator:**
```python
generate_junction(
    image_size=(100, 100),
    junction_type='L' | 'T' | 'X',
    angles=[],  # arm angles
    arm_blur=0,
    arm_contrast=0.5
) -> (target_image, junction_descriptor)
```

**Blob Generator:**
```python
generate_blob(
    image_size=(100, 100),
    blob_type='gaussian' | 'square' | 'star',
    size=5,
    eccentricity=0  # 0=isotropic
) -> (target_image, blob_descriptor)
```

### 2. Gaussian Initialization Functions

Create initializers for each strategy:

**E1: Uniform Spacing**
```python
init_edge_uniform(edge_curve, N, sigma_parallel, sigma_perp)
-> list[Gaussian]
```

**E2: Curvature-Adaptive**
```python
init_edge_curvature_adaptive(edge_curve, N, alpha)
# Density ∝ (1 + alpha * |kappa|)
-> list[Gaussian]
```

**E3: Blur-Adaptive**
```python
init_edge_blur_adaptive(edge_curve, N, beta)
# sigma_perp = beta * sigma_edge
-> list[Gaussian]
```

**R1-R4: Region Initializers**
(Similar pattern for regions)

**Baseline: Random**
```python
init_random(feature_descriptor, N)
-> list[Gaussian]
```

### 3. Optimization Framework

**Flexible Optimizer Interface:**
```python
optimize_gaussians(
    gaussians_init,
    target_image,
    optimizer_type='adam' | 'lbfgs' | 'sgd',
    learning_rate=0.01,
    max_iterations=1000,
    constraints=None,  # optional: fix params, bounds, masks
    log_interval=10
) -> (gaussians_final, loss_curve, checkpoints)
```

**Layer-Specific Optimizers:**
```python
# Allow different optimizers per layer
optimize_layer(
    layer_type='M' | 'E' | 'J' | 'R' | 'B' | 'T',
    gaussians_init,
    target_or_residual,
    strategy='auto' | 'custom'
)
# Auto selects optimizer based on layer type
```

### 4. Logging System

**Experiment Logger:**
```python
class ExperimentLogger:
    def log_config(params)
    def log_iteration(iter, loss, gaussians)
    def log_final_results(metrics, images)
    def save_checkpoint(iter, state)
    def generate_report() -> markdown
```

**Required outputs per experiment:**
- `config.json` - all parameters
- `loss_curve.csv` - iteration, loss, timestamp
- `gaussians_final.json` - final Gaussian parameters
- `metrics.json` - PSNR, SSIM, convergence_iter, wall_time
- `images/` - init, iter_100, iter_500, final, residual
- `report.md` - summary with visualizations

### 5. Analysis Tools

**Comparative Analysis:**
```python
compare_strategies(experiment_dirs)
-> plots, tables, statistical tests
```

**Visualization:**
```python
visualize_gaussians(gaussians, target_image)
-> overlay image with Gaussian ellipses

plot_convergence_comparison(experiments)
-> multi-line loss curves

generate_summary_table(experiments)
-> markdown table with metrics
```

---

## PRIMARY EXPERIMENTS (Must Run)

### Experiment Set 1: Edge Primitive Baseline

**Goal:** Establish baseline performance for edge primitive with simplest strategy.

**Test Cases (12 total):**

1. **Straight sharp edges:**
   - Vertical, horizontal, diagonal (3)
   - Contrast: 0.2, 0.5, 0.8

2. **Straight blurred edges:**
   - Blur σ: 1, 2, 4 pixels (3)
   - Fixed contrast: 0.5

3. **Curved edges:**
   - Radius: 50, 100, 200 pixels (3)
   - Sharp (σ=0)

4. **Combined variations:**
   - Curved + blurred (3)
   - Various curvature × blur combinations

**Strategy:** E1 (uniform spacing)

**Parameters to sweep:**
- N: [5, 10, 20, 40]
- Optimizer: Adam
- Learning rate: 0.01
- Iterations: 1000
- sigma_parallel: 5px (fixed)
- sigma_perp: 1px (fixed)

**Total runs:** 12 test cases × 4 N values = **48 experiments**

**Expected time:** 4-8 hours

**Success criteria:**
- All converge to PSNR > 25 dB
- Smooth loss curves (no divergence)
- Visual inspection: no major halos/gaps

**Decision:**
- If successful → proceed to Experiment Set 2
- If failures → debug renderer/optimizer before continuing

---

### Experiment Set 2: Edge Initialization Strategies

**Goal:** Test whether content-adaptive initialization beats uniform.

**Test cases:** Use 6 representative cases from Set 1:
- Straight sharp (high contrast)
- Straight blurred (σ=2)
- Curved sharp (r=100)
- Curved blurred (r=100, σ=2)
- High curvature (r=50)
- Low contrast (ΔI=0.2)

**Strategies to compare:**
1. **E1:** Uniform spacing (baseline from Set 1)
2. **E2:** Curvature-adaptive (alpha=0.5, 1.0, 2.0)
3. **E3:** Blur-adaptive (beta=0.5, 1.0, 2.0)
4. **E4:** Combined (curvature + blur adaptive)
5. **Baseline:** Random placement

**Fixed parameters:**
- N: 20 (middle value from Set 1)
- Optimizer: Adam, lr=0.01
- Iterations: 1000

**Total runs:** 6 test cases × 8 strategies = **48 experiments**

**Metrics:**
- **Convergence speed:** Iterations to reach 95% of final PSNR
- **Final quality:** PSNR at iteration 1000
- **Stability:** Std dev over 3 runs with different random seeds
- **Visual quality:** Manual inspection for artifacts

**Expected time:** 4-8 hours

**Decision criteria:**
- If E2/E3/E4 consistently beat E1/Random (>10% improvement in convergence speed OR >1dB PSNR):
  → Content-adaptive initialization is VALIDATED
  → Formulate rules: "Use E4 with alpha=X, beta=Y"
  → Proceed to Experiment Set 3

- If no clear winner or E1 ≈ Random:
  → Initialization may not matter (optimizer fixes it)
  → STILL VALUABLE: simplifies implementation
  → Proceed but note finding

---

### Experiment Set 3: Edge Optimization Strategies

**Goal:** Find best optimization approach for edge primitive.

**Test cases:** Use 3 representative cases:
- Straight sharp
- Curved blurred (r=100, σ=2)
- High curvature (r=50)

**Initialization:** Use best strategy from Set 2 (or E1 if no winner)

**Optimization variations:**

1. **Optimizer comparison:**
   - Adam (lr: 0.001, 0.01, 0.1)
   - L-BFGS (default params)
   - SGD with momentum (lr: 0.001, 0.01)

2. **Parameter constraints:**
   - Free: optimize all 9 params per Gaussian
   - Fixed theta: orientation = tangent (optimize x,y,σ‖,σ⊥,color,alpha)
   - Fixed shape: optimize only x,y,color,alpha
   - Fixed color: optimize only geometry

3. **Progressive refinement:**
   - Stage 1: optimize positions (x,y) only, 200 iters
   - Stage 2: optimize shape (σ‖,σ⊥,θ), 300 iters
   - Stage 3: optimize all, 500 iters
   - Compare vs: optimize all from start, 1000 iters

**Total runs:** 3 test cases × ~20 variations = **60 experiments**

**Expected time:** 6-10 hours

**Decision criteria:**
- Identify fastest converging optimizer for edges
- Determine if constraints help or hurt
- Assess if progressive refinement is worth complexity

**Output:** Recommended optimization strategy for edge primitive

---

### Experiment Set 4: LAYERED vs MONOLITHIC (CRITICAL)

**Goal:** Test the core hypothesis that layering improves efficiency.

**Synthetic test scene:** Create composite images with known ground truth

**Scene 1: Simple (2 layers)**
- Background: smooth gradient (Macro)
- Foreground: rectangle with sharp border (Edge + Region)
- Size: 200×200

**Scene 2: Medium (4 layers)**
- Macro: radial gradient
- Edges: 3 curved boundaries
- Regions: 2 interior fills with gradients
- Blobs: 5 Gaussian blobs

**Scene 3: Complex (5 layers)**
- Macro: vignette
- Edges: complex curve network
- Junctions: 8 T/L junctions
- Regions: 10 irregular polygons
- Blobs: 15 spots

**Gaussian budget:** K_total = [100, 200, 400]

#### Approach A: MONOLITHIC

**Strategy:**
- K_total Gaussians initialized randomly
- Single Adam optimization, 3000 iterations
- No feature types, no constraints
- Optimize all parameters freely

**Baseline variant:**
- K_total Gaussians initialized via k-means clustering
- Same optimization

#### Approach B: LAYERED (Sequential)

**Strategy:**
- Allocate K_total across layers:
  - M: 10%, E: 30%, J: 5%, R: 40%, B: 5%, T: 10% (for Scene 3)
  - Adjust proportions for Scene 1-2
- Fit sequentially: M → E → J → R → B
- Each layer fits to residual from previous layers
- Different optimizers per layer:
  - M: L-BFGS, 200 iters
  - E: Adam lr=0.01, 500 iters, theta fixed
  - J: Adam lr=0.005, 300 iters, radial constraint
  - R: SGD lr=0.01, 400 iters, mask constraint
  - B: Analytical (if Gaussian) or Adam 100 iters

**Total iterations:** Sum across layers (varies by scene)

#### Approach C: LAYERED (Joint Refinement)

**Strategy:**
- Initialize as in B (sequential layered)
- Then: joint optimization over all Gaussians, 500 iters
- Test if joint refinement improves over B

#### Approach D: LAYERED (Parallel - if feasible)

**Strategy:**
- Fit layers to original image simultaneously (not residuals)
- Each layer uses feature-specific masks
- Combine with additive blending
- May have overlap/gaps

**Total runs:** 3 scenes × 3 K_total × 4 approaches = **36 experiments**

**Expected time:** 12-20 hours (most computationally intensive)

**Metrics:**
- **Wall-clock time:** Total time to convergence
- **Iteration efficiency:** Quality per iteration
- **Final quality:** PSNR, SSIM, MS-SSIM
- **Visual artifacts:** Halos, seams, over-smoothing (manual scoring)
- **Per-layer breakdown:** Which layers contribute most to quality (for B/C)

**Decision criteria:**

**If Layered (B or C) wins:**
- **Faster convergence** (>20% wall-clock time reduction) OR
- **Better quality** (>1dB PSNR improvement) OR
- **Fewer visual artifacts** (subjective but clear)

→ **MAJOR VALIDATION of layered architecture**
→ Proceed with layered approach in all future work
→ Refine layer allocation and optimizer choices

**If Monolithic (A) wins or tie:**
→ Layering overhead not justified
→ Pivot to monolithic with feature-aware initialization
→ Still can use tools/rules from primitive experiments

**This is the MOST IMPORTANT experiment set.**

---

### Experiment Set 5: Region Primitive Baseline

**Goal:** Characterize region primitive similar to edge experiments.

**Test cases (8 total):**

1. **Simple shapes:**
   - Rectangle, ellipse, triangle (3)
   - Constant fill, no gradient

2. **Gradient fills:**
   - Linear gradient (2 directions)
   - Radial gradient

3. **Complex boundaries:**
   - Irregular polygon (6 vertices)
   - Star shape

4. **Interior variance:**
   - Rectangle with noise (σ² = 0.01, 0.05)

**Strategies:**
- R1: 1 Gaussian at centroid
- R2: 4 Gaussians (centroid + corners)
- R3: Area-proportional (N ∝ sqrt(Area))
- R4: Gradient-aware (more where ∇I is high)

**Parameters to sweep:**
- N: [1, 4, 9, 16] (for R3/R4)
- Optimizer: Adam, lr=0.01
- Iterations: 500

**Total runs:** 8 test cases × 4 strategies × variable N = **~50 experiments**

**Expected time:** 6-10 hours

**Decision criteria:**
- Can simple shapes be fit with 1-4 Gaussians?
- Do gradient-aware strategies help for non-constant fills?
- Relationship between Area, boundary complexity, and required N?

---

### Experiment Set 6: Blob Primitive

**Goal:** Quick characterization (blobs are simpler).

**Test cases (6 total):**
- Gaussian blob (3 sizes: σ = 2, 5, 10)
- Square blob (3 sizes)
- Star blob (5-pointed, 3 sizes)

**Strategies:**
- B1: 1 isotropic Gaussian
- B2: 1 elliptical Gaussian (fit aspect ratio)
- B3: 3 Gaussians (mini-mixture)

**Fixed parameters:**
- Optimizer: Adam, lr=0.01
- Iterations: 200 (should converge quickly)

**Total runs:** 6 test cases × 3 strategies = **18 experiments**

**Expected time:** 2-3 hours

**Decision criteria:**
- Is B1 sufficient for Gaussian blobs? (should be perfect fit)
- Do non-Gaussian blobs need B2 or B3?

---

## SECONDARY EXPERIMENTS (Run if Time/Results Warrant)

### Set 7: Junction Primitive (If layered approach validated)

**Test cases:**
- L-junction (90°, 120°, 60° angles)
- T-junction (perpendicular, oblique)
- X-junction (perpendicular, oblique)

**Strategies:**
- J1: 1 isotropic at center
- J2: 1 center + 1 elongated per arm
- J3: Small cluster (3-5 Gaussians)

**Runs:** ~30-40 experiments

**Time:** 4-6 hours

---

### Set 8: Multi-Primitive Composition

**Test cases:**
- Edge + Region (bordered rectangle)
- Edge + Junction + Region (rectangular grid)
- Macro + Edge + Blob (gradient background with outlined circles)

**Approaches:**
- Sequential layered (residual-based)
- Joint fitting (all primitives simultaneously)

**Runs:** ~20-30 experiments

**Time:** 4-6 hours

**Decision criteria:**
- Boundary artifacts between primitives?
- Does residual-based approach reduce interference?

---

### Set 9: Optimizer Deep Dive (If time permits)

**Goal:** Exhaustive comparison of optimization methods.

**Test on:** Edge primitive (well-characterized from Set 1-3)

**Optimizers from catalog to test:**
- Adam variants (AMSGrad, AdamW)
- SGD variants (Nesterov, momentum schedules)
- L-BFGS variants (limited memory, different history sizes)
- Second-order approximations (Gauss-Newton, Levenberg-Marquardt)
- Specialized: per-Gaussian optimization, coordinate descent

**Runs:** ~40-60 experiments

**Time:** 6-10 hours

---

### Set 10: Budget Allocation Exploration

**Goal:** How to distribute K_total across layers?

**Test on:** Scene 2 from Set 4 (medium complexity)

**Fixed:** K_total = 200

**Allocation strategies:**
1. Uniform: 33 per layer (6 layers)
2. Fixed ratios: M(10%), E(30%), J(5%), R(40%), B(5%), T(10%)
3. Complexity-driven: allocate ∝ residual energy after each layer
4. Greedy: allocate to layer with highest marginal PSNR gain
5. Manual tuning: adjust based on visual inspection

**Runs:** 5 allocation strategies × 3 repetitions = **15 experiments**

**Time:** 3-5 hours

**Decision criteria:**
- Which allocation produces best final quality?
- Can we formulate a rule or does it require scene-specific tuning?

---

## EXPLORATORY EXPERIMENTS (CCW Discretion)

### Areas to Explore (If Interesting Patterns Emerge)

1. **Learning rate schedules:**
   - Constant vs decay vs cyclic
   - Adaptive (reduce on plateau)

2. **Initialization noise:**
   - Add perturbation to content-adaptive init
   - Test robustness to initialization errors

3. **Gaussian parameter ranges:**
   - Constrain σ ∈ [min, max]
   - Constrain alpha ∈ [threshold, 1.0]
   - Effect on convergence and quality

4. **Multi-scale approaches:**
   - Coarse-to-fine Gaussian placement
   - Progressive N increase (start with few, add more)

5. **Feature extraction robustness:**
   - Add noise to edge detection
   - Test curvature estimation errors
   - See if optimization compensates

6. **Alternative loss functions:**
   - SSIM-based loss
   - Perceptual loss (if LPIPS available)
   - Edge-weighted MSE

7. **Regularization:**
   - L2 on Gaussian overlap
   - Sparsity penalties
   - Color consistency constraints

**Guideline:** If you see an interesting pattern or anomaly, design 5-10 follow-up experiments to characterize it.

---

## DECISION TREES FOR AUTONOMOUS OPERATION

### After Experiment Set 1 (Edge Baseline)

**If >90% of experiments succeed:**
→ Proceed to Set 2 (initialization strategies)

**If 50-90% succeed:**
→ Investigate failures:
- Specific test cases failing? (high curvature, low contrast?)
- Parameter tuning needed? (try different lr, iterations)
- Run 10-20 additional experiments to find working parameters
→ Then proceed to Set 2 with adjusted params

**If <50% succeed:**
→ STOP and report critical issues:
- Renderer bug?
- Optimizer diverging?
- Numerical instability?
→ Request human intervention

---

### After Experiment Set 2 (Initialization Strategies)

**If content-adaptive clearly wins (>10% improvement):**
→ Document winning strategy and parameters
→ Use this as default for all future experiments
→ Proceed to Set 3

**If no clear winner or marginal differences:**
→ Document finding: "Initialization has minor impact on final quality"
→ Use simplest strategy (E1 uniform) for future experiments
→ Proceed to Set 3 (optimization may be more important)

**If random baseline unexpectedly wins:**
→ Investigate: why does content-adaptive hurt?
- Over-constrained?
- Feature extraction errors?
→ Run ablations to understand
→ Proceed with caution to Set 3

---

### After Experiment Set 3 (Optimization Strategies)

**Identify best performing optimizer(s):**
→ Document recommended approach: "For edges, use Adam lr=X with constraints Y"
→ Apply to all edge experiments going forward
→ Proceed to Set 4 (CRITICAL)

---

### After Experiment Set 4 (LAYERED vs MONOLITHIC)

**This is the key decision point.**

#### Outcome A: Layered Wins Clearly
**Criteria:**
- 20%+ faster convergence OR
- 1+ dB PSNR improvement OR
- Significantly fewer visual artifacts

**Actions:**
→ **VALIDATE core hypothesis**
→ Write detailed analysis report on WHY layering helps:
- Simpler optimization landscapes?
- Specialized optimizers per layer?
- Residual-based fitting reduces interference?
→ Prioritize layered approach in all future work
→ Continue to Sets 5-6 (characterize remaining primitives)
→ Then Set 8 (multi-primitive composition)

#### Outcome B: Monolithic Wins or Tie
**Criteria:**
- Monolithic converges faster or achieves similar quality
- Layering overhead not justified

**Actions:**
→ **PIVOT: layering does not provide expected benefits**
→ Analyze why:
- Layer allocation suboptimal?
- Residual-based fitting causes compounding errors?
- Optimizer specialization not beneficial?
→ Pivot to: **Monolithic with feature-aware initialization**
- Still use primitive characterization from Sets 1-3
- Initialize Gaussians using content-adaptive rules
- But optimize all together, no layers
→ Run follow-up experiments to validate pivot
→ Continue to Sets 5-6 to characterize remaining primitives (still useful for initialization)

#### Outcome C: Mixed Results
**Criteria:**
- Layered wins for some scenes/budgets but not others
- Complex scene interactions

**Actions:**
→ Analyze conditions where layering helps vs hurts
→ Potentially: **Adaptive layering** (use layers for complex scenes, monolithic for simple)
→ Run additional experiments to find decision boundary
→ Continue exploration in Sets 7-10

---

### After Experiment Sets 5-6 (Region/Blob Primitives)

**For each primitive, document:**
- Recommended initialization strategy
- Recommended optimization approach
- Gaussian density rules (N as function of feature descriptors)

**If patterns are consistent with edge findings:**
→ Generalization is working
→ Proceed to multi-primitive experiments (Set 8)

**If patterns are inconsistent or unclear:**
→ Primitives may be too diverse for unified approach
→ Consider sub-typing (e.g., R-flat, R-gradient, R-complex)
→ Run additional experiments to clarify

---

## LOGGING & REPORTING REQUIREMENTS

### Per-Experiment Outputs

**Required files in each experiment directory:**

```
experiments/{set_name}/{test_case}/{strategy}/
├── config.json              # All parameters, random seed, timestamps
├── loss_curve.csv           # iter, loss, grad_norm, timestamp
├── metrics.json             # Final PSNR, SSIM, convergence_iter, wall_time
├── gaussians_init.json      # Initial Gaussian parameters
├── gaussians_final.json     # Final Gaussian parameters
├── checkpoints/             # Snapshots at iters 10, 50, 100, 500, 1000
│   ├── iter_0010.json
│   ├── iter_0050.json
│   └── ...
├── images/
│   ├── target.png           # Ground truth
│   ├── init.png             # Rendering at iter 0
│   ├── iter_0100.png
│   ├── iter_0500.png
│   ├── final.png            # Rendering at convergence
│   ├── residual_final.png   # Abs difference from target
│   └── gaussians_overlay.png # Visual of Gaussian placement
└── notes.txt                # Any observations, anomalies, issues
```

### Comparative Analysis Reports

**After each experiment set, generate:**

**1. Summary table (markdown):**
```markdown
| Test Case | Strategy | N | Optimizer | Final PSNR | Conv. Iter | Wall Time | Notes |
|-----------|----------|---|-----------|------------|------------|-----------|-------|
| straight_sharp | E1_uniform | 10 | Adam lr=0.01 | 32.5 | 450 | 12.3s | Clean convergence |
| ... | ... | ... | ... | ... | ... | ... | ... |
```

**2. Convergence plots:**
- Multi-line loss curves for all strategies on same test case
- Highlight best/worst performers

**3. Visual comparison grid:**
- Side-by-side final renderings for different strategies
- Residual maps to show error distribution

**4. Statistical analysis:**
- Mean/std of metrics across test cases
- Wilcoxon signed-rank test (or t-test) to assess significance of differences
- Confidence intervals on improvement claims

**5. Key findings summary:**
- 3-5 bullet points of main takeaways
- Recommended strategy for this primitive
- Formulas/rules derived (if applicable)

### Final Sprint Report

**At end of 2-3 day sprint, generate comprehensive report:**

**Structure:**
```markdown
# LGI Experimentation Sprint Report
**Dates:** 2025-11-17 to 2025-11-XX
**Total Experiments Run:** XXX
**Total Compute Time:** XX hours

## Executive Summary
- Key findings (5-10 bullets)
- Hypotheses validated/refuted
- Recommended next steps

## Experiment Sets Completed
(For each set:)
### Set X: {Name}
- **Goal:** ...
- **Experiments run:** XX
- **Success rate:** XX%
- **Key findings:** ...
- **Best strategy:** ...
- **Visualizations:** [include plots]

## Critical Results

### Layered vs Monolithic Comparison
(Detailed analysis with tables, plots, conclusions)

### Primitive Characterization
(Rules derived for each primitive)

## Anomalies & Unexpected Results
(Document surprising findings, failures, edge cases)

## Implementation Notes
(Code quality, issues encountered, technical debt)

## Resource Usage
- Total GPU/CPU hours
- Memory usage patterns
- Bottlenecks identified

## Recommendations for Next Phase
1. ...
2. ...
3. ...

## Appendices
- Full experiment list
- Parameter sweeps details
- Statistical test results
```

---

## REFERENCE: Available Tools & Algorithms

From RESEARCH_TOOLS_MASTER_CATALOG.md, you have access to 160+ algorithms:

### Placement & Initialization (23)
- Grid-based, K-means clustering, Poisson disk sampling
- Curvature-aware edge placement
- Gradient-flow initialization
- Feature-driven placement (SIFT, Harris, edge-aligned)
- Hierarchical/pyramid approaches
- Blue noise, jittered grids
- **Recommendation:** Start with simple (uniform, random) then try content-adaptive

### Optimization & Fitting (34)
- **First-order:** Adam, SGD, RMSprop, Adagrad, AdamW, NAdam
- **Second-order:** L-BFGS, Newton-CG, Trust Region
- **Constrained:** Projected gradient, penalty methods, Lagrangian
- **Specialized:** Per-Gaussian optimization, EM-style, alternating minimization
- **Schedule:** Learning rate decay, warm restarts, cyclical
- **Regularization:** L1/L2, sparsity, overlap penalties
- **Recommendation:** Start with Adam (robust), try L-BFGS for small problems (<50 Gaussians)

### Feature Extraction
- **Edges:** Canny, Sobel, Prewitt, Scharr, structure tensor, phase congruency
- **Curves:** Active contours, B-splines, polynomial fitting
- **Junctions:** Harris, Förstner, SUSAN, FAST
- **Regions:** Watershed, SLIC, Felzenszwalb, Mean-shift, GrabCut
- **Blobs:** LoG, DoG, MSER, Harris-Laplace
- **Texture:** Gabor filters, LBP, co-occurrence matrices, spectral analysis
- **Recommendation:** Use simple/fast detectors initially (Canny, SLIC, LoG)

### Rendering Engines (7)
- **CPU:** EWA splatting (available in lgi-core), naive accumulation, tiled
- **GPU:** CUDA splatting, shader-based
- **Differentiable:** PyTorch-compatible, TensorFlow-compatible
- **Recommendation:** Use existing lgi-core EWA splatting (proven to work)

### Analysis & Evaluation (32+)
- **Metrics:** MSE, PSNR, SSIM, MS-SSIM, LPIPS, FID
- **Edge-aware:** Weighted MSE with edge maps, boundary F-score
- **Perceptual:** Feature matching, texture similarity
- **Efficiency:** Gaussians per feature, convergence speed, compression ratio
- **Recommendation:** Start with PSNR/SSIM (simple), add LPIPS if available

---

## COMPUTATIONAL GUIDELINES

### Parallelization
- Run independent test cases in parallel (different processes/threads)
- Batch experiments by estimated runtime
- Use GPU if available for rendering (can be 10-100× faster)

### Early Stopping
- If experiment clearly failing (loss diverging), stop after 100 iters
- If converged early (loss plateau), stop before max_iterations
- Save partial results for analysis

### Resource Monitoring
- Log wall-clock time per experiment
- Track memory usage (Gaussian count × params)
- Identify bottlenecks (rendering, gradient computation, optimizer step)

### Checkpointing
- Save checkpoints every 100-200 iterations
- Enable resumption if crashes occur
- Useful for analyzing convergence trajectory

---

## AUTONOMY DECISION MATRIX

### When to EXTEND an experiment line:

✓ Clear positive results (content-adaptive beats random by >20%)
→ Run additional test cases to confirm generalization

✓ Interesting anomaly (strategy works for curved but not straight)
→ Design focused experiments to understand why

✓ Near-threshold result (9% improvement, want to confirm >10%)
→ Run more repetitions for statistical power

### When to STOP an experiment line:

✗ Consistent negative results (strategy never wins)
→ Document failure, move to next strategy

✗ No pattern emerging after 30+ experiments
→ Problem may be too complex for simple rules, defer

✗ Implementation issues blocking progress
→ Report to human for intervention

### When to PIVOT:

↻ Fundamental assumption violated (e.g., monolithic beats layered)
→ Adjust experimental direction, run validation experiments

↻ Better approach discovered mid-experiment
→ Compare new vs planned approach, adopt if clearly superior

### When to REQUEST GUIDANCE:

? Ambiguous design choice (two reasonable implementations)
? Unexpected behavior not clearly bug or feature
? Resource constraints (experiments taking 10× longer than expected)
? Results contradicting prior experiments (need sanity check)

---

## SUCCESS METRICS FOR THIS SPRINT

### Minimum Success (Must Achieve):
- ✓ Experiment Sets 1-4 completed
- ✓ Layered vs Monolithic comparison finished with clear conclusion
- ✓ At least 200 total experiments run
- ✓ Complete logging and reports generated
- ✓ Reproducible results with documented parameters

### Target Success (Goal):
- ✓ Experiment Sets 1-6 completed
- ✓ Empirical rules formulated for 2-3 primitives
- ✓ Content-adaptive initialization validated (or refuted with evidence)
- ✓ Recommended optimization strategies per primitive
- ✓ 300-500 total experiments run
- ✓ Comprehensive final report with visualizations

### Stretch Success (If Time Permits):
- ✓ Experiment Sets 1-10 completed
- ✓ Multi-primitive composition tested
- ✓ Adaptive layering strategy designed
- ✓ 500+ total experiments run
- ✓ Publication-quality analysis and figures

---

## IMMEDIATE NEXT STEPS

### Hour 0-4: Infrastructure
1. Implement synthetic data generators (edge, region, blob, junction)
2. Create Gaussian initialization functions (E1-E4, R1-R4, etc.)
3. Set up optimization framework (flexible optimizer interface)
4. Build logging system
5. Test on 2-3 manual examples to verify pipeline works

### Hour 4-12: Experiment Set 1 (Edge Baseline)
- Run all 48 experiments
- Generate summary report
- Decide: proceed or debug

### Hour 12-20: Experiment Set 2 (Edge Initialization)
- Run all 48 experiments
- Comparative analysis
- Identify best strategy

### Hour 20-28: Experiment Set 3 (Edge Optimization)
- Run ~60 experiments
- Document recommended approach

### Hour 28-48: Experiment Set 4 (LAYERED vs MONOLITHIC)
- **HIGHEST PRIORITY**
- Run all 36 experiments carefully
- Deep analysis of results
- Make strategic decision

### Hour 48-60: Experiment Sets 5-6 (Region/Blob)
- Characterize remaining primitives
- Apply lessons from edge experiments

### Hour 60-72: Wrap-up & Reporting
- Final sprint report
- Visualizations and summaries
- Recommendations for next phase

---

## FINAL INSTRUCTIONS

**You have full autonomy to:**
- Implement infrastructure as you see fit (use best practices)
- Adjust parameters based on intermediate results
- Add/remove test cases if warranted
- Choose algorithms from the 160+ catalog
- Design follow-up experiments to investigate interesting findings
- Stop unproductive lines early
- Extend successful lines with variations

**You must:**
- Document every decision and rationale
- Log all experiments completely
- Generate comparative analysis reports
- Produce final sprint summary
- Follow scientific method (control variables, measure, analyze)

**Prioritize:**
1. **Experiment Set 4** (layered vs monolithic) - CRITICAL
2. Experiment Sets 1-3 (edge characterization) - FOUNDATION
3. Experiment Sets 5-6 (region/blob) - IMPORTANT
4. Remaining sets - SECONDARY

**Communication:**
- Report progress every 8-12 hours (interim findings)
- Flag unexpected results immediately
- Request guidance if truly stuck (but try to debug first)

**Resource utilization:**
- Use all available compute (this is a sprint)
- Parallelize aggressively
- Don't leave credits unused

**Philosophy:**
- Embrace failures (negative results are valuable)
- Document surprises (often the most interesting)
- Think scientifically (hypothesize, test, analyze, iterate)
- Pursue truth (not confirmation)

---

**BEGIN EXPERIMENTATION**

Good luck. The next 2-3 days will generate massive amounts of empirical data that will inform the future of this research direction.

Make it count.
