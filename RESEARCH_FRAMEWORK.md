# LGI Research Framework: Experimental Phase

**Status:** ACTIVE EXPERIMENTATION
**Phase:** Primitive Discovery & Characterization
**Duration:** Indefinite (as long as yielding insights)
**Last Updated:** 2025-11-17

---

## Current Understanding

### What We're NOT Building (Yet)

- ❌ A complete image codec
- ❌ A production system
- ❌ A fully specified standard
- ❌ Optimized performance code

### What We ARE Doing

✅ **Exploring Gaussian primitives** - understanding how different feature types (edges, regions, junctions, blobs, textures) can be efficiently represented with Gaussians

✅ **Discovering content-adaptive rules** - finding empirical relationships between feature descriptors (curvature, blur, contrast, variance) and optimal Gaussian configurations

✅ **Testing initialization strategies** - determining whether content-aware placement outperforms naive approaches

✅ **Comparing iteration methods** - evaluating different optimizers, learning rates, and constraint strategies per primitive type

✅ **Building experimental knowledge** - accumulating data to inform future architectural decisions

---

## Core Research Questions

### Q1: Primitive Efficiency
**For each feature type, what is the minimal Gaussian configuration that achieves acceptable quality?**

- Edges: Gaussians per unit length as f(curvature, blur, contrast)
- Regions: Gaussians per area as f(variance, gradient energy, boundary complexity)
- Junctions: Gaussian cluster patterns for L/T/X configurations
- Blobs: Single vs multi-Gaussian representations
- Textures: Dense micro-Gaussians vs parametric models

### Q2: Initialization Impact
**Does content-adaptive initialization provide measurable benefits over random/uniform placement?**

- Convergence speed (iterations to target quality)
- Final quality (PSNR at convergence)
- Stability (variance across runs)
- Visual artifacts (halos, gaps, over-smoothing)

### Q3: Optimization Strategies
**What iteration methods work best for each primitive type?**

- Optimizers: Adam, L-BFGS, SGD, custom
- Learning rates: adaptive vs fixed
- Parameter constraints: which to fix, which to optimize
- Progressive refinement: coarse-to-fine strategies

### Q4: Layered vs Monolithic Architecture
**KEY INSIGHT: Does decomposing into layers enable better overall efficiency?**

**The Computational Tradeoff:**
- **Monolithic:** 1 optimization × 500 Gaussians
  - Single large parameter space
  - Complex loss landscape
  - One optimizer for all features

- **Layered:** 6 optimizations × ~75 Gaussians each
  - 6 simpler parameter spaces
  - Specialized optimizers per layer
  - Potentially parallel fitting
  - Different iteration budgets per layer

**Critical questions:**
- Do layered optimizations converge faster individually?
- Does specialization (different methods per primitive) improve efficiency?
- What's the overhead of multiple passes vs single pass?
- Can layers be parallelized or must they be sequential?

### Q5: Harmonization
**When combining primitives, how do we handle overlaps and interactions?**

- Additive vs compositional blending
- Boundary artifacts between feature types
- Gaussian budget allocation across layers
- Joint vs sequential refinement

---

## Methodological Flexibility: The Layering Advantage

### Key Insight
**Because primitives are separate, we can use COMPLETELY DIFFERENT iteration strategies for each.**

#### Example: Optimization Method per Layer

**Macro (M):** Few large Gaussians, smooth landscape
- **Optimizer:** L-BFGS (good for smooth, low-dimensional)
- **Iterations:** 100-200 (converges quickly)
- **Constraints:** Minimum size (prevent collapse)

**Edges (E):** Many elongated Gaussians along curves
- **Optimizer:** Adam with momentum
- **Iterations:** 500-1000 (needs more refinement)
- **Constraints:** Fix orientation θ = tangent, optimize position + width

**Junctions (J):** Small clusters at feature points
- **Optimizer:** Constrained optimization (keep Gaussians near junction)
- **Iterations:** 200-300
- **Constraints:** Radial distance < threshold from junction center

**Regions (R):** Interior fill Gaussians
- **Optimizer:** SGD with region mask constraint
- **Iterations:** 300-500
- **Constraints:** Stay inside region polygon

**Blobs (B):** Simple isotropic Gaussians
- **Optimizer:** Analytical solution (if blob IS a Gaussian)
- **Iterations:** 1 (closed form)
- **Constraints:** None

**Texture (T):** Dense micro-Gaussians or parametric
- **Optimizer:** Stochastic methods or dictionary learning
- **Iterations:** Varies by strategy
- **Constraints:** Frequency-domain constraints

### This flexibility is IMPOSSIBLE in monolithic approaches

---

## Computational Hypothesis

**H-Layered-Efficiency:** For equal total Gaussian budget K_total, a layered approach with specialized optimizers per layer will:

1. **Converge faster** (total wall-clock time)
   - Each layer solves a simpler problem
   - Specialized optimizers exploit structure

2. **Achieve better quality** (final PSNR/SSIM)
   - Content-adaptive initialization per layer
   - Residual-based fitting reduces interference

3. **Use fewer total iterations** (sum of iterations across layers < monolithic iterations)
   - Example: M(100) + E(500) + J(200) + R(300) + B(1) + T(400) = 1501 total
   - vs Monolithic: 3000+ iterations for same quality

**This is testable and will be a major focus of CCW experiments.**

---

## Current Experimental Strategy

### Phase 1: Single Primitive Characterization (Current)

**Focus:** Edge primitive (E)

**Approach:** Isolated experiments on synthetic edges
- Vary: curvature, blur, contrast, length
- Test: initialization strategies, optimizer choices, Gaussian density
- Measure: convergence speed, final quality, efficiency metrics
- Outcome: Empirical rules for edge primitive

**When complete:** Move to Region (R) or Blob (B) primitive

### Phase 2: Multi-Primitive Composition (Future)

**Focus:** Combining 2-3 primitives
- Test cases: Edge + Region (rectangle with border)
- Approaches: Sequential (M→E→R) vs joint fitting
- Measure: Boundary artifacts, total quality, efficiency
- Outcome: Understanding of harmonization requirements

### Phase 3: Layered vs Monolithic Comparison (Future)

**Focus:** Direct comparison at fixed Gaussian budget
- Monolithic: K Gaussians, no feature types, free optimization
- Layered: K Gaussians distributed across layers with specialized methods
- Measure: Quality, convergence, visual artifacts
- Outcome: Validation of core hypothesis

### Phase 4: Real Image Validation (Future)

**Focus:** Applying learned rules to detected features in real images
- Use existing feature detectors from 160+ tool catalog
- Test generalization of synthetic-derived rules
- Identify failure modes and edge cases

---

## Available Resources

### Tool Catalog (160+ Algorithms)
From consolidated branches, we have extensive tools:

**Placement Strategies (23):**
- Grid-based, clustering, feature-driven, gradient-aware
- Poisson disk, blue noise, adaptive sampling
- Curvature-aware, edge-aligned, junction-centered

**Optimization Methods (34):**
- First-order: Adam, SGD, RMSprop, Adagrad
- Second-order: L-BFGS, Newton, Gauss-Newton
- Constrained: projected gradient, penalty methods
- Specialized: per-Gaussian, per-layer, joint refinement

**Feature Detectors:**
- Edges: Canny, Sobel, structure tensor, multi-scale
- Junctions: Harris, FAST, Förstner
- Regions: watershed, SLIC, mean-shift, graph-cut
- Blobs: LoG, DoG, MSER
- Textures: Gabor, LBP, spectral analysis

**Rendering Engines (7):**
- CPU: naive, tiled, hierarchical
- GPU: CUDA splatting, shader-based
- Differentiable: autograd-compatible

**Analysis Tools (32+):**
- Metrics: PSNR, SSIM, MS-SSIM, LPIPS
- Visualization: layer decomposition, Gaussian overlays, residuals
- Profiling: timing, memory, convergence tracking

### Existing Baselines
From recovered research branches:
- 3 baseline implementations from reference papers
- Corrected gradient computation modules
- Convergence testing frameworks
- N-variation study results (N=20-100)

---

## Experimental Infrastructure

### Required Components

**1. Gaussian Renderer** ✓ (exists in lgi-core)
- EWA splatting implementation
- Differentiable for gradient computation
- Supports layered rendering

**2. Synthetic Data Generator** (to build)
- Edge generator: straight, curved, blurred
- Region generator: polygons with gradients
- Junction generator: L/T/X configurations
- Blob generator: Gaussian and non-Gaussian
- Texture generator: periodic and stochastic

**3. Optimizer Framework** (exists in lgi-encoder-v2)
- Adam optimizer with rotation-aware gradients
- Loss functions: MSE, SSIM-based
- Correct gradient computation module

**4. Logging & Analysis** (to build)
- Experiment tracking: parameters, metrics, images
- Convergence visualization
- Comparative analysis across strategies
- Automated report generation

### Data Management

**Experiment outputs:**
- Checkpointed Gaussian parameters (every N iterations)
- Rendered images (initial, intermediate, final)
- Loss curves and convergence metrics
- Visual artifact analysis (manual and automated)

**Organization:**
```
experiments/
  edge/
    straight_sharp/
      uniform_N10_adam_lr001/
        config.json
        loss_curve.txt
        gaussians_final.json
        images/
      curvature_adaptive_N10_adam_lr001/
      ...
    curved_r100_blur2/
      ...
  region/
    ...
  comparative/
    layered_vs_monolithic/
      ...
```

---

## Success Criteria

### Experiment-Level Success
An individual experiment succeeds if:
- ✓ Converges to target quality (PSNR > threshold)
- ✓ Produces expected visual output (no major artifacts)
- ✓ Completes within iteration budget
- ✓ Generates complete logs and outputs

### Strategy-Level Success
An initialization/optimization strategy succeeds if:
- ✓ Consistently outperforms baseline (>10% improvement in metric)
- ✓ Works across multiple test cases (not tuned to one scenario)
- ✓ Demonstrates clear pattern (can be formulated as rule)
- ✓ Is computationally practical (not 10× slower than alternatives)

### Research-Level Success
The overall experimental phase succeeds if:
- ✓ We can write empirical rules for 2-3 primitives
- ✓ Content-adaptive initialization shows measurable benefits
- ✓ Layered approach demonstrates advantages over monolithic
- ✓ We have enough data to design next research phase

### Acceptable Failure
Experiments that "fail" but provide insights:
- ✓ Clear negative results (X doesn't work because Y)
- ✓ Boundary cases identified (strategy works for A but not B)
- ✓ Unexpected behaviors documented (opens new questions)

**Unacceptable:** Experiments that fail due to bugs, incomplete data, or poor documentation.

---

## Decision Criteria for Continuing/Pivoting

### Continue Current Line if:
- Results show consistent patterns across test cases
- Improvements are measurable and significant (>10%)
- New questions emerge that are answerable with current setup
- Computational cost is reasonable

### Pivot to New Primitive if:
- Current primitive is well-characterized (rules established)
- Diminishing returns on further refinement
- Clear next primitive is needed for composition tests

### Pivot to Different Approach if:
- No clear patterns emerge after extensive testing
- Content-adaptive methods don't outperform baselines
- Computational costs are prohibitive
- Fundamental assumption is invalidated

### Stop Experimentation if:
- Core hypothesis is thoroughly refuted
- Gaussians prove inadequate for the task
- No path forward is apparent

**Current stance:** We expect experimentation to continue for weeks-months, yielding incremental insights.

---

## Relationship to LGI Specification

The `LGI_spec_and_experiments.md` document serves as:
- **Conceptual framework:** What we're ultimately exploring
- **Feature vocabulary:** Standard terminology for primitives
- **Hypothesis bank:** Long-term questions to answer
- **Experimental roadmap:** Structured progression of tests

But it is **NOT**:
- A rigid specification to implement
- A commitment to any particular architecture
- A guarantee that layering will work

**The spec evolves based on experimental findings.**

If experiments show:
- Junctions don't need special treatment → remove J layer
- Texture needs 3 sub-types → expand T layer
- Layering has no benefit → pivot to monolithic + feature-aware init
- Certain rules don't generalize → refine or discard

---

## CCW Experimentation Philosophy

### Autonomy & Flexibility
Claude Code Web (CCW) sessions are empowered to:
- ✓ Adjust experiment parameters based on intermediate results
- ✓ Extend successful experiments with additional test cases
- ✓ Prune unproductive lines of inquiry early
- ✓ Propose new experiments based on observed patterns
- ✓ Use any tools from the 160+ algorithm catalog
- ✓ Make implementation choices (within documented constraints)

### Structured Exploration
CCW must:
- ✓ Document all decisions and rationale
- ✓ Log complete experiment parameters and results
- ✓ Follow scientific method (control variables, measure, analyze)
- ✓ Produce reproducible experiments
- ✓ Summarize findings in human-readable reports

### Communication
CCW should:
- ✓ Report preliminary findings after each experiment batch
- ✓ Flag unexpected results or anomalies
- ✓ Request guidance when facing ambiguous design choices
- ✓ Propose next steps based on current results

---

## Next Steps (Immediate)

### For Human Researcher:
1. ✓ Review this framework document
2. Prepare CCW instructions with experimental protocol
3. Queue up massive parallel experimentation for 2-3 day sprint
4. Review results and synthesize findings

### For CCW Session:
1. Implement minimal experimental infrastructure
2. Run comprehensive edge primitive experiments (primary focus)
3. Explore layered vs monolithic comparison (key insight)
4. Begin region/blob experiments if time permits
5. Generate detailed analysis reports

**Time budget:** 2-3 days of intensive CCW experimentation with free credits.

---

## Long-Term Vision (Provisional)

If experimentation yields positive results over months:
- Formalize empirical rules into analytic functions
- Build prototype LGI encoder with learned rules
- Test on real images with detected features
- Evaluate against baseline codecs
- Consider learned components (small NNs) to augment rules
- Potentially publish findings

If experimentation yields negative/mixed results:
- Pivot to alternative representations (wavelets, learned dictionaries)
- Simplify to monolithic feature-aware Gaussians
- Focus on specific use cases (diagrams, line art, not natural photos)
- Extract useful sub-components (optimization methods, rendering techniques)

**Either outcome is valuable scientific knowledge.**

---

**Status:** Ready for intensive CCW experimentation phase
**Expected duration of current phase:** 2-3 days (CCW sprint), then synthesis and planning
