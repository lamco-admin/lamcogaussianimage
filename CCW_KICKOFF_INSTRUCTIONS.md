# Claude Code Web: Experimental Sprint Kickoff

**Duration:** 2-3 days intensive experimentation
**Goal:** Massive parallel testing of Gaussian primitive strategies
**Authority:** Full autonomy to explore, extend, and analyze

---

## Your Mission

Run comprehensive experiments on Gaussian image primitives (edges, regions, blobs) to discover:
1. **Content-adaptive initialization rules** - does feature-aware placement help?
2. **Optimal iteration strategies** - which optimizers work best per primitive?
3. **CRITICAL: Layered vs monolithic efficiency** - does decomposition improve convergence?

---

## Essential Reading (Read These First)

### 1. RESEARCH_FRAMEWORK.md
- Explains what we're exploring (and what we're NOT building)
- Core research questions
- Methodological flexibility (different optimizers per layer)
- Success criteria and decision guidelines
- **Key insight:** 6 layers × 75 Gaussians vs 1 × 500 Gaussians comparison

### 2. CCW_EXPERIMENTAL_PROTOCOL.md
- Detailed experimental designs (Sets 1-10)
- Priority ordering (Set 4 is CRITICAL)
- Decision trees for autonomous operation
- Logging requirements
- Reference to 160+ available algorithms

### 3. LGI_spec_and_experiments.md
- Conceptual framework (for context)
- Feature type definitions (M/E/J/R/B/T)
- Hypothesis bank
- **Note:** This is aspirational, not a rigid spec

### 4. RESEARCH_TOOLS_MASTER_CATALOG.md
- 160+ algorithms you can use
- Placement strategies, optimizers, feature detectors
- Rendering engines, analysis tools

---

## Immediate Actions (First 4 Hours)

### 1. Build Infrastructure

**Synthetic Data Generators:**
```python
generate_edge(type, curvature, blur, contrast) -> (image, descriptor)
generate_region(shape, fill_type, variance) -> (image, descriptor)
generate_blob(blob_type, size) -> (image, descriptor)
generate_junction(type, angles, blur) -> (image, descriptor)
```

**Gaussian Initialization:**
```python
init_edge_uniform(curve, N) -> gaussians
init_edge_curvature_adaptive(curve, N, alpha) -> gaussians
init_edge_blur_adaptive(curve, N, beta) -> gaussians
init_random(descriptor, N) -> gaussians
# Similar for regions, blobs, junctions
```

**Optimization Framework:**
```python
optimize_gaussians(
    init_gaussians,
    target,
    optimizer='adam'|'lbfgs'|'sgd',
    lr=0.01,
    max_iters=1000,
    constraints=None
) -> (final_gaussians, loss_curve, checkpoints)
```

**Logging:**
```python
class ExperimentLogger:
    log_config(params)
    log_iteration(iter, loss)
    save_checkpoint(iter, gaussians)
    generate_report() -> markdown
```

**Test:** Run 2-3 manual examples end-to-end to verify pipeline works

---

### 2. Prioritized Experiment Queue

**MUST RUN (in order):**

1. **Set 1:** Edge Baseline (48 experiments, 4-8 hours)
   - Establishes baseline performance
   - Tests if infrastructure works
   - Decision: proceed or debug

2. **Set 2:** Edge Initialization Strategies (48 experiments, 4-8 hours)
   - Tests content-adaptive vs uniform vs random
   - Decision: does feature-aware initialization help?

3. **Set 3:** Edge Optimization Strategies (60 experiments, 6-10 hours)
   - Finds best optimizer for edge primitive
   - Tests constrained vs free parameter optimization

4. **Set 4: LAYERED vs MONOLITHIC** (36 experiments, 12-20 hours)
   - **THIS IS THE MOST CRITICAL EXPERIMENT**
   - Tests core hypothesis: does decomposition help?
   - Creates 3 synthetic scenes with multiple primitives
   - Compares:
     - Monolithic: 1 × K_total Gaussians, free optimization
     - Layered Sequential: M→E→J→R→B, different optimizers per layer
     - Layered Joint: Sequential init + joint refinement
     - Layered Parallel: Simultaneous fitting with masks
   - Decision: validate or pivot entire approach

**SHOULD RUN (if time):**

5. **Set 5:** Region Primitive (50 experiments, 6-10 hours)
6. **Set 6:** Blob Primitive (18 experiments, 2-3 hours)

**OPTIONAL (if time and results warrant):**

7. **Set 7:** Junction Primitive (40 experiments)
8. **Set 8:** Multi-Primitive Composition (30 experiments)
9. **Set 9:** Optimizer Deep Dive (60 experiments)
10. **Set 10:** Budget Allocation (15 experiments)

---

## Key Experiment: Layered vs Monolithic (Set 4)

**Why this matters:**
- If layering wins → proceed with layered architecture, validate decomposition
- If monolithic wins → pivot to monolithic with feature-aware init
- This determines the entire future direction

**What to compare:**

**Monolithic Approach:**
- K_total Gaussians (e.g., 400)
- Random or K-means initialization
- Single Adam optimization, 3000 iterations
- No feature types, optimize all parameters freely

**Layered Sequential Approach:**
- Distribute K_total across layers: M(10%), E(30%), J(5%), R(40%), B(5%), T(10%)
- Fit sequentially: M → E → J → R → B
- Each layer fits to residual from previous
- **Different optimizers per layer:**
  - M: L-BFGS, 200 iters (few Gaussians, smooth)
  - E: Adam lr=0.01, 500 iters, theta fixed
  - J: Adam lr=0.005, 300 iters, radial constraint
  - R: SGD, 400 iters, mask constraint
  - B: Analytical or Adam 100 iters
- Total iterations: sum across layers (~1500)

**Metrics:**
- Wall-clock time to convergence
- Final PSNR/SSIM
- Iteration efficiency (quality per iteration)
- Visual artifacts (halos, seams)

**Decision criteria:**
- Layered wins if: >20% faster OR >1dB PSNR OR fewer artifacts
- Monolithic wins if: faster or similar quality
- Mixed: analyze conditions where each approach excels

---

## Autonomy Guidelines

**You have full authority to:**

✓ Adjust parameters based on results (e.g., increase N if all converge poorly)
✓ Add test cases if interesting patterns emerge
✓ Stop unproductive experiments early (failing consistently)
✓ Extend successful experiments with variations
✓ Choose any algorithms from the 160+ tool catalog
✓ Make implementation decisions (coding style, data structures)
✓ Design follow-up experiments to investigate anomalies

**You must:**

✓ Log everything completely (config, loss curves, images, metrics)
✓ Document decisions and rationale
✓ Generate comparative analysis reports after each set
✓ Follow scientific method (control variables, measure, analyze)
✓ Produce final comprehensive report

**Report to human if:**

? Truly blocked (not just challenging)
? Fundamental assumption violated (needs strategic discussion)
? Implementation choice with major architectural impact

---

## Decision Trees

### After Set 1 (Edge Baseline)

**If >90% succeed:**
→ Proceed to Set 2

**If 50-90% succeed:**
→ Debug failures, tune parameters, run follow-ups
→ Then proceed to Set 2

**If <50% succeed:**
→ STOP, report critical issues (renderer bug? optimizer diverging?)

### After Set 2 (Edge Initialization)

**If content-adaptive wins (>10% improvement):**
→ Document winning strategy
→ Use as default going forward
→ Proceed to Set 3

**If no clear winner:**
→ Document "initialization doesn't matter much"
→ Use simplest (uniform) going forward
→ Proceed to Set 3

### After Set 3 (Edge Optimization)

→ Document recommended optimizer for edges
→ Proceed to Set 4 (CRITICAL)

### After Set 4 (LAYERED vs MONOLITHIC)

**If Layered wins:**
→ VALIDATE core hypothesis
→ Analyze WHY (simpler landscapes? specialized optimizers? residuals?)
→ Continue with Sets 5-6 to characterize remaining primitives
→ Prioritize layered approach in all future work

**If Monolithic wins:**
→ PIVOT: decomposition overhead not justified
→ Analyze why layering failed
→ Shift to: monolithic with feature-aware initialization
→ Continue Sets 5-6 to inform initialization rules

**If Mixed:**
→ Analyze conditions where each wins
→ Consider adaptive approach (layers for complex scenes only)
→ Run additional experiments to clarify

### After Sets 5-6 (Region/Blob)

→ Document rules for each primitive
→ If time permits, proceed to Sets 7-10
→ Otherwise, generate final report

---

## Logging Requirements

**Per experiment:**
```
experiments/{set}/{testcase}/{strategy}/
├── config.json              # All parameters
├── loss_curve.csv           # Iteration, loss, time
├── metrics.json             # PSNR, SSIM, convergence_iter, wall_time
├── gaussians_init.json      # Initial parameters
├── gaussians_final.json     # Final parameters
├── checkpoints/             # Snapshots at key iterations
├── images/
│   ├── target.png
│   ├── init.png
│   ├── iter_0100.png
│   ├── final.png
│   ├── residual_final.png
│   └── gaussians_overlay.png
└── notes.txt                # Observations
```

**After each set:**
- Summary table (markdown)
- Convergence plots
- Visual comparison grid
- Statistical analysis
- Key findings summary

**Final sprint report:**
- Executive summary
- Results per experiment set
- Layered vs monolithic analysis
- Primitive characterization rules
- Anomalies and unexpected results
- Recommendations for next phase

---

## Success Metrics

**Minimum (must achieve):**
- ✓ Sets 1-4 completed
- ✓ 200+ experiments run
- ✓ Complete logs and reports
- ✓ Clear conclusion on layered vs monolithic

**Target (goal):**
- ✓ Sets 1-6 completed
- ✓ 300-500 experiments run
- ✓ Empirical rules for 2-3 primitives
- ✓ Comprehensive final report

**Stretch (if time):**
- ✓ Sets 1-10 completed
- ✓ 500+ experiments run
- ✓ Multi-primitive composition tested

---

## Timeline Estimate

**Hour 0-4:** Infrastructure setup
**Hour 4-12:** Experiment Set 1
**Hour 12-20:** Experiment Set 2
**Hour 20-28:** Experiment Set 3
**Hour 28-48:** Experiment Set 4 (CRITICAL, allocate most time here)
**Hour 48-60:** Experiment Sets 5-6
**Hour 60-72:** Final reporting

**Total: 72 hours (3 days intensive)**

---

## Available Resources

**Codebase:**
- lgi-core: Gaussian renderer (EWA splatting) - USE THIS
- lgi-encoder-v2: Optimizers, baseline implementations
- Catalogs: 160+ algorithms documented

**Reference implementations:**
- 3 baseline Gaussian codecs
- Corrected gradient computation
- N-variation study results

**Feature extractors:**
- Edges: Canny, structure tensor
- Regions: SLIC, watershed
- Blobs: LoG, DoG
- Use simple/fast ones initially

**Optimizers:**
- Adam (robust, use as default)
- L-BFGS (for small problems <50 Gaussians)
- SGD (baseline comparison)

---

## Critical Insights to Test

1. **Content-adaptive initialization:** Does feature-aware placement beat random?
2. **Specialized optimizers per layer:** Can we use different methods for different primitives?
3. **Layered decomposition:** Does 6 × 75 Gaussians beat 1 × 500 Gaussians?
4. **Residual-based fitting:** Does sequential residual fitting help or hurt?

---

## Final Notes

**Philosophy:**
- This is **exploration**, not production code
- Negative results are valuable (document WHY things fail)
- Unexpected findings are often most interesting
- Follow the data, not preconceptions

**Communication:**
- Interim reports every 8-12 hours
- Flag anomalies immediately
- Final comprehensive report at end

**Resource usage:**
- USE ALL AVAILABLE COMPUTE
- This is a sprint, maximize throughput
- Parallelize aggressively
- Don't leave free credits unused

---

## Begin

Read the framework and protocol documents thoroughly. Understand the research questions. Build the infrastructure. Run the experiments. Analyze the results. Report findings.

You have full autonomy and 2-3 days of intensive compute.

**Make it count.**

Generate knowledge that will inform the next phase of this research.

Good luck.
