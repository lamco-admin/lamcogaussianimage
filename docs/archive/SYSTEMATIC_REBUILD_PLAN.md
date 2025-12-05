# Systematic Rebuild Plan - LGI Codec Foundation

**Date**: November 14, 2025
**Status**: STRATEGIC PAUSE - Building Foundation
**Approach**: Theory First, Then Implementation

---

## Executive Summary

**Current State**: We have a toolkit of algorithms but lack deep understanding of how Gaussian splatting should work as an image codec. Multiple attempts to fix bugs have failed because we're operating without a theoretical foundation.

**New Approach**:
1. **Research exhaustively** - Study papers, implementations, experiments
2. **Document theory** - Build shared understanding of how this should work
3. **Analyze our tools** - Map what we have to what we need
4. **Test primitives** - Validate basic operations work correctly
5. **Iterate systematically** - Build up from working foundation

**Timeline**: No deadline - focus on correctness and understanding

---

## Phase 1: THEORY & RESEARCH (Weeks 1-2)

### Objective
Build deep, documented understanding of Gaussian splatting as an image codec.

### Research Areas

#### 1.1 Core Gaussian Splatting Theory
**Questions to Answer**:
- What is the mathematical representation? (equations, not just code)
- How does it differ from pixel, wavelet, DCT representations?
- What are the theoretical capacity limits?
- What's the information-theoretic analysis?
- Why use Gaussians specifically? (vs. other basis functions)

**Sources**:
- [ ] Original 3D Gaussian Splatting paper (Kerbl et al. 2023)
- [ ] 2D Gaussian splatting adaptations
- [ ] EWA splatting (Zwicker et al. 2001) - our renderer is based on this
- [ ] Differentiable rendering literature
- [ ] Image representation theory

**Deliverable**: `docs/theory/GAUSSIAN_REPRESENTATION.md`

#### 1.2 Initialization Strategies
**Questions to Answer**:
- Grid-based: When appropriate? What density formula?
- Structure-tensor: What does it actually detect? When to use?
- Error-driven: How to measure "error"? Where to place new Gaussians?
- Content-aware: How to classify image content? How does it affect initialization?

**Sources**:
- [ ] Structure tensor mathematics (edge detection theory)
- [ ] Image analysis techniques
- [ ] Adaptive sampling literature
- [ ] Our experiments in `docs/research/EXPERIMENTS.md`

**Deliverable**: `docs/theory/INITIALIZATION_THEORY.md`

#### 1.3 Optimization Theory
**Questions to Answer**:
- What loss function? (MSE, MS-SSIM, perceptual?)
- What's the gradient flow? (how do gradients propagate?)
- What convergence guarantees exist?
- How should learning rates be set?
- When does optimization fail? (local minima, saddle points?)

**Sources**:
- [ ] Differentiable rendering theory
- [ ] Gradient-based optimization literature (Adam, LBFGS theory)
- [ ] Our gradient fix (is it now correct? verify mathematically)
- [ ] 3DGS optimization pipeline

**Deliverable**: `docs/theory/OPTIMIZATION_THEORY.md`

#### 1.4 Densification & Adaptation
**Questions to Answer**:
- When to add Gaussians? (error threshold? gradient threshold?)
- Where to add them? (error map? saliency map?)
- When to remove Gaussians? (low opacity? low coverage?)
- When to split Gaussians? (high error? large scale?)
- What parameters for new Gaussians?

**Sources**:
- [ ] 3DGS densification algorithm
- [ ] Adaptive mesh refinement literature
- [ ] Our adaptive_densification module
- [ ] Split/clone/prune strategies

**Deliverable**: `docs/theory/DENSIFICATION_THEORY.md`

#### 1.5 Rendering Pipeline
**Questions to Answer**:
- Forward rendering: How do Gaussians become pixels? (we sort of know this)
- EWA splatting: Why use it? What does it guarantee?
- Alpha compositing vs. weighted average: Which is correct?
- Antialiasing: How does it work? Is it needed?
- Zoom stability: What does it mean? Why does it matter?

**Sources**:
- [ ] EWA splatting paper (Zwicker et al. 2001)
- [ ] Our renderer_v2.rs implementation
- [ ] Graphics rendering literature
- [ ] Verify our renderer matches theory

**Deliverable**: `docs/theory/RENDERING_THEORY.md`

### Research Method

For each topic:
1. **Read primary sources** (papers, not blog posts)
2. **Understand the math** (work through equations by hand)
3. **Document in plain language** (explain like teaching)
4. **Draw diagrams** (visualize the process)
5. **List key equations** (with meaning explained)
6. **Note assumptions & limitations** (where does theory break?)

### Success Criteria

- [ ] Can explain Gaussian splatting to someone unfamiliar
- [ ] Can derive key equations from first principles
- [ ] Understand WHY each component exists
- [ ] Know what "good" looks like at each stage
- [ ] Can predict behavior before running code

---

## Phase 2: SYSTEM AUDIT (Week 3)

### Objective
Map our existing toolkit to the theory, identify gaps and issues.

### 2.1 Component Inventory

**Create**: `docs/COMPONENT_INVENTORY.md`

For each module in our codebase:
- What does it do? (actual behavior)
- What SHOULD it do? (theory says)
- Does it match theory? (verify)
- Is it tested? (test coverage)
- Do we trust it? (validated or suspected?)

**Components to audit**:
- [ ] renderer_v2.rs - Does it match EWA theory?
- [ ] adam_optimizer.rs - Are gradients correct now?
- [ ] lib.rs (EncoderV2) - Does initialization match theory?
- [ ] geodesic_edt.rs - What does it actually prevent?
- [ ] structure_tensor.rs - Does it compute what we think?
- [ ] content_detection.rs - Is classification meaningful?
- [ ] error_driven.rs - Does it place Gaussians optimally?

### 2.2 Data Flow Mapping

**Create**: `docs/DATAFLOW.md`

Visual diagram showing:
```
Input Image
    ↓
[Preprocessing] → Structure Tensor, Geodesic EDT
    ↓
[Initialization] → Initial Gaussians (N=?, σ=?, positions=?)
    ↓
[Optimization] → Refined Gaussians (loss↓, PSNR↑)
    ↓
[Densification] → More Gaussians (where? how many?)
    ↓
[Loop] → Repeat optimize/densify
    ↓
[Render] → Output Image
    ↓
[Encode] → Bitstream (future)
```

For each arrow:
- What's the data format?
- What's the expected quality/properties?
- What can go wrong?

### 2.3 Gap Analysis

**Create**: `docs/GAPS.md`

Compare theory to implementation:
- **Missing components**: Theory says we need X, we don't have it
- **Broken components**: We have X, but it doesn't work correctly
- **Unvalidated components**: We have X, unsure if it works
- **Over-engineered components**: We have X, Y, Z when theory needs just X

### Success Criteria

- [ ] Complete inventory of all components
- [ ] Clear understanding of what works vs. broken vs. unknown
- [ ] Identified all gaps between theory and implementation
- [ ] Prioritized list of what to fix/build/remove

---

## Phase 3: PRIMITIVE VALIDATION (Week 4)

### Objective
Build confidence in basic operations through rigorous testing.

### 3.1 Test Pyramid

**Create**: `packages/lgi-rs/lgi-encoder-v2/tests/primitives/`

```
Level 1: Unit Tests (individual functions)
├── test_gaussian_evaluation.rs
│   └── Single Gaussian at point → correct weight
├── test_renderer_basic.rs
│   └── Single Gaussian → produces smooth blob
├── test_gradient_computation.rs
│   └── Finite difference vs. analytic gradient
└── test_initialization.rs
    └── Uniform image → uniform Gaussians

Level 2: Component Tests (modules)
├── test_renderer_complete.rs
│   └── N Gaussians → expected coverage, PSNR
├── test_optimizer_simple.rs
│   └── 1 Gaussian, wrong params → converges to right params
└── test_initializer_patterns.rs
    └── Known patterns → expected Gaussian distribution

Level 3: Integration Tests (pipeline)
├── test_simple_images.rs
│   └── Single pixel, edge, gradient → can encode at high quality
└── test_synthetic_patterns.rs
    └── Checkerboard, stripes, circles → PSNR targets
```

### 3.2 Test Cases (Simplest to Complex)

**Test 0: Sanity**
```rust
// 16×16 black image
// Expected: 1 black Gaussian (or minimal set)
// PSNR: Infinity (perfect match)
// Purpose: Verify renderer produces black when it should
```

**Test 1: Single Pixel**
```rust
// 16×16, white pixel at center (8,8)
// Expected: 1 white Gaussian, centered, σ ≈ 1 pixel
// PSNR: >40 dB (nearly perfect)
// Purpose: Can we represent the simplest non-trivial image?
```

**Test 2: Gradient**
```rust
// 16×16, left=black, right=white, linear gradient
// Expected: ~3-5 Gaussians aligned horizontally
// PSNR: >30 dB
// Purpose: Can we represent smooth variation?
```

**Test 3: Sharp Edge**
```rust
// 16×16, left=black, right=white, sharp vertical edge
// Expected: Gaussians aligned to edge, anisotropic
// PSNR: >25 dB (edges are hard)
// Purpose: Does structure tensor alignment work?
```

**Test 4: Known Pattern**
```rust
// 64×64 checkerboard (8×8 squares)
// Expected: Gaussians at square centers or edges
// PSNR: >20 dB
// Purpose: Complex but predictable pattern
```

### 3.3 Diagnostic Framework

**Create**: `packages/lgi-rs/lgi-debug/`

```rust
pub struct DiagnosticSuite {
    // Visualization
    fn render_gaussians_as_ellipses(image, gaussians) -> ImageBuffer;
    fn plot_coverage_heatmap(gaussians, width, height) -> ImageBuffer;
    fn show_error_map(rendered, target) -> ImageBuffer;

    // Metrics
    fn compute_coverage_stats(gaussians, width, height) -> CoverageStats;
    fn analyze_gaussian_distribution(gaussians) -> DistributionStats;
    fn check_gradient_sanity(gradients) -> GradientStats;

    // Validation
    fn verify_renderer_correctness(gaussians) -> Result<(), String>;
    fn verify_gradients_finite_diff(optimizer, gaussians) -> f32; // max error
    fn compare_to_reference(our_result, reference) -> ComparisonReport;
}
```

### 3.4 Reference Data

**Create**: `test-data/primitives/`

Store expected outputs:
```
test-data/primitives/
├── single_pixel/
│   ├── input.png (16×16, white pixel at 8,8)
│   ├── expected_gaussians.json (1 Gaussian, params)
│   └── expected_psnr.txt (>40 dB)
├── gradient/
│   ├── input.png
│   ├── expected_gaussian_count.txt (3-5)
│   └── expected_psnr.txt (>30 dB)
└── ...
```

### Success Criteria

- [ ] All primitive tests pass
- [ ] Renderer verified correct (matches theory)
- [ ] Gradients verified correct (finite diff check)
- [ ] Optimizer converges on simple cases
- [ ] Can explain any test failures (not mysterious)

---

## Phase 4: COMPONENT CONTRACTS (Week 5)

### Objective
Define clear interfaces and guarantees for each component.

### 4.1 Contract Template

For each component:

```rust
/// # Component: [Name]
///
/// ## Purpose
/// [What this does in the pipeline]
///
/// ## Theory
/// [Mathematical foundation - link to theory doc]
///
/// ## Input Contract
/// - Format: [Type, constraints]
/// - Preconditions: [What must be true]
/// - Example: [Concrete example]
///
/// ## Output Contract
/// - Format: [Type, properties]
/// - Postconditions: [Guarantees]
/// - Quality: [Expected metrics]
///
/// ## Failure Modes
/// - [Condition] → [Behavior]
/// - [Edge case] → [How handled]
///
/// ## Validation
/// - Unit tests: [List]
/// - Integration tests: [List]
/// - Reference: [Expected behavior doc]
///
/// ## Dependencies
/// - Requires: [Other components]
/// - Assumptions: [About inputs]
```

### 4.2 Critical Contracts

**Renderer**:
```rust
/// Postconditions:
/// - PSNR increases monotonically with Gaussian count (until saturation)
/// - Coverage ∈ [0, ∞) at every pixel (typically >0.5 for good quality)
/// - Gaussians outside image bounds contribute zero weight
/// - Empty Gaussian set produces black image
```

**Optimizer**:
```rust
/// Postconditions:
/// - Loss decreases monotonically (or early stops)
/// - Gradients are finite (no NaN/Inf)
/// - Converges in <1000 iterations typically
/// - Final PSNR ≥ initial PSNR (should not make quality worse!)
```

**Initializer**:
```rust
/// Postconditions:
/// - Returns exactly N Gaussians (±10%)
/// - All positions ∈ [0,1] × [0,1]
/// - All scales > 0 (positive definite)
/// - Coverage > 0.3 everywhere (no large gaps)
/// - Initial PSNR > grid baseline
```

### Success Criteria

- [ ] Every major component has documented contract
- [ ] Contracts are testable (can verify programmatically)
- [ ] Contracts reference theory documents
- [ ] Team understands what each component guarantees

---

## Phase 5: SYSTEMATIC DEBUGGING (Week 6+)

### Objective
Fix the current regression using the framework we've built.

### 5.1 Debug Protocol

**When something fails**:

1. **Measure** - Collect all relevant data
   ```rust
   let before = DiagnosticSuite::capture_state(&gaussians);
   // ... operation ...
   let after = DiagnosticSuite::capture_state(&gaussians);
   let diff = DiagnosticSuite::compare(before, after);
   ```

2. **Visualize** - See what's happening
   ```rust
   DiagnosticSuite::render_gaussians_as_ellipses(image, &gaussians);
   DiagnosticSuite::plot_coverage_heatmap(&gaussians, width, height);
   DiagnosticSuite::show_error_map(&rendered, &target);
   ```

3. **Compare to Theory** - What should happen?
   - Check theory document
   - Verify component contract
   - Identify discrepancy

4. **Hypothesize** - Form testable hypothesis
   - "I think X is happening because Y"
   - "If true, we should see Z"
   - "Test: check if Z is present"

5. **Test** - Minimal test case
   - Isolate the issue
   - Create smallest reproduction
   - Add to regression suite

6. **Fix** - Targeted change
   - One change at a time
   - Document reasoning
   - Reference theory

7. **Validate** - Verify fix
   - Re-run failing test (should pass)
   - Run all tests (no regressions)
   - Check metrics improved

### 5.2 Current Regression Debug Plan

**Issue**: Adam optimizer second pass stuck (loss = 0.375001)

**Step 1: Measure**
```rust
// Add detailed logging between passes
log::info!("=== BEFORE PASS {} ===", pass);
log::info!("Gaussian count: {}", gaussians.len());
log::info!("Min/mean/max scale: {:?}", compute_scale_stats(&gaussians));
log::info!("Coverage: {:?}", compute_coverage(&gaussians));

// ... optimization pass ...

log::info!("=== AFTER OPTIMIZATION ===");
log::info!("Loss: {:.6}", loss);
log::info!("Scale changes: {:?}", scale_changes);
log::info!("Gradient norms: {:?}", gradient_norms);

// ... geodesic clamping ...

log::info!("=== AFTER CLAMPING ===");
log::info!("Scales clamped: {}", count_clamped);
log::info!("New min scale: {:.6}", min_scale_after);
```

**Step 2: Visualize**
```rust
// Render Gaussians as ellipses on the image
DiagnosticSuite::save_gaussian_visualization(
    &image,
    &gaussians,
    "debug/pass_0_before.png"
);
DiagnosticSuite::save_gaussian_visualization(
    &image,
    &gaussians_after_opt,
    "debug/pass_0_after_opt.png"
);
DiagnosticSuite::save_gaussian_visualization(
    &image,
    &gaussians_after_clamp,
    "debug/pass_0_after_clamp.png"
);
```

**Step 3: Compare to Theory**
- Check OPTIMIZATION_THEORY.md
- What should happen during second pass?
- Are new Gaussians initialized correctly?
- Is clamping preserving minimum coverage?

**Step 4: Hypothesize**
Possible causes:
- New Gaussians have wrong parameters
- Geodesic clamping too aggressive
- Gradient computation breaks with mixed old/new Gaussians
- Learning rate needs adjustment for later passes

**Step 5-7: Test, Fix, Validate**
- Create minimal test: 2 passes, simple image
- Test each hypothesis
- Fix confirmed issue
- Verify with full benchmark

### Success Criteria

- [ ] Understand why second pass fails
- [ ] Have minimal reproduction
- [ ] Fix verified with tests
- [ ] No regressions introduced
- [ ] Documented in theory/code

---

## Phase 6: ITERATIVE IMPROVEMENT (Ongoing)

### Objective
Build up quality systematically, one validated improvement at a time.

### 6.1 Improvement Protocol

```
1. Identify opportunity (from theory or experiments)
   ↓
2. Design minimal change
   ↓
3. Predict impact (theory says should improve X by Y)
   ↓
4. Implement change
   ↓
5. Measure impact (did X improve by ~Y?)
   ↓
6. If success: commit, document, move on
   If failure: analyze why prediction wrong, adjust theory
```

### 6.2 Metrics Dashboard

**Create**: `tools/benchmark_dashboard.rs`

Track over time:
```
Metric                  Baseline    Current    Target    Status
─────────────────────────────────────────────────────────────────
Synthetic (128×128)
  Baseline PSNR        15.2 dB     15.2 dB    15.0 dB   ✅
  Adam PSNR            4.3 dB      ???        21.0 dB   ❌
  Convergence          STUCK       ???        <100 it   ❌

Kodak (768×512)
  Average PSNR         N/A         N/A        28-32 dB  ⏸️

Real Photos
  Average PSNR         N/A         N/A        30-35 dB  ⏸️
```

### 6.3 Documentation Updates

After each successful improvement:
- Update theory docs if learned something new
- Update component contracts if behavior changed
- Add test case to regression suite
- Note in CHANGELOG.md

### Success Criteria

- [ ] Every change is measured and validated
- [ ] Metrics dashboard shows progress over time
- [ ] Theory docs stay in sync with code
- [ ] No mysterious regressions

---

## Deliverables Summary

### Week 1-2: Theory
- [ ] `docs/theory/GAUSSIAN_REPRESENTATION.md`
- [ ] `docs/theory/INITIALIZATION_THEORY.md`
- [ ] `docs/theory/OPTIMIZATION_THEORY.md`
- [ ] `docs/theory/DENSIFICATION_THEORY.md`
- [ ] `docs/theory/RENDERING_THEORY.md`

### Week 3: Audit
- [ ] `docs/COMPONENT_INVENTORY.md`
- [ ] `docs/DATAFLOW.md`
- [ ] `docs/GAPS.md`

### Week 4: Testing
- [ ] `tests/primitives/*.rs` (comprehensive test suite)
- [ ] `lgi-debug` crate (diagnostic tools)
- [ ] `test-data/primitives/` (reference data)

### Week 5: Contracts
- [ ] Updated doc comments with contracts for all components

### Week 6+: Debugging & Improvement
- [ ] Current regression fixed and documented
- [ ] Metrics dashboard tracking progress
- [ ] Iterative improvements with validation

---

## Success Metrics

**We'll know we're successful when**:

1. **Understanding**: Can explain system to newcomer from theory up
2. **Predictability**: Can predict behavior before running code
3. **Reliability**: Tests pass consistently, no mysterious failures
4. **Progress**: Metrics dashboard shows steady improvement
5. **Confidence**: Changes don't break things because contracts enforced
6. **Quality**: Hitting PSNR targets from Session 7/8

---

## Principles

1. **Theory Before Code** - Understand before implementing
2. **Test Everything** - If not tested, assume broken
3. **One Change at a Time** - Isolate cause and effect
4. **Document Continuously** - Understanding fades, write it down
5. **Visualize Actively** - See what's happening, don't guess
6. **Measure Rigorously** - Numbers don't lie, hunches do
7. **Fail Fast** - Find problems early with simple tests
8. **Trust Process** - Framework over heroics

---

## Timeline

**Phase 1-2**: 2-3 weeks (theory + audit)
**Phase 3-4**: 1-2 weeks (testing + contracts)
**Phase 5**: 1-2 weeks (fix current regression)
**Phase 6**: Ongoing (iterative improvement)

**Total to working baseline**: 4-6 weeks
**Total to production quality**: Months (no deadline)

This is the RIGHT way. Slow is smooth, smooth is fast.

---

**Status**: Plan approved, proceeding to Phase 1 - Theory & Research

**Next Action**: Begin exhaustive research on Gaussian splatting theory
