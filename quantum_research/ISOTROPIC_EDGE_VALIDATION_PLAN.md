# Isotropic Edge Validation - Testing Quantum Discovery

**Date**: 2025-12-04
**Hypothesis**: Small isotropic Gaussians work better than anisotropic elongated ones for edges
**Source**: Quantum channel discovery - Channels 3, 4, 7 all isotropic despite high coherence
**Status**: READY TO TEST

---

## Executive Summary

**Quantum Finding**: All high-quality Gaussians (loss < 0.05) are **isotropic** (σ_x ≈ σ_y), even at strong edges (coherence > 0.96).

**Current Approach**: Uses **anisotropic** Gaussians at edges (σ_parallel >> σ_perp, elongated along edge tangent).

**Current Edge Quality**: 1.56 dB PSNR (catastrophic failure)

**Hypothesis**: Switching to small isotropic Gaussians will achieve **10-15 dB PSNR** at edges (6-8× improvement).

---

## Test Design

### Experiment 1: Pure Isotropic Initialization

**Modification**: Force σ_x = σ_y for ALL Gaussians, regardless of edge coherence

**Current Code** (`lib.rs:512-521`):
```rust
let (mut sig_para_px, mut sig_perp_px, rotation_angle) = if coherence < 0.2 {
    // Flat region → isotropic
    (sigma_base_px, sigma_base_px, 0.0)
} else {
    // Edge → anisotropic (ELONGATED)
    let sigma_perp = sigma_base_px / (1.0 + 3.0 * coherence);
    let sigma_para = 4.0 * sigma_perp;  // 4× elongation!
    let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
    (sigma_para, sigma_perp, angle)
};
```

**Modified Code** (isotropic):
```rust
let (mut sig_para_px, mut sig_perp_px, rotation_angle) = if coherence < 0.2 {
    // Flat region → isotropic
    (sigma_base_px, sigma_base_px, 0.0)
} else {
    // Edge → ISOTROPIC (quantum-guided)
    // Use small isotropic Gaussian instead of elongated
    let sigma_iso = sigma_base_px / (1.0 + 2.0 * coherence);  // Smaller at edges
    (sigma_iso, sigma_iso, 0.0)  // ISOTROPIC!
};
```

**Key Changes**:
1. `sigma_para = sigma_perp` (isotropic)
2. `rotation = 0` (no rotation needed)
3. Scale decreases with coherence (smaller at stronger edges)

### Experiment 2: Quantum-Channel-Guided Initialization

**Modification**: Sample Gaussian parameters from high-quality quantum channels

**Implementation**:
```rust
let (sig_x, sig_y, rotation) = if coherence < 0.2 {
    // Smooth regions: Use Channel 1 parameters
    let sigma = sample_from_normal(0.0283, 0.0795);  // Channel 1 mean/std
    (sigma, sigma, 0.0)
} else if coherence > 0.9 {
    // Strong edges: Use Channel 4 (highest quality)
    let sigma = sample_from_normal(0.0018, 0.0011);  // Channel 4 mean/std
    (sigma, sigma, 0.0)
} else {
    // Medium edges: Use Channel 3
    let sigma = sample_from_normal(0.0011, 0.0003);  // Channel 3 mean/std
    (sigma, sigma, 0.0)
};
```

### Experiment 3: Control (Current Method)

**Keep current anisotropic approach for comparison**

Run baseline to measure improvement.

---

## Test Images

**Use 5 Kodak images** with varying edge content:

1. **kodim03** - Architecture, sharp edges
2. **kodim05** - Building with many edges
3. **kodim08** - Woman portrait (soft + sharp edges)
4. **kodim15** - Outdoor scene
5. **kodim23** - Garden with texture

**Why 5**: Enough for statistical significance, fast to run (~40 minutes total)

---

## Success Criteria

### Baseline (Current Anisotropic)
```
Average PSNR: ~15-18 dB (based on kodim03 = 17.14 dB)
Edge quality: ~1.56 dB (from previous composition test)
```

### Success Thresholds

**Validation Success** (confirms quantum finding):
- Overall PSNR: ≥ 18 dB (matches or exceeds current)
- Edge PSNR: ≥ 8-10 dB (5-6× improvement over 1.56 dB)

**Major Breakthrough**:
- Overall PSNR: ≥ 22 dB (+4 dB over current)
- Edge PSNR: ≥ 15 dB (10× improvement!)

**Transformative**:
- Overall PSNR: ≥ 25 dB (target quality achieved!)
- Edge PSNR: ≥ 20 dB (problem solved!)

---

## Implementation Plan

### Step 1: Create Isotropic Variant Method

**File**: `packages/lgi-rs/lgi-encoder-v2/src/lib.rs`

**Add new method**: `encode_error_driven_adam_isotropic()`
- Copy `encode_error_driven_adam()`
- Modify initialization to force isotropic
- Keep all other logic identical

**Location**: After line 569 (after existing method)

### Step 2: Create Test Binary

**File**: `packages/lgi-rs/lgi-encoder-v2/examples/test_isotropic_edges.rs`

**Purpose**: Compare isotropic vs anisotropic on same images

**Output**:
```
IMAGE: kodim03
  Anisotropic (current): 17.14 dB
  Isotropic (quantum):   ?? dB
  Improvement:           +?? dB

IMAGE: kodim05
  ...
```

### Step 3: Run Benchmark

```bash
cd packages/lgi-rs
cargo build --release --example test_isotropic_edges
./target/release/examples/test_isotropic_edges
```

**Expected runtime**: 8-10 minutes per image × 5 = 40-50 minutes

### Step 4: Analyze Results

**If isotropic wins** (+3 dB or more):
- Quantum discovery validated!
- Adopt isotropic edges as new standard
- Investigate WHY isotropic works better (Q4 research)

**If anisotropic still better**:
- Quantum channels might need different interpretation
- Maybe it's not isotropic vs anisotropic, but SCALE that matters
- Re-analyze quantum channels for other patterns

**If roughly equal**:
- Both approaches valid
- Use conditional: anisotropic for some edges, isotropic for others
- Quantum channels might reveal WHEN to use each

---

## Detailed Implementation Specifications

### Modification Location 1: Hotspot Gaussian Creation

**Current**: `lib.rs:512-521` (in `encode_error_driven_adam`)

**Context**: When adding Gaussians at high-error regions

**Change**:
```rust
// OLD:
} else {
    // Edge → anisotropic
    let sigma_perp = sigma_base_px / (1.0 + 3.0 * coherence);
    let sigma_para = 4.0 * sigma_perp;  // 4× elongation
    let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
    (sigma_para, sigma_perp, angle)
};

// NEW (ISOTROPIC):
} else {
    // Edge → isotropic (quantum-guided)
    // Quantum channels 3,4,7 show isotropic works better
    let sigma_iso = sigma_base_px / (1.0 + 2.0 * coherence);  // Smaller at edges
    (sigma_iso, sigma_iso, 0.0)  // σ_x = σ_y, no rotation
};
```

### Modification Location 2: Grid Initialization

**Current**: `lib.rs:150-250` (in `initialize_gaussians_with_options`)

**Context**: Initial grid of Gaussians

**Change** (if guided = true):
```rust
// Around line 200
let (sig_para, sig_perp, rotation) = if use_structure_tensor {
    let tensor = self.structure_tensor.get(x, y);
    let coherence = tensor.coherence;

    if coherence < 0.3 {
        // Isotropic for smooth
        (sigma, sigma, 0.0)
    } else {
        // OLD: Anisotropic for edges
        // NEW: Isotropic but smaller
        let sigma_small = sigma * 0.5;  // Quantum shows edges need small scales
        (sigma_small, sigma_small, 0.0)
    }
} else {
    (sigma, sigma, 0.0)
};
```

---

## Expected Outcomes

### Scenario 1: Isotropic Wins Big (+5 dB or more)

**Interpretation**: Quantum discovery is correct and transformative
- Anisotropic assumption was fundamentally wrong
- Edge representation problem SOLVED
- Adopt isotropic as new standard immediately

**Next steps**:
- Implement in all encoders
- Test on full dataset (67 real photos)
- Publish finding (contradicts 3D splatting literature)

### Scenario 2: Isotropic Modest Win (+2-4 dB)

**Interpretation**: Quantum found improvement, but not complete solution
- Isotropic helps but doesn't solve edge problem entirely
- Combined approach might be needed
- Other quantum channels (scale, not shape) matter more

**Next steps**:
- Analyze which specific edge types benefit from isotropic
- Conditional logic: use isotropic for sharp edges, anisotropic for soft
- Investigate Channels 3 vs 4 vs 7 differences

### Scenario 3: No Significant Difference (< 1 dB)

**Interpretation**: Shape (isotropic vs anisotropic) isn't the key factor
- SCALE matters more than shape
- Quantum channels differ by scale, not shape ratios
- Focus on getting optimizer to find small scales

**Next steps**:
- Re-examine quantum channels for scale patterns
- Bias initialization toward σ = 0.001-0.002 ranges
- Investigate learning rate per-scale strategies

### Scenario 4: Anisotropic Still Better

**Interpretation**: Need deeper analysis of quantum channels
- Channels might not directly map to initialization strategy
- Isotropic in final state doesn't mean isotropic initialization works
- Optimizer dynamics matter (Q2 question)

**Next steps**:
- Analyze optimizer trajectories in quantum channels
- Do successful Gaussians START isotropic or BECOME isotropic?
- Might need different iteration strategy, not initialization

---

## Benchmark Design

### Metrics to Collect

**Per Image**:
1. **Overall PSNR** (dB) - full image quality
2. **Edge PSNR** (dB) - quality at edges only (coherence > 0.7)
3. **Smooth PSNR** (dB) - quality in flat regions (coherence < 0.3)
4. **Gaussian count** - how many Gaussians used
5. **Convergence speed** - iterations to target quality

**Aggregated** (across 5 images):
- Mean PSNR ± std
- Win rate (how many images improved)
- Statistical significance (t-test)

### Visualization

Create comparison table:
```
Image    | Aniso PSNR | Iso PSNR | Δ    | Edge Aniso | Edge Iso | Edge Δ
---------|------------|----------|------|------------|----------|--------
kodim03  | 17.14      | ??       | ??   | 1.56       | ??       | ??
kodim05  | ??         | ??       | ??   | ??         | ??       | ??
...
---------|------------|----------|------|------------|----------|--------
AVERAGE  | ??         | ??       | ??   | 1.56       | ??       | ??
```

---

## Implementation Timeline

### Step 1: Create Isotropic Encoder Method (15-20 min)
- Copy `encode_error_driven_adam()`
- Modify two locations (grid init, hotspot placement)
- Change anisotropic logic to isotropic
- Compile and test

### Step 2: Create Benchmark Tool (10-15 min)
- Load image
- Encode with anisotropic (baseline)
- Encode with isotropic (quantum-guided)
- Compute PSNR for both
- Report comparison

### Step 3: Run 5-Image Validation (40-50 min automated)
- Process kodim03, 05, 08, 15, 23
- Collect PSNR data
- Save results to JSON

### Step 4: Analyze Results (10-15 min)
- Load results
- Calculate mean improvement
- Statistical significance test
- Interpret findings

**Total Time**: ~2 hours (30 min human, 50 min compute)

---

## Success Documentation

### If Successful (Isotropic Wins)

Create `BREAKTHROUGH_ISOTROPIC_EDGES.md`:
- Document quantum prediction
- Show experimental validation
- Explain why anisotropic was wrong
- Update encoder standards
- Plan full dataset validation

### If Unsuccessful (Anisotropic Better)

Create `QUANTUM_CHANNEL_REINTERPRETATION.md`:
- Analyze why prediction failed
- Re-examine quantum channel data
- Alternative hypotheses
- Next experiments

---

## Risk Assessment

### Risk 1: Code Changes Break Existing Functionality

**Mitigation**: Create NEW method (don't modify existing)
- `encode_error_driven_adam()` - unchanged (anisotropic)
- `encode_error_driven_adam_isotropic()` - new (isotropic)
- Can compare side-by-side without breaking working code

### Risk 2: Results Are Ambiguous

**Mitigation**: Run on 5 diverse images
- Architecture (sharp edges)
- Portraits (soft edges)
- Outdoor (mixed)
- Clear winner should emerge

### Risk 3: Improvement Is Small (<1 dB)

**Mitigation**: Analyze edge-specific PSNR
- Overall might not change much
- Edge regions might show large improvement
- Separate metrics for edges vs smooth

---

## Code Locations

### Files to Modify

1. **`packages/lgi-rs/lgi-encoder-v2/src/lib.rs`**
   - Line 512-521: Hotspot Gaussian creation (change anisotropic to isotropic)
   - Line 150-250: Grid initialization (change edge handling)
   - Add new method: `encode_error_driven_adam_isotropic()`

2. **Create `packages/lgi-rs/lgi-encoder-v2/examples/test_isotropic_edges.rs`**
   - Benchmark comparing both approaches
   - Run on 5 Kodak images
   - Output comparison table

---

## Expected Output

```
================================================================================
ISOTROPIC EDGE VALIDATION
Testing Quantum Discovery: σ_x = σ_y for edges
================================================================================

[1/5] kodim03.png
  Anisotropic (current):  17.14 dB | Edge: 1.56 dB | Gaussians: 500
  Isotropic (quantum):    21.35 dB | Edge: 12.4 dB | Gaussians: 500
  Improvement:            +4.21 dB | Edge: +10.8 dB ⭐ MAJOR WIN!

[2/5] kodim05.png
  Anisotropic (current):  18.23 dB | Edge: 2.1 dB | Gaussians: 500
  Isotropic (quantum):    23.67 dB | Edge: 14.8 dB | Gaussians: 500
  Improvement:            +5.44 dB | Edge: +12.7 dB ⭐ BREAKTHROUGH!

[3/5] kodim08.png
  Anisotropic (current):  19.12 dB | Edge: 3.2 dB | Gaussians: 500
  Isotropic (quantum):    22.45 dB | Edge: 11.6 dB | Gaussians: 500
  Improvement:            +3.33 dB | Edge: +8.4 dB ⭐ CONFIRMED!

[4/5] kodim15.png
  ...

[5/5] kodim23.png
  ...

================================================================================
SUMMARY
================================================================================

Average PSNR:
  Anisotropic: 18.5 ± 1.2 dB
  Isotropic:   23.2 ± 1.8 dB
  Improvement: +4.7 dB (p < 0.001) ⭐⭐⭐

Edge PSNR:
  Anisotropic: 2.1 ± 0.8 dB (FAILURE)
  Isotropic:   12.6 ± 2.1 dB (SUCCESS!)
  Improvement: +10.5 dB (6× better!) ⭐⭐⭐

Conclusion: QUANTUM DISCOVERY VALIDATED!
  → Isotropic edges are fundamentally better than anisotropic
  → Edge representation problem SOLVED
  → Adopt isotropic as new standard immediately
```

---

## Next Steps After Validation

### If Isotropic Wins (Expected)

1. **Update all encoder methods** to use isotropic edges
2. **Run full benchmark** on 67 real photos + 24 Kodak
3. **Measure overall improvement** (expect +5-8 dB)
4. **Document breakthrough** in research log
5. **Investigate Q2**: Why does isotropic work better? (Theory)

### Q2 Follow-Up Questions

- **Why does 2D differ from 3D?** (3D splatting uses anisotropic successfully)
- **What about oriented textures?** (Fabric, wood grain - need elongation?)
- **Is there a hybrid approach?** (Isotropic for sharp edges, anisotropic for gradients?)

---

## Quantum Validation

If classical isotropic validates quantum finding:

**This proves**:
1. Quantum discovered hidden structure in parameter space
2. Structure is actionable (improves real encoding)
3. Quantum found something humans missed (anisotropic assumption wrong)
4. Q1 methodology works - Q2, Q3, Q4 worth pursuing

**Impact**:
- Justifies quantum research investment
- Opens door to Q2 (iteration methods), Q3 (discrete optimization), Q4 (basis functions)
- Potential for multiple breakthroughs using quantum approach

---

**READY TO EXECUTE**

Next: Implement isotropic encoder and run 5-image validation
