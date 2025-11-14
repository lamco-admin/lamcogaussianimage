# Analysis of Commit 776ffdd - Test Fix That Broke Optimization

**Commit**: 776ffdd7b5925af2b5ae5bd5e5fbeccad6b36dc5
**Author**: Claude (Claude Code Web session)
**Date**: November 14, 2025, 19:18:48 UTC
**Message**: "Fix 3 failing tests: EWA zoom stability, content detection, LOD system"

---

## What Was Changed

### File 1: `lgi-core/src/ewa_splatting_v2.rs`

**Change**: Removed zoom multiplication from position and scale

**Before**:
```rust
let mu_x = gaussian.position.x * width as f32 * zoom;
let mu_y = gaussian.position.y * height as f32 * zoom;
let sx = gaussian.shape.scale_x * width as f32 * zoom;
let sy = gaussian.shape.scale_y * height as f32 * zoom;
```

**After**:
```rust
// Note: width/height already account for zoom in render_multiscale
let mu_x = gaussian.position.x * width as f32;
let mu_y = gaussian.position.y * height as f32;
let sx = gaussian.shape.scale_x * width as f32;
let sy = gaussian.shape.scale_y * height as f32;
```

**Rationale**: "Remove redundant zoom multiplication (applied twice)"

**Impact**: Probably NOT the cause - this only affects EWA rendering at zoom levels ≠ 1.0

---

### File 2: `lgi-core/src/content_detection.rs`

**Change**: Modified Sharp/Smooth classification logic

**Before**:
```rust
if avg_gradient < 0.01 && avg_coherence < 0.1 {
    ContentType::Smooth
} else if edge_density > 0.15 {
    ContentType::Sharp
```

**After**:
```rust
if entropy < 2.0 && avg_coherence < 0.5 {
    ContentType::Sharp  // Low entropy + low coherence = sharp edges
} else if avg_coherence > 0.8 || avg_gradient < 0.01 {
    ContentType::Smooth  // High coherence OR low gradient = smooth
```

**Rationale**: "Use entropy + coherence for Sharp/Smooth classification"

**Impact**: **POSSIBLY THE CAUSE** - Changes how content is classified, which affects Gaussian initialization

---

### File 3: `lgi-core/src/lod_system.rs` 🚨 SMOKING GUN

**Change**: **Dramatically increased LOD thresholds (10× larger!)**

**Before**:
```rust
// Typical σ in normalized coords: 0.005-0.02
// det(Σ) = σ_x × σ_y ranges from 0.000025 to 0.0004
if det > 0.0004 {
    LODBand::Coarse  // σ > 0.02 (large Gaussians)
} else if det > 0.0001 {
    LODBand::Medium  // σ ~ 0.01-0.02 (typical)
} else {
    LODBand::Fine    // σ < 0.01 (small details)
}
```

**After**:
```rust
// Typical σ in normalized coords: 0.05-0.3  ← 10× LARGER!
// det(Σ) = σ_x × σ_y:
//   Coarse: > 0.04 (σ > 0.2)
//   Medium: 0.01-0.04 (σ ~ 0.1-0.2)
//   Fine: < 0.01 (σ < 0.1)
if det > 0.04 {           // Was 0.0004 → Now 0.04 (100× larger!)
    LODBand::Coarse
} else if det > 0.01 {    // Was 0.0001 → Now 0.01 (100× larger!)
    LODBand::Medium
} else {
    LODBand::Fine
}
```

**Rationale**: "Update thresholds to realistic scale ranges (0.04, 0.01)"

**Impact**: 🚨 **HIGHLY LIKELY THE CAUSE**

---

## Why This Breaks Optimization

### The LOD System Threshold Problem

**Original thresholds** were calibrated for:
```
σ range: 0.005-0.02 (normalized coordinates)
det range: 0.000025-0.0004
```

**New thresholds** expect:
```
σ range: 0.05-0.3 (10× LARGER!)
det range: 0.01-0.04 (100× LARGER threshold!)
```

### Impact on Gaussian Classification

**If actual Gaussians have σ ≈ 0.01** (reasonable for 128×128):
```
det = σ_x × σ_y = 0.01 × 0.01 = 0.0001

OLD classification:
  det = 0.0001
  Threshold: 0.0001 < det < 0.0004
  Classification: Medium ✅

NEW classification:
  det = 0.0001
  Threshold: det < 0.01
  Classification: Fine ✅ BUT...

  Wait, if det < 0.01, it's classified as Fine
  But the threshold comment says "σ < 0.1"
  If σ = 0.01, that IS < 0.1, so Fine is correct

  UNLESS: The thresholds are being used to FILTER or CLAMP Gaussians!
```

### Hypothesis: LOD System Used for Scale Validation?

**If LOD system is used to validate/clamp Gaussian scales**:

```rust
// Somewhere in encoder?
if gaussian.get_lod_band() == LODBand::Fine {
    // Old: det < 0.0001 → σ < 0.01 → Fine (many Gaussians)
    // New: det < 0.01 → σ < 0.1 → Fine (MOST Gaussians)

    // If code says "Fine means too small, reject/clamp it"
    // Then NEW thresholds reject MOST Gaussians!
}
```

**Or**:
```rust
// Validation logic?
if det < MIN_THRESHOLD {
    // Gaussian too small, clamp it
    // OLD: MIN = 0.000025 (very permissive)
    // NEW: MIN = 0.01 (very restrictive!)
    // MOST Gaussians now "too small" and get clamped!
}
```

---

## Evidence Supporting This Hypothesis

### From Benchmark Output

**Baseline works** (15.21 dB):
- Uses simple initialization
- Probably doesn't use LOD system for clamping
- Gaussians keep their natural scales

**All optimization methods fail** (4.26 dB):
- Adam, Error-Driven, GPU all use encoder
- Encoder might use LOD for scale validation
- New thresholds clamp everything
- Result: Zero coverage → no optimization

**Loss stuck at 0.375001**:
- Indicates no gradient signal
- Consistent with zero coverage
- Matches FAIL-001 pattern exactly

---

## The Exact Problem (Likely)

**Step 1**: Optimizer initializes Gaussians with σ ≈ 0.01-0.02 (reasonable)

**Step 2**: LOD classification or validation runs:
```rust
det = 0.01 × 0.01 = 0.0001

if det < LOD_THRESHOLD {  // LOD_THRESHOLD = 0.01 (NEW)
    // "Gaussian too small, clamp/reject it"
    // OR mark as "needs larger scale"
    sigma = clamp_to_minimum(sigma);  // Clamps to match threshold?
}
```

**Step 3**: Gaussians get clamped/filtered incorrectly

**Step 4**: Zero coverage → zero gradients → loss stuck → quality 4 dB

---

## Supporting Evidence

**From FAIL-001** (October 2025):
```
Problem: Geodesic EDT clamping Gaussians to 1 pixel
Evidence: σ_base = 38.4 pixels expected, σ_actual = 1.0 pixel
Result: W_median = 0.000, quality stuck at 5-7 dB
```

**Current symptoms**:
```
Loss: Stuck at 0.375001 (matches zero coverage pattern)
PSNR: 4.26 dB (matches zero optimization pattern)
All methods affected (matches systematic clamping issue)
```

---

## How The Test "Fix" Broke Production

**The test likely tested**:
```rust
#[test]
fn test_lod_classification() {
    let large_gaussian = Gaussian2D::new(σ=0.21);  // Large scale
    assert_eq!(classify(large_gaussian), LODBand::Coarse);
    // ✅ Test passes with new thresholds
}
```

**But production code uses**:
```rust
let normal_gaussian = Gaussian2D::new(σ=0.01);  // Normal scale for 128×128
let band = classify(normal_gaussian);
// Old: Medium
// New: Fine (if det=0.0001 < 0.01)

// If somewhere there's:
if band == LODBand::Fine && det < 0.01 {
    reject_or_clamp(gaussian);  // "Too small"
}
// BOOM - all normal Gaussians rejected!
```

**Result**: Test passes, production breaks

---

## Action Required (For Web Session)

### Immediate Investigation

1. **Check if LOD system is used for scale validation/clamping**:
   ```bash
   grep -r "LODBand\|lod_system\|classify" packages/lgi-rs/lgi-encoder-v2/
   ```

2. **Search for scale clamping logic**:
   ```bash
   grep -r "clamp.*scale\|MIN_SCALE\|threshold.*det" packages/lgi-rs/lgi-encoder-v2/
   ```

3. **Add diagnostic logging** to encoder:
   ```rust
   log::info!("Gaussian scale: σ_x={:.4}, σ_y={:.4}, det={:.6}",
       sigma_x, sigma_y, det);
   log::info!("LOD band: {:?}", lod_band);
   log::info!("W_median: {:.4} (expect > 0.5)", w_median);
   ```

4. **Run single iteration** to see values:
   ```rust
   // In fast_benchmark, change iterations to 1
   let result = encoder.optimize_n_iterations(gaussians, 1);
   ```

### Likely Fixes

**Option A: Revert LOD threshold changes**
```rust
// Restore original thresholds
if det > 0.0004 {      // Was: 0.04
    LODBand::Coarse
} else if det > 0.0001 {  // Was: 0.01
    LODBand::Medium
} else {
    LODBand::Fine
}
```

**Option B: Fix how LOD thresholds are used**
```rust
// If LOD is used for validation, update validation logic
// Don't clamp/reject based on LOD band
// OR adjust validation thresholds separately from LOD thresholds
```

**Option C: Both**
- Fix LOD thresholds back to original
- AND ensure they're not used for inappropriate clamping
- Tests should test LOD classification only, not scale clamping

---

## Test Strategy for Fix

**After fixing**:

1. Run fast_benchmark:
   ```bash
   cargo run --release --example fast_benchmark
   ```

2. Expected results:
   ```
   Complex Pattern: 15.X → 21-24 dB (+6-8 dB) ✅
   Loss: Should decrease from ~0.8 to ~0.09 ✅
   ```

3. If still broken:
   - Add more instrumentation
   - Check W_median
   - Check Gaussian scales in pixels
   - Review content detection changes too

4. If fixed:
   - Commit the fix
   - **Request local testing resume**
   - Continue with Test 1 (Kodak) and Test 2 (Real photos)

---

## Conclusion

**Commit 776ffdd EXISTS** - it's on the Claude Code Web branch

**What it does**: Fixed 3 failing tests (55→58 passing)

**Side effect**: **Broke all optimization methods** (-10 dB regression)

**Likely culprit**: LOD threshold changes (100× larger thresholds)

**Impact**: If LOD system used for scale validation, most Gaussians now classified as "too small" and get clamped/rejected

**Fix needed**: Revert LOD thresholds OR fix how they're used in optimization

**Validation**: fast_benchmark must show +6-8 dB after fix

---

**This analysis committed to main branch for web session review.**

**Files**:
- COMMIT_776ffdd_ANALYSIS.md (this file)
- HANDBACK_TO_WEB_SESSION.md (handback report)
- CRITICAL_REGRESSION_FOUND.md (regression evidence)
- fast_benchmark_results.txt (full output)

**Status**: Awaiting fix from web session

**Last Updated**: November 14, 2025, 10:20 PM EET
