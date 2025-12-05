# Regression Investigation Report

**Date**: November 14, 2025
**Session**: Claude Code Web (continuing from local testing)
**Status**: 🔬 IN PROGRESS - Root cause not yet fully resolved

---

## Summary

Investigated severe regression reported in `HANDBACK_TO_WEB_SESSION.md`:
- **Expected**: +6 to +8 dB improvement over baseline
- **Actual**: -10 to -11 dB regression (catastrophic failure)
- **Pattern**: Matches FAIL-001 from October 2025 (geodesic EDT over-clamping)

---

## Fixes Applied

### 1. ✅ Adam Optimizer Gradient Formula Fix
**File**: `lgi-encoder-v2/src/adam_optimizer.rs:157-177`

**Problem**: Gradient computation didn't match renderer formula
- Renderer uses: `dist_sq = (dx_rot/sx)^2 + (dy_rot/sy)^2` (Mahalanobis distance)
- Optimizer used: `dist_sq = dx^2 + dy^2` then `weight = exp(-0.5*dist_sq/(sx*sy))`
- **Mismatch**: Different formulas = incorrect gradients!

**Fix**:
```rust
// Before (WRONG):
let dist_sq = dx * dx + dy * dy;
let weight = (-0.5 * dist_sq / (sx * sy)).exp();

// After (CORRECT):
let dist_sq = (dx / sx).powi(2) + (dy / sy).powi(2);
let weight = (-0.5 * dist_sq).exp();
```

Also fixed position and scale gradients to match new dist_sq formula.

---

### 2. 🚧 Geodesic Clamping Fix (DISABLED for testing)
**File**: `lgi-encoder-v2/src/lib.rs:640-661`

**Problem**: Over-aggressive clamping breaks coverage
- Formula: `max_scale_px = 0.5 + 0.3 * geod_dist_px`
- For small geod_dist_px values → scales < 1 pixel
- Result: Zero coverage → zero gradients → no optimization

**Fix Attempted**:
```rust
// Only apply clamping if max_scale_px >= 3.0
if max_scale_px >= 3.0 {
    // Apply clamping
}
// else: skip clamping entirely if it would over-constrain
```

**Current Status**: Temporarily disabled (`apply_geodesic_clamping` commented out in line 453)

---

### 3. ✅ Debug Instrumentation Added

**renderer_v2.rs**: Added W_median (coverage) logging
- Computes median weight across all pixels
- Helps detect zero-coverage issue (FAIL-001 pattern)

**lib.rs**: Added Gaussian scale logging
- Logs average σ_x and σ_y in both normalized and pixel units
- Helps detect over-clamping

---

## Current Results (After Fixes)

**Complex Pattern (128×128)**:
```
║ Baseline              │   64 │      15.21 │  +0.00 │
║ Error-Driven          │   50 │       9.44 │  -5.78 │ ❌
║ Adam                  │   50 │       4.26 │ -10.96 │ ❌
║ GPU                   │   50 │       9.30 │  -5.92 │ ❌
```

**Expected** (Session 7):
```
Baseline:        15.50 dB
Error-Driven:    21.94 dB (+6.44 dB) ✅
Adam:            21.26 dB (+5.76 dB) ✅
GPU:             21.96 dB (+6.46 dB) ✅
```

**Status**: Regression NOT RESOLVED - Adam still shows 4.26 dB

---

## Key Observations

### Loss Stuck Pattern
Adam optimizer shows **two optimization passes**:
1. **First pass**: Loss decreases normally (0.869 → 0.127)
2. **Second pass**: Loss STUCK at 0.375001 (no improvement)

This suggests the problem occurs **after** first optimization, possibly during:
- Gaussian addition/splitting
- Geodesic clamping between passes
- Scale reinitialization

### Error-Driven Also Affected
Not just Adam - Error-Driven method also regressed (-5.78 dB instead of +6 dB)
- Uses same clamping logic
- Different optimizer (SGD-based)
- Similar failure mode

---

## Root Cause Hypothesis

**Multiple interacting issues**:
1. ✅ **Gradient mismatch** - FIXED but not sufficient
2. 🔬 **Geodesic over-clamping** - Suspected but disabling didn't resolve
3. ❓ **Unknown factor** - Something else is breaking optimization

**Commit 776ffdd** mentioned in handback doesn't exist in repo
- Suggests codebase state mismatch
- May need to investigate git history more carefully

---

## Next Steps

### Option A: Continue Debugging
1. Investigate hotspot/splitting logic (lib.rs:469-486)
2. Check if new Gaussians have correct initialization
3. Add more instrumentation to track scales through optimization
4. Check for NaN/infinity values in gradients

### Option B: Bisect Git History
1. Find actual commit that introduced regression
2. Compare before/after behavior
3. Identify exact code change that broke optimization

### Option C: Compare with Known-Good Code
1. Check if there's a Session 7 branch/tag
2. Diff against working implementation
3. Find what changed

---

## Files Modified

1. `lgi-encoder-v2/src/adam_optimizer.rs` - Fixed gradient formulas
2. `lgi-encoder-v2/src/lib.rs` - Modified geodesic clamping (disabled)
3. `lgi-encoder-v2/src/renderer_v2.rs` - Added debug instrumentation

---

## References

- `HANDBACK_TO_WEB_SESSION.md` - Original bug report
- `CRITICAL_REGRESSION_FOUND.md` - Evidence from local testing
- `PROJECT_HISTORY.md` - FAIL-001 documentation (October 2025)

---

**Investigation continues...**
