# Baseline Fix Status - November 14, 2025

## Current State: REGRESSION NOT YET RESOLVED

**Target**: Session 8 baseline (Oct 7, 2025)
- Adam: 21.26 dB (+5.76 dB) ✅
- GPU: 21.96 dB (+6.46 dB) ✅

**Current Results**:
- Adam: 4.26 dB (-10.96 dB) ❌
- GPU: 9.30 dB (-5.92 dB) ❌

---

## Root Cause Analysis

### 1. ✅ CONFIRMED: Gradient Formula Mismatch (NOW FIXED)

**Problem**: Adam optimizer computed gradients with wrong formula

**Renderer** (`renderer_v2.rs:45-52`):
```rust
// Rotate coordinates to Gaussian's local frame
let cos_t = theta.cos();
let sin_t = theta.sin();
let dx_rot = dx * cos_t + dy * sin_t;
let dy_rot = -dx * sin_t + dy * cos_t;

// Mahalanobis distance in rotated frame
let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);
let weight = (-0.5 * dist_sq).exp();
```

**Original Optimizer** (WRONG):
```rust
let dist_sq = dx * dx + dy * dy;  // Euclidean, no rotation!
let weight = (-0.5 * dist_sq / (sx * sy)).exp();  // Different formula!
```

**Fixed Optimizer**  (`adam_optimizer.rs:163-174`):
```rust
// Now matches renderer exactly
let cos_t = theta.cos();
let sin_t = theta.sin();
let dx_rot = dx * cos_t + dy * sin_t;
let dy_rot = -dx * sin_t + dy * cos_t;
let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);
let weight = (-0.5 * dist_sq).exp();
```

**Status**: Fixed in commit 299f69c

---

### 2. ❓ UNRESOLVED: Something Breaks Between Optimization Passes

**Observation**: Adam optimizer shows TWO distinct behaviors:

**Pass 0** (First optimization):
```
Iteration 10: loss = 0.126823
Iteration 20: loss = 0.128541
...
Iteration 100: loss = 0.259055
```
Loss is changing (not stuck) ✅

**Pass 1** (Second optimization):
```
Iteration 10: loss = 0.375001
Iteration 20: loss = 0.375001
...
Iteration 100: loss = 0.375001
```
Loss COMPLETELY STUCK ❌

**This Pattern Indicates**:
- First pass works (gradients are correct)
- Something happens BETWEEN passes that breaks optimization
- Newly added Gaussians or clamping destroying gradient signal

---

### 3. Suspected Culprits

**A. Geodesic Clamping** (`lib.rs:640-661`)

Currently skips clamping if `max_scale_px < 3.0`:
```rust
if max_scale_px >= 3.0 {
    // Apply clamping
}
// else: skip if would over-constrain
```

But this might not be enough - needs investigation of:
- What scales are ACTUALLY being clamped to
- Whether new Gaussians added between passes get over-clamped

**B. New Gaussian Initialization** (`lib.rs:485-544`)

When adding Gaussians at hotspots:
```rust
let sigma_base_px = gamma * ((width_px * height_px) / current_n as f32).sqrt();
```

Questions:
- Are new Gaussians getting correct scales?
- Do they match initialization formula?
- Are they being clamped too aggressively immediately after creation?

**C. Potential NaN/Inf Issues**

Gradients might be producing NaN or Inf values that propagate

---

## What We Know FOR SURE

1. **LOD thresholds are CORRECT** (0.0004, 0.0001) - not the cause
2. **LOD not used in EncoderV2** - only in encoder_v3_adaptive
3. **Gradient formula was WRONG** - now fixed with rotation
4. **Problem occurs BETWEEN optimization passes** - not during first pass
5. **Initial commit already had broken code** - Nov 14 reorganization introduced bugs

---

## User's Strategy: Incremental Baseline

**Recommended Approach** (user's suggestion):
1. Find known-good baseline where everything works
2. Add ONE feature at a time
3. Thoroughly validate each addition
4. Trust code and results, NOT documentation

**Problem**: We don't have access to Session 7/8 working code (Oct 6-7, 2025)
- Repository reorganized Nov 14, 2025
- Initial commit may have already included regression
- Need to BUILD a working baseline from scratch

---

## Next Steps

### Immediate Debugging

1. **Add comprehensive logging** to see what happens between passes:
   ```rust
   log::info!("Min scale before opt: {:.6}", ...);
   log::info!("Min scale after opt: {:.6}", ...);
   log::info!("Min scale after clamp: {:.6}", ...);
   log::info!("New Gaussian scales: {:.6}", ...);
   ```

2. **Check for NaN/Inf**:
   ```rust
   for grad in gradients {
       assert!(grad.scale_x.is_finite(), "NaN/Inf in gradients!");
   }
   ```

3. **Disable geodesic clamping entirely** to isolate issue:
   ```rust
   // self.apply_geodesic_clamping(&mut gaussians);  // DISABLED
   ```

4. **Test with just ONE optimization pass**:
   ```rust
   for pass in 0..1 {  // Was 0..10
   ```

### Build Known-Good Baseline

If debugging doesn't reveal issue:

1. **Start from simplest possible optimizer**:
   - No geodesic clamping
   - No splitting/densification
   - Just single-pass optimization
   - Verify this works

2. **Add features incrementally**:
   - Step 1: Basic optimization (verify +6dB)
   - Step 2: Add splitting (verify still works)
   - Step 3: Add geodesic clamping (verify still works)
   - Step 4: Add multiple passes (verify still works)

3. **Document each working state**

---

## Files Modified

1. `adam_optimizer.rs` - Fixed gradient formula with rotation
2. `lib.rs` - Added scale diagnostics, re-enabled clamping
3. `renderer_v2.rs` - Added W_median logging (from earlier)

---

## Commit History

- `299f69c`: WIP: Investigate and attempt fixes for -10dB regression
- `c696d1b`: Add complete handback report with regression evidence
- `299b2f4`: CRITICAL: Severe regression detected in local testing

---

**Status**: Investigation continuing
**Blocker**: Pass 2 loss stuck at 0.375001
**Strategy**: Incremental baseline building per user's recommendation

**Next**: Add detailed diagnostics to identify what breaks between passes
