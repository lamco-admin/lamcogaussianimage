# V1 Implementation Specification

**Goal**: Single-pass baseline that achieves +6 dB improvement
**Success Criteria**: PSNR 15.5 dB → 21.5 dB in single 100-iteration pass
**Strategy**: Strip all complexity, fix core optimization, rebuild incrementally

---

## Changes Required

### Change 1: Simplify EncoderV2 main loop

**File**: `packages/lgi-rs/lgi-encoder-v2/src/lib.rs`

**Current code** (lines ~430-610):
```rust
// Multi-pass optimization with densification
for pass in 0..self.config.num_passes {
    log::info!("  Pass {}/{}", pass + 1, self.config.num_passes);

    // Optimize
    let loss = optimizer.optimize(&mut gaussians, &self.target);

    // Apply geodesic clamping
    self.apply_geodesic_clamping(&mut gaussians);

    // Add Gaussians at hotspots
    if pass < self.config.num_passes - 1 {
        self.add_gaussians_at_hotspots(&mut gaussians, &error_map);
    }
}
```

**New code** (V1 simplified):
```rust
// V1: Single-pass optimization with fixed N
log::info!("  V1 single-pass optimization (100 iterations)");

// Configure optimizer for single pass
let iterations = 100;
let loss = optimizer.optimize_iterations(&mut gaussians, &self.target, iterations);

log::info!("  Final loss: {:.6}", loss);

// NO geodesic clamping
// NO densification
// NO multi-pass
```

**Rationale**:
- First pass works (loss changes)
- Second pass breaks (loss stuck)
- Therefore: Eliminate second pass entirely
- Focus: Get single pass to full convergence

---

### Change 2: Update AdamOptimizer API

**File**: `packages/lgi-rs/lgi-encoder-v2/src/adam_optimizer.rs`

**Current**: `optimize()` runs single iteration, returns loss

**New**: Add `optimize_iterations()` that runs N iterations

```rust
impl AdamOptimizer {
    /// Run optimization for N iterations (V1 API)
    pub fn optimize_iterations(
        &mut self,
        gaussians: &mut [Gaussian2D],
        target: &RgbImage,
        iterations: usize,
    ) -> f32 {
        let mut final_loss = 0.0;

        for iter in 0..iterations {
            final_loss = self.optimize_single_iteration(gaussians, target);

            // Logging every 10 iterations
            if iter % 10 == 0 || iter == iterations - 1 {
                log::info!("    Iteration {}: loss = {:.6}", iter + 1, final_loss);
            }

            // Diagnostics: Check for NaN/Inf
            self.check_for_numerical_issues(gaussians, iter)?;
        }

        final_loss
    }

    /// Single iteration (refactored from old optimize())
    fn optimize_single_iteration(
        &mut self,
        gaussians: &mut [Gaussian2D],
        target: &RgbImage,
    ) -> f32 {
        // Existing optimization code (compute gradients, apply updates)
        // ... (keep all existing logic)
    }

    /// Detect NaN/Inf/scale collapse
    fn check_for_numerical_issues(
        &self,
        gaussians: &[Gaussian2D],
        iteration: usize,
    ) -> Result<(), String> {
        // Check 1: NaN/Inf in positions
        for (i, g) in gaussians.iter().enumerate() {
            if !g.position.x.is_finite() || !g.position.y.is_finite() {
                return Err(format!(
                    "Iteration {}: NaN/Inf in position of Gaussian {}",
                    iteration, i
                ));
            }
        }

        // Check 2: Scale collapse
        let min_scale = gaussians
            .iter()
            .map(|g| g.shape.scale_x.min(g.shape.scale_y))
            .fold(f32::INFINITY, |a, b| a.min(b));

        if min_scale < 1e-6 {
            log::warn!(
                "Iteration {}: Scale collapse detected! min_scale = {:.8}",
                iteration, min_scale
            );
        }

        // Check 3: All gradients zero (optimization stuck)
        // (would need gradient history, skip for V1)

        Ok(())
    }
}
```

**Rationale**:
- Cleaner API for multi-iteration optimization
- Centralized diagnostics
- Early detection of numerical issues

---

### Change 3: Add minimum scale bounds

**File**: `packages/lgi-rs/lgi-encoder-v2/src/adam_optimizer.rs`

**Location**: In gradient application (line ~250-280)

**Current**:
```rust
// Apply Adam updates without bounds
gaussian.shape.scale_x -= self.learning_rate * m_corrected_sx / (v_corrected_sx.sqrt() + 1e-8);
gaussian.shape.scale_y -= self.learning_rate * m_corrected_sy / (v_corrected_sy.sqrt() + 1e-8);
```

**New** (add bounds):
```rust
// Apply Adam updates
gaussian.shape.scale_x -= self.learning_rate * m_corrected_sx / (v_corrected_sx.sqrt() + 1e-8);
gaussian.shape.scale_y -= self.learning_rate * m_corrected_sy / (v_corrected_sy.sqrt() + 1e-8);

// V1: Add minimum scale bounds to prevent collapse
const MIN_SCALE: f32 = 0.001; // ~1 pixel at 1000px image
gaussian.shape.scale_x = gaussian.shape.scale_x.max(MIN_SCALE);
gaussian.shape.scale_y = gaussian.shape.scale_y.max(MIN_SCALE);
```

**Rationale**:
- Research shows scale collapse is common failure mode
- Minimum bounds prevent gradients from pushing scales to zero
- Conservative threshold: 0.001 ≈ 1 pixel in normalized coordinates

---

### Change 4: Disable geodesic clamping (comment out)

**File**: `packages/lgi-rs/lgi-encoder-v2/src/lib.rs`

**Location**: Lines ~640-680

**Current**:
```rust
fn apply_geodesic_clamping(&self, gaussians: &mut [Gaussian2D]) {
    // ... (existing implementation)
}
```

**New** (V1: disable entirely):
```rust
/// V1: DISABLED - Geodesic clamping suspected to over-constrain
/// Will re-enable in V2 after baseline works
#[allow(dead_code)]
fn apply_geodesic_clamping_DISABLED(&self, gaussians: &mut [Gaussian2D]) {
    // ... (keep code for reference, but don't call)
}
```

**Rationale**:
- Suspected to over-constrain scales
- Test hypothesis: Does optimization work WITHOUT clamping?
- Can re-enable later if needed

---

### Change 5: Disable densification (comment out)

**File**: `packages/lgi-rs/lgi-encoder-v2/src/lib.rs`

**Location**: Lines ~485-544

**Current**:
```rust
fn add_gaussians_at_hotspots(&self, gaussians: &mut Vec<Gaussian2D>, error_map: &[f32]) {
    // ... (existing implementation)
}
```

**New** (V1: disable):
```rust
/// V1: DISABLED - Densification breaks multi-pass optimization
/// Will re-enable in V2 after single-pass baseline works
#[allow(dead_code)]
fn add_gaussians_at_hotspots_DISABLED(&self, gaussians: &mut Vec<Gaussian2D>, error_map: &[f32]) {
    // ... (keep code for reference, but don't call)
}
```

**Rationale**:
- Suspected to break second optimization pass
- V1 uses fixed N Gaussians throughout
- Can re-enable in V2 with proper testing

---

### Change 6: Improve diagnostics

**File**: `packages/lgi-rs/lgi-encoder-v2/src/renderer_v2.rs`

**Current**: W_median logged once

**New**: Log more comprehensive stats

```rust
pub fn render_with_diagnostics(
    &self,
    gaussians: &[Gaussian2D],
) -> (RgbImage, RenderStats) {
    let mut output = RgbImage::new(self.width, self.height);
    let mut all_weights = Vec::new();
    let mut max_weight_per_pixel = Vec::new();

    // ... (existing rendering code)

    // Compute diagnostics
    all_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let w_median = all_weights[all_weights.len() / 2];
    let w_mean = all_weights.iter().sum::<f32>() / all_weights.len() as f32;
    let w_max = *all_weights.last().unwrap();

    let stats = RenderStats {
        w_median,
        w_mean,
        w_max,
        num_pixels: (self.width * self.height) as usize,
        num_gaussians: gaussians.len(),
    };

    log::info!(
        "  Rendering: W_median={:.4}, W_mean={:.4}, W_max={:.4}",
        stats.w_median, stats.w_mean, stats.w_max
    );

    (output, stats)
}

#[derive(Debug, Clone)]
pub struct RenderStats {
    pub w_median: f32,
    pub w_mean: f32,
    pub w_max: f32,
    pub num_pixels: usize,
    pub num_gaussians: usize,
}
```

**Rationale**:
- More data points for debugging
- W_median alone not sufficient
- Need mean and max to understand distribution

---

## Testing Strategy

### Test 1: Gradient verification (unit test)

**File**: `packages/lgi-rs/lgi-encoder-v2/tests/gradient_test.rs` (new)

```rust
use lgi_encoder_v2::*;

#[test]
fn test_gradients_finite_difference() {
    // Create simple test case
    let gaussian = Gaussian2D {
        position: Vec2 { x: 0.5, y: 0.5 },
        shape: GaussianShape {
            scale_x: 0.05,
            scale_y: 0.03,
            rotation: 0.2,
        },
        color: Rgb([128, 64, 200]),
    };

    let target = create_test_image(100, 100);
    let optimizer = AdamOptimizer::new(0.01);

    // Compute analytical gradient
    let analytical_grad = optimizer.compute_gradient_for_gaussian(
        &gaussian,
        &target,
        0, // gaussian index
    );

    // Compute numerical gradient (finite difference)
    let eps = 1e-4;
    let numerical_grad_x = compute_numerical_gradient(
        &gaussian,
        &target,
        |g| g.position.x,
        eps,
    );

    // Should match within 1%
    let error_pct = (analytical_grad.position.x - numerical_grad_x).abs()
        / numerical_grad_x.abs() * 100.0;

    assert!(
        error_pct < 1.0,
        "Gradient mismatch: analytical={}, numerical={}, error={}%",
        analytical_grad.position.x, numerical_grad_x, error_pct
    );
}
```

**Success criteria**: All gradient components match within 1%

---

### Test 2: Single-pass convergence (integration test)

**File**: `packages/lgi-rs/lgi-encoder-v2/tests/integration_test.rs`

```rust
#[test]
fn test_v1_single_pass_convergence() {
    // Create simple test image (solid color gradient)
    let target = create_gradient_image(100, 100);

    // Initialize encoder
    let config = EncoderConfig {
        num_gaussians: 100,
        learning_rate: 0.01,
        num_passes: 1, // V1: single pass
        iterations_per_pass: 100,
    };
    let encoder = EncoderV2::new(config, target.clone());

    // Run encoding
    let gaussians = encoder.encode();

    // Render result
    let rendered = renderer.render(&gaussians);

    // Measure PSNR
    let psnr = compute_psnr(&target, &rendered);

    // V1 success criteria
    assert!(
        psnr > 21.0,
        "V1 failed: PSNR = {:.2} dB (expected > 21.0 dB)",
        psnr
    );

    // Check no scale collapse
    let min_scale = gaussians
        .iter()
        .map(|g| g.shape.scale_x.min(g.shape.scale_y))
        .fold(f32::INFINITY, |a, b| a.min(b));

    assert!(
        min_scale > 0.001,
        "Scale collapse: min_scale = {:.6}",
        min_scale
    );
}
```

**Success criteria**:
- PSNR > 21.0 dB
- No scale collapse (min_scale > 0.001)
- Loss decreasing trend

---

### Test 3: Regression test against Session 8 baseline

**File**: `packages/lgi-rs/lgi-encoder-v2/benches/baseline_bench.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_v1_vs_session8(c: &mut Criterion) {
    let target = load_test_image("test_image.png");

    c.bench_function("v1_encoder_100_gaussians", |b| {
        b.iter(|| {
            let encoder = EncoderV2::new_v1(100, target.clone());
            black_box(encoder.encode())
        });
    });
}

criterion_group!(benches, benchmark_v1_vs_session8);
criterion_main!(benches);
```

**Success criteria**:
- Matches Session 8 baseline: PSNR ≈ 21.26 dB
- Takes < 5 seconds per image

---

## Implementation Checklist

- [ ] **Step 1**: Update EncoderV2 main loop (lib.rs:430-610)
  - Remove multi-pass loop
  - Remove geodesic clamping call
  - Remove densification call
  - Call `optimizer.optimize_iterations(100)`

- [ ] **Step 2**: Add AdamOptimizer::optimize_iterations() (adam_optimizer.rs)
  - Create new method wrapping iteration loop
  - Add diagnostics logging
  - Add numerical issue detection

- [ ] **Step 3**: Add minimum scale bounds (adam_optimizer.rs:250-280)
  - Clamp scale_x, scale_y to MIN_SCALE = 0.001
  - After each Adam update

- [ ] **Step 4**: Disable geodesic clamping (lib.rs:640-680)
  - Rename method to `*_DISABLED`
  - Add comment explaining why

- [ ] **Step 5**: Disable densification (lib.rs:485-544)
  - Rename method to `*_DISABLED`
  - Add comment explaining why

- [ ] **Step 6**: Improve rendering diagnostics (renderer_v2.rs)
  - Return RenderStats struct
  - Log W_median, W_mean, W_max

- [ ] **Step 7**: Write gradient verification test
  - Create tests/gradient_test.rs
  - Implement finite difference check

- [ ] **Step 8**: Write integration test
  - Create tests/integration_test.rs
  - Test full V1 pipeline

- [ ] **Step 9**: Run tests
  - `cargo test --release`
  - Verify all pass

- [ ] **Step 10**: Run benchmark
  - `cargo bench`
  - Compare to Session 8 baseline

- [ ] **Step 11**: Update documentation
  - BASELINE_FIX_STATUS.md
  - V1_IMPLEMENTATION_SPEC.md (this file)

---

## Expected Results

### Iteration Log (Expected)

```
Pass 1/1
V1 single-pass optimization (100 iterations)
  Iteration 10: loss = 0.280532
  Iteration 20: loss = 0.251034
  Iteration 30: loss = 0.228541
  Iteration 40: loss = 0.210338
  Iteration 50: loss = 0.195221
  Iteration 60: loss = 0.182847
  Iteration 70: loss = 0.172109
  Iteration 80: loss = 0.162891
  Iteration 90: loss = 0.154982
  Iteration 100: loss = 0.148203
Final loss: 0.148203
Rendering: W_median=0.542, W_mean=0.498, W_max=0.987
PSNR: 21.48 dB (+5.98 dB from baseline)
```

### Scale Statistics (Expected)

```
Initial scales:
  Min: 0.0183, Max: 0.0547, Mean: 0.0312

After optimization:
  Min: 0.0058, Max: 0.0891, Mean: 0.0402
  (No collapse: min > 0.001 ✅)
```

### Coverage Statistics (Expected)

```
W_median: 0.542 (> 0.5 ✅)
W_mean: 0.498
W_max: 0.987
(Good coverage across pixels)
```

---

## Failure Modes and Debugging

### If loss still stuck:

**Symptom**: Loss = 0.375001 at all iterations

**Debug steps**:
1. Check gradient computation:
   ```rust
   log::info!("Gradient for Gaussian 0: {:?}", gradients[0]);
   ```
   - Expected: Non-zero values
   - If all zero: Gradient formula broken

2. Check rendering:
   ```rust
   log::info!("Rendered pixel (50,50): {:?}", rendered.get_pixel(50, 50));
   log::info!("Target pixel (50,50): {:?}", target.get_pixel(50, 50));
   ```
   - Expected: Different values
   - If identical: Rendering not using Gaussians

3. Check Adam updates:
   ```rust
   log::info!("m[0] before: {:?}", self.m[0]);
   log::info!("v[0] before: {:?}", self.v[0]);
   // ... apply update ...
   log::info!("m[0] after: {:?}", self.m[0]);
   log::info!("v[0] after: {:?}", self.v[0]);
   ```
   - Expected: Values changing
   - If frozen: Adam state issue

---

### If scale collapse occurs:

**Symptom**: min_scale < 0.001 during optimization

**Debug steps**:
1. Check gradient magnitudes:
   ```rust
   let grad_magnitude = (grad.scale_x.powi(2) + grad.scale_y.powi(2)).sqrt();
   log::info!("Gradient magnitude for Gaussian {}: {:.6}", i, grad_magnitude);
   ```
   - Expected: Reasonable values (0.001 - 1.0)
   - If too large: Gradient explosion

2. Check if minimum bounds work:
   ```rust
   assert!(gaussian.shape.scale_x >= MIN_SCALE, "Bounds violated!");
   ```

3. Reduce learning rate:
   ```rust
   let learning_rate = 0.001; // 10× smaller
   ```

---

### If coverage is low:

**Symptom**: W_median < 0.3

**Debug steps**:
1. Check initialization:
   ```rust
   log::info!("Initial sigma_base: {:.6}", sigma_base);
   ```
   - Expected: ~0.03 for 1000px image with 100 Gaussians
   - If too small: Increase gamma parameter

2. Visualize Gaussian positions:
   ```rust
   for g in &gaussians {
       log::info!("Gaussian at ({:.3}, {:.3}), scale=({:.4}, {:.4})",
           g.position.x, g.position.y, g.shape.scale_x, g.shape.scale_y);
   }
   ```
   - Check if positions clustered or spread out

---

## Success Criteria Summary

V1 is **COMPLETE** when:

1. ✅ Single pass achieves **PSNR > 21.0 dB**
2. ✅ Loss **decreases monotonically** (mostly)
3. ✅ No scale collapse: **min_scale > 0.001**
4. ✅ Adequate coverage: **W_median > 0.5**
5. ✅ Gradients verified via **finite difference** (< 1% error)
6. ✅ All tests pass
7. ✅ Documented in BASELINE_FIX_STATUS.md

---

## Timeline Estimate

| Task | Time | Cumulative |
|------|------|------------|
| Code changes | 30 min | 30 min |
| Write tests | 20 min | 50 min |
| Run tests | 10 min | 60 min |
| Debug (if needed) | 30 min | 90 min |
| Documentation | 10 min | 100 min |
| **TOTAL** | **100 min** | **~1.5 hours** |

---

## Rollback Plan

**If V1 doesn't work after debugging**:

1. Revert to commit 827d186 (gradient fix)
2. Try alternative approach:
   - Switch to Cholesky parameterization
   - Switch to alpha compositing
   - Try different learning rate
3. Consult user for strategy adjustment

---

**Status**: Ready to implement
**Next**: Execute implementation checklist
**Confidence**: High (based on research and analysis)
