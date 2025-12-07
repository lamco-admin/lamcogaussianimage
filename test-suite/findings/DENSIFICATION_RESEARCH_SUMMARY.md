# Densification Research Summary

## The Problem We're Solving

**Goal**: Start with few Gaussians (16-128) and grow to many (256-512) while achieving **equal or better** quality than starting with the full count.

**Why This Matters for Codec**:
- Rate-distortion optimization: Fewer Gaussians = smaller bitstream
- Progressive encoding: Coarse → fine transmission
- Adaptive quality: Match Gaussian count to image complexity

**Current Reality**: ~~All our densification approaches hit a ceiling **0.5-1.3 dB below** simply starting with 256 Gaussians.~~ **UPDATE 2025-12-06**: Research revealed that this gap may be due to tests bypassing the actual 3DGS implementation. See "Critical Discovery" section below.

---

## ⚠️ CRITICAL DISCOVERY (2025-12-06)

### The Implementation-Test Gap

**Key Finding**: The `AdaptiveDensifier` module ALREADY EXISTS with clone/split/prune operations, but the densification experiments were **bypassing it entirely** and using broken inline logic.

| Component | What Exists | What Tests Used |
|-----------|-------------|-----------------|
| Clone operation | ✅ `AdaptiveDensifier::clone_gaussian()` | ❌ Not used |
| Split operation | ✅ `AdaptiveDensifier::split_gaussian()` | ⚠️ Inline reimplementation |
| Prune operation | ✅ `AdaptiveDensifier::prune()` | ❌ Not used |
| Gradient signal | ✅ Adam's `m_position` (accumulated EMA) | ❌ Used instantaneous error proxy |

### The V2 Bug: Parent Conservation Violation

The `test_densification_v2.rs` file implements `densify_v2()` which **keeps the parent when splitting**:

```rust
// BROKEN: Creates 3 Gaussians from 1 (not 2 from 1)
gaussians.extend(new_gaussians.clone());  // Adds children BUT KEEPS PARENT
```

In 3DGS, split replaces 1 Gaussian with 2 smaller ones. The V2 logic creates 3 Gaussians where 1 existed, violating the conservation principle and preventing proper optimization.

### Real vs Reported Results

| Test Type | Implementation | Gap to Baseline |
|-----------|----------------|-----------------|
| V2 inline (reported) | Broken logic | -0.5 to -1.3 dB |
| **AdaptiveDensifier (actual)** | Real 3DGS-style | **-0.13 dB** |

The actual `test_adaptive_densification.rs` using `AdaptiveDensifier` achieved **-0.13 dB** vs baseline—nearly within the 0.1 dB success criterion!

### Adam's Hidden Gradient Accumulation

**Breakthrough insight**: Adam optimizer already computes accumulated position gradients in `m_position`:

```rust
// Adam's first moment IS accumulated gradient (EMA with β₁=0.9)
self.m_position[i].x = 0.9 * self.m_position[i].x + 0.1 * grad.position.x;
```

This is effectively what 3DGS accumulates over 100 iterations. We don't need a separate accumulation system—just expose Adam's internal state for densification decisions.

### 3DGS Paper Thresholds vs LGI

| Parameter | 3DGS Paper | LGI AdaptiveDensifier | Fix Applied |
|-----------|------------|----------------------|-------------|
| Gradient threshold | 0.0005 | 0.0002 (2.5× aggressive) | Set to 0.0005 |
| Size threshold | 1% of scene | 0.05 (5% absolute) | Set to 0.01 |
| Pruning threshold | 0.005 | 0.005 | ✅ Match |
| Interval | 100 iters | 100 iters | ✅ Match |

### Test Results (2025-12-06)

Using proper AdaptiveDensifier with Adam's m_position gradients:

| Image | Baseline (256 G) | Densify (~120 G) | Gap | Gaussians |
|-------|------------------|------------------|-----|-----------|
| kodim03 | 20.01 dB | **20.89 dB** | **+0.88 dB** | 120 (47%) |
| kodim05 | 17.62 dB | 17.36 dB | **-0.26 dB** | 123 (48%) |
| kodim08 | 13.15 dB | 12.81 dB | **-0.34 dB** | 123 (48%) |

**Key findings:**
1. ✅ Densification WORKS with proper implementation
2. ✅ Achieves ~50% Gaussian reduction with <0.35 dB quality loss
3. ⚠️ Only 1 densification event per image (threshold too conservative)
4. ⚠️ Baseline test has LR decay bug (kodim03 peaked at 22.46, dropped to 20.01)

**Next steps:**
- Lower gradient threshold to 0.0002-0.0003 for more densification events
- Fix LR decay formula to maintain minimum LR floor
- Test reaching full 256 Gaussian target

---

## What We've Tried (Exhaustively)

| Parameter | Range Tested | Best Result |
|-----------|--------------|-------------|
| Starting count | 16, 32, 64, 100, 128 | 100 (-0.55 dB) |
| Delay before start | 0, 30, 50, 80, 100 iterations | 50 (marginal) |
| Split threshold | 0.00003 to 0.0005 | No clear winner |
| Densify interval | 8 to 40 iterations | No clear winner |
| Child opacity | 0.5, 1.0 | 1.0 (marginal) |
| State inheritance | None, Full, Partial | Image-dependent |

**Conclusion**: Parameter tuning is exhausted. The algorithm itself is wrong.

---

## 3DGS Densification: The Reference Implementation

The original 3D Gaussian Splatting paper (Kerbl et al., SIGGRAPH 2023) describes a sophisticated densification system we haven't faithfully implemented.

### 1. Gradient Accumulation (NOT Instantaneous Error)

**What 3DGS does**:
```python
# Accumulate position gradients over iterations
for iteration in range(densify_interval):
    loss.backward()

    # Accumulate gradient magnitude per Gaussian
    grad_accum[i] += torch.norm(gaussian.position.grad)
    grad_count[i] += 1

# Trigger based on average accumulated gradient
avg_grad = grad_accum / grad_count
should_densify = avg_grad > threshold
```

**What we do**: Check instantaneous local error at Gaussian center.

**Why it matters**: Position gradient magnitude indicates how much the optimizer "wants" to move the Gaussian. High gradient = Gaussian is in wrong place or wrong shape. Accumulated over iterations, this is more stable than instantaneous error.

### 2. Clone vs Split Distinction

**3DGS uses TWO different operations**:

| Condition | Operation | Intuition |
|-----------|-----------|-----------|
| High grad + **small** Gaussian | **Clone** | Under-reconstruction: Gaussian too small to cover needed area |
| High grad + **large** Gaussian | **Split** | Over-reconstruction: One Gaussian covering too much variation |

**Clone operation**:
```python
# Duplicate the Gaussian and shift it in gradient direction
new_gaussian = copy(gaussian)
new_gaussian.position += gradient_direction * offset
# Keep same size
```

**Split operation**:
```python
# Replace with two smaller Gaussians offset along major axis
child1 = gaussian.clone()
child2 = gaussian.clone()
child1.position += major_axis * offset
child2.position -= major_axis * offset
child1.scale *= 0.6  # Shrink
child2.scale *= 0.6
# Remove parent
```

**What we do**: Only split, no distinction based on Gaussian size.

### 3. Pruning (We Don't Have This)

**3DGS prunes Gaussians that**:
1. Have opacity < threshold (e.g., 0.005) - "transparent" Gaussians
2. Are too large in screen-space - "bloated" Gaussians covering too much
3. Are too small to render - sub-pixel Gaussians

**Why it matters**: Without pruning:
- Dead Gaussians accumulate and interfere with rendering
- Optimizer wastes capacity on useless primitives
- Memory/compute grows without quality benefit

### 4. Periodic Opacity Reset

**3DGS does**:
```python
if iteration % opacity_reset_interval == 0:
    for gaussian in gaussians:
        gaussian.opacity = small_value  # e.g., 0.01
    # Let optimizer re-learn which Gaussians matter
```

**Why it matters**: After densification, some original Gaussians become redundant. Resetting opacity lets Adam "vote" on which Gaussians are actually needed by increasing their opacity through gradient descent.

---

## What the Research Literature Says

### BOGausS (Better Optimized Gaussian Splatting)

Key contributions relevant to densification:
1. **State inheritance**: Children inherit parent's Adam (m, v, t) state
2. **Precision-aware scaling**: Scale updates by confidence per Gaussian
3. **Rate-distortion pruning**: Prune based on PSNR contribution, not just opacity

Our test showed inheritance helps on difficult images (+0.48 dB on kodim08).

### Taming 3DGS

Key insight: **Sparse Adam** - only update Gaussians that contributed to rendered pixels in the batch. Combined with fused CUDA kernels for 1.6x speedup.

Relevance: When we densify, we're adding Gaussians that may not immediately contribute. Sparse updates could prevent wasted computation.

### 3DGS-LM (Levenberg-Marquardt)

Replaces Adam with second-order optimization. Reports:
- 20-30% faster training
- Similar quality
- Better convergence after densification (curvature information helps)

May be overkill for 2D, but the insight is: Adam may struggle after the optimization landscape suddenly changes (densification).

---

## Hypotheses for Why Our Densification Fails

### Hypothesis 1: Position Lock-In
Early Gaussians settle into locally-optimal positions. When we split, children inherit these positions and can't escape the local minimum.

**Test**: After densification, run a "position perturbation" phase with high position LR.

### Hypothesis 2: Coverage Gaps
Sparse initialization leaves some image regions uncovered. Densification only splits existing Gaussians - it can't fill gaps where no Gaussian exists.

**Test**: Implement **Clone** operation that can place new Gaussians in high-error regions regardless of existing Gaussian positions.

### Hypothesis 3: Gradient Signal Loss
After split, the two children compete for the same gradient signal. With half the opacity each (our old approach) or in the same region, they may interfere.

**Test**: Use gradient accumulation over iterations, not instantaneous error.

### Hypothesis 4: No Cleanup
Dead/redundant Gaussians from failed densification attempts accumulate and interfere with good Gaussians.

**Test**: Implement pruning of low-opacity Gaussians.

### Hypothesis 5: Optimizer State Disruption
Splitting resets optimizer state (in our current approach), causing children to "start from scratch" despite inheriting positions/colors.

**Test**: State inheritance (we tested this - helps on hard images).

---

## Implementation Plan for Faithful 3DGS Densification

```rust
struct DensificationState {
    // Per-Gaussian gradient accumulation
    position_grad_accum: Vec<f32>,
    grad_count: Vec<u32>,

    // Per-Gaussian Adam state (for inheritance)
    adam_states: Vec<AdamState>,

    // Configuration
    densify_interval: usize,      // e.g., 100 iterations
    densify_start: usize,         // e.g., 500 iterations
    densify_end: usize,           // e.g., 15000 iterations
    grad_threshold: f32,          // e.g., 0.0002
    size_threshold: f32,          // Clone if < this, Split if > this
    opacity_reset_interval: usize, // e.g., 3000 iterations
    prune_opacity_threshold: f32,  // e.g., 0.005
}

fn densification_step(
    gaussians: &mut Vec<Gaussian>,
    state: &mut DensificationState,
    iteration: usize,
) {
    // 1. Accumulate gradients (every iteration)
    for (i, g) in gaussians.iter().enumerate() {
        state.position_grad_accum[i] += g.position_gradient_magnitude();
        state.grad_count[i] += 1;
    }

    // 2. Periodic densification
    if iteration >= state.densify_start
        && iteration < state.densify_end
        && iteration % state.densify_interval == 0
    {
        for i in 0..gaussians.len() {
            let avg_grad = state.position_grad_accum[i] / state.grad_count[i] as f32;

            if avg_grad > state.grad_threshold {
                let size = gaussians[i].max_scale();

                if size < state.size_threshold {
                    // CLONE: Under-reconstruction
                    clone_gaussian(gaussians, i, &mut state.adam_states);
                } else {
                    // SPLIT: Over-reconstruction
                    split_gaussian(gaussians, i, &mut state.adam_states);
                }
            }
        }

        // Reset accumulators
        state.position_grad_accum.fill(0.0);
        state.grad_count.fill(0);
    }

    // 3. Periodic opacity reset
    if iteration % state.opacity_reset_interval == 0 {
        for g in gaussians.iter_mut() {
            g.opacity = 0.01;
        }
    }

    // 4. Pruning
    gaussians.retain(|g| g.opacity > state.prune_opacity_threshold);
}
```

---

## Metrics to Track in Research

1. **PSNR vs iteration curve** - Does quality dip after densification then recover?
2. **Gaussian count over time** - Is pruning removing enough? Too many?
3. **Clone vs Split ratio** - Which operation is firing more?
4. **Gradient accumulation distribution** - Are gradients concentrated or spread?
5. **Position movement after densification** - Do children move significantly?

---

## Research Questions to Answer

1. **Is gradient accumulation necessary, or can we use a better instantaneous metric?**
   - 2D may be simpler than 3D - maybe accumulated error works?

2. **What's the optimal Clone vs Split size threshold for 2D images?**
   - 3DGS uses 3D extent; we need 2D equivalent

3. **How aggressive should pruning be?**
   - Too aggressive: lose useful Gaussians
   - Too lenient: accumulate dead weight

4. **Should we use state inheritance always, or only for difficult images?**
   - Our data suggests image-dependent benefit

5. **Is periodic opacity reset necessary for 2D?**
   - May be less critical without view-dependent effects

---

## References to Study

1. **3D Gaussian Splatting** (Kerbl et al., 2023) - Section 5: Optimization
2. **BOGausS** (EUSIPCO 2025) - State inheritance, precision-aware updates
3. **Taming 3DGS** - Sparse Adam implementation
4. **AbsGS** (2024) - Absolute gradient-based densification
5. **Mini-Splatting** (2024) - Simplified densification for efficiency

---

## Expected Outcome

This research should take 3-4 hours of focused implementation and testing to either:
- **Breakthrough**: Match or exceed baseline with densification
- **Definitive conclusion**: 2D Gaussian images fundamentally differ from 3DGS and need a different approach

---

## Test Data

- Primary: kodim03.png (768×512) - Medium complexity
- Secondary: kodim05.png - Lower complexity
- Tertiary: kodim08.png - High complexity (inheritance helps here)

## Baseline to Beat

| Image | Baseline (256 Gaussians) |
|-------|--------------------------|
| kodim03 | 21.68 dB |
| kodim05 | 17.08 dB |
| kodim08 | 12.29 dB |

**Success criteria**: Densification from 64→256 achieves within 0.1 dB of baseline.
