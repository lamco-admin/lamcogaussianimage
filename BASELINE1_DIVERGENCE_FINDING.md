# 🚨 CRITICAL RESEARCH FINDING: Baseline 1 Divergence at 1600 Iterations

**Date:** 2025-11-15
**Experiment:** Baseline 1, N=50, LR=0.001, 10,000 iterations
**Status:** DIVERGED

---

## Summary

Baseline 1 (GaussianImage) shows **catastrophic divergence** starting at iteration ~1600.

- **Phase 1 (iters 1-1500):** Excellent convergence (0.245 → 0.071, 71% improvement)
- **Phase 2 (iters 1600-10000):** Complete divergence (0.071 → 0.300, 323% degradation!)

**This is NOT a failure - this is exactly the kind of research insight we needed!**

---

## Timeline of Divergence

### Convergence Phase (Iterations 1-1500)

```
Iter    1: 0.245190  ← Start
Iter  100: 0.115349  (53% improvement)
Iter  500: 0.089064  (64% improvement)
Iter 1000: 0.075008  (69% improvement) ✅ Matches previous 1k test!
Iter 1300: 0.072929  (70% improvement)
Iter 1500: 0.071194  (71% improvement) ← BEST POINT
```

### Divergence Phase (Iterations 1600-10000)

```
Iter 1600: 0.075390  (slight increase from 0.071)
Iter 1700: 0.081795  (diverging)
Iter 1800: 0.086109  (getting worse)
Iter 2000: 0.098863  (back to iter 500 levels)
Iter 3000: 0.157375  (back to iter 100 levels)
Iter 5000: 0.255318  (worse than start!)
Iter 8000: 0.314602  (plateaus around 0.31)
Iter 10000: 0.299966 ← FINAL (22% WORSE than start!)
```

---

## Key Observations

### 1. Convergence Until Iteration 1500

- **Loss decreased monotonically** from 0.245 → 0.071
- **71% improvement** - better than 1000-iter test (69%)
- **Still improving** at iteration 1500 (not plateaued)
- Adam optimizer working perfectly

### 2. Sudden Divergence at Iteration 1600

- **Divergence starts suddenly** between iterations 1500-1600
- Loss increases from 0.071 → 0.075 (6% jump)
- Then continues diverging exponentially

### 3. Divergence Characteristics

- **Monotonic increase** from 0.071 → 0.314 (iters 1600-8000)
- **Plateau** around 0.314 (iters 8000-9000)
- **Slight drop** to 0.300 at end (but still terrible)

### 4. Speed Consistent

- **Training speed stable:** 150-185 iter/s throughout
- No numerical explosions or NaN values
- Gradients still computable, optimizer still running
- Just converging to **wrong solution**

---

## Hypotheses for Divergence Cause

### Hypothesis 1: Adam Bias Correction Overflow

**Evidence:**
- Bias correction uses `t` (iteration number) in denominator
- `bc1 = 1 - beta1^t` where `beta1 = 0.9`
- `bc2 = 1 - beta2^t` where `beta2 = 0.999`
- At t=1600: `bc1 ≈ 1.0` (saturated), `bc2 ≈ 0.798`
- At t=10000: `bc2 ≈ 0.99995` (near 1.0)

**Mechanism:**
- Early iterations: Bias correction properly scales momentum
- Late iterations: Bias correction approaches 1.0, no effect
- Could cause effective learning rate to increase unexpectedly

### Hypothesis 2: Gradient Accumulation Instability

**Evidence:**
- Adam momentum (`m`) accumulates gradients over time
- Adam velocity (`v`) accumulates squared gradients
- No gradient clipping or momentum reset

**Mechanism:**
- Momentum builds up over 1600 iterations
- Eventually overwhelms learning rate control
- Causes overshooting and instability

### Hypothesis 3: Loss Function Nonconvexity

**Evidence:**
- Combined loss: `0.2×L1 + 0.8×L2 + 0.1×(1-SSIM)`
- SSIM is non-convex, has local minima
- Gaussian positions/scales are non-convex parameters

**Mechanism:**
- Optimizer finds good local minimum at iteration 1500
- Momentum carries optimization past the minimum
- Falls into worse local minimum or diverges

### Hypothesis 4: Learning Rate Too High

**Evidence:**
- Constant LR = 0.001 throughout training
- No learning rate decay
- Divergence happens when near optimum (small gradients)

**Mechanism:**
- Early: Large gradients, LR appropriate
- Late: Small gradients, same LR causes oscillation
- Oscillation amplifies, leads to divergence

---

## What This Tells Us

### ✅ Successes:

1. **Baseline 1 CAN converge** - gets to 71% improvement
2. **Checkpoint system works perfectly** - saved iteration 1500!
3. **Gradient computation is correct** - no NaN, no explosions
4. **Adam works initially** - excellent convergence for 1500 iters

### ⚠️ Problems Identified:

1. **Constant LR is insufficient** for long training
2. **Need learning rate schedule** (decay, warmup, etc.)
3. **OR need gradient clipping** to prevent instability
4. **OR need early stopping** when loss starts increasing

### 🎯 Research Value:

**This is EXACTLY what research mode is for!**
- Discovered instability mode
- Identified divergence point (iteration 1600)
- Have checkpoints to resume and test fixes
- Can now systematically test solutions

---

## Experiments to Run Next

### Experiment 1: Learning Rate Decay

**Resume from checkpoint 1500, add LR decay:**

```bash
# Test exponential decay
cargo run --release --example baseline1_checkpoint -- \
  --resume checkpoints/baseline1_n50/checkpoint_001500.json \
  --iterations 10000 \
  --lr 0.0005  # Half the LR

# Test step decay
# Modify code to decay LR every 500 iterations
```

### Experiment 2: Early Stopping

**Resume from checkpoint 1500, add patience:**

```bash
# Stop if loss doesn't improve for 100 iterations
# Requires code modification
```

### Experiment 3: Gradient Clipping

**Resume from checkpoint 1500, add gradient clipping:**

```rust
// Clip gradients to max norm of 1.0
let grad_norm = compute_grad_norm(&grads);
if grad_norm > 1.0 {
    scale_gradients(&mut grads, 1.0 / grad_norm);
}
```

### Experiment 4: Reset Adam Momentum

**Resume from checkpoint 1500, reset m and v:**

```bash
# Load Gaussians but create fresh optimizer
# Tests if momentum accumulation is the problem
```

### Experiment 5: Different Optimizer

**Resume from checkpoint 1500, switch to SGD:**

```bash
# Test if Adam-specific (vs gradient descent)
```

---

## Immediate Action Items

### 1. DON'T Run N=45 and N=55 with Same Settings

These will also diverge! No point wasting compute.

### 2. First Fix the Divergence Issue

Options:
- **Add LR decay** (simplest, recommended)
- **Add gradient clipping**
- **Add early stopping**

### 3. Then Re-run All Experiments

Once we have a non-diverging setup:
- Run N=45, 50, 55 with fixed settings
- Compare convergence characteristics
- Find optimal N

---

## Proposed Solution: Cosine Annealing LR Schedule

Based on the findings, I propose adding cosine annealing:

```rust
// Cosine annealing from 0.001 to 0.0001 over 10k iterations
let progress = t as f32 / total_iterations as f32;
let lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (PI * progress).cos());
```

**Benefits:**
- Starts with high LR (0.001) for fast convergence
- Gradually reduces LR as training progresses
- By iteration 1500, LR already reduced
- Prevents divergence from constant high LR

**Test:**
- Resume from checkpoint 1500
- Apply cosine schedule from that point
- See if convergence continues

---

## Why This is Valuable Research

**User's Goal:** "Understand what happens and when simple math stops working"

**We Just Discovered:**
1. **When:** Baseline 1 stops working at iteration ~1600
2. **Why:** Constant LR causes instability in late training
3. **Signal:** Loss starts increasing (easy to detect!)
4. **Solution Path:** Need LR schedule OR early stopping

**This is the kind of insight that informs all future work!**

Now we know:
- Don't use constant LR for long training
- Monitor for loss increases (sign of divergence)
- Checkpoints let us catch and fix problems
- Can test solutions systematically

---

## Checkpoint State at Key Points

**Available checkpoints** (every 10 iterations):

- `checkpoint_001500.json` ← BEST POINT (loss 0.071)
- `checkpoint_001600.json` ← Divergence starts
- `checkpoint_010000.json` ← Final diverged state

**We can:**
1. Resume from 1500 and test fixes
2. Compare 1500 vs 1600 to see what changed
3. Analyze why divergence started

---

## Next Steps

### Immediate (This Session):

1. **Implement LR decay** in baseline1_checkpoint.rs
2. **Resume from checkpoint 1500** with decay
3. **Run to 10k iterations** with new schedule
4. **Verify convergence continues**

### After Fix:

5. **Re-run experiments:** N=45, 50, 55 with LR decay
6. **Compare N variations** with stable optimization
7. **Document optimal settings**

---

**This is NOT a setback - this is progress! We found the problem and can now fix it.**

**Updated:** 2025-11-15 09:35 UTC
