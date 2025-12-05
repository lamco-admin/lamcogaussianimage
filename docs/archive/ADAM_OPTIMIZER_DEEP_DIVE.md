# Adam Optimizer Deep Dive: Baseline 1 Implementation

**Last Updated:** 2025-11-15
**Context:** Understanding divergence behavior and optimization dynamics

---

## Adam Algorithm Overview

**Adam = Adaptive Moment Estimation**

Combines two ideas:
1. **Momentum (first moment):** Smooths gradient direction (like a ball rolling downhill)
2. **Adaptive learning rates (second moment):** Different step sizes for each parameter

### The Math

For each parameter θ:

```
m_t = β1 × m_(t-1) + (1 - β1) × g_t          # First moment (momentum)
v_t = β2 × v_(t-1) + (1 - β2) × g_t²         # Second moment (adaptive LR)

m_hat_t = m_t / (1 - β1^t)                    # Bias correction for m
v_hat_t = v_t / (1 - β2^t)                    # Bias correction for v

θ_t = θ_(t-1) - α × m_hat_t / (√v_hat_t + ε)  # Parameter update
```

Where:
- `g_t` = gradient at iteration t
- `β1` = momentum decay rate (default 0.9)
- `β2` = adaptive LR decay rate (default 0.999)
- `α` = learning rate (0.001 in our experiments)
- `ε` = numerical stability constant (1e-8)
- `t` = iteration number

---

## Baseline 1 Implementation

### Hyperparameters Used

```rust
let beta1: f32 = 0.9;      // Momentum decay
let beta2: f32 = 0.999;    // Adaptive LR decay
let epsilon: f32 = 1e-8;   // Numerical stability
let lr: f32 = 0.001;       // Learning rate (CONSTANT - no decay!)
```

These are the **standard Adam defaults** from the original paper (Kingma & Ba, 2015).

### Bias Correction

```rust
let bc1 = 1.0 - beta1.powi(t as i32);  // Bias correction for momentum
let bc2 = 1.0 - beta2.powi(t as i32);  // Bias correction for adaptive LR
```

**Why needed:**
- `m` and `v` are initialized to zero
- Early iterations are biased toward zero
- Bias correction compensates by dividing by (1 - β^t)

**Behavior over iterations:**
- t=1: bc1 = 0.1, bc2 = 0.001 (strong correction)
- t=10: bc1 = 0.651, bc2 = 0.00995
- t=100: bc1 = 0.99997, bc2 = 0.0951
- t=1000: bc1 ≈ 1.0, bc2 = 0.632
- t=10000: bc1 ≈ 1.0, bc2 = 0.99995 ≈ 1.0

**Key observation:** bc2 takes MUCH longer to reach 1.0 than bc1 because β2 = 0.999 is very close to 1.

---

## Parameter-Specific Implementation

Adam maintains **separate moment estimates for each parameter type**.

### Per Gaussian, Adam Tracks:

**For Color (RGB - 3 values):**
```rust
m_color: Vec<Color4<f32>>   // Momentum for R, G, B
v_color: Vec<Color4<f32>>   // Adaptive LR for R, G, B
```

**For Position (XY - 2 values):**
```rust
m_position: Vec<Vector2<f32>>   // Momentum for X, Y
v_position: Vec<Vector2<f32>>   // Adaptive LR for X, Y
```

**For Scale (scale_x, scale_y - 2 values):**
```rust
m_scale: Vec<(f32, f32)>   // Momentum for scale_x, scale_y
v_scale: Vec<(f32, f32)>   // Adaptive LR for scale_x, scale_y
```

**Note:** Rotation is NOT tracked separately (not currently optimized in Baseline 1).

**Total Adam state per Gaussian:**
- 3 momentum + 3 adaptive LR (color) = 6
- 2 momentum + 2 adaptive LR (position) = 4
- 2 momentum + 2 adaptive LR (scale) = 4
- **Total: 14 optimizer state values per Gaussian**

**For N=50:** 50 Gaussians × 14 state values = 700 optimizer state values!

---

## Update Process (Step by Step)

### For Each Gaussian, For Each Parameter:

#### 1. Color Update Example (Red Channel)

```rust
// Compute momentum (first moment)
self.m_color[i].r = beta1 * self.m_color[i].r + (1.0 - beta1) * grads[i].color.r;

// Compute adaptive LR (second moment)
self.v_color[i].r = beta2 * self.v_color[i].r + (1.0 - beta2) * grads[i].color.r.powi(2);

// Apply bias-corrected update
gaussians[i].color.r -= lr * (self.m_color[i].r / bc1) / ((self.v_color[i].r / bc2).sqrt() + epsilon);

// Clamp to valid range
gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
```

**Breakdown:**

**Step 1: Update momentum**
- `m = 0.9 × old_m + 0.1 × gradient`
- Smooths gradient direction over time
- Acts like "velocity" - gradient is "acceleration"

**Step 2: Update adaptive LR**
- `v = 0.999 × old_v + 0.001 × gradient²`
- Accumulates squared gradients
- Tracks "how much this parameter varies"
- Parameters with large gradients get smaller effective LR

**Step 3: Compute update**
- `step = lr × (m / bc1) / (√(v / bc2) + ε)`
- Effective LR = `lr / √v` (adaptive)
- Direction = `m` (smoothed gradient)
- Bias correction applied via bc1, bc2

**Step 4: Apply update**
- `param = param - step`
- Gradient descent (minimize loss)

**Step 5: Clamp**
- Enforce constraints (e.g., color ∈ [0, 1])

#### 2. Position Update (Same Pattern)

```rust
// X coordinate
self.m_position[i].x = beta1 * self.m_position[i].x + (1.0 - beta1) * grads[i].position.x;
self.v_position[i].x = beta2 * self.v_position[i].x + (1.0 - beta2) * grads[i].position.x.powi(2);
gaussians[i].position.x -= lr * (self.m_position[i].x / bc1) / ((self.v_position[i].x / bc2).sqrt() + epsilon);
gaussians[i].position.x = gaussians[i].position.x.clamp(0.0, 1.0);

// Y coordinate (same)
// ...
```

#### 3. Scale Update (Same Pattern)

```rust
// scale_x
self.m_scale[i].0 = beta1 * self.m_scale[i].0 + (1.0 - beta1) * grads[i].scale_x;
self.v_scale[i].0 = beta2 * self.v_scale[i].0 + (1.0 - beta2) * grads[i].scale_x.powi(2);
gaussians[i].shape.scale_x -= lr * (self.m_scale[i].0 / bc1) / ((self.v_scale[i].0 / bc2).sqrt() + epsilon);
gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);

// scale_y (same)
// ...
```

---

## Effective Learning Rate Dynamics

### The Actual Step Size

The **effective learning rate** for each parameter is:

```
α_effective = α / (√v + ε)
```

Where `v` is the accumulated squared gradients.

### What This Means

**Parameters with large gradients:**
- Large v → small effective LR
- Prevents overshooting

**Parameters with small gradients:**
- Small v → large effective LR
- Accelerates progress

**Example:**
- If v = 0.001, effective LR = 0.001 / √0.001 ≈ 0.0316
- If v = 0.1, effective LR = 0.001 / √0.1 ≈ 0.00316
- **10× difference in effective LR!**

### Adaptation Over Time

Early iterations (t < 100):
- v is small (just starting to accumulate)
- Effective LR is large (bc2 correction amplifies this)
- Large steps, fast initial convergence

Mid iterations (t = 1000-1500):
- v has accumulated moderate values
- Effective LR decreases
- Smaller steps, refinement phase

Late iterations (t > 1500):
- v continues growing
- **Problem:** With constant nominal LR, momentum can overwhelm
- This is where divergence starts!

---

## Why Divergence Happens at ~1500 Iterations

### The Momentum Accumulation Problem

**Key insight:** Adam's momentum (`m`) can accumulate over time.

#### Normal Behavior (Converging)

```
Iteration 1:   m = 0.1 × gradient
Iteration 2:   m = 0.9 × 0.1g + 0.1g = 0.19g
Iteration 10:  m ≈ 0.651g (smoothed gradient)
```

As long as gradients point consistently toward a minimum, momentum helps.

#### Divergence Behavior (Near Minimum)

**What happens near a minimum:**
1. Gradients get small (good!)
2. But momentum is still large from previous iterations
3. Update = lr × (large_momentum) / (sqrt(small_v) + ε)
4. Step size can actually **increase** even as gradients decrease
5. Overshoots the minimum
6. Gradients reverse direction
7. Momentum takes time to reverse (β1 = 0.9 means 90% of old momentum remains)
8. Oscillation begins, then diverges

### The Bias Correction Factor

At iteration 1500:
- bc1 ≈ 1.0 (momentum bias correction done)
- bc2 ≈ 0.777 (adaptive LR bias correction still significant)

**Effect:** Dividing v by bc2 < 1 makes effective LR **larger** than it should be.

```
effective_LR = lr × (1 / bc1) / (√(v / bc2) + ε)
             = lr / (√v × √bc2 + ε)
             = lr / (√v × 0.88 + ε)
             ≈ 1.14 × (lr / √v)      # 14% larger than if bc2 = 1!
```

At t=1500, the bias correction is still inflating the effective LR by ~14%.

### Why It Happens at ~1500 Specifically

**Multiple factors converge:**

1. **Approaching minimum:** Gradients getting small
2. **Momentum still large:** From previous iterations
3. **Bias correction still active:** bc2 ≈ 0.777, amplifies effective LR
4. **Constant nominal LR:** No decay to compensate

**Around iteration 1500:**
- Loss has decreased significantly (0.245 → 0.071)
- Gradients are small (near optimum for this LR)
- Momentum from earlier (larger) gradients is still present
- Effective LR is still high due to bc2 correction
- **First overshoot** occurs
- Momentum can't reverse quickly enough (β1 = 0.9)
- Divergence cascade begins

---

## Adam Hyperparameter Effects

### β1 = 0.9 (Momentum Decay)

**Meaning:** Keep 90% of previous momentum, add 10% of current gradient

**Effect on convergence:**
- Higher β1 (e.g., 0.95): More momentum, faster convergence, less stable
- Lower β1 (e.g., 0.8): Less momentum, slower convergence, more stable

**In our experiments:**
- β1 = 0.9 is standard, works well initially
- But: Momentum takes ~10 iterations to decay significantly
- At divergence point, old momentum lingers too long

### β2 = 0.999 (Adaptive LR Decay)

**Meaning:** Keep 99.9% of previous squared gradient accumulation, add 0.1% of current squared gradient

**Effect:**
- Higher β2 → slower adaptation, more stable
- Lower β2 → faster adaptation, can be unstable

**In our experiments:**
- β2 = 0.999 is very close to 1.0
- v changes very slowly (0.1% per iteration)
- Provides stability... until near minimum where it prevents LR from decreasing enough

### ε = 1e-8 (Numerical Stability)

**Purpose:** Prevent division by zero when v is very small

**Effect:**
- Typically negligible (v >> 1e-8 after a few iterations)
- Only matters for parameters with consistently near-zero gradients

### α = 0.001 (Learning Rate) - THE KEY PARAMETER

**Our experiments:** CONSTANT α = 0.001 for all 10,000 iterations

**Standard practice:** Use LR schedule (decay over time)

**Why constant LR causes divergence:**
- Early: LR is appropriate for large gradients
- Mid: LR is appropriate for moderate gradients
- Late: LR is TOO LARGE for small gradients near minimum
- Result: Overshoots, oscillates, diverges

**Common LR schedules:**
1. **Step decay:** Reduce by 10× every N iterations
2. **Exponential decay:** α_t = α_0 × e^(-λt)
3. **Cosine annealing:** α_t = α_min + 0.5(α_max - α_min)(1 + cos(πt/T))
4. **1/sqrt(t) decay:** α_t = α_0 / √t

**None of these are used in Baseline 1.** This is deliberate - we want to study the "simple math" behavior.

---

## Adam State Evolution During Training

### Example: One Gaussian's Color (Red Channel)

#### Iteration 1

```
gradient[i].color.r = 0.05       # Initial gradient
m[i].r = 0.9 × 0 + 0.1 × 0.05 = 0.005
v[i].r = 0.999 × 0 + 0.001 × 0.05² = 0.0000025
bc1 = 1 - 0.9^1 = 0.1
bc2 = 1 - 0.999^1 = 0.001
m_hat = 0.005 / 0.1 = 0.05
v_hat = 0.0000025 / 0.001 = 0.0025
step = 0.001 × 0.05 / (√0.0025 + 1e-8) ≈ 0.001
```

Large step due to bias correction.

#### Iteration 100

```
gradient[i].color.r = 0.02       # Smaller gradient (converging)
m[i].r ≈ 0.018                   # Accumulated momentum
v[i].r ≈ 0.0005                  # Accumulated squared gradients
bc1 ≈ 0.99997 ≈ 1.0
bc2 ≈ 0.0951
m_hat ≈ 0.018
v_hat ≈ 0.0005 / 0.0951 ≈ 0.00526
step = 0.001 × 0.018 / (√0.00526 + 1e-8) ≈ 0.000248
```

Moderate step, still making progress.

#### Iteration 1500 (Near Divergence)

```
gradient[i].color.r = 0.001      # Very small gradient (near optimum)
m[i].r ≈ 0.005                   # Momentum from previous iterations
v[i].r ≈ 0.0008                  # Accumulated squared gradients
bc1 ≈ 1.0
bc2 ≈ 0.777
m_hat ≈ 0.005                    # Still significant momentum!
v_hat ≈ 0.0008 / 0.777 ≈ 0.00103  # Bias correction inflates this
step = 0.001 × 0.005 / (√0.00103 + 1e-8) ≈ 0.000156
```

**Problem:** Gradient is 0.001 but step is 0.000156.
Step size is **156× the current gradient!**
This is because:
1. Momentum from earlier (larger) gradients
2. Bias correction still inflating v_hat

**Result:** Overshoots → gradient flips → momentum takes ~10 iterations to reverse → overshoot again → divergence.

---

## Comparison to Plain Gradient Descent

### Plain SGD (No Momentum, No Adaptive LR)

```
θ_t = θ_(t-1) - α × g_t
```

**Baseline 1 with SGD instead of Adam:**
- Slower convergence (no momentum to accelerate)
- More stable (no momentum accumulation to cause divergence)
- Would likely NOT diverge at iteration 1500

**Why we use Adam:**
- Much faster initial convergence
- Better handling of different parameter scales
- Industry standard for neural networks

### SGD with Momentum (No Adaptive LR)

```
m_t = β × m_(t-1) + g_t
θ_t = θ_(t-1) - α × m_t
```

**Would also diverge** due to momentum accumulation, but:
- Might diverge later (no adaptive LR to amplify effect)
- Easier to fix (just tune β and α)

### Adam's Advantage (When Used Correctly)

Adam works incredibly well with:
- **LR schedule** (decay over time)
- **Gradient clipping** (prevent huge steps)
- **Early stopping** (stop when validation loss increases)
- **Warmup** (start with small LR, increase, then decay)

**Baseline 1 uses NONE of these.** Again, this is deliberate for research.

---

## Why N Affects Divergence Timing

### Observation from Experiments

- N=45: Diverges at ~1400 iterations
- N=50: Diverges at ~1600 iterations
- N=55: Diverges at ~1700 iterations

**Hypothesis:** More Gaussians = more parameters = more "momentum capacity"

### Explanation

**More parameters means:**
1. Gradients distributed across more Gaussians
2. Each individual Gaussian has smaller gradient
3. Momentum accumulates more slowly per Gaussian
4. Takes longer to reach critical momentum level
5. Divergence delayed by ~200-300 iterations per +5 Gaussians

**Alternative explanation:**
- More Gaussians = better representation
- Better representation = smaller gradients earlier (already well-fit)
- Smaller gradients = less momentum accumulation
- Less momentum = delayed divergence

**Data needed to distinguish:**
- Gradient magnitudes over time for different N
- Momentum magnitudes over time for different N
- Can extract from checkpoints!

---

## Adam Optimizer State in Checkpoints

### What's Saved

Every checkpoint includes:
```json
{
  "m_color": [[r, g, b], ...],
  "v_color": [[r, g, b], ...],
  "m_position": [[x, y], ...],
  "v_position": [[x, y], ...],
  "m_scale": [[sx, sy], ...],
  "v_scale": [[sx, sy], ...],
  "learning_rate": 0.001,
  "iteration": 1500
}
```

**This allows:**
1. Resume training with exact optimizer state
2. Analyze momentum/velocity at divergence point
3. Test "what if we reset momentum at iteration 1500?"
4. Test "what if we reduce LR at iteration 1500?"

---

## Experiments Enabled by Understanding Adam

### 1. LR Decay Test

Resume from iteration 1500, reduce LR:
```bash
cargo run --release --example baseline1_checkpoint -- \
  --resume checkpoints/baseline1_n50/checkpoint_001500.json \
  --lr 0.0005 --iterations 5000
```

**Hypothesis:** Halving LR will prevent divergence.

### 2. Momentum Reset Test

Modify checkpoint_001500.json:
- Set all m_* arrays to zero (reset momentum)
- Keep v_* arrays (keep adaptive LR info)

Resume:
```bash
cargo run --release --example baseline1_checkpoint -- \
  --resume checkpoints/baseline1_n50_momentum_reset/checkpoint_001500.json \
  --iterations 5000
```

**Hypothesis:** Resetting momentum will prevent divergence.

### 3. Different β1 Test

Modify code to use β1 = 0.8 (less momentum):
- Would need code change (currently hardcoded)
- Run full 10k iterations
- Compare divergence timing

**Hypothesis:** Lower β1 will delay or prevent divergence.

### 4. Different β2 Test

Modify code to use β2 = 0.99 (more adaptive):
- More responsive to gradient changes
- Less bias correction needed

**Hypothesis:** Lower β2 might help or hurt depending on interaction with momentum.

---

## Summary of Adam Behavior in Baseline 1

### What Works Well

✅ **Fast initial convergence** (iterations 1-1000)
- Momentum accelerates progress
- Adaptive LR handles different parameter scales
- Gets to ~70% improvement quickly

✅ **Consistent across different N**
- Same Adam hyperparameters work for N=20 to N=100
- Robust to parameter count changes

✅ **Stable mid-training** (iterations 500-1500)
- Loss decreases smoothly
- No oscillations or instability
- Predictable behavior

### What Fails

❌ **Divergence at iteration ~1500** (varies by N)
- Momentum accumulation overwhelms small gradients
- Bias correction still active (bc2 ≈ 0.777)
- Constant LR doesn't compensate

❌ **No recovery mechanism**
- Once divergence starts, momentum points wrong direction
- Takes ~10 iterations to reverse (β1 = 0.9)
- By then, overshoot again
- Oscillations grow

❌ **Poor late-stage optimization**
- Needs LR decay for refinement
- Needs momentum damping or reset
- Current config not suitable for >2000 iterations

### How to Fix (Future Work)

**Option 1: LR Schedule** (simplest)
- Cosine annealing from 0.001 to 0.0001 over 10k iterations
- **Estimated effect:** Prevents divergence, continues converging

**Option 2: Gradient Clipping**
- Clip gradient norm to max value (e.g., 1.0)
- **Estimated effect:** Limits momentum growth, may prevent divergence

**Option 3: Early Stopping**
- Monitor validation loss (or just loss)
- Stop when loss stops decreasing
- **Estimated effect:** Stops at iteration ~1500, avoids divergence

**Option 4: Periodic Momentum Reset**
- Reset m_* to zero every N iterations (e.g., every 500)
- **Estimated effect:** Prevents momentum accumulation, slower but more stable

**Option 5: Switch to L-BFGS**
- Second-order optimizer, no momentum issues
- **Estimated effect:** Different convergence behavior, likely no divergence
- **Cost:** Much more expensive per iteration

---

## Adam Optimizer: The Bottom Line

**Adam is powerful but requires care.**

**For Baseline 1:**
- Works excellently for first ~1500 iterations
- Standard hyperparameters (β1=0.9, β2=0.999) are fine
- **Constant LR is the problem**, not Adam itself

**For future work:**
- Add LR schedule (cosine annealing recommended)
- Consider gradient clipping for robustness
- Monitor gradients and momentum magnitudes
- Use checkpoints to experiment with fixes

**Key insight from experiments:**
The divergence at ~1500 iterations is **predictable, understandable, and fixable**. It's not a fundamental flaw in the approach - it's a configuration choice (constant LR) that we deliberately made for research purposes.

---

**Next:** Full results analysis across all N values and multi-image research plan.
