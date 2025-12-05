# Session Handover - December 5, 2025

**Purpose**: Begin new session to implement optimization toolkit
**VM**: High-resource but run ONE experiment at a time (previous multi-experiment runs crashed)

---

## CRITICAL: Copy This Prompt to Start New Session

```
I'm continuing work on the LGI Gaussian image codec. Before doing anything, read these files in order:

1. /home/greg/gaussian-image-projects/lgi-project/GROUND_TRUTH.md (verified facts, fundamental principles)
2. /home/greg/gaussian-image-projects/lgi-project/docs/research/OPTIMIZER_RESEARCH_2025-12-05.md (today's research findings)
3. /home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs/lgi-encoder-v2/src/adam_optimizer.rs (current optimizer)
4. /home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs/lgi-core/src/lbfgs.rs (working L-BFGS we can use)

STRICT STANDARDS:
- NO tiling or 2D geometric segmentation of images. Use semantic LAYERS only.
- We build an optimization TOOLKIT, not crown a single winner. Each method needs its own tuning.
- Run ONE experiment/test at a time. This VM crashed when running multiple simultaneously.
- Do NOT use lgi-encoder-v2/src/optimizer_lbfgs.rs - it's broken. Use lgi-core/src/lbfgs.rs instead.

TASK: Implement three optimizer improvements SEQUENTIALLY (not in parallel):

Task 1: Add per-parameter learning rates to Adam optimizer
- Position LR: 0.0002 (100× smaller than current)
- Color LR: 0.02
- Scale LR: 0.005
- Opacity LR: 0.05
- Add exponential decay for position LR
- Test on ONE Kodak image before proceeding

Task 2: Create Adam→L-BFGS hybrid optimizer
- Use existing lgi-core/src/lbfgs.rs (it works, tested)
- Run Adam for 100 iterations, then switch to L-BFGS for 50 iterations
- Test on ONE Kodak image before proceeding

Task 3: Add Levenberg-Marquardt optimizer
- Add levenberg-marquardt = "0.14" to Cargo.toml
- Implement LeastSquaresProblem trait for Gaussian optimization
- Start with finite differences for Jacobian (correct first, fast later)
- Test on ONE Kodak image

After EACH task: Report results before proceeding to next task.
```

---

## What Was Accomplished This Session

### 1. Project Reorganization

**Root directory cleaned**: Reduced from 55+ markdown files to 4:
- `README.md` - Project overview
- `GROUND_TRUTH.md` - Verified facts only (NEW)
- `NEXT_STEPS.md` - Action plan (NEW)
- `ACKNOWLEDGMENTS.md` - Credits

**Archives created** at `docs/archive/`:
- `sessions/` - All session handovers
- `quantum_exploration/` - Speculative quantum docs
- `phase_experiments/` - Phase 0, 0.5 experiment logs

### 2. Fundamental Principles Documented

**Principle 1: Layers, Never Tiles**
- NO tiling or 2D geometric segmentation
- Use semantic layers (edges, textures, smooth) if needed
- Gaussians are global primitives

**Principle 2: Optimization Toolkit, Not Single Method**
- Adam is what we've learned, NOT the mathematical answer
- Each optimizer needs its own hyperparameter tuning and initialization
- Build multiple well-understood methods

### 3. Research Completed

**Four parallel research tracks** (all completed):

| Track | Finding |
|-------|---------|
| L-BFGS | Working implementation exists in `lgi-core/src/lbfgs.rs`. Hybrid Adam→L-BFGS gives 20-30% speedup. |
| Levenberg-Marquardt | Ideal for our least-squares problem. `levenberg-marquardt` crate available. 2-3× fewer iterations. |
| 3DGS Papers | All use per-parameter learning rates. Position LR should be ~100× smaller than color. |
| Our Code Status | `lgi-core/src/lbfgs.rs` works. `encoder-v2/optimizer_lbfgs.rs` is BROKEN (ignore it). |

### 4. Key Discovery: Missing Per-Parameter Learning Rates

**Current Adam**: Single learning rate (0.01) for ALL parameters

**3DGS best practice** (Kerbl et al.):
| Parameter | Learning Rate |
|-----------|---------------|
| Position | 0.00016 → 0.0000016 (exponential decay) |
| Color | 0.0025 |
| Scale | 0.005 |
| Rotation | 0.001 |
| Opacity | 0.05 |

**This is LOW-HANGING FRUIT** - significant improvement expected from this change alone.

---

## File Reference Guide

### Must-Read Files (in order)

1. **`GROUND_TRUTH.md`** - Verified facts, fundamental principles, what's working
2. **`docs/research/OPTIMIZER_RESEARCH_2025-12-05.md`** - Today's research findings
3. **`packages/lgi-rs/lgi-encoder-v2/src/adam_optimizer.rs`** - Current optimizer to modify
4. **`packages/lgi-rs/lgi-core/src/lbfgs.rs`** - Working L-BFGS (291 lines, tested)

### DO NOT USE

- `packages/lgi-rs/lgi-encoder-v2/src/optimizer_lbfgs.rs` - BROKEN (missing argmin dep, incomplete gradients)

### Key Implementation Files

| File | Purpose | Lines |
|------|---------|-------|
| `lgi-encoder-v2/src/lib.rs` | Main encoder, all encode methods | 1,773 |
| `lgi-encoder-v2/src/adam_optimizer.rs` | Adam implementation to modify | ~300 |
| `lgi-core/src/lbfgs.rs` | Working L-BFGS to integrate | 291 |
| `lgi-math/src/gaussian.rs` | Gaussian2D struct definition | 293 |
| `lgi-math/src/parameterization.rs` | Euler, Cholesky, etc. | 403 |

### Test Commands

```bash
cd /home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs

# Build
cargo build --release

# Run L-BFGS test (verifies it works)
cargo run --example lbfgs_test

# Run isotropic edge test on one image
cargo run --release --example test_isotropic_edges -- --image /path/to/kodak/kodim01.png

# Run comprehensive benchmark (SLOW - one image at a time)
cargo run --release --example comprehensive_benchmark
```

---

## Implementation Details for Next Session

### Task 1: Per-Parameter Learning Rates

**File to modify**: `lgi-encoder-v2/src/adam_optimizer.rs`

**Current structure** (around line 15-30):
```rust
pub struct AdamOptimizer {
    pub learning_rate: f32,  // Single LR - CHANGE THIS
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub max_iterations: usize,
    // ...
}
```

**Change to**:
```rust
pub struct AdamOptimizer {
    pub lr_position: f32,      // 0.0002
    pub lr_color: f32,         // 0.02
    pub lr_scale: f32,         // 0.005
    pub lr_opacity: f32,       // 0.05
    pub lr_rotation: f32,      // 0.001
    pub position_lr_decay: f32, // 0.99 per iteration
    // ... rest unchanged
}
```

**Update the optimization loop** to use different LRs for different parameters.

### Task 2: Adam→L-BFGS Hybrid

**New file**: `lgi-encoder-v2/src/hybrid_optimizer.rs`

**Structure**:
```rust
use lgi_core::lbfgs::LBFGS;

pub struct HybridOptimizer {
    pub adam_iterations: usize,   // 100
    pub lbfgs_iterations: usize,  // 50
}

impl HybridOptimizer {
    pub fn optimize(&self, gaussians: &mut [Gaussian2D], target: &ImageBuffer) {
        // Phase 1: Adam warm-start
        let mut adam = AdamOptimizer::default();
        adam.max_iterations = self.adam_iterations;
        adam.optimize(gaussians, target);

        // Phase 2: L-BFGS refinement
        // Convert Gaussians to flat params, run L-BFGS, convert back
    }
}
```

**Key**: The L-BFGS in lgi-core takes closures for `f` and `grad_f`. You'll need to create these from the Gaussian rendering pipeline.

### Task 3: Levenberg-Marquardt

**Add to Cargo.toml**:
```toml
levenberg-marquardt = "0.14"
nalgebra = "0.32"
```

**New file**: `lgi-encoder-v2/src/lm_optimizer.rs`

**Implement `LeastSquaresProblem` trait**:
```rust
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};

struct GaussianLeastSquares {
    gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
    target: ImageBuffer<f32>,
    width: u32,
    height: u32,
}

impl LeastSquaresProblem<f32, Dyn, Dyn> for GaussianLeastSquares {
    type ParameterStorage = nalgebra::DVector<f32>;
    type ResidualStorage = nalgebra::DVector<f32>;
    type JacobianStorage = nalgebra::DMatrix<f32>;

    fn set_params(&mut self, params: &Self::ParameterStorage) { ... }
    fn params(&self) -> Self::ParameterStorage { ... }
    fn residuals(&self) -> Option<Self::ResidualStorage> { ... }
    fn jacobian(&self) -> Option<Self::JacobianStorage> { ... }
}
```

**Start with finite differences for Jacobian** (correct first, optimize later).

---

## Resource Constraints

### VM Specifications
- High memory (experiments with >1500 samples crashed)
- Multi-core CPU
- GPU available (RTX series)

### CRITICAL: Run ONE Test at a Time

**Previous failures**:
- Running Q1 + Q2 quantum experiments simultaneously: Both crashed
- Running 1,483 samples on 76GB RAM: OOM
- Multiple encodings in parallel: Resource exhaustion

**Safe practice**:
1. Run ONE encoding test
2. Wait for completion
3. Check results
4. Then run next test

---

## What's NOT Done Yet

1. **Isotropic edges not validated on full Kodak** - Only tested on 5 images (+1.87 dB)
2. **Compression ratio not measured** - No comparison vs JPEG/WebP
3. **GPU gradient computation** - Would give 100-1500× speedup but not implemented
4. **Video codec (LGIV)** - Spec exists, no implementation

---

## Git Status (Uncommitted Work)

**Modified files**:
```
M packages/lgi-rs/lgi-core/src/structure_tensor.rs
M packages/lgi-rs/lgi-encoder-v2/src/adam_optimizer.rs
M packages/lgi-rs/lgi-encoder-v2/src/lib.rs
```

**Untracked important files**:
```
packages/lgi-rs/lgi-encoder-v2/src/gaussian_logger.rs
packages/lgi-rs/lgi-encoder-v2/examples/test_isotropic_edges.rs
packages/lgi-rs/BREAKTHROUGH_ISOTROPIC_EDGES.md
GROUND_TRUTH.md
NEXT_STEPS.md
docs/research/OPTIMIZER_RESEARCH_2025-12-05.md
SESSION_HANDOVER_2025-12-05.md
```

**Recommend committing** before starting new work:
```bash
git add GROUND_TRUTH.md NEXT_STEPS.md docs/research/OPTIMIZER_RESEARCH_2025-12-05.md
git add packages/lgi-rs/lgi-encoder-v2/src/gaussian_logger.rs
git add packages/lgi-rs/BREAKTHROUGH_ISOTROPIC_EDGES.md
git commit -m "Reorganize project, document optimizer research findings"
```

---

## Success Criteria for Next Session

### Task 1 Complete When:
- [ ] Adam has per-parameter learning rates
- [ ] Position LR decays exponentially
- [ ] Tested on kodim01.png - reports PSNR
- [ ] No crashes or errors

### Task 2 Complete When:
- [ ] HybridOptimizer exists and compiles
- [ ] Successfully runs Adam (100 iter) → L-BFGS (50 iter)
- [ ] Tested on kodim01.png - reports PSNR
- [ ] Compare: hybrid vs Adam-only (same total iterations)

### Task 3 Complete When:
- [ ] levenberg-marquardt crate added
- [ ] LeastSquaresProblem implemented
- [ ] Tested on kodim01.png - reports PSNR
- [ ] Compare: LM vs Adam (same iteration budget)

---

## Contact Points in Code

### Where gradients are computed (for L-BFGS/LM integration):

**Adam gradient computation**: `adam_optimizer.rs` around line 150-250
- Computes ∂loss/∂position, ∂loss/∂color, ∂loss/∂scale
- This logic can be reused for L-BFGS `grad_f` closure

**Renderer**: `renderer_v2.rs`
- `render()` method produces the image from Gaussians
- Needed for computing residuals in LM

### Where to add new optimizers:

**lib.rs exports**: Around line 16-32
```rust
pub mod adam_optimizer;
// Add:
pub mod hybrid_optimizer;
pub mod lm_optimizer;
```

**New encode methods**: Add after existing `encode_error_driven_adam()` (line ~500)
```rust
pub fn encode_error_driven_hybrid(&self, ...) -> Vec<Gaussian2D> { ... }
pub fn encode_error_driven_lm(&self, ...) -> Vec<Gaussian2D> { ... }
```

---

---

## ADDENDUM: Later Session Work (Context Loss Recovery)

**CRITICAL UPDATES:**

### Track 1 is DEAD/MOTHBALLED
Do NOT focus on the 150 techniques. User explicitly confirmed this is no longer active.

### Focus Areas (User-Confirmed)
1. **Option A alternatives**: Gabor functions, separable 1D×1D Gaussians, truncation radius
2. **Optimizer tuning**: Figure out how to tune EACH optimizer properly
3. **Systematic comparison**: Same init, same images, different optimizers
4. **Full Kodak testing**: Only AFTER optimizers are tuned

### Test Results Infrastructure Created

Location: `packages/lgi-rs/lgi-encoder-v2/test-results/`

**Files:**
- `README.md` - Complete framework documentation
- `2025-12-05/` - Today's experiment results
- `rendered-images/` - Visual outputs

**Rust API** (`src/test_results.rs`):
```rust
let mut result = TestResult::new("experiment_name", "/path/to/image.png");
result.save(RESULTS_DIR)?;
save_rendered_image(&image, RESULTS_DIR, "name", "suffix")?;
```

### Experiments Completed This Session

| File | Finding |
|------|---------|
| `per_param_lr_comparison.json` | Per-param LR +4.79 dB vs single LR |
| `multi_kodak_per_param_lr.json` | +2.25 dB avg, huge variance by content type |
| `placement_strategy_comparison.json` | PPM/Laplacian **8 dB WORSE** than uniform grid |

### IMPORTANT: Results Are Evidence, Not Conclusions

The placement experiment showing "smart" placement 8 dB worse is **evidence about tool behavior with current optimizer**, NOT a conclusion that PPM placement is universally bad.

### Alternative Techniques to Implement (Option A)

From `docs/archive/EXISTING_TECHNIQUES_RESEARCH.md`:

1. **Truncation radius** - Standard practice, 100× speedup claimed
2. **Separable 1D×1D** - Product of 1D Gaussians, 8× speedup
3. **Gabor functions** - Gaussian × sinusoid, better for edges/textures

### What NOT To Do (User Feedback)
- DO NOT poll background processes (forbidden)
- DO NOT draw conclusions from limited experiments
- DO NOT run full Kodak until optimizers are properly tuned
- DO NOT focus on Track 1 (150 techniques)

*End of handover. Next session should begin with the prompt at the top of this document.*
