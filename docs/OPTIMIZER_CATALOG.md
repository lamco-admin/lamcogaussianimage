# OPTIMIZER CATALOG - Gaussian Image Codec
## Complete Parameter Update Algorithms Inventory

**Document Version:** 2.0
**Date:** 2025-12-07
**Repository:** lgi-project
**Code Base:** packages/lgi-rs/
**Total Lines:** 4,671 lines of optimizer code

---

## TABLE OF CONTENTS

1. [Overview](#overview)
2. [CPU Optimizers (6 implementations)](#cpu-optimizers)
3. [GPU Optimizers (3 implementations)](#gpu-optimizers)
4. [Loss Functions](#loss-functions)
5. [Gradient Computation](#gradient-computation)
6. [Learning Rate Schedules](#learning-rate-schedules)
7. [File Location Index](#file-location-index)

---

## OVERVIEW

This catalog documents all parameter optimization algorithms implemented for Gaussian splatting image encoding. These algorithms update Gaussian parameters (position, scale, rotation, color, opacity) to minimize the difference between rendered and target images.

### Architecture Layers

```
ENCODER (entry point)
    ↓
OPTIMIZER (this catalog)
    ├─ Forward: calls RENDERER to get pixels
    ├─ Loss: computes error vs target
    ├─ Backward: computes gradients
    └─ Update: modifies parameters
```

### Categories

**CPU Optimizers (lgi-encoder-v2/src/):**
- 6 implementations
- 2,003 total lines
- Suitable for research and CPU-only systems

**GPU Optimizers (lgi-gpu/src/):**
- 3 implementations
- 2,668 total lines
- Includes complex JtJ strategy system for L-M
- 100-1000× faster gradient computation

---

## CPU OPTIMIZERS

### 1. ADAM OPTIMIZER

**File:** `lgi-encoder-v2/src/adam_optimizer.rs`
**Lines:** 487
**Algorithm:** Adaptive Moment Estimation (Kingma & Ba, 2014)

#### Description
First-order gradient descent with adaptive per-parameter learning rates using exponential moving averages of gradients (momentum) and squared gradients (RMSprop).

#### Key Components

**Struct:** `AdamOptimizer` (Lines 30-49)
- Momentum buffers: `m_position`, `m_color`, `m_scale`, `m_opacity`
- RMSprop buffers: `v_position`, `v_color`, `v_scale`, `v_opacity`
- Per-parameter learning rates (Lines 9-28)

**Hyperparameters:**
```rust
pub struct LearningRates {
    pub position: f32,        // Default: 0.0002
    pub color: f32,           // Default: 0.02
    pub scale: f32,           // Default: 0.005
    pub opacity: f32,         // Default: 0.05
    pub position_final: f32,  // Default: 0.00002
}
```

**Adam Parameters:**
- β₁ (beta1): 0.9 - Momentum decay
- β₂ (beta2): 0.999 - RMSprop decay
- ε (epsilon): 1e-8 - Numerical stability

#### Methods

**Core Functions:**
- `new()` (Line 75) - Create optimizer with per-parameter LRs
- `optimize()` (Line 112) - Main optimization loop
- `compute_gradients()` (Line 240) - Analytical gradient computation
- `apply_adam_updates()` (Line 365) - Parameter update with bias correction

#### Update Equations

For parameter θ:
```
m_t = β₁·m_{t-1} + (1-β₁)·∇θ
v_t = β₂·v_{t-1} + (1-β₂)·(∇θ)²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)
```

#### Features
- Per-parameter learning rates (position, color, scale, opacity)
- Learning rate decay for position parameter
- Bias correction for momentum and RMSprop
- Gradient clipping support
- Integration with `GaussianLogger` for visualization

---

### 2. OPTIMIZER V2 (Gradient Descent)

**File:** `lgi-encoder-v2/src/optimizer_v2.rs`
**Lines:** 379
**Algorithm:** Vanilla gradient descent with optional MS-SSIM loss

#### Description
Simple gradient descent optimizer serving as baseline and validation tool. Supports CPU and GPU rendering, MS-SSIM perceptual loss, and edge-weighted gradients.

#### Key Components

**Struct:** `OptimizerV2` (Lines 33-48)
- Per-parameter learning rates
- Optional GPU renderer for 454× speedup
- MS-SSIM loss module
- Edge-weighted loss module
- Cached structure tensor field

**Learning Rates:**
- Position: 0.10 (default)
- Scale: 0.10
- Color: 0.6
- Rotation: 0.02

#### Methods

**Constructors:**
- `new_with_gpu()` (Line 65) - GPU-accelerated rendering
- `new_with_ms_ssim()` (Line 80) - MS-SSIM perceptual loss
- `new_with_gpu_and_ms_ssim()` (Line 86) - Both GPU + MS-SSIM

**Core Functions:**
- `optimize()` (Line 94) - Main optimization loop
- `compute_gradients_cpu()` (Line 155) - CPU gradient computation
- `compute_gradients_gpu()` (Line 230) - GPU gradient computation (454× faster)

#### Features
- Optional GPU acceleration (wgpu/Vulkan)
- Optional MS-SSIM loss instead of L2
- Optional edge-weighted gradients (3× weight at edges)
- Analytical gradient computation for all parameters
- Structure tensor integration for edge detection

---

### 3. OPTIMIZER V3 (Perceptual)

**File:** `lgi-encoder-v2/src/optimizer_v3_perceptual.rs`
**Lines:** 201
**Algorithm:** MS-SSIM perceptual loss with edge weighting

#### Description
Specialized optimizer using multi-scale structural similarity (MS-SSIM) as the primary loss function, with 3× gradient weighting at detected edges.

#### Key Components

**Struct:** `OptimizerV3` (line information in file)
- MS-SSIM loss module
- Edge-weighted loss computation
- Structure tensor for edge detection

**Loss Combination:**
- α = 0.84 (MS-SSIM weight)
- (1-α) = 0.16 (L2 MSE weight)
- Combined: `L = α·(1-MS-SSIM) + (1-α)·MSE`

#### Methods
- Perceptual loss computation
- Edge-weighted gradient scaling
- Multi-scale analysis integration

---

### 4. L-BFGS OPTIMIZER (CPU)

**File:** `lgi-encoder-v2/src/optimizer_lbfgs.rs`
**Lines:** 281
**Algorithm:** Limited-memory Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method

#### Description
Second-order optimization using L-BFGS from the `lgi-core/src/lbfgs.rs` module. Approximates the Hessian using gradient history without storing the full matrix.

#### Key Components

**Struct:** `LBFGSOptimizer` (Line 165)
- History size (m): number of gradient/parameter pairs stored
- Line search parameters
- Convergence tolerance

**Configuration:**
- Default history: 7-10 iterations
- Wolfe conditions for line search
- Gradient norm tolerance

#### Methods
- `optimize()` (Line 200) - Main L-BFGS loop using lgi-core implementation

#### Characteristics
- Memory: O(m·n) where m=history, n=parameters
- Convergence: Super-linear for well-conditioned problems
- No explicit Hessian computation

---

### 5. LEVENBERG-MARQUARDT (CPU)

**File:** `lgi-encoder-v2/src/lm_optimizer.rs`
**Lines:** 295
**Algorithm:** Levenberg-Marquardt damped least-squares

#### Description
Implements L-M algorithm using the `levenberg_marquardt` Rust crate. Interpolates between gradient descent (high λ) and Gauss-Newton (low λ) for nonlinear least-squares problems.

#### Key Components

**Struct:** `LMOptimizer` (Line 38)
- Configuration for iterations, tolerance, damping
- Finite difference or analytical Jacobian option

**Configuration:** `LMConfig` (Lines 17-30)
```rust
pub struct LMConfig {
    pub max_iterations: usize,        // Default: 100
    pub tolerance: f64,               // Default: 1e-8
    pub initial_lambda: f64,          // Default: 0.01
    pub use_finite_diff: bool,        // Default: true
    pub fd_epsilon: f64,              // Default: 1e-6
}
```

#### Methods
- `optimize()` (Line 56) - L-M optimization using nalgebra DVector/DMatrix
- Implements `LeastSquaresProblem` trait for levenberg_marquardt crate

#### Parameter Flattening
Maps Gaussians to parameter vector (7 params per Gaussian):
- x, y (position)
- r, g, b (color)
- scale_x, scale_y

#### Notes from Code Comments
- "Best used AFTER Adam warm-start (100+ iterations), not from scratch"
- Research reference: 3DGS-LM (ICCV 2025) shows 20-30% speedup over Adam for refinement

---

### 6. HYBRID OPTIMIZER (Adam → L-BFGS)

**File:** `lgi-encoder-v2/src/hybrid_optimizer.rs`
**Lines:** 360
**Algorithm:** Two-phase optimization combining Adam and L-BFGS

#### Description
Runs Adam for warm-start, then switches to L-BFGS for final refinement. Based on research showing L-BFGS from scratch provides no benefit over Adam alone.

#### Key Components

**Struct:** `HybridOptimizer` (Line 47)

**Configuration:** `HybridConfig` (Lines 14-26)
```rust
pub struct HybridConfig {
    pub adam_iterations: usize,      // Default: 30
    pub lbfgs_iterations: usize,     // Default: 20
    pub lbfgs_history: usize,        // Default: 10
    pub adam_lr: LearningRates,      // Per-parameter LRs
}
```

#### Methods
- `optimize()` (Line 58) - Two-phase optimization
  - Phase 1: Adam warm-start (lines 67-77)
  - Phase 2: L-BFGS refinement (lines 79+)

#### Notes from Code Comments
- "IMPORTANT: Uses analytical gradients (not finite differences) for L-BFGS phase"
- "Finite differences would require O(2×7×n_gaussians) renders per gradient"
- Research reference: 3DGS-LM (ICCV 2025)

---

## GPU OPTIMIZERS

### 7. GPU ADAM OPTIMIZER

**File:** `lgi-gpu/src/adam_optimizer.rs`
**Lines:** 907
**Algorithm:** GPU-accelerated Adam with compute shaders

#### Description
Complete GPU implementation of Adam optimizer using wgpu compute shaders. Computes gradients and applies parameter updates entirely on GPU.

#### Key Components

**Struct:** `GpuAdamOptimizer` (line in file)
- GPU device and queue references
- Compute pipeline for Adam updates
- Buffer management for parameters and gradients
- Per-parameter learning rates

**Learning Rates:** `GpuLearningRates` (structure in file)
- Same defaults as CPU Adam
- Transferred to GPU as uniform buffer

#### Compute Shader
**File:** `lgi-gpu/src/shaders/adam_update.wgsl`
- Per-Gaussian workgroup processing
- In-place momentum and RMSprop updates
- Bias correction
- Parameter update

#### Methods
- GPU gradient computation via `GpuGradientComputer`
- GPU parameter updates via compute shader
- Async GPU operations with sync points

#### Performance
- 100-1000× faster gradient computation than CPU
- Full pipeline on GPU reduces CPU-GPU transfers

---

### 8. GPU L-BFGS OPTIMIZER

**File:** `lgi-gpu/src/lbfgs_optimizer.rs`
**Lines:** 1,038
**Algorithm:** GPU-accelerated L-BFGS

#### Description
Hybrid GPU/CPU L-BFGS implementation. Gradients computed on GPU, line search and two-loop recursion on CPU.

#### Key Components

**Struct:** `GpuLbfgsOptimizer` (line in file)
- History storage for gradient/parameter pairs
- Line search configuration
- GPU gradient computer

**Configuration:** `LbfgsConfig`
- History size m (default: 10)
- Line search parameters
- Convergence tolerance

#### Methods
- GPU gradient computation
- CPU two-loop recursion for search direction
- Line search with Wolfe conditions
- History management (FIFO for gradient/step pairs)

---

### 9. GPU LEVENBERG-MARQUARDT OPTIMIZER

**File:** `lgi-gpu/src/lm_optimizer.rs`
**Lines:** 723
**Algorithm:** GPU-accelerated L-M with JtJ strategies

#### Description
Advanced L-M implementation computing J^T J (Gauss-Newton Hessian) and J^T r (gradient) on GPU, solving the linear system on CPU.

#### Architecture

**Hybrid GPU/CPU Design:**
- **GPU:** Computes J^T J and J^T r in parallel
- **CPU:** Solves (J^T J + λI)δ = -J^T r
- **GPU:** Applies parameter updates

#### Why This Approach?

Full Jacobian J has shape (n_residuals × n_params):
- For 768×512 image, 256 Gaussians: 1.18M × 1.79K = 2.1 billion elements (8GB)
- Too large to store explicitly

Instead compute J^T J and J^T r directly:
- J^T J: (n_params × n_params) = 1792 × 1792 = 12MB
- J^T r: (n_params,) = 1792 floats = 7KB

#### Key Components

**Struct:** `GpuLMOptimizer` (line in file)

**Configuration:** `LMConfig` (lines in file)
```rust
pub struct LMConfig {
    pub max_iterations: usize,    // Default: 50
    pub tolerance: f32,           // Default: 1e-6
    pub initial_lambda: f32,      // Default: 1e-3
    pub lambda_up: f32,           // Default: 10.0
    pub lambda_down: f32,         // Default: 0.1
    pub lambda_min: f32,          // Default: 1e-7
}
```

#### JtJ Strategy System

**File:** `lgi-gpu/src/jtj_strategy.rs`
**Lines:** (part of GPU ecosystem)

Multiple strategies for computing J^T J:

1. **BlockDiagonalJtJ** - Only within-Gaussian blocks (current default)
2. **FullDenseJtJ** - All interactions (not implemented)
3. **SparseJtJ** - Spatial locality (not implemented)
4. **HierarchicalJtJ** - Multi-level (not implemented)
5. **MatrixFreeJtJ** - Jacobian-vector products only (not implemented)

**Trait:** `JtJStrategy`
- `compute_jtj()` - Strategy-specific computation
- `analyze()` - Complexity analysis
- `select()` - Auto-selection based on problem size

**Current Limitations (from code comments):**
- Block-diagonal approximation ignores cross-Gaussian interactions
- Causes poor Hessian approximation when Gaussians overlap significantly
- Lambda stays high (1e6-1e7) for regularization

#### Methods
- GPU J^T J computation via compute shader
- GPU J^T r computation
- CPU linear system solve
- GPU parameter update

#### Parameters Per Gaussian
10 parameters optimized:
- x, y (position)
- scale_x, scale_y
- rotation
- r, g, b (color)
- opacity

---

## LOSS FUNCTIONS

### Implemented Loss Functions

#### 1. L2 MSE (Mean Squared Error)
**Location:** Used by all optimizers
**Formula:** `L = Σ(I - Î)² / n_pixels`
**Gradient:** `∇L = 2(Î - I) / n_pixels`

#### 2. MS-SSIM Loss
**File:** `lgi-core/src/ms_ssim_loss.rs`
**Algorithm:** Multi-scale Structural Similarity (Wang et al. 2003)
**Scales:** 5 levels (default)
**Formula:** `L = α·(1-MS-SSIM) + (1-α)·MSE` where α=0.84

**Integration:**
- OptimizerV2 with `use_ms_ssim = true`
- OptimizerV3 (perceptual)

#### 3. Edge-Weighted Loss
**File:** `lgi-core/src/edge_weighted_loss.rs`
**Formula:** `L = Σ w(x,y)·(I - Î)²` where w = 1 + 2·coherence
**Effect:** 3× gradient weight at detected edges

**Integration:**
- OptimizerV2 with `use_edge_weighted = true`
- OptimizerV3 (perceptual)

---

## GRADIENT COMPUTATION

### Analytical Gradients

All CPU optimizers compute analytical gradients with respect to:
- Position (x, y)
- Scale (σ_x, σ_y)
- Rotation (θ)
- Color (R, G, B, A)
- Opacity (α)

### Gradient Methods

#### 1. CPU Analytical (OptimizerV2, Adam)
**Location:** `adam_optimizer.rs` lines 240+
**Method:** Chain rule through rendering equation
**Speed:** Baseline

#### 2. GPU Analytical (All GPU optimizers)
**Shader:** `lgi-gpu/src/shaders/gradient_compute.wgsl`
**Workgroup:** 256 threads
**Features:**
- Per-pixel gradient accumulation
- 3.5σ spatial culling
- Parallel reduction
**Speed:** 100-1000× faster than CPU

#### 3. MS-SSIM Analytical
**File:** `lgi-core/src/ms_ssim_gradients.rs`
**Method:** Analytical derivatives of MS-SSIM through multi-scale pyramid
**Complexity:** Higher than L2 but still tractable

#### 4. Finite Differences (L-M fallback)
**Method:** (f(x+ε) - f(x)) / ε for each parameter
**Cost:** 2·n_params forward passes
**Use:** Fallback only, not recommended

---

## LEARNING RATE SCHEDULES

### Implemented Schedules

#### 1. Per-Parameter Fixed (Adam, OptimizerV2)
Different LRs for different parameter types:
- Position: 0.0002 (small - spatial)
- Color: 0.02 (large - perceptual)
- Scale: 0.005 (medium)
- Opacity: 0.05 (large)

#### 2. Exponential Decay (Adam position)
Position LR decays from 0.0002 → 0.00002 over iterations

#### 3. Adaptive (Adam via bias correction)
Effective LR adapts based on gradient statistics

---

## FILE LOCATION INDEX

### CPU Optimizers (lgi-encoder-v2/src/)
```
lgi-encoder-v2/src/
├── adam_optimizer.rs (487 lines) - Adam algorithm
├── optimizer_v2.rs (379 lines) - Gradient descent + MS-SSIM
├── optimizer_v3_perceptual.rs (201 lines) - Perceptual loss
├── optimizer_lbfgs.rs (281 lines) - L-BFGS
├── lm_optimizer.rs (295 lines) - Levenberg-Marquardt
└── hybrid_optimizer.rs (360 lines) - Adam → L-BFGS
```

**Total CPU:** 2,003 lines

### GPU Optimizers (lgi-gpu/src/)
```
lgi-gpu/src/
├── adam_optimizer.rs (907 lines) - GPU Adam
├── lbfgs_optimizer.rs (1,038 lines) - GPU L-BFGS
├── lm_optimizer.rs (723 lines) - GPU L-M with JtJ strategies
└── jtj_strategy.rs - JtJ computation strategies
```

**Total GPU:** 2,668 lines

### Supporting Modules
```
lgi-core/src/
├── lbfgs.rs - Core L-BFGS implementation (used by CPU optimizer)
├── ms_ssim_loss.rs - MS-SSIM loss function
├── ms_ssim_gradients.rs - Analytical MS-SSIM gradients
└── edge_weighted_loss.rs - Edge-weighted loss

lgi-gpu/src/shaders/
├── adam_update.wgsl - GPU Adam compute shader
├── gradient_compute.wgsl - GPU gradient shader (235 lines)
└── lm_jacobian*.wgsl - L-M Jacobian shaders
```

---

## SUMMARY STATISTICS

### Implementation Count
- **Total Optimizers:** 9 (6 CPU + 3 GPU)
- **Total Lines:** 4,671
- **Loss Functions:** 3 (L2, MS-SSIM, Edge-weighted)
- **Gradient Methods:** 4 (CPU analytical, GPU analytical, MS-SSIM analytical, finite diff)

### Optimizer Characteristics

| Optimizer | Type | Lines | Gradient Method | Parameters Optimized |
|-----------|------|-------|----------------|---------------------|
| Adam (CPU) | First-order | 487 | Analytical | All 9 |
| OptimizerV2 | First-order | 379 | Analytical + GPU | All 9 |
| OptimizerV3 | First-order | 201 | MS-SSIM analytical | All 9 |
| L-BFGS (CPU) | Quasi-Newton | 281 | Analytical | All 9 |
| L-M (CPU) | Gauss-Newton | 295 | Analytical or FD | 7 (no rotation/opacity) |
| Hybrid | Two-phase | 360 | Analytical | All 9 |
| Adam (GPU) | First-order | 907 | GPU analytical | All 9 |
| L-BFGS (GPU) | Quasi-Newton | 1,038 | GPU analytical | All 9 |
| L-M (GPU) | Gauss-Newton | 723 | GPU analytical | 10 |

---

*End of Optimizer Catalog - 2025-12-07*
