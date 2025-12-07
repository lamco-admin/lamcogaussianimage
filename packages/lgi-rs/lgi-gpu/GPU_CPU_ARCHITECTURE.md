# GPU/CPU Optimizer Architecture and Alignment Issues

## Executive Summary

This document chronicles the development and debugging of GPU-accelerated optimizers for LGI (Lamco Gaussian Image) encoding. The primary challenge was achieving **quality parity** between CPU and GPU implementations, which required solving a critical **WGSL struct alignment mismatch**.

**Key Results:**
- Fixed -1.07 dB quality gap between GPU and CPU Adam optimizers
- Achieved GPU quality within -0.06 dB of CPU (effectively identical)
- Maintained 12x speedup (GPU: 10s vs CPU: 120s for 200 iterations)

---

## Architecture Overview

### Optimization Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    GPU Optimization Pipeline                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐             │
│  │   Render    │ → │  Gradient   │ → │   Update    │             │
│  │   Pass      │   │   Pass      │   │   Pass      │             │
│  └─────────────┘   └─────────────┘   └─────────────┘             │
│        ↓                 ↓                 ↓                      │
│  gaussian_render   gradient_compute   adam_update.wgsl           │
│      .wgsl             .wgsl          lbfgs_update.wgsl          │
│                                                                   │
│  Buffer Layout (48 bytes each, 8-byte aligned):                  │
│  ┌────────────────────────────────────────────────┐              │
│  │ ShaderGaussian | GpuGradient | AdamMoment     │              │
│  │ (48 bytes)     | (48 bytes)  | (48 bytes)     │              │
│  └────────────────────────────────────────────────┘              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Files Involved

| File | Purpose |
|------|---------|
| `lgi-gpu/src/adam_optimizer.rs` | GPU Adam optimizer (Rust) |
| `lgi-gpu/src/lbfgs_optimizer.rs` | GPU L-BFGS optimizer (Rust) |
| `lgi-gpu/src/gradient.rs` | GPU gradient computation (Rust) |
| `lgi-gpu/src/shaders/gaussian_render.wgsl` | Gaussian splatting renderer |
| `lgi-gpu/src/shaders/gradient_compute.wgsl` | Backprop gradient computation |
| `lgi-gpu/src/shaders/adam_update.wgsl` | Adam parameter update |
| `lgi-gpu/src/shaders/lbfgs_update.wgsl` | L-BFGS parameter update |

---

## The Critical Bug: WGSL Array Stride Alignment

### Symptom

GPU Adam optimizer plateaued at loss ~0.051 while CPU continued improving to ~0.037, resulting in a **-1.07 dB quality gap**.

### Root Cause

**WGSL storage buffer arrays pad struct elements to the alignment of the largest field.**

Our `GpuGradient` struct had `vec2<f32>` as the largest aligned type (8-byte alignment). The actual content was 44 bytes, but WGSL array stride was **48 bytes** due to padding.

```
Rust struct (BEFORE FIX):
┌────────────────────────────────────────────────┐
│ d_position: [f32; 2]  │ 8 bytes @ offset 0     │
│ d_scale_x: f32        │ 4 bytes @ offset 8     │
│ d_scale_y: f32        │ 4 bytes @ offset 12    │
│ d_rotation: f32       │ 4 bytes @ offset 16    │
│ d_color: [f32; 4]     │ 16 bytes @ offset 20   │
│ d_opacity: f32        │ 4 bytes @ offset 36    │
│ _padding: f32         │ 4 bytes @ offset 40    │ ← WRONG!
└────────────────────────────────────────────────┘
Total: 44 bytes (but WGSL expects 48-byte stride!)
```

When reading an array of these structs, element N was at offset `N * 44` in Rust but `N * 48` in WGSL. This caused a **4-byte shift per element**, corrupting all gradient values after the first Gaussian.

### The Fix

Add explicit padding to match WGSL's 48-byte array stride:

```rust
// AFTER FIX:
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuGradient {
    pub d_position: [f32; 2],   // 8 bytes @ offset 0
    pub d_scale_x: f32,         // 4 bytes @ offset 8
    pub d_scale_y: f32,         // 4 bytes @ offset 12
    pub d_rotation: f32,        // 4 bytes @ offset 16
    pub d_color: [f32; 4],      // 16 bytes @ offset 20
    pub d_opacity: f32,         // 4 bytes @ offset 36
    pub _padding: [f32; 2],     // 8 bytes @ offset 40 ← CRITICAL: 48 total!
}
```

Similarly fixed in WGSL:

```wgsl
struct GaussianGradient {
    d_position: vec2<f32>,    // 8 bytes @ offset 0
    d_scale_x: f32,           // 4 bytes @ offset 8
    d_scale_y: f32,           // 4 bytes @ offset 12
    d_rotation: f32,          // 4 bytes @ offset 16
    d_color_r: f32,           // 4 bytes @ offset 20
    d_color_g: f32,           // 4 bytes @ offset 24
    d_color_b: f32,           // 4 bytes @ offset 28
    d_color_a: f32,           // 4 bytes @ offset 32
    d_opacity: f32,           // 4 bytes @ offset 36
    _padding: vec2<f32>,      // 8 bytes @ offset 40 (48 bytes total)
}
```

### Affected Structs

| Struct | Location | Issue |
|--------|----------|-------|
| `GpuGradient` | `gradient.rs` | `_padding: f32` → `_padding: [f32; 2]` |
| `GaussianGradient` | `gradient_compute.wgsl` | `_padding: f32` → `_padding: vec2<f32>` |
| `GaussianGradient` | `adam_update.wgsl` | Same fix |
| `AdamMoment` | `adam_optimizer.rs` + WGSL | Same fix |
| `AdamVariance` | `adam_optimizer.rs` + WGSL | Same fix |

---

## Diagnostic Process

### 1. Initial Observation

Loss curves diverged after ~10 iterations:
- CPU: 0.143 → 0.043 → 0.039 → 0.037
- GPU: 0.143 → 0.051 → 0.051 → 0.051 (stuck!)

### 2. Hypothesis: Gradient Mismatch

Created `debug_gradient_comparison.rs` to compare exact gradient values:

```rust
// Test setup: 3 Gaussians, 128×128 image
// Compare GPU vs CPU gradient values per-Gaussian
```

### 3. Discovery: Pattern in Errors

```
Gaussian 0: ✓ Perfect match (max diff: 7.45e-9)
Gaussian 1: ✗ Completely wrong values
Gaussian 2: ✗ Completely wrong values

Pattern: GPU values for Gaussian 1 matched CPU values shifted by 4 bytes
         GPU values for Gaussian 2 matched CPU values shifted by 8 bytes
```

This **4-byte shift per element** pattern revealed the array stride mismatch.

### 4. Verification After Fix

```
Gaussian 0: ✓ Match (max diff: 7.45e-9)
Gaussian 1: ✓ Match (max diff: 7.45e-9)
Gaussian 2: ✓ Match (max diff: 7.45e-9)

✓ Gradients MATCH (max diff: 7.45e-9)
```

---

## WGSL Alignment Rules

### Key Rules

1. **Primitive types**: `f32` = 4 bytes, `u32` = 4 bytes, `vec2<f32>` = 8 bytes, `vec4<f32>` = 16 bytes

2. **Struct alignment**: Aligned to the alignment of its largest member

3. **Array stride**: Elements padded to struct alignment
   - If struct has `vec2<f32>`, stride rounds up to 8-byte multiple
   - If struct has `vec4<f32>`, stride rounds up to 16-byte multiple

4. **Rust `#[repr(C)]`**: Does NOT automatically match WGSL stride. Explicit padding required.

### Golden Rule

**Always verify struct sizes in tests:**

```rust
#[test]
fn test_struct_sizes() {
    assert_eq!(std::mem::size_of::<ShaderGaussian>(), 48, "ShaderGaussian must be 48 bytes");
    assert_eq!(std::mem::size_of::<GpuGradient>(), 48, "GpuGradient must be 48 bytes");
    assert_eq!(std::mem::size_of::<AdamMoment>(), 48, "AdamMoment must be 48 bytes");
    assert_eq!(std::mem::size_of::<AdamVariance>(), 48, "AdamVariance must be 48 bytes");
}
```

---

## Optimizer Architecture

### GPU Adam Optimizer

```
Per-iteration pipeline:
1. RENDER: Composite Gaussians → rendered image
2. GRADIENT: Backprop loss → per-Gaussian gradients
3. ADAM UPDATE: Update parameters using moment estimates

State buffers (persistent across iterations):
- gaussian_buffer: ShaderGaussian[N] (48 bytes each)
- moment_buffer: AdamMoment[N] (first moment m)
- variance_buffer: AdamVariance[N] (second moment v)
- gradient_buffer: GpuGradient[N] (overwritten each iteration)
```

### GPU L-BFGS Optimizer (Hybrid)

```
Hybrid architecture:
- GPU: Gradient computation, rendering, parameter updates
- CPU: Two-loop recursion (sequential, history size m=10 is small)

Per-iteration pipeline:
1. GPU: Render + compute gradients
2. CPU: Read gradients, compute search direction (two-loop)
3. GPU: Line search (multiple render passes for loss evaluation)
4. GPU: Apply update (x += alpha * direction)
5. CPU: Update L-BFGS history (s_k, y_k vectors)
```

---

## Performance Characteristics

### GPU Adam vs CPU Adam

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Time (200 iter) | 120s | 10s | **12x** |
| Final PSNR | 14.30 dB | 14.24 dB | -0.06 dB |
| Iter/sec | 1.7 | 20 | **12x** |

### GPU Overhead

- Initial GPU setup: ~1.7s (one-time)
- Per-iteration overhead: ~50ms (mostly render + gradient)
- Buffer readback (periodic loss check): ~5ms

---

## Lessons Learned

### 1. Test CPU/GPU Parity Early

Create diagnostic tests that compare **exact numerical values**, not just aggregate metrics. The gradient comparison test immediately revealed the byte-shift pattern.

### 2. Explicit Struct Alignment

Never assume Rust `#[repr(C)]` matches WGSL layout. Calculate expected sizes manually and add explicit `_padding` fields.

### 3. Document Struct Layouts

Include byte offset comments in both Rust and WGSL:

```rust
pub struct GpuGradient {
    pub d_position: [f32; 2],   // 8 bytes @ offset 0
    pub d_scale_x: f32,         // 4 bytes @ offset 8
    // ... etc
}
```

### 4. Size Assertion Tests

Add compile-time or runtime checks for struct sizes:

```rust
assert_eq!(std::mem::size_of::<T>(), EXPECTED_SIZE);
```

### 5. Isotropic vs Anisotropic Gradients

Both CPU and GPU use simplified **isotropic** gradient formulas (ignoring rotation) for stability. This is intentional - the anisotropic formulas can be unstable during optimization.

### 6. Line Search Must Use Actual Target

L-BFGS line search computes trial losses to check the Armijo condition. A subtle bug had the line search computing loss against a **black placeholder image** instead of the actual target:

```rust
// WRONG: Loss against black image
let f_new = self.compute_loss(&trial_gaussians, &ImageBuffer {
    width, height,
    data: vec![Color4::new(0.0, 0.0, 0.0, 1.0); ...],
});

// CORRECT: Loss against actual target
let f_new = self.compute_loss(&trial_gaussians, target);
```

This caused incorrect step size selection, potentially accepting bad steps or rejecting good ones.

### 7. Armijo Condition Must Use Current Loss

The Armijo line search condition compares `f(x + αp)` against `f(x) + c1·α·∇f^T·p`. A critical bug used `initial_loss` (computed once at the start) instead of the **current iteration's loss**:

```rust
// WRONG: Using initial_loss (stale after many iterations)
let step_size = self.line_search_gpu(
    ...,
    initial_loss,   // ← BUG: This never updates!
    directional_deriv, target,
)?;

// CORRECT: Compute current loss before line search
let current_loss = self.compute_loss(&current_gaussians, target);
let step_size = self.line_search_gpu(
    ...,
    current_loss,   // ← Current iteration's actual loss
    directional_deriv, target,
)?;
```

**Symptom**: Loss oscillated instead of monotonically decreasing:
```
Iter 10: 0.0456 → Iter 20: 0.0429 ✓ → Iter 40: 0.0465 ✗ (got worse!)
```

**Root cause**: After many improvements, the current loss was much lower than initial. When checking `f_new <= f_x + c1·α·∇f^T·p`, using a stale high `f_x` made the condition too permissive, accepting steps that increased the actual loss.

---

## Complete Optimizer Inventory

### CPU Implementations (`lgi-encoder-v2/src/`, `lgi-core/src/`)

| File | Algorithm | Status | Notes |
|------|-----------|--------|-------|
| `adam_optimizer.rs` | Adam | **Production** | Per-parameter learning rates, +2.25 dB avg |
| `optimizer_v2.rs` | GD + corrections | Working | Gradient sign corrections |
| `optimizer_lbfgs.rs` | L-BFGS | Working | Uses `lgi-core/src/lbfgs.rs` |
| `lm_optimizer.rs` | Levenberg-Marquardt | Working | Full Jacobian, very slow |
| `hybrid_optimizer.rs` | Adam→L-BFGS | Experimental | Warm start L-BFGS from Adam |
| `optimizer_v3_perceptual.rs` | Adam + perceptual | Experimental | MS-SSIM loss |

### GPU Implementations (`lgi-gpu/src/`)

| File | Algorithm | Status | Parity vs CPU | Notes |
|------|-----------|--------|---------------|-------|
| `adam_optimizer.rs` | Adam | **Production** | -0.06 dB | 12x speedup, gradient alignment fixed |
| `lbfgs_optimizer.rs` | L-BFGS | **Needs Investigation** | -4 to -6 dB gap | Armijo fix applied but quality gap remains |
| `lm_optimizer.rs` | L-M (hybrid) | Working | N/A (different arch) | Block-diagonal J^T J, slow convergence |

### GPU Shaders (`lgi-gpu/src/shaders/`)

| File | Purpose | Verified |
|------|---------|----------|
| `gaussian_render.wgsl` | Gaussian splatting compositor | ✓ |
| `gradient_compute.wgsl` | Backprop gradients | ✓ (struct alignment fixed) |
| `adam_update.wgsl` | Adam moment/variance + update | ✓ (struct alignment fixed) |
| `lbfgs_update.wgsl` | L-BFGS direction application | Partial |
| `lm_jacobian.wgsl` | J^T J and J^T r accumulation | ✓ (sum-of-outer-products fixed) |
| `lm_update.wgsl` | L-M parameter update | ✓ |
| `hybrid_loss.wgsl` | MS-SSIM + MSE combined loss | Experimental |

---

## Current GPU/CPU Benchmark Results (Dec 7, 2025)

### Adam Optimizer (Production Quality)

| Config | CPU PSNR | GPU PSNR | Gap | Speedup |
|--------|----------|----------|-----|---------|
| 256 Gaussians, 200 iter | 14.30 dB | 14.24 dB | **-0.06 dB** | 12x |

**Status**: ✓ GPU matches CPU quality. Production ready.

### L-BFGS Optimizer (Quality Gap Needs Investigation)

| Config | CPU PSNR | GPU PSNR | Gap | Speedup |
|--------|----------|----------|-----|---------|
| 64 Gaussians, 50 iter | 20.13 dB | 16.10 dB | **-4.03 dB** | 139x |
| 128 Gaussians, 50 iter | 20.58 dB | 14.10 dB | **-6.47 dB** | 309x |

**Status**: ⚠ GPU significantly worse. Root cause investigation needed.

**Bugs Fixed (but gap remains)**:
1. Line search was computing loss against black image (fixed)
2. Armijo condition was using stale `initial_loss` (fixed)

**Hypotheses for remaining gap**:
- GPU gradient precision issues at float32
- Line search parameter differences (c1, c2)
- Two-loop recursion numerical stability
- Different initial conditions or random seeds

### L-M Optimizer (Reference Implementation)

| Config | PSNR | Improvement | Notes |
|--------|------|-------------|-------|
| 64 Gaussians, 100 iter | 14.12 dB | 0.38% | Block-diagonal approximation |

**Status**: GPU L-M works but converges slowly due to block-diagonal J^T J approximation ignoring cross-Gaussian interactions. λ stays at 1e6-1e7 (heavily regularized).

**GPU-only features**:
- Implicit Jacobian via atomic accumulation
- Fixed-point atomic arithmetic (FIXED_SCALE = 1e6)
- Per-channel outer product sum: `Jr^T*Jr + Jg^T*Jg + Jb^T*Jb`

**Potential improvements (not implemented)**:
- Full dense J^T J with off-diagonal blocks
- Sparse Jacobian using pixel-Gaussian associations
- Jacobian-free Newton-CG

---

## Architectural Differences: GPU vs CPU

### What's Fundamentally Different

| Aspect | CPU | GPU |
|--------|-----|-----|
| Parallelism | Per-Gaussian or per-iteration | Per-pixel (massive) |
| Memory model | Sequential access, cache-friendly | Coalesced access, limited bandwidth |
| Atomics | Not needed | Required for reduction |
| Precision | f64 possible | f32 only (for atomics) |
| Linear algebra | BLAS/LAPACK available | Must implement or transfer to CPU |

### What Could Only Be Done on GPU

1. **Real-time encoding**: 20+ iter/sec enables live preview
2. **Massive parallelism**: 768×512×256 = 100M pixel-Gaussian interactions/iteration
3. **Video encoding**: Per-frame optimization at 30+ FPS decode target

### What Could Only Be Done on CPU

1. **L-BFGS two-loop recursion**: Sequential algorithm, history access pattern
2. **Sparse linear solvers**: UMFPACK, CHOLMOD for L-M with sparse Jacobian
3. **Double precision**: Critical for high-accuracy L-M/L-BFGS
4. **Complex control flow**: Trust region, adaptive strategies

### Hybrid Approaches (Current)

1. **GPU L-BFGS**: GPU gradients → CPU two-loop → GPU line search → CPU history update
2. **GPU L-M**: GPU J^T J/J^T r → CPU solve → GPU update

---

## J^T J Strategy System (GPU L-M)

### Why L-M is Fundamentally GPU-Native

Levenberg-Marquardt requires computing J^T J (Gauss-Newton Hessian) and J^T r (gradient). The Jacobian J has shape `(n_residuals × n_params)`:

```
For 768×512 image, 256 Gaussians:
- Residuals: 768 × 512 × 3 = 1.18M (RGB per pixel)
- Parameters: 256 × 10 = 2,560
- Jacobian J: 1.18M × 2,560 = 3 billion elements (12 GB)
```

This is **impractical to store explicitly** on either CPU or GPU. However:

- **GPU**: Can accumulate J^T J and J^T r *implicitly* via per-pixel atomics
- **CPU**: Would need explicit Jacobian or very slow per-pixel accumulation

This makes L-M a case where **GPU enables an algorithm class that's impractical on CPU**.

### Strategy Abstraction (`lgi-gpu/src/jtj_strategy.rs`)

The L-M solver is generic over HOW J^T J is computed:

```rust
pub trait JtJStrategy: Send + Sync {
    /// Compute J^T J and J^T r for current state
    fn compute(&self, device: &Device, queue: &Queue, ctx: &JtJContext)
        -> Result<JtJResult>;

    /// Whether this strategy captures cross-Gaussian interactions
    fn captures_interactions(&self) -> bool;

    /// Expected computational complexity
    fn complexity(&self) -> ComplexityClass;
}
```

### Available Strategies

| Strategy | Complexity | Captures Interactions | Status |
|----------|------------|----------------------|--------|
| `BlockDiagonalJtJ` | O(P·pixels) | **No** | Implemented |
| `FullDenseJtJ` | O(G²·pixels) | Yes | Placeholder |
| `SparseJtJ` | O(overlap·pixels) | Yes | Placeholder |
| `MatrixFreeJtJ` | O(iter·G·pixels) | Yes | Placeholder |

Where P = params per Gaussian (10), G = number of Gaussians.

### The Interaction Problem

The current `BlockDiagonalJtJ` ignores cross-Gaussian interactions. This is mathematically incorrect when Gaussians overlap - the Hessian approximation is poor, requiring heavy regularization (λ ~ 1e6-1e7) and causing slow convergence.

**However**: The importance of interactions depends on the image. For images where Gaussians don't overlap much, block-diagonal may be sufficient. For highly overlapping cases, full dense or sparse strategies would converge faster.

### Usage

```rust
// Switch strategies at runtime
let strategy: Box<dyn JtJStrategy> = JtJStrategyRegistry::create(&device, "block-diagonal")?;
optimizer.set_strategy(strategy);

// Or use factory
let strategy = JtJStrategyRegistry::create(&device, "sparse")?;
```

---

## Investigation Queue

1. **L-BFGS Quality Gap** [HIGH PRIORITY]
   - Compare exact gradient values GPU vs CPU (like we did for Adam)
   - Check line search acceptance rate
   - Compare L-BFGS history vectors (s_k, y_k)
   - Try identical initial conditions

2. **L-M Dense J^T J** [RESEARCH]
   - Implement `FullDenseJtJ` shader computing all off-diagonal blocks
   - Measure memory/compute cost vs quality gain
   - Compare convergence rate vs block-diagonal

3. **L-M Sparse J^T J** [RESEARCH]
   - Build spatial index (grid or k-d tree) of Gaussians
   - Only compute J^T J blocks for overlapping pairs
   - Measure sparsity ratio on Kodak dataset

4. **Mixed Precision** [OPTIMIZATION]
   - FP16 gradients for 2x bandwidth
   - FP32 accumulation for numerical stability

---

## References

- WebGPU Spec: https://www.w3.org/TR/WGPU/
- wgpu-rs: https://github.com/gfx-rs/wgpu
- WGSL Alignment Rules: https://www.w3.org/TR/WGSL/#alignment-and-size
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
