# RENDERER CATALOG - Gaussian Image Codec
## Complete Rendering Implementations Inventory

**Document Version:** 2.0
**Date:** 2025-12-07
**Updated From:** Nov 15, 2025 RENDERING_ENGINES_CATALOG.md
**Repository:** lgi-project
**Total Renderer Code:** 1,326 lines (Rust) + 1,995 lines (WGSL shaders)

---

## TABLE OF CONTENTS

1. [Overview](#overview)
2. [CPU Renderers](#cpu-renderers)
3. [GPU Renderer](#gpu-renderer)
4. [GPU Shaders](#gpu-shaders)
5. [Rendering Modes](#rendering-modes)
6. [File Location Index](#file-location-index)

---

## OVERVIEW

Renderers convert Vec<Gaussian2D> → ImageBuffer (pixels). This is the **forward pass** in optimization - pure evaluation with no learning.

### Renderer vs Optimizer vs Encoder

```
RENDERER: Gaussians → pixels (forward only)
OPTIMIZER: Updates Gaussians (calls renderer repeatedly)
ENCODER: Complete pipeline (initialization + optimizer selection)
```

### Renderer Count
- **CPU Renderers:** 6 implementations (1,326 lines)
- **GPU Renderer:** 1 implementation (279 lines + 1,995 shader lines)
- **Total:** 7 rendering engines

---

## CPU RENDERERS

### 1. PRIMARY RENDERER (Alpha Compositing)

**File:** `lgi-core/src/renderer.rs`
**Lines:** 327
**Purpose:** Production-quality multi-threaded renderer

#### Rendering Modes

**AlphaComposite Mode (default):**
- Front-to-back Porter-Duff alpha compositing
- Early termination at 99.9% opacity
- Physically-based blending

**AccumulatedSum Mode:**
- GaussianImage ECCV 2024 method
- Weighted normalization: C = Σ(w×c) / Σ(w)
- No early termination

#### Configuration (RenderConfig struct)

```rust
pub struct RenderConfig {
    pub background: Color4<f32>,        // Default: black transparent
    pub alpha_mode: AlphaMode,          // Straight or Premultiplied
    pub render_mode: RenderMode,        // AlphaComposite or AccumulatedSum
    pub cutoff_threshold: f32,          // Default: 1e-5
    pub n_sigma: f32,                   // Default: 3.5 (bounding box)
    pub termination_threshold: f32,     // Default: 0.999 (early term)
    pub parallel: bool,                 // Default: true (Rayon)
}
```

#### Key Methods
- `render()` - Auto dispatch (parallel or sequential)
- `render_parallel()` - Multi-threaded scanline (Rayon)
- `render_alpha_composite()` - Alpha blending mode
- `render_accumulated_sum()` - Weighted average mode

#### Bounding Box Optimization
3.5σ cutoff radius - covers 99.99% of Gaussian mass
Per-Gaussian AABB culling before per-pixel evaluation

#### Early Termination
In AlphaComposite mode: stop when accumulated opacity ≥ 0.999
Performance gain: ~20-30%

---

### 2. EWA SPLATTING V1

**File:** `lgi-core/src/ewa_splatting.rs`
**Lines:** 101
**Purpose:** Alias-free Gaussian rendering (Zwicker et al. 2001)

#### Description
Elliptical Weighted Average algorithm for high-quality Gaussian evaluation.

#### Algorithm
- Reconstruction filter (Gaussian)
- Mahalanobis distance computation
- Alias-free rendering
- Rotation-invariant

#### Features
- Bounding box culling
- Configurable filter radius
- No early termination (all Gaussians contribute)

---

### 3. EWA SPLATTING V2

**File:** `lgi-core/src/ewa_splatting_v2.rs`
**Lines:** 228
**Purpose:** Enhanced EWA with zoom and robustness

#### Enhancements over V1
- Zoom support (covariance transformation)
- Degenerate Gaussian handling
- Improved numerical stability
- Reconstruction filter options

#### Additional Complexity
~2× code size vs V1
Trades speed for quality and robustness

---

### 4. RENDERER V2 (Encoder Testing)

**File:** `lgi-encoder-v2/src/renderer_v2.rs`
**Lines:** 113
**Purpose:** Fast validation renderer for encoder development

#### Description
Simplified weighted average renderer optimized for speed during encoder development.

#### Algorithm
```
C(x,y) = Σ(w_i × c_i) / Σ(w_i)
where w_i = α_i × G_i(x,y)
```

No alpha compositing - pure weighted average.

#### Usage
Primary renderer for all CPU optimizers:
- adam_optimizer.rs
- adaptive_densification.rs
- commercial_encoder.rs
- debug_logger.rs
- encoder_v3_adaptive.rs

#### Performance
Faster than alpha compositing (~20% for typical Gaussian counts)

---

### 5. RENDERER V3 (Textured)

**File:** `lgi-encoder-v2/src/renderer_v3_textured.rs`
**Lines:** 204
**Purpose:** Per-primitive texture support for detail preservation

#### Description
Extends RendererV2 with optional per-Gaussian texture maps for fine detail.

#### Features
- Embedded texture support
- Local texture sampling within Gaussian extent
- Color blending (texture × base color)
- Adaptive quality toggle

#### Overhead
10-20% slower than V2 when textures enabled

#### Usage
Used by commercial_encoder.rs for production-quality encoding

---

### 6. RENDERER_GPU (Wrapper)

**File:** `lgi-encoder-v2/src/renderer_gpu.rs`
**Lines:** 74
**Purpose:** Convenience wrapper for GPU renderer in encoder context

#### Description
Thin wrapper around `lgi-gpu::GpuRenderer` for use in encoder-v2 optimizers.

---

## GPU RENDERER

### 7. GPU RENDERER (wgpu/WebGPU)

**File:** `lgi-gpu/src/renderer.rs`
**Lines:** 279
**Backend:** Vulkan, Metal, DirectX 12, OpenGL, WebGPU
**Purpose:** High-performance GPU rendering (1000+ FPS)

#### Description
Complete GPU-accelerated renderer using wgpu for cross-platform compute.

#### Architecture
- Automatic adapter selection (performance preference)
- Vulkan first, then Metal, DX12, GL fallback
- Compute pipeline for Gaussian evaluation
- Buffer management for parameters and output

#### Key Components

**Struct:** `GpuRenderer`
- Device and queue
- Render pipeline
- Buffer pool
- Profiler

#### Supported Render Modes
- AlphaComposite (front-to-back)
- AccumulatedSum (weighted average)

#### Performance Characteristics
- Vulkan/NVIDIA: 1000+ FPS @ 1080p
- Metal/Apple: 800+ FPS
- DX12/Intel: 500+ FPS
- OpenGL: 300+ FPS

For 10,000 Gaussians at 1920×1080 resolution

#### Buffer Layout

GpuGaussian struct (40 bytes, aligned):
```
position: vec2<f32>   (8 bytes)
scale: vec2<f32>      (8 bytes)
rotation: f32         (4 bytes)
_padding: f32         (4 bytes)
color: vec3<f32>      (12 bytes)
opacity: f32          (4 bytes)
```

---

## GPU SHADERS

### GAUSSIAN_RENDER.WGSL

**File:** `lgi-gpu/src/shaders/gaussian_render.wgsl`
**Lines:** 145
**Purpose:** Per-pixel Gaussian evaluation shader

#### Algorithm
```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // For each pixel (x,y):
    for (each Gaussian) {
        // Evaluate Gaussian at pixel
        // Accumulate color/opacity
        // Early terminate if opacity > 0.999 (AlphaComposite)
    }
}
```

#### Workgroup
16×16 = 256 threads per workgroup
Each thread processes one pixel

#### Optimizations
- 3.5σ bounding box culling
- Early termination (AlphaComposite mode)
- Local memory for accumulation
- Zero-initialization critical (Line 80 in earlier version)

#### Modes (from earlier catalog)
- AlphaComposite: Lines 99-110
- AccumulatedSum: Lines 112-118

---

### GRADIENT_COMPUTE.WGSL

**File:** `lgi-gpu/src/shaders/gradient_compute.wgsl`
**Lines:** 287
**Purpose:** Backpropagation for all Gaussian parameters

#### Description
Computes analytical gradients with respect to all 9 parameters per Gaussian.

#### Workgroup
256 threads

#### Derivatives Computed
- ∂L/∂position (x, y)
- ∂L/∂scale (σ_x, σ_y)
- ∂L/∂rotation (θ)
- ∂L/∂color (R, G, B)
- ∂L/∂opacity (α)

#### Algorithm
- Per-pixel loss gradient
- Chain rule through Gaussian evaluation
- Accumulate per-Gaussian gradients
- 3.5σ spatial culling
- Parallel reduction

#### Speedup
100-1000× faster than CPU gradient computation

---

### ADAM_UPDATE.WGSL

**File:** `lgi-gpu/src/shaders/adam_update.wgsl`
**Lines:** 328
**Purpose:** GPU Adam parameter updates

#### Description
Compute shader implementing Adam update equations for all Gaussian parameters.

#### Operations Per Gaussian
- Update momentum buffers (m)
- Update RMSprop buffers (v)
- Bias correction
- Compute update: θ ← θ - lr·m̂/(√v̂ + ε)

---

### L-BFGS UPDATE SHADER

**File:** `lgi-gpu/src/shaders/lbfgs_update.wgsl`
**Lines:** 133
**Purpose:** Apply L-BFGS parameter updates

---

### L-M JACOBIAN SHADERS

**Files:**
- `lm_jacobian.wgsl` (248 lines) - Base Jacobian computation
- `lm_jacobian_dense.wgsl` (271 lines) - Dense J^T J strategy
- `lm_jacobian_sparse.wgsl` (268 lines) - Sparse J^T J strategy

**Purpose:** Compute J^T J and J^T r for Levenberg-Marquardt

---

### L-M UPDATE SHADER

**File:** `lgi-gpu/src/shaders/lm_update.wgsl`
**Lines:** 72
**Purpose:** Apply L-M parameter updates after CPU solves linear system

---

### HYBRID LOSS SHADER

**File:** `lgi-gpu/src/shaders/hybrid_loss.wgsl`
**Lines:** 243
**Purpose:** Combined loss functions (L2 + MS-SSIM + edge-weighted)

---

## RENDERING MODES

### AlphaComposite (Physically-Based)

**Algorithm:** Front-to-back Porter-Duff compositing
```
for each Gaussian (back to front):
    α_acc ← α_acc + (1 - α_acc) × α_i × w_i
    C_acc ← C_acc + (1 - α_acc) × c_i × α_i × w_i
    if α_acc > 0.999: break  // Early termination
```

**Characteristics:**
- Physically correct blending
- Order-dependent (requires sorting)
- Early termination possible
- Standard for production

**Used By:**
- lgi-core/renderer.rs (default mode)
- GPU renderer (default mode)

---

### AccumulatedSum (Weighted Average)

**Algorithm:** Normalized weighted sum
```
Σ_w = Σ(α_i × G_i(x,y))
Σ_c = Σ(c_i × α_i × G_i(x,y))
C(x,y) = Σ_c / Σ_w
```

**Characteristics:**
- Order-independent
- No early termination
- Simpler mathematics
- GaussianImage ECCV 2024 method

**Used By:**
- lgi-core/renderer.rs (optional mode)
- RendererV2 (default)
- GPU renderer (optional mode)

---

## FILE LOCATION INDEX

### CPU Renderers
```
packages/lgi-rs/
├── lgi-core/src/
│   ├── renderer.rs (327 lines) - Primary multi-threaded renderer
│   ├── ewa_splatting.rs (101 lines) - EWA V1
│   └── ewa_splatting_v2.rs (228 lines) - EWA V2 enhanced
└── lgi-encoder-v2/src/
    ├── renderer_v2.rs (113 lines) - Fast validation renderer
    ├── renderer_v3_textured.rs (204 lines) - Textured Gaussians
    └── renderer_gpu.rs (74 lines) - GPU wrapper
```

**Total CPU:** 1,047 lines

### GPU Renderer & Shaders
```
packages/lgi-rs/lgi-gpu/src/
├── renderer.rs (279 lines) - Main GPU renderer
└── shaders/
    ├── gaussian_render.wgsl (145 lines) - Forward rendering
    ├── gradient_compute.wgsl (287 lines) - Gradient computation
    ├── adam_update.wgsl (328 lines) - Adam updates
    ├── lbfgs_update.wgsl (133 lines) - L-BFGS updates
    ├── lm_jacobian.wgsl (248 lines) - L-M Jacobian base
    ├── lm_jacobian_dense.wgsl (271 lines) - Dense J^T J
    ├── lm_jacobian_sparse.wgsl (268 lines) - Sparse J^T J
    ├── lm_update.wgsl (72 lines) - L-M updates
    └── hybrid_loss.wgsl (243 lines) - Combined loss

```

**Total GPU:** 279 + 1,995 = 2,274 lines

---

## RENDERER COMPARISON TABLE

| Renderer | Algorithm | Lines | Features | Used By |
|----------|-----------|-------|----------|---------|
| renderer.rs | Alpha composite or Accumulated sum | 327 | Multi-threaded, early termination, dual mode | Production, viewer |
| ewa_splatting.rs | EWA (Zwicker 2001) | 101 | Alias-free, rotation-invariant | Legacy/research |
| ewa_splatting_v2.rs | EWA enhanced | 228 | Zoom support, robust | Legacy/research |
| renderer_v2.rs | Weighted average | 113 | Fast, simple | All CPU optimizers |
| renderer_v3_textured.rs | Textured Gaussians | 204 | Per-primitive textures | Commercial encoder |
| renderer_gpu.rs | GPU wrapper | 74 | Convenience wrapper | Encoder-v2 integration |
| GPU renderer.rs | wgpu compute | 279 | Cross-platform, 1000+ FPS | GPU optimizers, viewer |

---

## RENDERING CONFIGURATION

### Bounding Box Cutoff
Standard: 3.5σ radius (covers 99.99% of Gaussian)
Optimization: Early culling before per-pixel evaluation
Configurable via `n_sigma` parameter

### Rotation Handling
Representation: Single angle θ (radians)
Transformation:
```
dx_rot = dx·cos(θ) + dy·sin(θ)
dy_rot = -dx·sin(θ) + dy·cos(θ)
Distance metric: (dx_rot/σx)² + (dy_rot/σy)²
```

### Opacity Handling
Range: [0, 1]
Application: Multiplied with Gaussian weight
Compositing: Tracked separately from color

---

## SUMMARY STATISTICS

### Implementation Count
- **Total Renderers:** 7 (6 CPU + 1 GPU)
- **Total Code:** 3,321 lines (1,326 Rust + 1,995 WGSL)
- **Rendering Modes:** 2 (AlphaComposite, AccumulatedSum)
- **GPU Shaders:** 9 compute shaders

### GPU Shader Breakdown
- **Rendering:** 1 shader (145 lines)
- **Gradients:** 1 shader (287 lines)
- **Optimizer Updates:** 6 shaders (1,076 lines total)
- **Loss Functions:** 1 shader (243 lines)

---

*End of Renderer Catalog - 2025-12-07*
