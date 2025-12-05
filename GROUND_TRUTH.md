# LGI Project Ground Truth

**Last Updated**: 2025-12-05
**Purpose**: Single source of verified facts about this project. Nothing speculative.

---

## What This Project Is

A **Gaussian splatting image codec** that represents images as collections of 2D Gaussian primitives instead of pixel grids. The premise is that this representation has advantages for:
- **Scaling**: Gaussians are resolution-independent (render at any size)
- **Compositing**: Layers can be merged mathematically
- **Video potential**: Temporal coherence via Gaussian tracking (not implemented yet)

---

## FUNDAMENTAL PRINCIPLES

### Principle 1: Layers, Never Tiles

**CRITICAL**: This project fundamentally rejects dividing images into 2D geometric regions (tiles, patches, segments).

- **NO tiling** of image geometry
- **NO patch-based processing**
- **NO spatial segmentation**

If a technique doesn't work on an entire image, the solution is **semantic layers** (edges, textures, smooth regions) - NOT geometric subdivision.

**Why**: Tiling introduces boundary artifacts and loses global coherence. Gaussians are global primitives that should represent the entire image holistically.

### Principle 2: Optimization Toolkit, Not Single Method

We develop **multiple well-understood optimization methods**, not crown a single winner.

Adam works because we tuned everything around it - that's circular reasoning. Each optimizer (Adam, L-BFGS, Levenberg-Marquardt, etc.) needs its own hyperparameter tuning, initialization strategy, and understanding of when it excels.

---

## Part 1: Gaussian Dimensional Characteristics (Verified)

### A 2D Gaussian Splat Has These Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| **Position (x, y)** | Vector2<f32> | [0, 1] normalized | Center location in image space |
| **Scale (σ_x, σ_y)** | f32, f32 | > 0 | Standard deviation along principal axes |
| **Rotation (θ)** | f32 | radians | Orientation of principal axes |
| **Color (R, G, B)** | Color4<f32> | [0, 1] | RGB color of the Gaussian |
| **Opacity (α)** | f32 | [0, 1] | Transparency/weight |
| **Optional: weight** | Option<f32> | > 0 | For progressive rendering priority |

**Total intrinsic parameters**: 9 (position: 2, shape: 3, color: 3, opacity: 1)

### Shape Parameterizations (4 Equivalent Representations)

The covariance matrix can be represented multiple ways:

1. **Euler** (most intuitive): `(σ_x, σ_y, θ)` - scales and rotation
   - Best for initialization and human understanding
   - Used in our encoder

2. **Cholesky**: `L` where `Σ = L·Lᵀ` - lower triangular factor
   - Numerically stable, guarantees positive definite
   - Used in research literature (log-Cholesky)

3. **LogRadius**: `(log_r, eccentricity, θ)` - good for compression
   - Better dynamic range for quantization
   - Specified in format, not heavily used yet

4. **InverseCovariance**: `Σ⁻¹` directly - fastest for rendering
   - No matrix inversion needed at render time
   - GPU shader format

**Code reference**: `packages/lgi-rs/lgi-math/src/parameterization.rs`

### How Gaussians Combine to Form Images

Unlike pixels (discrete, independent), Gaussians **overlap and blend**:

```
pixel_color(x, y) = Σᵢ (colorᵢ × αᵢ × G(x, y | μᵢ, Σᵢ))
```

Where `G(x, y | μ, Σ)` is the Gaussian function:
```
G(x, y) = exp(-0.5 × (p - μ)ᵀ × Σ⁻¹ × (p - μ))
```

This is **alpha compositing** - each Gaussian contributes to pixels within its spatial extent.

---

## Part 2: What's Actually Working (Validated)

### Core Codec Quality

**Benchmark results** (128×128 synthetic images):

| Method | Sharp Edges | Complex Patterns | Average Gain |
|--------|-------------|------------------|--------------|
| Baseline | 14.67 dB | 15.50 dB | - |
| Adam Optimizer | 24.36 dB | 21.26 dB | **+7.7 dB** |
| Error-Driven | 24.26 dB | 21.94 dB | **+8.0 dB** |
| GPU | 24.31 dB | 21.96 dB | **+8.0 dB** |

**Validated on**: Synthetic gradients, Kodak dataset (24 images), 67 real photos

### Performance (Validated)

| Operation | Performance | Platform |
|-----------|-------------|----------|
| GPU Decode | 1,176 FPS @ 1080p | RTX 4060 via wgpu/Vulkan |
| CPU Decode | ~60 FPS @ 1080p | Modern x86 |
| Encode | 1.4s for 128×128 | CPU (gradient computation bottleneck) |

**Cross-platform**: Works on Windows, Linux, macOS via wgpu (not CUDA-dependent)

### Production-Ready Methods

These are the recommended encoder functions:

1. **`encode_error_driven_adam()`** - General purpose, +8 dB
2. **`encode_error_driven_gpu()`** - For large images, GPU-accelerated
3. **`encode_error_driven_gpu_msssim()`** - Highest perceptual quality
4. **`encode_error_driven_adam_isotropic()`** - New, +1.87 dB for edges

**Code reference**: `packages/lgi-rs/lgi-encoder-v2/src/lib.rs`

---

## Part 3: One Validated Quantum Finding

### Isotropic Edges Work Better

**Discovery source**: Quantum kernel clustering (Q1 research, December 2025)

**Finding**: High-quality Gaussians at edges are **isotropic** (σ_x ≈ σ_y), not elongated

**Classical validation** on 5 Kodak images:

| Image | Anisotropic | Isotropic | Improvement |
|-------|-------------|-----------|-------------|
| kodim03 | 8.17 dB | 10.94 dB | +2.77 dB |
| kodim05 | 9.55 dB | 10.52 dB | +0.97 dB |
| kodim08 | 5.50 dB | 6.89 dB | +1.39 dB |
| kodim15 | 7.73 dB | 10.04 dB | +2.31 dB |
| kodim23 | 7.05 dB | 8.97 dB | +1.92 dB |
| **Average** | **7.60 dB** | **9.47 dB** | **+1.87 dB** |

**Win rate**: 100% (5/5 images)

**Why this matters**: Contradicts intuition that edges need elongated Gaussians. Small isotropic Gaussians provide precision in both directions along the edge.

**Implementation**: `encode_error_driven_adam_isotropic()` in lib.rs:744-888

**Status**: Validated but not yet the default encoder. Needs full benchmark.

---

## Part 4: How Optimization Works (Verified)

### The Optimization Loop

1. **Initialize** Gaussians (grid, k-means, SLIC, gradient peaks, etc.)
2. **Render** current Gaussians to pixel image
3. **Compute loss** (MSE or MS-SSIM vs target)
4. **Compute gradients** w.r.t. each Gaussian parameter
5. **Update parameters** via optimizer
6. **Repeat** until convergence or iteration limit

### Optimization Toolkit (Multiple Methods)

**PRINCIPLE**: Adam is what we've learned to use, but it is NOT "the answer." We need a toolkit of well-understood optimization methods, each tuned for its strengths.

| Optimizer | File | Status | Best For |
|-----------|------|--------|----------|
| **Adam** | `adam_optimizer.rs` | Working, tuned | General purpose, noisy gradients |
| **L-BFGS** | `optimizer_lbfgs.rs` | Implemented | Precise convergence, smooth problems |
| **Gradient Descent** | `optimizer_v2.rs` | Working | Simple baseline, MS-SSIM option |
| **Perceptual** | `optimizer_v3_perceptual.rs` | Working | Edge-weighted quality |
| **Levenberg-Marquardt** | NOT IMPLEMENTED | Needed | Least-squares (our exact problem!) |

**Research needed**: Each optimizer requires its own:
- Hyperparameter tuning
- Initialization strategy
- Understanding of when it excels vs fails

**Adam is NOT mathematically optimal** - it has known issues with sharp minima and can fail on well-conditioned problems. L-BFGS and Levenberg-Marquardt may be superior for our least-squares objective.

### Initialization Strategies (Implemented)

| Strategy | Code | Best For |
|----------|------|----------|
| Grid | `initialize_gaussians()` | Uniform coverage |
| K-means | `initialize_kmeans()` | Color-based clustering |
| SLIC | `initialize_slic()` | Superpixel regions |
| Gradient Peaks | `initialize_gradient_peaks()` | Edge-focused |
| GDGS | `initialize_gdgs()` | Laplacian peaks |

---

## Part 5: What's NOT Validated (Speculative/Incomplete)

### Quantum Research Status

| Question | Status | Conclusion |
|----------|--------|------------|
| Q1: Channel Discovery | Completed | Classical RBF 60× better than quantum |
| Q2: Per-channel Optimizers | Incomplete | Experiment design flawed |
| Q3: Quantum Iteration | Not started | Speculative |
| Q4: Alternative Basis Functions | Not started | Speculative |

**Honest assessment**: Quantum found ONE useful thing (isotropic edges). Classical methods are better for clustering Gaussian configurations.

### Feature Separation (Layers)

**Status**: NOT TESTED

**Your hypothesis**: Separate image features into layers, optimize each with best technique, merge later.

**Reality**: No experiments have validated this. The layer-based approach is promising but unproven.

### Video (LGIV)

**Status**: Specification exists (1,027 lines), NO implementation

**Code reference**: `packages/lgi-rs/docs/specifications/LGIV_VIDEO_FORMAT.md`

---

## Part 6: Data Assets

### Validated Datasets

| Dataset | Location | Size | Description |
|---------|----------|------|-------------|
| Kodak Gaussians | `quantum_research/kodak_gaussian_data/` | 84MB | 682,059 real Gaussian configs from encoding |
| Processed Features | `quantum_research/kodak_gaussians_quantum_ready.pkl` | 117KB | 1,000 samples, 6D features |
| Enhanced Features | `quantum_research/kodak_gaussians_quantum_ready_enhanced.pkl` | 243KB | 1,483 samples, 10D features |

### Test Images

| Set | Location | Count | Description |
|-----|----------|-------|-------------|
| Kodak | Downloaded | 24 | 768×512, industry standard |
| Real Photos | `lgi-benchmarks` | 67 | 4K resolution, real-world content |

---

## Part 7: Architecture Summary

```
lgi-rs/
├── lgi-math/         # Gaussian primitives, parameterizations
├── lgi-core/         # Rendering, structure tensor, geodesic EDT
├── lgi-encoder-v2/   # Main optimizer (Adam, error-driven, etc.)
├── lgi-gpu/          # wgpu GPU acceleration
├── lgi-format/       # File I/O, serialization
├── lgi-cli/          # Command-line tools
├── lgi-viewer/       # GUI viewer
└── lgi-pyramid/      # Multi-resolution
```

**Total Rust code**: ~5,700 lines production
**Tests**: 65/65 passing

---

## Part 8: Open Questions (Honest Assessment)

### Questions We Can Answer

1. **What defines a Gaussian?** → 9 parameters (position, shape, color, opacity)
2. **How do we optimize?** → Adam with error-driven placement, ~200 iterations
3. **What quality do we get?** → 21-24 dB on test images, +8 dB over baseline
4. **Is isotropic better for edges?** → YES, validated +1.87 dB

### Questions We Cannot Yet Answer

1. **Optimal N for a given image?** → Entropy-based heuristics exist but not tuned
2. **Can layers be optimized separately?** → Not tested
3. **Can quantum help with optimization?** → Unknown, Q2-Q4 abandoned
4. **What's the compression ratio vs JPEG?** → Not properly benchmarked
5. **Does this work well on video?** → No implementation exists

---

## What To Do Next

### Immediate (Before More Research)

1. **Adopt isotropic edges** as default encoder
2. **Run full benchmark** on 24 Kodak + 67 real photos
3. **Commit untracked code** (gaussian_logger, isotropic method)
4. **Archive quantum research docs** - done, keep only actionable findings

### Short-term

1. **GPU gradient computation** - 1500× speedup potential
2. **Compression testing** - actual file sizes vs JPEG/WebP
3. **Format finalization** - quantization profiles

### Medium-term (If Pursuing Quantum)

1. **Abandon Q1 quantum clustering** - classical RBF is better
2. **Q2 needs redesign** - current experiment is flawed
3. **Q3-Q4 are speculative** - decide if worth pursuing

---

## References

- **Core math**: `lgi-math/src/gaussian.rs`, `parameterization.rs`
- **Encoder**: `lgi-encoder-v2/src/lib.rs` (1,773 lines)
- **History**: `docs/research/PROJECT_HISTORY.md` (1,109 lines)
- **Isotropic finding**: `packages/lgi-rs/BREAKTHROUGH_ISOTROPIC_EDGES.md`

---

*This document contains only verified facts. Speculation belongs elsewhere.*
