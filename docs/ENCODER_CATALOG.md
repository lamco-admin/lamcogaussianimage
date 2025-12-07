# ENCODER CATALOG - Gaussian Image Codec
## Complete Encoder Entry Points Inventory

**Document Version:** 2.0
**Date:** 2025-12-07
**Repository:** lgi-project
**Source File:** `packages/lgi-rs/lgi-encoder-v2/src/lib.rs` (1,778 lines)
**Total Encoder Functions:** 13

---

## TABLE OF CONTENTS

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Encoder Functions](#encoder-functions)
4. [Supporting Methods](#supporting-methods)
5. [File Location Index](#file-location-index)

---

## OVERVIEW

Encoders are **entry point functions** that combine initialization, optimization, and refinement strategies into complete encoding pipelines. Each encoder represents a different combination of:

- Initialization strategy (how Gaussians are placed)
- Optimizer algorithm (how parameters are updated)
- Loss function (what is minimized)
- Refinement strategy (adaptive densification, pruning, etc.)

### Architecture Flow

```
EncoderV2::new(target_image)
    ↓ Preprocessing
    ├─ Compute structure tensor (edge detection)
    └─ Compute geodesic EDT (distance transform)

EncoderV2::encode_*()
    ↓ Initialization
    ├─ Place initial Gaussians (grid, k-means, gradient peaks, etc.)
    ↓ Optimization Loop
    ├─ Render Gaussians → pixels
    ├─ Compute loss (L2, MS-SSIM, edge-weighted, etc.)
    ├─ Compute gradients
    ├─ Update parameters (Adam, L-BFGS, gradient descent, etc.)
    ├─ Apply constraints (geodesic clamping)
    └─ Refine (add Gaussians at high-error regions)
    ↓ Output
    └─ Return optimized Gaussians
```

---

## ENCODER FUNCTIONS

### 1. ENCODE_AUTO

**Function:** `encode_auto()`
**Line:** 156-160
**Purpose:** Automatic Gaussian count selection and encoding

#### Description
Convenience method that selects Gaussian count based on image entropy, then encodes using structure-tensor-aware initialization.

#### Implementation
- Calls `auto_gaussian_count()` to determine N
- Computes grid size: √N
- Calls `initialize_gaussians()` with computed grid size

#### Returns
Initial Gaussians (no optimization - just initialization)

---

### 2. ENCODE_ERROR_DRIVEN

**Function:** `encode_error_driven(initial_gaussians, max_gaussians)`
**Line:** 343-422
**Purpose:** Error-driven placement with OptimizerV2 (gradient descent)

#### Description
Iterative refinement encoder that adds Gaussians at high-error regions adaptively. Uses simple gradient descent optimizer.

#### Algorithm
1. Initialize with structure-tensor-aware grid (initial_gaussians)
2. For up to 10 passes:
   - Optimize with OptimizerV2 (100 iterations)
   - Apply geodesic EDT clamping (anti-bleeding)
   - Compute error map
   - Find hotspots (top 10% error)
   - Add Gaussians at hotspots with structure-tensor shapes
   - Warmup new Gaussians with LR ramp (0.1× → 1.0×)
3. Stop when: loss < 0.001 OR N ≥ max_gaussians OR no hotspots

#### Components
- **Initialization:** Structure-tensor-aware grid
- **Optimizer:** OptimizerV2 (gradient descent)
- **Renderer:** RendererV2 (CPU)
- **Refinement:** Error-driven Gaussian addition
- **Constraints:** Geodesic EDT clamping

#### Warmup
New Gaussians get proportional warmup iterations (10-50) with LR ramping to prevent instability.

---

### 3. ENCODE_ERROR_DRIVEN_ADAM

**Function:** `encode_error_driven_adam(initial_gaussians, max_gaussians)`
**Line:** 434-571
**Purpose:** Error-driven placement with Adam optimizer

#### Description
Same as `encode_error_driven()` but uses Adam optimizer instead of gradient descent for faster convergence.

#### Algorithm
Identical to `encode_error_driven()` except:
- Uses `AdamOptimizer` instead of `OptimizerV2`
- Per-parameter learning rates
- Momentum and RMSprop buffers

#### Components
- **Initialization:** Structure-tensor-aware grid
- **Optimizer:** AdamOptimizer (per-parameter LR)
- **Renderer:** RendererV2 (CPU)
- **Refinement:** Error-driven Gaussian addition
- **Constraints:** Geodesic EDT clamping

#### Hotspot Addition
Uses structure-tensor-aware shapes:
- Flat regions (coherence < 0.2): Isotropic
- Edge regions: Anisotropic aligned to edge (σ∥ = 4·σ⊥)
- Adaptive gamma: γ = f(N) for coverage

---

### 4. ENCODE_ERROR_DRIVEN_ADAM_WITH_LOGGER

**Function:** `encode_error_driven_adam_with_logger(initial_gaussians, max_gaussians, logger_callback)`
**Line:** 583-761
**Purpose:** Error-driven Adam with data logging for research

#### Description
Same as `encode_error_driven_adam()` but captures Gaussian configurations during optimization via callback function for quantum research data collection.

#### Additional Features
- Callback: `fn(&[Gaussian2D], iteration, psnr, loss)`
- Logs after each optimization pass
- Used for building datasets

---

### 5. ENCODE_ERROR_DRIVEN_ADAM_ISOTROPIC

**Function:** `encode_error_driven_adam_isotropic(initial_gaussians, max_gaussians)`
**Line:** 763-890
**Purpose:** Error-driven Adam with isotropic edge Gaussians (quantum-guided)

#### Description
Variant of error-driven Adam that uses isotropic Gaussians (σ_x = σ_y) even at edges, based on quantum research findings.

#### Key Difference from Standard Error-Driven Adam
At hotspots in edge regions:
- Standard: Anisotropic (σ∥ = 4·σ⊥, aligned to edge)
- Isotropic: Isotropic (σ_x = σ_y = smaller_scale, no rotation)

#### Hotspot Addition Logic (Lines 842-871)
```rust
let sigma_iso_px = if coherence < 0.2 {
    sigma_base_px  // Smooth: standard
} else {
    sigma_base_px / (1.0 + 2.0 * coherence)  // Edges: SMALLER isotropic
};
```

Creates isotropic Gaussian: `Euler::new(sigma, sigma, 0.0)`

#### Research Note (from code comments)
"QUANTUM-GUIDED: Use isotropic for ALL coherence levels"
"Quantum shows: smaller scales work better at edges"

---

### 6. ENCODE_ERROR_DRIVEN_V2

**Function:** `encode_error_driven_v2(initial_gaussians, max_gaussians, use_ms_ssim)`
**Line:** 902-1003
**Purpose:** Error-driven with OptimizerV2 and optional MS-SSIM loss

#### Description
Error-driven refinement using OptimizerV2 with optional perceptual (MS-SSIM) loss function.

#### Parameters
- `use_ms_ssim`: if true, uses MS-SSIM perceptual loss; if false, uses L2 MSE

#### Components
- **Optimizer:** OptimizerV2 with configurable loss
- **Loss:** L2 MSE (default) or MS-SSIM (perceptual)
- **Refinement:** Error-driven adaptive placement

#### Research Purpose (from code comments)
"Q2 RESEARCH: Tests if gradient descent with perceptual loss works better than Adam for certain quantum channels"

---

### 7. ENCODE_ERROR_DRIVEN_V3

**Function:** `encode_error_driven_v3(initial_gaussians, max_gaussians)`
**Line:** 1005-1218
**Purpose:** Error-driven with OptimizerV3 (perceptual + edge-weighted)

#### Description
Uses OptimizerV3 which combines MS-SSIM perceptual loss with edge-weighted gradients (3× weight at detected edges).

#### Components
- **Optimizer:** OptimizerV3 (perceptual optimizer)
- **Loss:** MS-SSIM (α=0.84) + L2 (α=0.16)
- **Gradient Weighting:** 3× at edges via structure tensor
- **Refinement:** Error-driven adaptive placement

---

### 8. ENCODE_ERROR_DRIVEN_GPU

**Function:** `encode_error_driven_gpu(initial_gaussians, max_gaussians)`
**Line:** 1220-1334
**Purpose:** Error-driven with GPU-accelerated rendering

#### Description
Uses OptimizerV2 with GPU renderer for 100-1000× faster rendering during optimization. Falls back to CPU if GPU unavailable.

#### Components
- **Optimizer:** OptimizerV2 with GPU renderer
- **Renderer:** GPU renderer (wgpu/Vulkan) with CPU fallback
- **Rendering Speed:** 100-1000× faster than CPU
- **Refinement:** Error-driven adaptive placement

#### GPU Acceleration
Created with `OptimizerV2::new_with_gpu()` which initializes wgpu device and queue.

---

### 9. ENCODE_ERROR_DRIVEN_GPU_MSSSIM

**Function:** `encode_error_driven_gpu_msssim(initial_gaussians, max_gaussians)`
**Line:** 1345-1433
**Purpose:** Error-driven with GPU rendering and MS-SSIM loss

#### Description
Combines GPU-accelerated rendering with MS-SSIM perceptual loss for both speed and perceptual quality.

#### Components
- **Optimizer:** OptimizerV2 with GPU + MS-SSIM enabled
- **Loss:** MS-SSIM perceptual
- **Renderer:** GPU (100-1000× faster)
- **Refinement:** Error-driven adaptive placement

Created with `OptimizerV2::new_with_gpu_and_ms_ssim()`.

---

### 10. ENCODE_FOR_PSNR

**Function:** `encode_for_psnr(target_psnr, profile)`
**Line:** 1435-1453
**Purpose:** Target-driven encoding for specific PSNR goal

#### Description
Uses rate-distortion optimizer to select optimal Gaussian count for target PSNR, then encodes.

#### Parameters
- `target_psnr`: Desired quality in dB (e.g., 30.0)
- `profile`: Quantization profile (LGIQ-B/S/H/X)

#### Algorithm
1. Create RateDistortionOptimizer with profile
2. Select optimal N for target PSNR using R-D model
3. Call `encode_error_driven_adam()` with optimal N

#### Rate-Distortion Integration
Uses `lgi_core::rate_distortion::RateDistortionOptimizer` to model quality vs Gaussian count relationship.

---

### 11. ENCODE_FOR_PERCEPTUAL_QUALITY

**Function:** `encode_for_perceptual_quality(target_msssim, profile)`
**Line:** 1466-1495
**Purpose:** Target-driven encoding for perceptual quality (MS-SSIM)

#### Description
Similar to `encode_for_psnr()` but targets perceptual quality (MS-SSIM metric) instead of PSNR.

#### Parameters
- `target_msssim`: Desired MS-SSIM score (0.0-1.0, typically 0.95+)
- `profile`: Quantization profile

#### Algorithm
1. Convert MS-SSIM target to approximate PSNR:
   - MS-SSIM ≥ 0.99 → 40 dB
   - MS-SSIM ≥ 0.98 → 35 dB
   - MS-SSIM ≥ 0.95 → 30 dB
   - Else → 25 dB
2. Use R-D optimizer to select N
3. Call `encode_error_driven_gpu_msssim()` for perceptual encoding

---

### 12. ENCODE_FOR_BITRATE

**Function:** `encode_for_bitrate(target_bits, profile)`
**Line:** 1507-1525
**Purpose:** Target-driven encoding for file size constraint

#### Description
Encodes to fit within target bitrate by computing maximum Gaussians that fit in target size.

#### Parameters
- `target_bits`: Target file size in bits
- `profile`: Quantization profile (determines bits per Gaussian)

#### Algorithm
1. Estimate bits per Gaussian from profile
2. Account for header overhead (512 bits = 64 bytes)
3. Compute max N: (target_bits - overhead) / bits_per_gaussian
4. Call `encode_error_driven_adam()` with max N constraint

#### Profile Bits Per Gaussian
- LGIQ-B: 13 bytes = 104 bits
- LGIQ-S: 14 bytes = 112 bits
- LGIQ-H: 20 bytes = 160 bits
- LGIQ-X: 36 bytes = 288 bits

(Plus compression overhead/savings)

---

### 13. ENCODE_WITH_RD_PRUNING

**Function:** `encode_with_rd_pruning(initial_gaussians, lambda)`
**Line:** 1538-1778 (end of file)
**Purpose:** Dense initialization with rate-distortion pruning

#### Description
Starts with dense Gaussian set, optimizes, then prunes Gaussians that don't contribute enough to justify their rate cost.

#### Parameters
- `initial_gaussians`: Starting count (dense)
- `lambda`: R-D tradeoff (higher = prefer smaller files)

#### Algorithm
1. Initialize dense grid
2. Optimize with Adam (200 iterations)
3. Apply geodesic clamping
4. Render and compute per-pixel error
5. Estimate each Gaussian's contribution to error reduction
6. Prune Gaussians where: rate_cost > lambda × distortion_benefit
7. Return pruned set

#### Rate-Distortion Criterion
For each Gaussian:
```
contribution = reduction in total error if removed
cost = bits required to encode this Gaussian
keep if: contribution / cost > 1/lambda
```

---

## SUPPORTING METHODS

### Preprocessing Methods

#### new()
**Line:** 56-75
**Purpose:** Create encoder with preprocessed image analysis

Computes:
- Structure tensor (σ_gradient=1.2, σ_smooth=1.0)
- Geodesic EDT (coherence_threshold=0.7, edge_penalty=50.0)

---

### Initialization Methods

#### initialize_gaussians(grid_size)
**Line:** 85-86
**Purpose:** Structure-tensor-aware grid initialization

Creates √N × √N grid with:
- Positions at grid points
- Colors sampled from target
- Anisotropic shapes aligned to structure tensor
- Scales clamped by geodesic EDT

#### initialize_gaussians_guided(grid_size)
**Line:** 90-92
**Purpose:** Grid initialization with guided filter colors

Same as `initialize_gaussians()` but uses guided filter for better color initialization.

#### initialize_gdgs(target_n)
**Line:** 168-176
**Purpose:** GDGS (Gradient Domain Gaussian Splatting) initialization

Places Gaussians at Laplacian peaks (edges, corners). Sparse initialization strategy.

#### initialize_gradient_peaks(target_n)
**Line:** 185-195
**Purpose:** Gradient peak placement (80% peaks, 20% background)

Hybrid:
- 80% at gradient maxima
- 20% on background grid
Better than GDGS (doesn't neglect smooth regions)

---

### Gaussian Count Methods

#### auto_gaussian_count()
**Line:** 98-102
**Purpose:** Entropy-based automatic N selection

Uses `lgi_core::entropy::adaptive_gaussian_count()` to analyze image complexity.

#### hybrid_gaussian_count()
**Line:** 116-151
**Purpose:** Combined entropy + gradient count selection

Formula:
- Base: entropy-based N
- Gradient factor: mean gradient magnitude normalized to [0,1]
- Hybrid: 60% entropy + 40% gradient
- Result: `N_hybrid = N_entropy × (0.6 + 0.4 × gradient_factor)`

Range: 50 to 50,000 Gaussians

---

### Utility Methods

#### adaptive_gamma(n)
**Line:** (static method, line in file)
**Purpose:** Compute coverage parameter γ based on Gaussian count

Coverage: σ = γ · √(area/N)

#### apply_geodesic_clamping()
**Line:** (instance method, line in file)
**Purpose:** Clamp Gaussian scales to not exceed edge distance

For each Gaussian:
```rust
max_sigma = 0.8 × geodesic_distance_to_nearest_edge
sigma = min(sigma, max_sigma)
```

Prevents bleeding across edges.

#### compute_error_map()
**Line:** (instance method, line in file)
**Purpose:** Compute per-pixel error between render and target

Returns: Vec<f32> of pixel-wise errors

#### find_hotspots()
**Line:** (instance method, line in file)
**Purpose:** Find high-error locations above percentile threshold

Returns: Vec<(x, y, error)> sorted by error descending

---

## ENCODER COMPARISON TABLE

| Encoder | Initialization | Optimizer | Loss | Rendering | Refinement |
|---------|---------------|-----------|------|-----------|------------|
| encode_auto | Grid (auto N) | None | - | - | No |
| encode_error_driven | ST grid | OptimizerV2 | L2 | CPU | Error-driven |
| encode_error_driven_adam | ST grid | Adam | L2 | CPU | Error-driven |
| encode_error_driven_adam_with_logger | ST grid | Adam | L2 | CPU | Error-driven + logging |
| encode_error_driven_adam_isotropic | ST grid | Adam | L2 | CPU | Error-driven (isotropic edges) |
| encode_error_driven_v2 | ST grid | OptimizerV2 | L2 or MS-SSIM | CPU | Error-driven |
| encode_error_driven_v3 | ST grid | OptimizerV3 | MS-SSIM + edge-weighted | CPU | Error-driven |
| encode_error_driven_gpu | ST grid | OptimizerV2 | L2 | GPU | Error-driven |
| encode_error_driven_gpu_msssim | ST grid | OptimizerV2 | MS-SSIM | GPU | Error-driven |
| encode_for_psnr | (delegates) | (delegates) | (delegates) | (delegates) | Via R-D selection |
| encode_for_perceptual_quality | (delegates) | (delegates) | MS-SSIM | GPU | Via R-D selection |
| encode_for_bitrate | (delegates) | (delegates) | (delegates) | (delegates) | Via size constraint |
| encode_with_rd_pruning | ST grid | Adam | L2 | CPU | R-D pruning |

**ST = Structure Tensor**

---

## PARAMETER DIMENSIONS

All encoders optimize these Gaussian parameters:

| Parameter | Dimensions | Range | Parameterization |
|-----------|------------|-------|------------------|
| Position | 2 (x, y) | [0, 1] normalized | Euler |
| Scale | 2 (σ_x, σ_y) | > 0 | Euler |
| Rotation | 1 (θ) | radians | Euler |
| Color | 3 (R, G, B) | [0, 1] | Linear RGB |
| Opacity | 1 (α) | [0, 1] | Linear |

**Total:** 9 parameters per Gaussian

---

## FILE LOCATION INDEX

### Primary Source
```
packages/lgi-rs/lgi-encoder-v2/src/
└── lib.rs (1,778 lines)
    ├── EncoderV2 struct (Lines 39-48)
    ├── new() - Lines 56-75
    ├── Initialization methods - Lines 85-340
    ├── encode_auto() - Lines 156-160
    ├── encode_error_driven() - Lines 343-422
    ├── encode_error_driven_adam() - Lines 434-571
    ├── encode_error_driven_adam_with_logger() - Lines 583-761
    ├── encode_error_driven_adam_isotropic() - Lines 763-890
    ├── encode_error_driven_v2() - Lines 902-1003
    ├── encode_error_driven_v3() - Lines 1005-1218
    ├── encode_error_driven_gpu() - Lines 1220-1334
    ├── encode_error_driven_gpu_msssim() - Lines 1345-1433
    ├── encode_for_psnr() - Lines 1435-1453
    ├── encode_for_perceptual_quality() - Lines 1466-1495
    ├── encode_for_bitrate() - Lines 1507-1525
    └── encode_with_rd_pruning() - Lines 1538-1778
```

### Dependencies

**Optimizers:**
- `adam_optimizer.rs` - Lines 436, 446, 776, 1551
- `optimizer_v2.rs` - Lines 345, 355, 904, 1221
- `optimizer_v3_perceptual.rs` - Line 1005+

**Renderers:**
- `renderer_v2.rs` - Used by all CPU encoders
- GPU renderer - Via OptimizerV2::new_with_gpu()

**Analysis Modules:**
- `lgi_core::StructureTensorField` - Lines 44, 62
- `lgi_core::geodesic_edt::GeodesicEDT` - Lines 46, 66
- `lgi_core::entropy` - Lines 99, 118
- `lgi_core::rate_distortion` - Lines 1436, 1467, 1508, 1539

---

## SUMMARY STATISTICS

### Implementation Count
- **Total Encoder Functions:** 13
- **Total Lines:** 1,778
- **Initialization Strategies Used:** 2 (ST grid, GDGS)
- **Optimizers Used:** 3 (Adam, OptimizerV2, OptimizerV3)
- **Renderers Used:** 2 (RendererV2 CPU, GPU)
- **Loss Functions Used:** 3 (L2, MS-SSIM, Edge-weighted)
- **Refinement Strategies:** 2 (Error-driven, R-D pruning)

### Encoder Categories

**By Refinement Strategy:**
- Error-driven adaptive: 8 encoders
- R-D optimization: 3 encoders
- No refinement: 1 encoder (auto)
- Pruning: 1 encoder

**By Optimizer:**
- Adam: 4 encoders
- OptimizerV2: 4 encoders
- OptimizerV3: 1 encoder
- Delegated/R-D selected: 3 encoders

**By Rendering:**
- CPU: 8 encoders
- GPU: 2 encoders
- Delegated: 3 encoders

---

*End of Encoder Catalog - 2025-12-07*
