# Session Handover: Toolkit Research Framework
**Date:** 2025-12-07
**Focus:** Building composable toolkit components, NOT finding a single optimal configuration

---

## CRITICAL CONTEXT FOR NEXT SESSION

### Strict Constraint (NON-NEGOTIABLE)
**No geometric tiling/segmentation of images.** The image must remain whole. Semantic isolation by feature type and technique layering IS acceptable.

### Paradigm Shift: TOOLKIT vs MONOLITH
The goal is **NOT** to find a single "optimal" monolithic encoder configuration. Instead, we are building a **TOOLKIT of composable components** that can be selectively applied based on image content:

- Each module (structure tensor, geodesic EDT, saliency, etc.) is a **standalone tool**
- Tests should measure **isolated contribution** of each tool
- Tools can be **combined** based on detected image characteristics
- The encoder becomes a **pipeline** that selects appropriate tools per image region

### Critical Insight: Initialization vs Optimization-Time Effects
**Adam optimizer washes out initialization differences** - with 200 iterations and aggressive learning rates, initial parameter differences often converge to similar final states. This means:

1. **Initialization-based tests are less meaningful with Adam** - they measure "what Adam can recover from" rather than toolkit value
2. **Real toolkit value lies in:**
   - Weighted loss during optimization (edge regions get 3× gradient weight)
   - Constraint enforcement during updates (sigma clamped to edge distance)
   - Testing with weaker optimizers where initialization matters more

---

## SESSION ACCOMPLISHMENTS

### 1. Isolated Toolkit Tests Created (Initialization-Based)
These test the effect of module-informed initialization with Adam:

| Test File | Module Under Test | What It Measures |
|-----------|------------------|------------------|
| `isolated_structure_tensor_test.rs` | Structure Tensor | Anisotropic init (σ⊥<σ∥) vs isotropic |
| `isolated_geodesic_edt_test.rs` | Geodesic EDT | Scale proportional to edge distance |
| `isolated_saliency_test.rs` | Saliency Detection | More Gaussians in salient regions |
| `isolated_frangi_vesselness_test.rs` | Frangi Vesselness | Extreme anisotropy (1:8) for tubular structures |
| `isolated_text_stroke_test.rs` | Text Stroke Detection | Text preset σ⊥:σ∥ = 1:16.7 |
| `isolated_perceptual_weight_test.rs` | Perceptual Weighting | Edge-weighted placement (CDF sampling) |

**Run command:** `cargo run --release --example isolated_structure_tensor_test -p lgi-encoder-v2`

### 2. Optimization-Time Tests Created (HIGHER VALUE)
These inject module effects DURING optimization, not just at initialization:

| Test File | Module Under Test | Mechanism |
|-----------|------------------|-----------|
| `constraint_geodesic_edt_test.rs` | Geodesic EDT as Constraint | **σ ≤ 0.5 + 0.3×d** enforced every iteration |
| `weighted_loss_perceptual_test.rs` | Perceptual Weighting | **3× gradient weight** for Gaussians on edges |

**These tests use a manual Adam loop with `GpuGradientComputer`**, allowing injection of:
- Per-iteration constraint enforcement (scale clamping)
- Gradient weighting/scaling based on position

**Run commands:**
```bash
cargo run --release --example constraint_geodesic_edt_test -p lgi-encoder-v2
cargo run --release --example weighted_loss_perceptual_test -p lgi-encoder-v2
```

### 3. Bug Fixes Applied
- Added `f32` type annotations to `beta1`, `beta2`, `epsilon` in both optimization-time tests
- Fixed closure borrow issue in `weighted_loss_perceptual_test.rs` (replaced functional iterator with for-loop)

---

## KEY TECHNICAL DETAILS

### GpuGradientComputer API (for custom optimization loops)
```rust
use lgi_gpu::{GpuGradientComputer, GpuGradient};

// Create gradient computer
let gradient_computer = GpuGradientComputer::new(device)?;

// Compute gradients
let gpu_gradients = gradient_computer.compute_gradients(
    device, queue, &gaussians, &rendered, target
)?;

// GpuGradient struct:
// d_position: [f32; 2], d_scale_x: f32, d_scale_y: f32, d_rotation: f32,
// d_color: [f32; 4], d_opacity: f32, _padding: [f32; 2]
```

### Adam Hyperparameters (Standard Working Config)
```rust
let lr_position: f32 = 0.0002;
let lr_color: f32 = 0.02;
let lr_scale: f32 = 0.005;
let lr_opacity: f32 = 0.05;
let lr_rotation: f32 = 0.01;
let beta1: f32 = 0.9;
let beta2: f32 = 0.999;
let epsilon: f32 = 1e-8;
```

### Constraint Enforcement Pattern (Geodesic)
```rust
// After Adam update but before next iteration:
let px = (gaussian.position.x * width as f32) as u32;
let py = (gaussian.position.y * height as f32) as u32;
let image_size = width.max(height) as f32;

// Clamp sigma to edge distance
let pixel_scale = gaussian.shape.scale_x * image_size;
let clamped = gedt.clamp_sigma(px, py, pixel_scale);
gaussian.shape.scale_x = clamped / image_size;
```

### Gradient Weighting Pattern (Perceptual)
```rust
// Apply edge weight to gradients before Adam update:
let idx = (py * width + px) as usize;
let edge_weight = weight_map.get(idx).copied().unwrap_or(1.0);

// Scale gradient by edge importance (3x for edges)
let g = grad.d_position[j] * edge_weight;
// Use weighted gradient in Adam update
```

---

## AVAILABLE TOOLKIT MODULES (in lgi-core)

| Module | File | Purpose | Key Method |
|--------|------|---------|------------|
| Structure Tensor | `structure_tensor.rs` | Edge/coherence detection | `StructureTensorField::compute()` |
| Geodesic EDT | `geodesic_edt.rs` | Edge-aware distance transform | `GeodesicEDT::compute()`, `.clamp_sigma()` |
| Saliency | `saliency.rs` | Attention-important regions | `SaliencyDetector::detect()` |
| Frangi Vesselness | `frangi_vesselness.rs` | Tubular structure detection | `FrangiFilter::compute_vesselness()` |
| Text Stroke | `text_stroke_detection.rs` | Thin stroke detection | `TextStrokeDetector::detect_strokes()` |
| Perceptual Weight | `perceptual_weighting.rs` | Edge importance weights | `PerceptualWeighter::compute_weight_map()` |

---

## NEXT STEPS FOR CONSOLIDATION

### 1. Run All Isolated Tests
```bash
# Initialization-based tests (informative but Adam may wash out)
for t in structure_tensor geodesic_edt saliency frangi_vesselness text_stroke perceptual_weight; do
    cargo run --release --example isolated_${t}_test -p lgi-encoder-v2
done

# Optimization-time tests (HIGHER VALUE)
cargo run --release --example constraint_geodesic_edt_test -p lgi-encoder-v2
cargo run --release --example weighted_loss_perceptual_test -p lgi-encoder-v2
```

### 2. Analyze Results
- Look for **consistent** improvement across multiple Kodak images
- Note **correlation** with image characteristics (edge %, coherence %, etc.)
- Document which modules help which image types

### 3. Prioritize Based on Impact
- **High impact**: Modules showing >+0.1 dB average improvement
- **Content-specific**: Modules that help specific image types (text, tubular, etc.)
- **Low priority**: Modules showing <0.05 dB or inconsistent results

### 4. Design Combined Pipeline
Based on test results, design a pipeline that:
1. Analyzes input image (compute all analysis maps)
2. Selects appropriate tools based on image characteristics
3. Applies selected initialization strategies
4. Uses optimization-time constraints/weights where beneficial

---

## FILES MODIFIED THIS SESSION

| File | Change |
|------|--------|
| `examples/constraint_geodesic_edt_test.rs` | Fixed f32 type annotations |
| `examples/weighted_loss_perceptual_test.rs` | Fixed f32 types + closure borrow |

## FILES CREATED THIS SESSION

| File | Purpose |
|------|---------|
| `examples/isolated_structure_tensor_test.rs` | Structure tensor module test |
| `examples/isolated_geodesic_edt_test.rs` | Geodesic EDT init test |
| `examples/isolated_saliency_test.rs` | Saliency placement test |
| `examples/isolated_frangi_vesselness_test.rs` | Vesselness anisotropy test |
| `examples/isolated_text_stroke_test.rs` | Text stroke preset test |
| `examples/isolated_perceptual_weight_test.rs` | Edge-weighted placement test |
| `examples/constraint_geodesic_edt_test.rs` | **Geodesic constraint during opt** |
| `examples/weighted_loss_perceptual_test.rs` | **Perceptual gradient weighting** |

---

## RESEARCH PHILOSOPHY REMINDER

> "We're building a TOOLKIT, not finding an optimal configuration."

Each test answers a specific question:
- "Does module X help for image type Y?"
- "What is the isolated contribution of module X?"
- "Should module X be applied at init time, opt time, or both?"

The encoder will eventually be a **smart pipeline** that analyzes each image and selects the appropriate tools from this toolkit.

---

## COMPILE STATUS

All test files compile successfully. No runtime testing performed in this session due to consolidation focus.

```bash
# Verify all compile:
cargo build --release --example constraint_geodesic_edt_test -p lgi-encoder-v2  # OK
cargo build --release --example weighted_loss_perceptual_test -p lgi-encoder-v2  # OK
```
