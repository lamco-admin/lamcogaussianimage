# LGI Toolkit Modules - Component Reference

**Purpose:** Document each reusable module in lgi-core for building encoding strategies.
**Approach:** Toolkit of composable components, NOT a monolithic "optimal" configuration.

---

## Core Dependency Chain

```
structure_tensor  <─────────────────────────────┐
      │                                         │
      ├── frangi_vesselness (tubular detection) │
      │                                         │
      ├── text_stroke_detection (text strokes)  │
      │                                         │
      └── geodesic_edt (anti-bleeding sigma)    │
                                                │
saliency_detection (gradient-based importance) ─┘
```

---

## 1. Structure Tensor (`structure_tensor.rs`)

**Purpose:** Compute local edge orientation and strength at every pixel.

**Key Outputs:**
- `coherence`: 0-1 measure of edge strength (1 = strong linear edge)
- `eigenvector_minor`: Edge TANGENT direction (align Gaussians along this)
- `eigenvector_major`: Edge NORMAL direction (make Gaussians thin along this)
- `orientation_angle()`: Rotation angle in radians

**API:**
```rust
let tensor_field = StructureTensorField::compute(&image, sigma_gradient=1.2, sigma_smooth=2.5)?;
let tensor = tensor_field.get(x, y);
tensor.coherence  // 0-1
tensor.is_edge(threshold=0.7)  // bool
tensor.orientation_angle()  // radians
```

**Gaussian Application:**
- Set Gaussian rotation = `tensor.eigenvector_minor.y.atan2(tensor.eigenvector_minor.x)`
- Set anisotropy = `scale_major / scale_minor` proportional to coherence

---

## 2. Frangi Vesselness (`frangi_vesselness.rs`)

**Purpose:** Detect tubular structures (text strokes, hair, wood grain, blood vessels).

**Key Output:**
- `vesselness`: 0-1 map where 1 = strong tubular structure

**API:**
```rust
let filter = FrangiFilter::default();  // alpha=0.5, beta=15.0, c=500.0
let vesselness = filter.compute_vesselness(&tensor_field);
```

**Gaussian Application:**
- High vesselness regions need thin, elongated Gaussians
- Combine with structure tensor for orientation

---

## 3. Saliency Detection (`saliency_detection.rs`)

**Purpose:** Compute importance map for content-adaptive encoding.

**Key Output:**
- `saliency_map`: Per-pixel importance (gradient magnitude)

**API:**
```rust
let detector = SaliencyDetector::default();
let saliency_map = detector.compute_saliency_map(&image);
let salient_regions = detector.get_salient_regions(&saliency_map);
```

**Gaussian Application:**
- Place MORE Gaussians in high-saliency regions
- Use SMALLER Gaussians in high-saliency regions
- Use LARGER Gaussians in low-saliency (smooth) regions

---

## 4. Geodesic EDT (`geodesic_edt.rs`)

**Purpose:** Compute edge-respecting distance for anti-bleeding sigma clamping.

**Key Feature:** `clamp_sigma(x, y, desired_σ) → min(desired_σ, 0.5 + 0.3×d)`

**API:**
```rust
let gedt = GeodesicEDT::compute(&image, &tensor_field, edge_threshold=0.7, edge_penalty=50.0)?;
let safe_sigma = gedt.clamp_sigma(x, y, desired_sigma);
```

**Gaussian Application:**
- CRITICAL for preventing color bleeding across object boundaries
- Use clamped sigma for all Gaussians near edges
- Especially important for: text edges, face boundaries, object silhouettes

---

## 5. Text Stroke Detection (`text_stroke_detection.rs`)

**Purpose:** Detect text strokes for special handling with anisotropic Gaussians.

**Key Feature:** `create_stroke_preset()` creates σ⊥=0.3, σ∥=5 Gaussian

**API:**
```rust
let detector = TextStrokeDetector::default();
let is_stroke = detector.detect_strokes(&image, &tensor_field);
let preset = detector.create_stroke_preset(position, tangent_angle, color);
```

**Gaussian Application:**
- Strokes need EXTREME anisotropy (thin across, long along)
- Use preset ratio σ⊥:σ∥ = 0.3:5 = 1:16.7

---

## Module Composition Patterns

### Pattern 1: Edge-Aware Initialization
```rust
// 1. Compute structure tensor
let tensor_field = StructureTensorField::compute(&image, 1.2, 2.5)?;

// 2. For each Gaussian position
let tensor = tensor_field.get(x, y);
if tensor.is_edge(0.7) {
    // Align with edge
    rotation = tensor.orientation_angle();
    scale_x = base_scale * (1.0 + tensor.coherence);  // Long along edge
    scale_y = base_scale / (1.0 + tensor.coherence);  // Thin across edge
} else {
    // Isotropic in flat regions
    scale_x = base_scale;
    scale_y = base_scale;
    rotation = 0.0;
}
```

### Pattern 2: Anti-Bleeding Sigma Clamping
```rust
// 1. Compute geodesic EDT
let gedt = GeodesicEDT::compute(&image, &tensor_field, 0.7, 50.0)?;

// 2. For each Gaussian
let safe_scale_x = gedt.clamp_sigma(x, y, desired_scale_x);
let safe_scale_y = gedt.clamp_sigma(x, y, desired_scale_y);
```

### Pattern 3: Content-Adaptive Density
```rust
// 1. Compute saliency
let saliency_map = SaliencyDetector::default().compute_saliency_map(&image);

// 2. Allocate Gaussians proportional to saliency
for (i, &s) in saliency_map.iter().enumerate() {
    gaussian_density[i] = base_density * (1.0 + s * importance_factor);
}
```

### Pattern 4: Text-Specific Handling
```rust
// 1. Detect text strokes
let is_stroke = TextStrokeDetector::default().detect_strokes(&image, &tensor_field);

// 2. Use stroke presets for text regions
for (i, &stroke) in is_stroke.iter().enumerate() {
    if stroke {
        let tangent = tensor_field.get(x, y).orientation_angle() + PI/2;
        gaussians.push(detector.create_stroke_preset(position, tangent, color));
    }
}
```

---

## Testing Methodology

Each module should be tested IN ISOLATION:
1. Apply module to test images with known characteristics
2. Visualize module output (coherence map, vesselness map, etc.)
3. Measure effect on final PSNR when module is enabled vs disabled
4. Document which image types benefit most

**Do NOT combine all modules and claim "optimal" - test each contribution separately.**

---

## Module Status

| Module | Implemented | Tested | Documented | Quality Impact |
|--------|-------------|--------|------------|----------------|
| structure_tensor | ✓ | Unit tests | ✓ | Unknown |
| frangi_vesselness | ✓ | - | ✓ | Unknown |
| saliency_detection | ✓ | - | ✓ | Unknown |
| geodesic_edt | ✓ | Unit tests | ✓ | Unknown |
| text_stroke_detection | ✓ | - | ✓ | Unknown |
| perceptual_weighting | ✓ | - | - | Unknown |
| content_detection | ✓ | - | - | Unknown |

*"Unknown" = needs isolated testing to measure individual contribution*
