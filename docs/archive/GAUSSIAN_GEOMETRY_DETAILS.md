# Gaussian Geometry and Parameter Details

**Baseline 1 Configuration**
**Last Updated:** 2025-11-15

---

## Gaussian Representation

### 2D Elliptical Gaussian

Each Gaussian is defined by a **2D elliptical Gaussian** with the following parameters:

```
G(x, y) = opacity × exp(-0.5 × d^T × Σ^(-1) × d) × color
```

Where:
- `d = [x - position_x, y - position_y]` (distance from center)
- `Σ` is the covariance matrix (defined by scale and rotation)
- Blending uses standard alpha compositing

### Gaussian Parameters (9 total per Gaussian)

#### 1. Position (2 parameters)
- **position.x**: X coordinate of Gaussian center
- **position.y**: Y coordinate of Gaussian center
- **Range:** [0.0, 1.0] (normalized image coordinates)
- **Initialization:** Random uniform in [0.0, 1.0]
- **Constraint:** Clamped to [0.0, 1.0] after each optimization step

#### 2. Shape/Geometry (3 parameters)
- **scale_x**: Half-width of Gaussian ellipse along local X axis
- **scale_y**: Half-width of Gaussian ellipse along local Y axis
- **rotation**: Rotation angle of ellipse (radians)
- **Parameterization:** Euler (explicit scale_x, scale_y, rotation)
- **Initialization:**
  - `scale_x = scale_y = random uniform in [0.01, 0.1]` (isotropic)
  - `rotation = 0.0` (initially axis-aligned)
- **Constraints:**
  - `scale_x`: Clamped to [0.001, 0.5] (not too tiny, not too huge)
  - `scale_y`: Clamped to [0.001, 0.5]
  - `rotation`: No constraints (can rotate freely)

**Note:** Initial Gaussians are **isotropic** (scale_x = scale_y), meaning they start as circles, not ellipses. The optimizer can make them elliptical during training.

#### 3. Color (3 parameters)
- **color.r**: Red channel intensity
- **color.g**: Green channel intensity
- **color.b**: Blue channel intensity
- **Range:** [0.0, 1.0] each
- **Initialization:** Random uniform in [0.0, 1.0] per channel
- **Constraint:** Clamped to [0.0, 1.0] after each optimization step

#### 4. Opacity (1 parameter)
- **opacity**: Alpha value for blending
- **Range:** [0.0, 1.0]
- **Initialization:** 1.0 (fully opaque)
- **Optimization:** **FIXED at 1.0** (not optimized in Baseline 1)

### Why Euler Parameterization?

The code uses **Euler parameterization** which explicitly stores scale_x, scale_y, and rotation.

**Alternative:** Could use covariance matrix directly, but Euler is:
- ✅ More interpretable (can see ellipse dimensions directly)
- ✅ Easier to constrain (can clamp scales independently)
- ✅ Simpler initialization (isotropic = scale_x = scale_y)
- ❌ Requires rotation parameter (1 extra vs symmetric matrix)

---

## Gaussian Coverage: N-to-Pixels Ratios

### Image: 64×64 = 4096 pixels total

| N | Pixels per Gaussian | Gaussian Coverage | Gaussian Diameter* | Status |
|---|---------------------|-------------------|-------------------|--------|
| **20** | 204.8 | 0.488% | ~0.078 (5.0 px) | ✅ Complete |
| **25** | 163.8 | 0.611% | ~0.070 (4.5 px) | ✅ Complete |
| **30** | 136.5 | 0.732% | ~0.064 (4.1 px) | ✅ Complete |
| **35** | 117.0 | 0.854% | ~0.059 (3.8 px) | ✅ Complete |
| **40** | 102.4 | 0.976% | ~0.055 (3.5 px) | ✅ Complete |
| **45** | 91.0 | 1.099% | ~0.052 (3.3 px) | ✅ Complete |
| **50** | 81.9 | 1.221% | ~0.049 (3.1 px) | ✅ Complete |
| **55** | 74.5 | 1.343% | ~0.047 (3.0 px) | ✅ Complete |
| **60** | 68.3 | 1.465% | ~0.045 (2.9 px) | ✅ Complete |
| **65** | 63.0 | 1.587% | ~0.043 (2.8 px) | ✅ Complete |
| **70** | 58.5 | 1.709% | ~0.042 (2.7 px) | ✅ Complete |
| **80** | 51.2 | 1.953% | ~0.039 (2.5 px) | ✅ Complete |
| **100** | 41.0 | 2.441% | ~0.035 (2.2 px) | 🟡 Running |

*Gaussian diameter assumes isotropic Gaussian with scale ≈ sqrt(pixel_coverage_area). Actual Gaussians become elliptical during training.

### Interpretation

**Pixels per Gaussian:**
- N=20: Each Gaussian "responsible for" ~205 pixels (very sparse)
- N=50: Each Gaussian "responsible for" ~82 pixels (moderate)
- N=100: Each Gaussian "responsible for" ~41 pixels (dense)

**Gaussian Coverage %:**
- Percentage of image area each Gaussian would cover if uniformly distributed
- Assumes Gaussian footprint ≈ 4σ × 4σ (captures ~95% of Gaussian mass)
- Lower N = larger individual Gaussians needed to cover image

**Practical Meaning:**
- **N=20-30:** Very sparse, large Gaussians, limited detail capability
- **N=40-60:** Moderate density, can represent moderate detail
- **N=70-100:** Dense coverage, can represent fine detail (if needed)

---

## N-to-Pixels Ratio Analysis

### For 64×64 Images (4096 pixels)

**N/Pixels ratio tested:**
- Minimum: N=20, ratio = 0.00488 (1 Gaussian per 205 pixels)
- Maximum: N=100, ratio = 0.02441 (1 Gaussian per 41 pixels)
- **Range:** 0.005 to 0.024 (0.5% to 2.4%)

### Hypothesis About Optimal Ratio

Based on the diminishing returns observed (N=45-55 region):
- **Optimal range appears to be:** N/pixels ≈ 0.011-0.013 (1.1%-1.3%)
- **For 64×64:** N ≈ 45-55
- **Translates to:** ~75-91 pixels per Gaussian

**This is specific to this test image!** Different images (more complex, more detail) may have different optimal ratios.

### Extrapolation to Other Image Sizes (Hypothetical)

**IF the N/pixels ratio holds constant** (big IF, needs testing):
- **32×32 (1024 pixels):** Optimal N ≈ 11-13
- **128×128 (16384 pixels):** Optimal N ≈ 180-213
- **256×256 (65536 pixels):** Optimal N ≈ 720-850

**Caution:** These are just proportional extrapolations. Actual optimal N likely depends on:
- Image complexity (amount of detail to represent)
- Image content (smooth vs high-frequency)
- Desired quality level

---

## Gaussian Initialization Details

### Random Initialization (Baseline 1)

```rust
fn init_random(n: usize, seed: u64) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| {
        Gaussian2D::new(
            Vector2::new(rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)),  // Position
            Euler::isotropic(rng.gen_range(0.01..0.1)),                      // Scale (isotropic)
            Color4::new(
                rng.gen_range(0.0..1.0),  // Red
                rng.gen_range(0.0..1.0),  // Green
                rng.gen_range(0.0..1.0),  // Blue
                1.0                       // Alpha (always 1.0)
            ),
            1.0,  // Opacity (always 1.0)
        )
    }).collect()
}
```

### Initialization Strategy

**Position:**
- Uniform random across image [0.0, 1.0] × [0.0, 1.0]
- No spatial structure (not grid-based)
- Gaussians can start anywhere, including overlapping

**Scale:**
- Isotropic (circular, not elliptical initially)
- Random size between 0.01 and 0.1 (1% to 10% of image dimension)
- In pixel terms: 0.64px to 6.4px diameter for 64×64 image
- Allows variety of Gaussian sizes from start

**Rotation:**
- Starts at 0.0 (axis-aligned)
- `Euler::isotropic()` creates circular Gaussian with no rotation
- Optimizer will add rotation during training if needed

**Color:**
- Each channel random [0.0, 1.0]
- No bias toward any particular color
- Allows full color space exploration

**Opacity:**
- Fixed at 1.0 (fully opaque)
- Not optimized in Baseline 1

**Random Seed:**
- Fixed at 42 for reproducibility
- Same N always gives same initialization
- Allows fair comparison across experiments

---

## Parameter Constraints (Hard Limits)

### Position Constraints
```rust
gaussians[i].position.x = gaussians[i].position.x.clamp(0.0, 1.0);
gaussians[i].position.y = gaussians[i].position.y.clamp(0.0, 1.0);
```

**Why:** Keep Gaussians within image bounds
**Effect:** Gaussians near edges are "pushed back" into image

### Color Constraints
```rust
gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
gaussians[i].color.g = gaussians[i].color.g.clamp(0.0, 1.0);
gaussians[i].color.b = gaussians[i].color.b.clamp(0.0, 1.0);
```

**Why:** Valid RGB range
**Effect:** Prevents "super-bright" or negative colors
**Note:** Could allow HDR range [0.0, ∞) but keeps it standard RGB for now

### Scale Constraints
```rust
gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);
```

**Why:**
- **Lower bound (0.001):** Prevent "infinitely sharp" Gaussians (numerical stability)
- **Upper bound (0.5):** Prevent "huge" Gaussians that cover entire image

**Effect:**
- Minimum Gaussian diameter: ~0.001 = 0.064 pixels (sub-pixel)
- Maximum Gaussian diameter: ~0.5 = 32 pixels (half image width)

**Practical range in this experiment:**
- Most Gaussians settle between 0.02-0.08 scale (1.3-5.1 pixels)
- Constraints rarely hit (Gaussians naturally stay in reasonable range)

### Rotation Constraints
**None!** Rotation can be any value.

The Gaussian ellipse can rotate freely 0-2π radians. Typically rotations settle to align with image features (edges, gradients).

### Opacity Constraints
Fixed at 1.0, not optimized.

---

## Gaussian Rendering (Forward Pass)

### Splatting Process

For each pixel (x, y):
1. **Evaluate all N Gaussians** at that pixel location
2. **Compute Gaussian value** using elliptical formula:
   ```
   G_i(x,y) = exp(-0.5 × d^T × Σ_i^(-1) × d)
   ```
   Where d = [x - position_x, y - position_y]
3. **Weight by opacity:** `w_i = G_i(x,y) × opacity_i`
4. **Blend colors:**
   ```
   pixel_color = Σ(w_i × color_i) / Σ(w_i)
   ```
   (Weighted average of all Gaussian colors)

**Note:** This is **normalized** blending, not alpha compositing order. All Gaussians contribute regardless of depth.

### Computational Cost

**Per pixel:** Evaluate N Gaussians
**Total per forward pass:** 64 × 64 × N = 4096N Gaussian evaluations

**Scaling:**
- N=20: 81,920 evaluations
- N=50: 204,800 evaluations
- N=100: 409,600 evaluations

**Observed training speed:**
- N=20: ~220 iter/s
- N=50: ~185 iter/s
- N=100: ~150 iter/s (estimated)

Cost scales **linearly with N** (as expected).

---

## Why These Specific Constraints?

### Position [0, 1]
- **Image boundaries:** Gaussians outside image don't contribute
- **Normalization:** Makes code independent of pixel resolution

### Color [0, 1]
- **Standard RGB:** Matches most image formats
- **No HDR:** Could allow >1.0 for "super-bright" but adds complexity

### Scale [0.001, 0.5]
- **Lower bound:** Prevents numerical instability (division by ~zero)
- **Upper bound:** Prevents "meaningless" Gaussians covering entire image
- **Range:** Allows ~1000× size variation (0.001 to 0.5)

### Rotation (unbounded)
- **No constraint needed:** Rotation is periodic (θ = θ + 2π)
- **Optimizer handles it:** Adam naturally finds useful rotations

---

## Gaussian Parameters: Optimization Perspective

### Total Parameters per Gaussian
- Position: 2
- Scale: 2
- Rotation: 1
- Color: 3
- Opacity: 0 (fixed)
- **Total: 8 parameters per Gaussian**

### Total Parameters for Different N

| N | Total Optimized Parameters |
|---|----------------------------|
| 20 | 160 |
| 30 | 240 |
| 40 | 320 |
| 50 | 400 |
| 60 | 480 |
| 70 | 560 |
| 80 | 640 |
| 100 | 800 |

**Comparison to image pixels:**
- 64×64 image = 4096 pixels × 3 channels = 12,288 values
- N=50 = 400 parameters (3% of pixel data)
- **Highly compressed representation!**

This is the key: we're optimizing ~400 parameters to represent ~12k pixel values. That's a 30× compression.

---

## Gaussian Geometry Implications

### 1. Elliptical Gaussians Can Represent Oriented Features

**Example:** A diagonal edge
- Isotropic (circular) Gaussian: Poor fit, needs many Gaussians
- Elliptical Gaussian aligned with edge: Good fit, fewer Gaussians needed

The rotation parameter allows Gaussians to align with image features.

### 2. Scale Anisotropy Allows Different Widths

**Example:** A thin horizontal line
- Circular Gaussian: Too wide vertically
- Elliptical with scale_x > scale_y: Good fit

The separate scale_x and scale_y allow "skinny" Gaussians.

### 3. Constraint Bounds Affect Representable Detail

**Small scale limit (0.001):**
- Can represent very fine detail (sub-pixel features)
- But: Sub-pixel features may not be useful for 64×64 image

**Large scale limit (0.5):**
- Can represent smooth gradients efficiently
- One large Gaussian can cover half the image

### 4. Fixed Opacity Simplifies Optimization

**Baseline 1 choice:** opacity = 1.0 (not optimized)

**Why fixed:**
- One less parameter to optimize
- Simplifies blending (no semi-transparent Gaussians)
- Reduces search space

**Tradeoff:**
- Easier optimization (fewer parameters)
- Less representational power (can't have "faint" Gaussians)

---

## Comparison to Alternative Representations

### Baseline 1 (This Configuration)
- **Gaussians:** N elliptical 2D Gaussians
- **Parameters:** 8N (position, scale, rotation, color)
- **Opacity:** Fixed at 1.0
- **Rendering:** Normalized blending

### Alternative: Adaptive N (Baseline 3)
- **Gaussians:** Start with few, grow to many
- **Parameters:** 8N where N changes dynamically
- **Opacity:** Fixed at 1.0
- **Rendering:** Same

**Key difference:** N is optimized, not fixed.

### Alternative: Variable Opacity
- **Parameters:** 9N (add opacity to each Gaussian)
- **Benefit:** Can have faint background Gaussians
- **Cost:** More parameters, more complex optimization

### Alternative: 3D Gaussians (Splatting)
- **Gaussians:** 3D ellipsoids projected to 2D
- **Parameters:** 11-14 per Gaussian (3D position, 3D scale, rotation, color, opacity)
- **Use case:** 3D scene reconstruction, not 2D image compression

**Baseline 1 is simplest:** 2D, fixed opacity, fixed N.

---

## Summary

**Gaussian Geometry:**
- Elliptical 2D Gaussians (not circular)
- Euler parameterization (scale_x, scale_y, rotation)
- 8 parameters per Gaussian (9 total, but opacity fixed)

**Initialization:**
- Random positions, random isotropic scales [0.01, 0.1]
- Random colors, fixed seed (42) for reproducibility

**Constraints:**
- Position: [0, 1] (image bounds)
- Color: [0, 1] per channel (valid RGB)
- Scale: [0.001, 0.5] (not too tiny, not too huge)
- Rotation: Unbounded
- Opacity: Fixed at 1.0 (not optimized)

**N-to-Pixels Ratios (64×64 = 4096 pixels):**
- Tested: N=20 to N=100
- Ratios: 0.00488 to 0.02441 (0.5% to 2.4%)
- Optimal appears to be: ~1.1-1.3% (N=45-55 range)

**Parameter Count:**
- N=50: 400 parameters to represent 12,288 pixel values
- ~30× compression ratio

**These details are crucial** for understanding why certain N values work better and how the optimization behaves.

---

**Next:** Adam optimizer deep dive and loss function analysis.
