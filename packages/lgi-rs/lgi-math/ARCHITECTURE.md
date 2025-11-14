# LGI-Math Architecture

## Design Philosophy

The lgi-math library is designed with the following principles:

1. **Zero-Cost Abstractions**: Generic over float types and parameterizations without runtime overhead
2. **SIMD-First**: Data layouts optimized for vectorization from the start
3. **Numerically Stable**: Careful handling of edge cases and degenerate Gaussians
4. **Extensible**: Trait-based design allows custom parameterizations
5. **No-std Compatible**: Core functionality works without allocator

## Module Structure

```
lgi-math/
├── lib.rs              # Public API and trait definitions
├── gaussian.rs         # Core Gaussian2D type
├── parameterization.rs # Euler, Cholesky, LogRadius, InverseCovariance
├── covariance.rs       # 2×2 matrix utilities
├── evaluation.rs       # Gaussian evaluation and rendering
├── compositing.rs      # Alpha compositing operations
├── color.rs            # Color types and color space conversion
├── transform.rs        # Geometric transformations
└── utils.rs            # Utilities (space-filling curves, numerics)
```

## Type System Design

### Generic Float Trait

```rust
pub trait Float: Copy + Debug + Add + Sub + Mul + Div + Neg + PartialOrd {
    fn zero() -> Self;
    fn one() -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    // ... math operations
}
```

Allows the library to work with both `f32` and `f64` seamlessly.

### Parameterization Trait

```rust
pub trait Parameterization<T: Float>: Copy + Clone + Debug {
    fn covariance(&self) -> [[T; 2]; 2];
    fn inverse_covariance(&self) -> [[T; 2]; 2];
    fn is_valid(&self) -> bool;
}
```

Four implementations:

1. **Euler** `(σx, σy, θ)` - Most intuitive
   - σx, σy: Scales along principal axes
   - θ: Rotation angle
   - Fast covariance computation via rotation matrix

2. **Cholesky** `(L11, L21, L22)` - Numerically stable
   - Lower triangular Cholesky factor: Σ = L·Lᵀ
   - Ensures positive definiteness
   - Stable inversion

3. **LogRadius** `(log r, e, θ)` - Good for compression
   - log r: Log of geometric mean radius
   - e: Eccentricity (aspect ratio)
   - θ: Rotation
   - Better dynamic range for quantization

4. **InverseCovariance** `(a, b, c)` - Fastest rendering
   - Direct Σ⁻¹ storage
   - No matrix inversion needed
   - Optimal for hot path

### Gaussian2D Type

```rust
pub struct Gaussian2D<T: Float, P: Parameterization<T>> {
    pub position: Vec2,      // [0,1] normalized coordinates
    pub shape: P,            // Parameterization-specific
    pub color: Color4<T>,    // RGBA
    pub opacity: T,          // [0,1]
    pub weight: Option<T>,   // Progressive rendering priority
}
```

**Design choices:**
- Generic over `T` (f32/f64) and `P` (parameterization)
- Uses `glam::Vec2` for SIMD-friendly position
- Optional weight for LOD/progressive rendering
- `#[repr(C)]` layout for FFI compatibility (when using concrete types)

## Memory Layouts

### Array-of-Structures (AoS)

```
[Gaussian0][Gaussian1][Gaussian2]...
```

**Pros:**
- Simple to work with
- Good for random access
- Cache-friendly for single Gaussian operations

**Cons:**
- Poor SIMD utilization
- Scattered memory access in batch operations

### Structure-of-Arrays (SoA)

```
Positions: [x0, y0][x1, y1][x2, y2]...
Shapes:    [s0][s1][s2]...
Colors:    [r0, g0, b0, a0][r1, g1, b1, a1]...
Opacities: [o0][o1][o2]...
```

**Pros:**
- Excellent SIMD vectorization
- Cache-friendly for batch operations
- Optimal for GPU transfer

**Cons:**
- More complex to work with
- Poor for single-Gaussian access

**lgi-math provides both:**
- `Gaussian2D<T, P>` for ease of use
- `GaussianSoA<T, P>` for performance

## Gaussian Evaluation Pipeline

### 1. Bounding Box Culling

```rust
pub fn bounding_box(&self, n_sigma: T) -> (Vec2, Vec2)
```

- Compute eigenvalues of covariance matrix
- Find maximum extent: `radius = n_sigma * sqrt(max_eigenvalue)`
- Clamp to [0, 1] normalized space
- **Speedup**: Skip ~90% of Gaussians per pixel

### 2. Mahalanobis Distance

```rust
let dx = point.x - gaussian.position.x;
let dy = point.y - gaussian.position.y;
let mahal_sq = inv_cov[0][0] * dx * dx +
               (inv_cov[0][1] + inv_cov[1][0]) * dx * dy +
               inv_cov[1][1] * dy * dy;
```

- Compute squared Mahalanobis distance: `dᵀ Σ⁻¹ d`
- **Optimization**: Precompute `Σ⁻¹` per Gaussian

### 3. Gaussian Weight

```rust
let weight = (-0.5 * mahal_sq).exp();
if weight < cutoff { return 0.0; }
```

- Gaussian falloff: `exp(-0.5 * d²)`
- **Cutoff**: Early exit when `weight < 1e-5` (negligible contribution)

### 4. Alpha Compositing

```rust
let alpha_contrib = opacity * weight;
color_accum += (1 - alpha_accum) * color * alpha_contrib;
alpha_accum += (1 - alpha_accum) * alpha_contrib;
```

- Porter-Duff over operator (front-to-back)
- **Early termination**: Stop when `alpha_accum > 0.999`

## SIMD Strategy

### Current State (v0.1.0)

Data layouts are SIMD-ready but explicit vectorization not yet implemented.

**SIMD-friendly design:**
- `SoA` layout for batch operations
- Aligned data structures (`#[repr(C, align(16))]` where needed)
- Separate fields for independent SIMD lanes

### Future SIMD (v0.2.0 planned)

```rust
// Evaluate 8 Gaussians at once (AVX2)
use wide::f32x8;

pub fn evaluate_simd_x8(
    gaussians: &GaussianSoA<f32, InverseCovariance<f32>>,
    point: Vec2,
    start_idx: usize,
) -> f32x8 {
    // Load 8 positions
    let gx = f32x8::load(&gaussians.positions[start_idx..]);
    let gy = f32x8::load(&gaussians.positions[start_idx + 1..]);

    // Compute 8 Mahalanobis distances in parallel
    let dx = f32x8::splat(point.x) - gx;
    let dy = f32x8::splat(point.y) - gy;
    // ... SIMD math
    let weights = (-0.5 * mahal_sq).fast_exp();

    weights
}
```

**Expected speedup**: 4-8× on AVX2, 8-16× on AVX-512

## Numerical Stability

### Covariance Matrix Inversion

**Problem**: Direct inversion can be unstable for ill-conditioned matrices.

**Solutions:**

1. **Cholesky Decomposition** (preferred):
   ```rust
   Σ = L·Lᵀ
   Σ⁻¹ = L⁻ᵀ·L⁻¹
   ```
   - Numerically stable
   - Guaranteed positive definite

2. **Eigenvalue Check**:
   ```rust
   if det(Σ) < ε { return None; } // Reject degenerate
   ```

3. **Condition Number**:
   ```rust
   let cond = lambda_max / lambda_min;
   if cond > 1e6 { warn!("Ill-conditioned"); }
   ```

### Exponential Computation

**Problem**: `exp()` is expensive (~30-100 cycles).

**Optimizations:**

1. **Cutoff**: Skip `exp()` if Mahalanobis distance large
   ```rust
   if mahal_sq > 24.0 { return 0.0; } // exp(-12) ≈ 6e-6
   ```

2. **Fast approximation** (future):
   ```rust
   fn fast_exp(x: f32) -> f32 {
       // Schraudolph's approximation or rational polynomial
   }
   ```

3. **LUT (Lookup Table)**:
   ```rust
   const EXP_LUT: [f32; 4096] = precompute_exp_table();
   fn exp_lut(x: f32) -> f32 {
       let idx = ((x + 12.0) * 341.0) as usize; // map [-12, 0] to [0, 4095]
       EXP_LUT[idx.min(4095)]
   }
   ```
   **Speedup**: ~10× faster than `libm::exp()`

## Compositing Architecture

### Front-to-Back vs. Back-to-Front

**Front-to-Back (used by lgi-math):**
```rust
for gaussian in gaussians_sorted_front_to_back {
    let contrib = (1 - alpha_accum) * color * opacity * weight;
    color_accum += contrib;
    alpha_accum += (1 - alpha_accum) * opacity * weight;

    if alpha_accum > 0.999 { break; } // Early termination
}
```

**Advantages:**
- Early termination when alpha saturates
- Skip hidden Gaussians (e.g., behind opaque foreground)
- ~30-50% fewer Gaussians processed

**Back-to-Front:**
```rust
for gaussian in gaussians_sorted_back_to_front {
    color_out = alpha * color_fg + (1 - alpha) * color_bg;
}
```

**Disadvantages:**
- Must process all Gaussians
- No early termination

**lgi-math choice**: Front-to-back with early termination for maximum performance.

### Termination Threshold

```rust
pub termination_threshold: T = 0.999;
```

- **0.999**: 99.9% alpha coverage → terminate
- Remaining 0.1% contributes < 0.001 to color (imperceptible)
- **Speedup**: ~2-3× on scenes with many overlapping Gaussians

## Space-Filling Curves

### Morton (Z-Order) Curve

```rust
pub fn morton_index(x: u32, y: u32) -> u64 {
    // Interleave bits of x and y
    part1by1(x) | (part1by1(y) << 1)
}
```

**Properties:**
- Simple to compute
- Good spatial locality
- 2D → 1D mapping preserves some proximity

**Use case**: Sort Gaussians before encoding for better delta compression.

### Hilbert Curve

```rust
pub fn hilbert_index(x: u32, y: u32, order: u32) -> u32
```

**Properties:**
- Better locality than Morton
- Preserves more spatial proximity
- More complex to compute

**Use case**: Optimal ordering for progressive transmission.

**Performance comparison**:
- Morton: ~5 ns per index
- Hilbert: ~20 ns per index

For LGI, Morton is sufficient (simpler, faster).

## Parameterization Conversions

### Euler ↔ Cholesky

```rust
impl From<Euler<T>> for Cholesky<T> {
    fn from(euler: Euler<T>) -> Self {
        let cov = euler.covariance();
        Cholesky::from_covariance(cov)
    }
}
```

**Cost**: ~50 ns (covariance + Cholesky decomposition)

### Euler ↔ LogRadius

```rust
impl From<Euler<T>> for LogRadius<T> {
    fn from(euler: Euler<T>) -> Self {
        let geom_mean = (euler.scale_x * euler.scale_y).sqrt();
        let eccentricity = euler.scale_y / euler.scale_x;
        LogRadius::new(geom_mean.ln(), eccentricity, euler.rotation)
    }
}
```

**Cost**: ~30 ns (2 sqrt, 1 ln, 1 div)

### All → InverseCovariance

```rust
let inv_cov_gaussian = euler_gaussian.convert::<InverseCovariance<_>>();
```

**Rendering path**: Convert to `InverseCovariance` once, cache, reuse for all pixels.

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Gaussian evaluation | O(1) | ~10-20 ns |
| Bounding box | O(1) | Eigenvalue formula, ~15 ns |
| Covariance | O(1) | Matrix mult, ~10 ns |
| Inverse covariance | O(1) | Cached or O(1) formula, ~10 ns |
| Compositing | O(1) | ~7 ns per operation |
| Sort by Morton | O(n log n) | ~5 ns per index, std::sort |

### Memory Footprint

**Per Gaussian (AoS, f32, Euler)**:
```
position:  8 bytes (2×f32)
shape:    12 bytes (3×f32)
color:    16 bytes (4×f32)
opacity:   4 bytes (f32)
weight:    8 bytes (Option<f32>)
------
Total:    48 bytes
```

**Cache efficiency**:
- L1 cache line: 64 bytes → 1 Gaussian + metadata
- L2 cache: 512 KB → ~10K Gaussians
- L3 cache: 32 MB → ~600K Gaussians

For 1M Gaussians: 48 MB (fits in L3 on modern CPUs).

### Optimization Opportunities

1. **SIMD** (8×): Batch evaluate 8-16 Gaussians simultaneously
2. **GPU** (100×): Evaluate 1000s of Gaussians in parallel
3. **Tiling** (2×): Divide image into tiles, cull Gaussians per tile
4. **LOD** (3×): Progressive rendering, fewer Gaussians for preview
5. **Compression** (0.3×): zstd on quantized parameters

**Combined**: 8 (SIMD) × 2 (tiling) × 3 (LOD) = **48× speedup** over naive implementation.

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_gaussian_evaluation() {
    let g = Gaussian2D::new(...);
    let evaluator = GaussianEvaluator::default();
    let weight = evaluator.evaluate(&g, Vec2::new(0.5, 0.5));
    assert_relative_eq!(weight, 1.0, epsilon = 1e-5);
}
```

- Test each module independently
- Use `approx::assert_relative_eq!` for float comparison
- Cover edge cases (degenerate Gaussians, etc.)

### Property-Based Tests (using proptest)

```rust
proptest! {
    #[test]
    fn test_covariance_positive_definite(
        sx in 0.01f32..1.0,
        sy in 0.01f32..1.0,
        theta in -PI..PI,
    ) {
        let euler = Euler::new(sx, sy, theta);
        let cov = euler.covariance();
        assert!(CovarianceMatrix::is_positive_definite(cov));
    }
}
```

- Generate random inputs
- Verify mathematical invariants
- Find edge cases automatically

### Benchmarks (using criterion)

```rust
fn bench_gaussian_eval(c: &mut Criterion) {
    let gaussian = Gaussian2D::new(...);
    let evaluator = GaussianEvaluator::default();
    let point = Vec2::new(0.55, 0.55);

    c.bench_function("evaluate_single", |b| {
        b.iter(|| black_box(evaluator.evaluate(&gaussian, point)))
    });
}
```

- Measure performance regressions
- Compare implementations (scalar vs. SIMD)
- Profile hot paths

## Future Enhancements (v0.2.0+)

### 1. Explicit SIMD

```rust
#[cfg(target_feature = "avx2")]
mod simd {
    use wide::f32x8;

    pub struct GaussianEvaluatorSIMD {
        // SIMD-optimized evaluator
    }
}
```

### 2. GPU Compute Shaders

```rust
// Generate wgpu compute shader from template
pub fn generate_gaussian_shader(parameterization: &str) -> String {
    format!(r#"
        @compute @workgroup_size(16, 16)
        fn evaluate_gaussians(
            @builtin(global_invocation_id) gid: vec3<u32>,
            @group(0) @binding(0) gaussians: Gaussians,
            @group(0) @binding(1) output: texture_storage_2d<rgba8unorm, write>,
        ) {{
            let point = vec2<f32>(gid.xy) / vec2<f32>(resolution);
            var color = vec4<f32>(0.0);
            for (var i = 0u; i < num_gaussians; i++) {{
                // Evaluate Gaussian i at point
            }}
            textureStore(output, gid.xy, color);
        }}
    "#)
}
```

### 3. Quantization

```rust
pub struct QuantizedGaussian {
    position: [u16; 2],   // 16-bit normalized
    shape: [u16; 3],      // 16-bit log-encoded
    color: [u8; 4],       // 8-bit
    opacity: u8,          // 8-bit
}
// Total: 14 bytes (vs. 48 bytes unquantized)
```

### 4. Compression-Aware Ordering

```rust
pub fn optimize_for_compression(gaussians: &mut [Gaussian2D]) {
    // 1. Sort by Morton curve
    // 2. Delta encode positions
    // 3. Run-length encode similar parameters
    // 4. Entropy code deltas
}
```

---

**Document Version**: 1.0
**Last Updated**: October 2025
**Status**: v0.1.0 Architecture
