# Gaussian Representation Theory

**Status**: Phase 1 Research - In Progress
**Last Updated**: November 14, 2025

---

## Overview

This document establishes the theoretical foundation for representing images using 2D Gaussian primitives, adapting concepts from 3D Gaussian Splatting (Kerbl et al., 2023) to 2D image compression/representation.

---

## 1. What is a Gaussian Representation?

### 1.1 Fundamental Concept

Instead of representing an image as a grid of pixels, we represent it as a **collection of 2D Gaussian functions** that are **composited** to form the image.

**Analogy**:
- **Pixel representation**: Image = grid of colored squares
- **DCT/Wavelet**: Image = weighted sum of frequency basis functions
- **Gaussian representation**: Image = weighted blend of smooth blobs (Gaussians)

### 1.2 Single 2D Gaussian Definition

A 2D Gaussian is defined by its probability density function:

```
G(x, y; μ, Σ) = (1 / 2π√|Σ|) × exp(-½ (p - μ)ᵀ Σ⁻¹ (p - μ))
```

Where:
- **p = (x, y)**: Point in image space
- **μ = (μₓ, μᵧ)**: Center position (mean)
- **Σ**: 2×2 covariance matrix (controls shape and orientation)
- **|Σ|**: Determinant of covariance matrix

**In image representation, we extend this to include color and opacity**:

```
Gaussian := {
    position:  (μₓ, μᵧ) ∈ [0,1]² (normalized image coordinates)
    shape:     Σ or parameterization thereof
    color:     (r, g, b) ∈ [0,1]³
    opacity:   α ∈ [0,1]
}
```

### 1.3 Covariance Matrix Representation

The covariance matrix Σ controls the Gaussian's shape:

```
Σ = [ σₓ²    ρσₓσᵧ ]
    [ ρσₓσᵧ  σᵧ²   ]
```

Where:
- **σₓ, σᵧ**: Standard deviations (scale in x and y directions)
- **ρ ∈ [-1, 1]**: Correlation coefficient

**Geometric interpretation**:
- **σₓ = σᵧ, ρ = 0**: Circular Gaussian (isotropic)
- **σₓ ≠ σᵧ, ρ = 0**: Axis-aligned elliptical Gaussian
- **ρ ≠ 0**: Rotated elliptical Gaussian

### 1.4 Alternative Parameterizations

**Problem**: Optimizing Σ directly can produce invalid (non-positive-definite) matrices.

**Solution 1: Eigen Decomposition** (what we use in Euler parameterization)
```
Σ = R(θ) × S × R(θ)ᵀ

Where:
- R(θ) = rotation matrix (angle θ)
- S = diagonal scale matrix [σₓ² 0; 0 σᵧ²]

Parameters: (σₓ, σᵧ, θ) - always produces valid Σ
```

**Solution 2: Log-Cholesky** (mentioned in our code)
```
Σ = L × Lᵀ where L is lower triangular

L = [ exp(l₁₁)    0      ]
    [     l₂₁   exp(l₂₂)  ]

Parameters: (l₁₁, l₂₁, l₂₂) - always positive definite
```

**Solution 3: Scale + Quaternion** (from 3DGS, for 3D)
- Not applicable to 2D directly
- In 2D, rotation is single angle, not quaternion

---

## 2. Image Composition from Gaussians

### 2.1 Rendering Equation

Given N Gaussians {G₁, G₂, ..., Gₙ}, how do we produce pixel colors?

**Two approaches**:

#### A. Weighted Average (Our Current Approach - renderer_v2.rs)

```
For each pixel p = (x, y):

  W = Σᵢ wᵢ(p)                          // Total weight
  C = Σᵢ wᵢ(p) × cᵢ                     // Weighted color sum

  I(p) = C / max(W, ε)                  // Final pixel color

Where:
  wᵢ(p) = αᵢ × exp(-½ dᵢ²(p))          // Weight from Gaussian i
  dᵢ²(p) = (p - μᵢ)ᵀ Σᵢ⁻¹ (p - μᵢ)     // Mahalanobis distance
  αᵢ = opacity of Gaussian i
  cᵢ = color of Gaussian i
```

**Properties**:
- Normalized: if W ≥ 1, colors stay in [0,1]
- Background: if W ≈ 0, pixel is black (or could set background color)
- Smooth: inherently anti-aliased due to Gaussian falloff

#### B. Alpha Compositing (From 3DGS)

```
Sort Gaussians by depth (front to back or back to front)
For each pixel p, accumulate:

  C = 0, T = 1  // Accumulated color and transmittance

  For each Gaussian i in order:
    αᵢ' = αᵢ × exp(-½ dᵢ²(p))          // Modulated opacity
    C += T × αᵢ' × cᵢ                   // Add color contribution
    T × = (1 - αᵢ')                     // Update transmittance

  I(p) = C
```

**Properties**:
- Order-dependent (need depth sorting)
- More physically motivated (light occlusion)
- Used in 3DGS for 3D scenes

**For 2D images**: Weighted average is simpler and works well (no depth ordering needed).

### 2.2 Mahalanobis Distance

The distance metric is critical:

```
d²(p) = (p - μ)ᵀ Σ⁻¹ (p - μ)
```

For our Euler parameterization (σₓ, σᵧ, θ):
```
1. Rotate p to Gaussian's local frame:
   p' = R(-θ) × (p - μ)

2. Compute normalized squared distance:
   d² = (p'ₓ / σₓ)² + (p'ᵧ / σᵧ)²
```

**Geometric meaning**: d measures how many "standard deviations" away p is from μ.
- d = 0: At center
- d = 1: One standard deviation away (≈61% of max weight)
- d = 3: Three standard deviations (≈1% of max weight)
- d > 3.5: Typically cutoff (negligible contribution)

### 2.3 Coverage Concept

**Coverage** at pixel p: How much "Gaussian weight" exists there.

```
Coverage(p) = Σᵢ αᵢ × exp(-½ dᵢ²(p))
```

**Good coverage**: Coverage ≥ 0.5 everywhere
- Means every pixel is influenced by Gaussians
- Prevents "gaps" (black holes in reconstruction)

**Bad coverage**: Coverage ≈ 0 in some regions
- Those pixels will be black (or background color)
- Indicates insufficient Gaussians or poor placement

**Diagnostic**: W_median (median coverage across all pixels)
- W_median > 0.5: Good coverage
- W_median ≈ 0: Major problem (most pixels have no Gaussian influence)

---

## 3. Comparison to Other Representations

### 3.1 vs. Pixel Representation

| Aspect | Pixels | Gaussians |
|--------|--------|-----------|
| Basis | Delta functions (point samples) | Smooth Gaussians |
| Continuity | Discrete | Continuous |
| Zoom | Pixelation/aliasing | Smooth (with proper splatting) |
| Editing | Easy (per-pixel) | Hard (find relevant Gaussians) |
| Compression | Needs transform (JPEG, etc.) | Directly compact (fewer Gaussians) |
| Rendering | Trivial (lookup) | More complex (splat and blend) |

### 3.2 vs. DCT/Wavelet

| Aspect | DCT/Wavelet | Gaussians |
|--------|-------------|-----------|
| Basis | Global frequency | Local spatial |
| Support | Entire image (DCT) or fixed grid (wavelet) | Adaptive (place where needed) |
| Sharp edges | Ringing artifacts | Smooth (blurry but no ringing) |
| Optimization | Linear (FFT) | Non-linear (gradient descent) |
| Interpretability | Abstract frequencies | Intuitive (blobs with positions/colors) |

### 3.3 Theoretical Advantages of Gaussians

1. **Adaptive Density**
   - Can place more Gaussians in complex regions
   - Fewer in smooth regions
   - Unlike fixed grids (pixels, DCT blocks)

2. **Smooth Interpolation**
   - Naturally continuous representation
   - Good for zooming, rotation
   - No aliasing if splatted correctly (EWA)

3. **Direct Optimization**
   - Can optimize position, scale, color directly
   - Gradient-based refinement
   - Structure-aware (can align to edges)

4. **Compact for Smooth Content**
   - Smooth gradients: few large Gaussians
   - But struggles with high-frequency textures

### 3.4 Theoretical Disadvantages

1. **Non-linear Optimization**
   - No closed-form solution (unlike DCT)
   - Can get stuck in local minima
   - Sensitive to initialization

2. **High-Frequency Limitation**
   - Gaussians are smooth → blurry
   - Need many tiny Gaussians for sharp edges
   - Trade-off: compactness vs. sharpness

3. **Rendering Cost**
   - O(N × pixels) naively
   - Need acceleration (tiling, culling)
   - More expensive than pixel lookup

4. **Bitstream Encoding**
   - Positions are continuous (need quantization)
   - Scales, colors also continuous
   - Not as naturally compressible as transform coefficients

---

## 4. Information-Theoretic Considerations

### 4.1 Representational Capacity

**Question**: How many Gaussians are needed to represent an image?

**Theoretical bounds** (loose):
- **Lower bound**: 1 Gaussian (solid color image)
- **Upper bound**: ~(pixels / 9) Gaussians if placing at 3×3 grid
  - For 128×128: ~1820 Gaussians max useful

**Practical observations** (from 3DGS):
- Natural scenes: 10⁴-10⁶ Gaussians for high quality
- For 2D, should be fewer (no depth complexity)

**From our experiments** (128×128 images):
- Grid baseline: 64 Gaussians (8×8)
- Optimization target: 50-100 Gaussians
- Expected PSNR: 20-25 dB for synthetic patterns

### 4.2 Degrees of Freedom

Each Gaussian has:
- Position: 2 parameters (μₓ, μᵧ)
- Shape: 3 parameters (σₓ, σᵧ, θ) or (l₁₁, l₂₁, l₂₂)
- Color: 3 parameters (r, g, b)
- Opacity: 1 parameter (α)

**Total**: 9 parameters per Gaussian

**For N Gaussians**: 9N parameters

**Comparison**:
- 128×128 image: 128×128×3 = 49,152 values
- 100 Gaussians: 900 parameters
- **Compression ratio**: ~54× fewer parameters

**But**: Parameters are continuous (floats), pixels are quantized (8-bit)
- With quantization: closer to ~20-30× compression potentially

### 4.3 Rate-Distortion Theory

**Question**: For a given number of Gaussians N, what's the best achievable quality?

**No closed-form answer** (representation is non-linear)

**Empirical approach**:
- Measure PSNR vs. N curve
- Should see diminishing returns (logarithmic)
- Example target: 30 dB at N=200 for natural images

---

## 5. Gaussian Splatting Rendering (EWA)

### 5.1 The Aliasing Problem

**Naïve rendering**:
```
For each pixel p:
  Evaluate all Gaussians at pixel center
```

**Problem**: Undersampling (Nyquist theorem)
- High-frequency Gaussians (small σ) can alias
- Visible as jaggies, Moiré patterns

### 5.2 EWA Solution (Zwicker et al., 2001)

**Elliptical Weighted Average**:
- Convolve each Gaussian with a **reconstruction filter**
- The result is another Gaussian (Gaussians are closed under convolution!)

**Forward splatting**:
```
G_rendered = G_primitive ⊗ F_reconstruction

Where:
- G_primitive: Original Gaussian (σₓ, σᵧ, θ)
- F_reconstruction: Low-pass filter (typically Gaussian, bandwidth b)
- ⊗: Convolution operator

Result: G_rendered has covariance Σ' = Σ + b²I
```

**Properties**:
- Prevents aliasing (low-pass filtering)
- Maintains smooth appearance across zoom levels
- Properly handles anisotropic Gaussians

### 5.3 Our Implementation (renderer_v2.rs and ewa_splatting_v2.rs)

**renderer_v2.rs**: Simple weighted average
- Fast, straightforward
- Good enough for optimization
- May have aliasing at high zoom

**ewa_splatting_v2.rs**: EWA splatting
- Adds reconstruction filter bandwidth
- Zoom-stable rendering
- More expensive (used for final quality)

---

## 6. Connection to 3D Gaussian Splatting

### 6.1 Differences from 3DGS

**3D (Kerbl et al., 2023)**:
- Gaussians in 3D space (world coordinates)
- Projected to 2D screen space
- Depth ordering for alpha compositing
- Camera viewpoint dependent

**2D (Our LGI codec)**:
- Gaussians directly in 2D image space
- No projection, no camera
- No depth (order-independent blending)
- Single "view" (the image itself)

### 6.2 What We Can Borrow from 3DGS

1. **Optimization Strategy**
   - Adaptive density control (densification)
   - Split/clone/prune operations
   - Structure-aware initialization

2. **Parameterization**
   - Log-Cholesky for covariance
   - Ensures positive definiteness
   - Smooth optimization landscape

3. **Loss Functions**
   - L1 + SSIM combination (perceptual)
   - Better than pure L2/MSE

4. **Rendering Acceleration**
   - Tiling for parallel rendering
   - Screen-space culling
   - Gaussian sorting by contribution

### 6.3 What Doesn't Apply

1. **Depth-based operations**
   - Depth rendering
   - Alpha compositing with depth
   - 3D transformations

2. **Multi-view consistency**
   - No multiple camera views
   - No 3D geometry constraints

3. **SH (Spherical Harmonics)**
   - Used in 3DGS for view-dependent color
   - Not needed for 2D (color is just RGB)

---

## 7. Open Questions & Research Gaps

### 7.1 Initialization

**Q**: What's the optimal initial Gaussian distribution?
- Grid? Structure-tensor? Content-aware?
- How to set initial scales σ?

**Hypothesis**: Should match image complexity
- Smooth regions: fewer, larger Gaussians
- Detailed regions: more, smaller Gaussians

### 7.2 Optimization Convergence

**Q**: Why does our optimizer get stuck (loss = 0.375)?
- Wrong gradients? (we fixed rotation, but is it enough?)
- Bad initialization?
- Optimization hyperparameters (LR, iterations)?

**Need**: Verify gradients with finite differences

### 7.3 Densification Strategy

**Q**: When and where to add Gaussians?
- High error regions (obvious)
- But how to avoid over-densification?
- How to balance quality vs. Gaussian count?

**From 3DGS**: Use gradient magnitude of position
- If ∇_μ is large → Gaussian trying to cover too much → split it

### 7.4 Quality Limits

**Q**: What's the best achievable PSNR for N Gaussians?
- Theoretical bound unknown
- Empirical: need to measure PSNR(N) curves

**Hypothesis**: Logarithmic improvement
- N → 2N gives +3-6 dB (diminishing returns)

---

## 8. Next Steps in Research

1. **Verify our renderer** matches EWA theory
   - Read Zwicker 2001 paper
   - Compare equations to our code
   - Test: single Gaussian, known parameters → expected image

2. **Understand gradient flow**
   - Derive ∂Loss/∂μ, ∂Loss/∂Σ, ∂Loss/∂c analytically
   - Verify implementation with finite differences
   - Ensure our rotation-aware gradients are correct

3. **Study initialization theory**
   - Structure tensor mathematics
   - Why does it detect edges?
   - How to translate to Gaussian placement?

4. **Analyze densification**
   - Read 3DGS densification section
   - Adapt split/clone heuristics for 2D
   - Test: does adding Gaussians improve quality?

5. **Measure empirically**
   - PSNR vs. N curves for different images
   - Compare initialization strategies
   - Benchmark optimization methods

---

## References

### Papers (To Read/Study)

- **[Kerbl et al., 2023]** "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
  - ArXiv: https://arxiv.org/abs/2308.04079
  - Project: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

- **[Zwicker et al., 2001]** "Surface Splatting" (EWA splatting)
  - SIGGRAPH 2001
  - Key: Reconstruction filter theory, anti-aliasing

- **[Differentiable Rendering]** General theory
  - How to compute gradients through rendering
  - Soft rasterization, neural rendering

### Code References

- **3DGS Official Implementation**
  - https://github.com/graphdeco-inria/gaussian-splatting
  - CUDA kernels, optimization, densification

- **Our Implementation**
  - `renderer_v2.rs` - Weighted average rendering
  - `ewa_splatting_v2.rs` - EWA splatting
  - `adam_optimizer.rs` - Gradient-based optimization

---

## Glossary

- **Gaussian**: Smooth bell-curve function; in 2D, an elliptical blob
- **Covariance Matrix (Σ)**: 2×2 matrix defining Gaussian shape and orientation
- **Mahalanobis Distance**: Normalized distance accounting for Gaussian shape
- **Coverage**: Amount of Gaussian weight at a pixel; measure of representation quality
- **Splatting**: Rendering technique for point/blob primitives
- **EWA (Elliptical Weighted Average)**: Anti-aliased splatting with reconstruction filter
- **Densification**: Process of adding more Gaussians to improve quality
- **Opacity (α)**: Transparency/blending weight of a Gaussian

---

**Status**: Initial draft from existing knowledge
**TODO**:
- [ ] Read and integrate Kerbl 2023 paper details
- [ ] Read and integrate Zwicker 2001 EWA theory
- [ ] Add mathematical derivations (gradient flow)
- [ ] Add diagrams (Gaussian shape, covariance, splatting)
- [ ] Verify all equations against our code
- [ ] Expand information-theoretic analysis

**Next Document**: `RENDERING_THEORY.md` - Deep dive into EWA splatting mathematics
