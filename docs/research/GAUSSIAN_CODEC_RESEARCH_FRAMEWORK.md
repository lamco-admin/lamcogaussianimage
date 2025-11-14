# Gaussian Image Codec: Comprehensive Research Framework

## Executive Overview

Building a Gaussian image codec requires understanding three foundational pillars: **(1) theoretical frameworks** for representing images as collections of 2D Gaussians, **(2) existing experimental implementations** that have successfully applied this approach, and **(3) technical frameworks** for both rendering Gaussian representations and compressing them into viable codecs.

This document provides an exhaustive research foundation covering all three areas, with actionable implementation paths and architectural recommendations for your solution.

---

## Part 1: Theoretical Foundations

### 1.1 Gaussian Representation Mathematics

#### Core Mathematical Framework

A **2D Gaussian** in image space is defined by:

- **Position**: \(\boldsymbol{\mu} \in \mathbb{R}^2\) — the center coordinates (x, y)
- **Covariance**: \(\boldsymbol{\Sigma} \in \mathbb{R}^{2 \times 2}\) — the spread in 2D space
- **Color**: \(\boldsymbol{c} \in \mathbb{R}^3\) — RGB values
- **Opacity**: \(\alpha \in [0, 1]\) — transparency coefficient

The **2D Gaussian function** is:

\[G(\mathbf{x}) = \alpha \cdot \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)\]

#### Covariance Parameterization (Cholesky Decomposition)

Rather than storing the full 2×2 symmetric matrix, use **Cholesky factorization** for numerical stability:

\[\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^T\]

Where \(\mathbf{L}\) is a lower triangular matrix, represented as a 3-element vector:
- \(l_1\) = diagonal scale factor 1
- \(l_2\) = diagonal scale factor 2  
- \(l_3\) = off-diagonal covariance term

This reduces storage from 4 parameters to 3 and ensures positive-definiteness without explicit constraints.

#### Total Parameters per Gaussian

GaussianImage uses **8 parameters per Gaussian**:
- Position: 2 (μx, μy)
- Covariance via Cholesky: 3 (l₁, l₂, l₃)
- Color: 3 (R, G, B)

This is dramatically more efficient than 3D Gaussian Splatting, which requires ~59 parameters per Gaussian (3D position, 3D rotation matrix, 3D scales, spherical harmonics for view-dependent color).

### 1.2 Image Representation via Gaussian Decomposition

#### The Core Principle

Instead of representing an image as a grid of pixels, represent it as an **unordered collection of N 2D Gaussians**. The final pixel value at any location (x, y) is computed via **alpha blending**:

\[C(x,y) = \sum_{i=1}^{N} \alpha_i G_i(x,y) \cdot \mathbf{c}_i + (1 - \sum_{i=1}^{N} \alpha_i G_i(x,y)) \cdot C_{bg}\]

Where:
- The Gaussians blend in a specific order (front-to-back or with proper depth handling)
- \(C_{bg}\) is background color (typically white)
- **Accumulated transparency** is tracked for proper compositing

#### Why This Works

1. **Smoothness Bias**: Gaussian primitives naturally enforce spatial smoothness, acting as an implicit regularizer
2. **Efficiency**: Complex regions get more Gaussians; simple regions get fewer (adaptive density)
3. **Compressibility**: Gaussian attributes (position, scale, color) compress well with quantization
4. **Differentiability**: The entire rendering pipeline is differentiable through alpha-blending operations

### 1.3 Connection to Implicit Neural Representations (INRs)

**Key Distinction**: While traditional INRs (like SIREN, NeRF) use neural networks to implicitly map coordinates to values, **Gaussian codecs use explicit basis functions**—this is actually closer to classical signal processing than deep learning.

**Advantages over INRs**:
- No network weights to store → direct compression of Gaussian attributes
- Fast inference (2000 FPS vs 1-50 FPS for INRs)
- Lower GPU memory during training (3× less than SIREN)
- Better compatibility with traditional graphics pipelines

---

## Part 2: Existing Experimental Implementations

### 2.1 GaussianImage (ECCV 2024) — State-of-the-Art

**Paper**: "1000 FPS Image Representation and Compression by 2D Gaussian Splatting"

#### Architecture Overview

**Stage 1: Image Overfitting**
- Initialize N random 2D Gaussians
- Optimize all 8N parameters to minimize reconstruction error
- Loss: L1 + L2 + SSIM (mixed loss)
- Training time: ~3-5 seconds per image
- Rendering speed: **1500-2000 FPS**

**Stage 2: Attribute Quantization-Aware Fine-tuning**
- Quantize Gaussian attributes with awareness that quantization will occur at inference
- Quantization strategies per attribute:
  - **Position**: FP16 floating point
  - **Covariance (Cholesky)**: 6-bit integer quantization
  - **Color**: Residual Vector Quantization (RVQ)
  
**Stage 3: Compression Pipeline**
- Apply **vector quantization** to build codebook of common Gaussian configurations
- Use **bits-back coding** (optional, for further rate reduction)
- Entropy encoding with ANS (Asymmetric Numeral Systems)

#### Key Results

- **Compression ratio**: 7.375× at reasonable quality
- **Decoding speed**: ~2000 FPS (outpaces JPEG by 3-5×)
- **Quality metric**: Competitive with COIN++ at low bitrates
- **GPU memory**: 3× lower than SIREN
- **Training time**: 5× faster than NeRF-based approaches

#### Code Repository
- Official implementation: `https://github.com/Xinjie-Q/GaussianImage`
- Based on 2D Gaussian splatting with accumulated summation rendering

### 2.2 Investigation into 2D Gaussian Splatting (EPFL 2024)

**Focus**: Balancing compression and image quality via adaptive Gaussian placement

#### Key Contributions

1. **Progressive Gaussian Addition Strategy**:
   - Start with small number of Gaussians capturing broad features
   - Progressively add more Gaussians to refine finer details
   - Avoids over-densification in simple regions

2. **Compression-Quality Trade-off**:
   - Fewer Gaussians with larger support = better compression
   - More Gaussians with tighter clustering = better quality
   - Optimal balance found through adaptive density strategies

3. **Cropping Artifact Solutions**:
   - Identified boundary effects in naive Gaussian splatting
   - Proposed solutions for clean edge handling

#### Relevance to Your Solution
- Demonstrates that Gaussian density distribution matters greatly
- Shows adaptive allocation is superior to uniform initialization

### 2.3 Neural Video Compression using 2D Gaussian Splatting (2025)

**Focus**: Extending Gaussian codecs to video

#### Important Innovations

1. **Content-Aware Initialization**:
   - Use superpixel segmentation to initialize Gaussian means/covariances
   - Reduces encoding overhead from 13 seconds/frame to <1 second
   - Achieves target PSNR 30 on standard hardware

2. **Frame Types**:
   - **I-frames**: Independently coded (full spatial reconstruction)
   - **P-frames**: Reference previous frames (temporal redundancy reduction)
   - 78% bitstream reduction vs image-only codec through inter-frame optimization

3. **Entropy Coding Variants**:
   - Standard entropy coding for first K Gaussians
   - Bits-back coding for remaining N-K Gaussians
   - Can exploit inter-frame dependencies

### 2.4 Related Codec Baselines (For Comparison)

#### COIN / COIN++ (INR-based)
- Fits MLP to image, stores quantized weights
- Encoding: ~1 hour per image (extremely slow)
- Decoding: Fast but requires running neural network inference
- Performance: Competitive with JPEG2000 at low bitrates

#### WIRE (Wavelet INR)
- Uses continuous complex Gabor wavelets as activation
- Better than SIREN but still network-weight-based compression

#### LIVE (Layer-wise Image Vectorization)
- Converts raster to vector graphics (SVG)
- Generates layer-wise representations
- Different paradigm: geometric primitives (Bezier curves) vs Gaussians

---

## Part 3: Technical Frameworks and Rendering

### 3.1 Rendering Pipeline Architecture

#### Forward Pass: Accumulation-Based Rasterization

**Algorithm**:
```
for each pixel (x, y):
    accumulated_color = background_color
    accumulated_alpha = 0
    for each Gaussian i (in depth-sorted or z-order):
        gaussian_value = exp(-0.5 * (x-μ)^T Σ^-1 (x-μ))
        pixel_contribution = α_i * gaussian_value * c_i
        accumulated_color = accumulated_color * (1 - α_i * gaussian_value) + pixel_contribution
        accumulated_alpha += α_i * gaussian_value
        if accumulated_alpha ≈ 1.0: break  # Early termination
    output_pixel = accumulated_color
```

**Key Optimizations**:
1. **Tile-based rasterization** (GPU-friendly):
   - Divide image into small tiles (8×8 or 16×16 pixels)
   - Process all Gaussians per tile
   - Reduces memory bandwidth vs full-screen rasterization
   
2. **Early termination**: Stop processing Gaussians once alpha reaches ~0.99

3. **View-space optimization**: Compute Covariance in 2D screen space after projection

#### Backward Pass: Gradient Computation

The entire pipeline is **differentiable**, enabling gradient-based optimization:

\[\frac{\partial L}{\partial \boldsymbol{\mu}_i} = \text{gradient w.r.t. Gaussian center}\]
\[\frac{\partial L}{\partial \mathbf{L}_i} = \text{gradient w.r.t. Cholesky factors}\]
\[\frac{\partial L}{\partial \mathbf{c}_i} = \text{gradient w.r.t. color}\]

Used for both:
- Initial optimization to fit image
- Quantization-aware fine-tuning

### 3.2 Optimization Frameworks

#### Loss Function Design

**Multi-component loss** (standard in Gaussian codecs):

\[L = \lambda_1 \cdot L_1(\hat{I}, I) + \lambda_2 \cdot L_2(\hat{I}, I) + \lambda_3 \cdot L_{SSIM}(\hat{I}, I)\]

Where:
- **L1 loss**: \(\|I - \hat{I}\|_1\) — captures sparse high-frequency errors
- **L2 loss**: \(\|I - \hat{I}\|_2^2\) — smooth low-frequency fitting
- **SSIM loss**: Structural Similarity Index — perceptual quality
- Typical weights: λ₁=0.2, λ₂=0.8, λ₃=0.1

**Optional perceptual loss** (for improved visual quality):
- LPIPS (Learned Perceptual Image Patch Similarity)
- Measures distance in VGG feature space
- Better correlation with human perception than SSIM alone

#### Adaptive Density Control

Inspired by 3D Gaussian Splatting, but adapted for 2D:

**Densification Criteria** (during training):
- Monitor accumulated positional gradients
- Split Gaussians with high gradients (moving frequently)
- Clone Gaussians in under-reconstructed regions
- Prune Gaussians with opacity < threshold

**Key difference from 3D**: 2D codecs typically don't use explicit density control during initial fitting—simpler approach works better for images

#### Quantization-Aware Training (QAT)

**Challenge**: Quantization introduces discontinuities that break gradient flow

**Solution**: Simulate quantization during training:
```python
# Forward pass
quantized_value = round(value / step_size) * step_size
# Backward pass (straight-through estimator)
∇value = ∇quantized_value  # Treat round() as identity for gradients
```

**Implementation stages**:
1. Train with full precision initially
2. Add fake quantization to loss computation
3. Fine-tune all parameters with quantization active
4. Gradually reduce learning rate to convergence

### 3.3 Compression Pipeline

#### Attribute-Specific Quantization

**Position (FP16)**:
- Floating-point 16-bit representation
- Sufficient precision for pixel-level positioning
- ~2 bytes per Gaussian

**Covariance via Cholesky (6-bit integer)**:
- Quantize each of 3 Cholesky elements to 6 bits
- Provides ~64 levels of precision per dimension
- Ensures positive-definiteness maintained
- ~2.25 bytes per Gaussian

**Color (Residual Vector Quantization)**:
- Use RVQ with multiple codebook stages
- Stage 1: Coarse quantization (8 bits)
- Stage 2: Refine residuals (8 bits)
- Typical: 2-3 stages = 16-24 bits per Gaussian color

**Total per Gaussian**: ~6-8 bytes compressed

#### Entropy Coding Integration

**Standard Entropy Coding** (all Gaussians):
- Build probability distribution of quantized values
- Use Asymmetric Numeral Systems (ANS) or arithmetic coding
- Achieves theoretical Shannon entropy bound

**Bits-Back Coding** (optional refinement):
- Encode latent variables with iterative ANS operations
- Can reduce overall bitrate by 10-20%
- Trade-off: increased decoding complexity

### 3.4 Framework Implementation Considerations

#### GPU Acceleration Requirements

**CUDA/HIP Kernels Needed**:
1. **Gaussian rasterization**: Tile-based forward pass
2. **Gradient computation**: Backward pass for optimization
3. **Covariance transformation**: 2D → screen space projection
4. **Alpha-blending**: Accumulation with order-awareness

**Memory Requirements**:
- For 1000×1000 image with 10,000 Gaussians:
  - Gaussian parameters: 10,000 × 8 × 4 bytes = 320 MB (full precision)
  - Tile buffer: ~50 MB
  - Intermediate gradients: ~100 MB
  - **Total**: ~500 MB (compared to 2-3 GB for NeRF-based methods)

#### Software Stack Options

**Recommended**:
- **Python frontend**: PyTorch for high-level optimization
- **CUDA backend**: Custom kernels for rasterization (inspired by 3D-GS)
- **Entropy coding**: CompressAI library (PyTorch-compatible)

**Alternative to CUDA**:
- OpenGL compute shaders (cross-platform)
- Vulkan for modern graphics APIs

---

## Part 4: Techniques for Rendering Raster and Vector Images

### 4.1 From Raster Images → Gaussian Representation

#### Initialization Strategies

**Method 1: Random Initialization** (baseline)
- Sample N random positions
- Initialize scales from neighbor distances
- Initialize colors from image patches
- Optimize from scratch (13-20 seconds per image)

**Method 2: Content-Aware Initialization** (fast)
- Run superpixel segmentation (SLIC algorithm)
- One Gaussian per superpixel segment
- Gaussian μ = segment centroid
- Gaussian Σ = segment covariance
- Gaussian c = segment mean color
- Fine-tune 1-2 seconds per image

**Method 3: Multi-scale Progressive Addition**
- Start with coarse features (few Gaussians, large support)
- Iteratively add fine detail Gaussians
- Gradient-based splits for complex regions

#### Error-Driven Optimization

**Key insight**: Optimize for **reconstruction error** not classification

```
while not converged:
    1. Render image from current Gaussians
    2. Compute loss against original image
    3. Backprop to get gradients
    4. Update Gaussian parameters via Adam/SGD
    5. Adaptive densification (optional)
```

**Convergence criteria**:
- PSNR plateaus (typically 30-35 dB sufficient)
- Gradient norm < threshold
- Maximum iterations reached

### 4.2 From Vector Graphics → Gaussian Representation

#### SVG/Bezier Curve Conversion

**Challenge**: Vector graphics use parametric curves (Bezier paths), but Gaussians operate in spatial domain

**Solution Pipeline**:
1. **Rasterize** vector graphics to fine-resolution raster image (4-8× super-resolution)
2. **Apply Gaussian fitting** to rasterized result
3. **Optimize** jointly for spatial and color accuracy

#### Bezier Splatting (Advanced Alternative)

New technique: **Bezier Splatting** (2024) samples 2D Gaussians along Bezier curves:
- 30× faster than DiffVG for vector graphic rendering
- Maintains path control points and optimization
- Produces SVG-exportable representations

**For your codec**: Less critical unless targeting vector input specifically

### 4.3 Rendering Gaussian Representation Back to Raster

#### Real-time Rendering (1500-2000 FPS)

```python
# Pseudocode for GPU kernel
for each tile (in parallel):
    for each pixel (x, y) in tile:
        color = background
        alpha_accum = 0
        for each gaussian (sorted by depth):
            # Evaluate 2D Gaussian at pixel
            dist = (x - μx)² / σx² + (y - μy)² / σy²
            value = exp(-0.5 * dist)
            # Alpha blend
            contrib = α * value * gaussian_color
            color = color * (1 - α*value) + contrib
            alpha_accum += α * value
            if alpha_accum > 0.99: break
        output[x, y] = color
```

#### Quality Enhancement via Antialiasing

**Supersampling** (accurate but slow):
- Render at 4-8× resolution
- Downsample with Gaussian filter
- Critical for avoiding aliasing artifacts

**Analytical Antialiasing** (via prefiltering):
- Account for pixel footprint during Gaussian evaluation
- Approximate impact of pixel integration analytically
- Faster than supersampling but approximate

---

## Part 5: Rate-Distortion Optimization Framework

### 5.1 Joint Rate-Distortion Formulation

**Objective**: Minimize \(D + \lambda R\) where:
- D = Distortion (reconstruction error)
- R = Rate (bitrate in bits)
- λ = Lagrange multiplier controlling trade-off

#### During Training

```
L_total = D_loss + λ * bits_estimate
```

Where bits estimate comes from entropy model:
- \(\text{bits} \approx -\log_2 P(\text{quantized values})\)
- Probability distribution learned from training data

#### Balanced R-D Optimization

**Challenge**: Rate and distortion gradients have different scales/sensitivities

**Solution** (from 2025 research):
- Monitor improvement speeds: \(\|∇D\|\) vs \(\|∇R\|\)
- Dynamically adjust relative gradient weights
- Ensures balanced progress on both objectives

### 5.2 Controlling Gaussian Count vs Quality

**Trade-off Surface**:
- Fewer Gaussians → smaller bitrate, lower quality
- More Gaussians → larger bitrate, higher quality
- Relationship is non-linear (diminishing returns)

**Implementation**:
1. Train multiple models with different λ values
2. Measure (bitrate, PSNR) for each
3. Select operating point based on requirements

---

## Part 6: Hybrid Approaches and Future Directions

### 6.1 Combining Multiple Representation Types

**Potential Architecture**:
- **High-frequency content**: 2D Gaussians (many small ones)
- **Low-frequency content**: Larger Gaussians with lower density
- **Edges/boundaries**: Specialized high-precision Gaussians
- **Background**: Single large Gaussian or constant color

### 6.2 Integration with Deep Learning

**Possible enhancements**:
- Learn initialization network (predicts good starting Gaussians)
- Learned density predictor (where to place Gaussians)
- Joint optimization with perceptual networks

### 6.3 Multi-resolution Representation

**Hierarchical structure**:
- Level 0 (coarse): ~10-100 Gaussians, low-frequency details
- Level 1 (medium): ~1,000 Gaussians, mid-frequency
- Level 2 (fine): ~10,000 Gaussians, high-frequency details
- **Benefit**: Progressive rendering, scalable transmission

---

## Part 7: Practical Implementation Roadmap

### Phase 1: Foundation (2-3 weeks)
- [ ] Implement basic 2D Gaussian representation (CPU version)
- [ ] Build differentiable alpha-blending renderer
- [ ] Implement gradient-based optimization loop
- [ ] Test on small images (256×256)
- [ ] Achieve target PSNR ~30 dB within 30 seconds per image

### Phase 2: GPU Acceleration (2-3 weeks)
- [ ] Implement tile-based CUDA rasterizer
- [ ] Add gradient computation kernels
- [ ] Integrate with PyTorch autograd
- [ ] Achieve 1000+ FPS rendering speed

### Phase 3: Compression Pipeline (2-3 weeks)
- [ ] Implement quantization-aware training
- [ ] Integrate entropy coding (CompressAI/ANS)
- [ ] Build residual vector quantization
- [ ] Measure compression ratios

### Phase 4: Optimization (2 weeks)
- [ ] Implement adaptive density control
- [ ] Add content-aware initialization
- [ ] Benchmark against COIN++, JPEG2000
- [ ] Profile and optimize bottlenecks

### Phase 5: Advanced Features (2+ weeks)
- [ ] Vector graphics input support
- [ ] Video codec extension
- [ ] Multiple quality presets
- [ ] Format specification and serialization

---

## Part 8: Key Publications and Resources

### Essential Papers

1. **GaussianImage (ECCV 2024)**
   - "1000 FPS Image Representation and Compression by 2D Gaussian Splatting"
   - Status: Most current, production-ready approach
   - Implementation: Available on GitHub

2. **Investigation into 2D Gaussian Splatting (EPFL 2024)**
   - Addresses quality-compression trade-offs
   - Adaptive density strategies

3. **Neural Video Compression using 2D Gaussian Splatting (CVPR 2025)**
   - Content-aware initialization
   - Frame-based compression

4. **3D Gaussian Splatting (SIGGRAPH 2023)**
   - Original Gaussian splatting work (for 3D, but principles apply)
   - Adaptive density control mechanisms

5. **COIN & COIN++ (Implicit Neural Representation baselines)**
   - Understanding INR-based compression
   - Rate-distortion formulation

6. **Recursive Bits-Back Coding with ANS**
   - Entropy coding framework
   - Theoretical foundations

### Related Technique References

- **DiffVG / Bézier Splatting**: Vector graphics rendering
- **LIVE**: Layer-wise image vectorization
- **WIRE**: Wavelet implicit neural representations
- **SIREN**: Sinusoidal implicit neural networks
- **NeRF**: Neural radiance fields (3D reference)

### Implementation Resources

- **CompressAI**: PyTorch-based compression library (entropy coding)
- **3D-GS CUDA source**: Reference for GPU optimization patterns
- **OpenCV**: Superpixel segmentation (SLIC)
- **PyTorch**: Automatic differentiation framework

---

## Part 9: Benchmark Datasets and Metrics

### Evaluation Datasets

- **Kodak Lossless Image Database**: 24 standard test images, 768×512 resolution
- **DIV2K**: 1000 diverse high-resolution images (2560×1440)
- **COCO**: General image dataset with diverse content
- **Technical images**: Screenshots, diagrams, charts

### Quality Metrics

**Pixel-level**:
- PSNR (Peak Signal-to-Noise Ratio): standard but limited
- SSIM (Structural Similarity Index): better correlation with perception
- MS-SSIM: Multi-scale structural similarity

**Perceptual**:
- LPIPS (Learned Perceptual Image Patch Similarity): best correlation with human judgment
- VGG/Inception features: Deep perceptual distances

**Rate metrics**:
- Bits per pixel (bpp): normalized compression ratio
- Compression ratio: original size / compressed size
- Encoding/decoding time: practical performance

### Comparison Baselines

1. **Traditional codecs**: JPEG, JPEG2000, WebP, BPG
2. **Learned codecs**: COIN++, WIRE, established deep compression
3. **Other Gaussian approaches**: 2D-GS papers

---

## Conclusion and Recommendations

**Key Takeaways**:

1. **Gaussian image codecs are viable and practical** — proven by GaussianImage achieving 2000 FPS decoding with competitive compression

2. **Theoretical foundation is solid** — grounded in classical signal processing (basis functions) with modern neural optimization

3. **Implementation is feasible** — reasonable GPU memory (500 MB), well-understood rendering pipeline (alpha blending), standard optimization techniques

4. **Research is active and evolving** — multiple recent papers (2024-2025) show this is active frontier with clear trajectory toward improvement

**For Your Implementation**:

1. **Start with content-aware initialization** — 88% faster encoding than random init
2. **Focus on GPU tile-based rasterization** — critical bottleneck for speed
3. **Use quantization-aware training** — essential for compression performance
4. **Build progressive implementation** — get basics working before optimizations
5. **Use existing entropy coding libraries** — don't reimplement ANS/arithmetic coding

This is an excellent research direction combining principled signal processing with modern optimization—very different from the overhyped deep learning codecs but potentially more practical and interpretable.
