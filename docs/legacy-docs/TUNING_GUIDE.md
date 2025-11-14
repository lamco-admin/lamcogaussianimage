# LGI v2 Tuning & Calibration Guide

**Purpose**: Reference guide for tuning parameters, understanding process flows, and identifying which techniques to apply to different image regions.

**Created**: Session 8 (2025-10-07)
**Status**: Living document - update as we discover more

---

## Table of Contents

1. [Process Flows](#process-flows)
2. [Key Tuning Questions](#key-tuning-questions)
3. [Parameter Reference](#parameter-reference)
4. [Image Analysis for Technique Selection](#image-analysis)
5. [Debugging & Instrumentation](#debugging)
6. [Performance Bottlenecks](#performance)

---

## Process Flows

### 1. Complete Encoding Pipeline

```
User calls: encoder.encode_error_driven_adam(n_init, n_max)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PREPROCESSING (EncoderV2::new)                              │
├─────────────────────────────────────────────────────────────┤
│ 1. Compute structure tensor (σ_gradient=1.2, σ_smooth=1.0) │
│    → Finds edge orientations for Gaussian alignment        │
│    → ~50ms for Kodak, ~500ms for 4K                        │
│                                                             │
│ 2. Compute geodesic EDT (coherence=0.7, penalty=50.0)      │
│    → Anti-bleeding distance field                          │
│    → ~100ms for Kodak, ~1s for 4K                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ INITIALIZATION                                              │
├─────────────────────────────────────────────────────────────┤
│ 1. Create grid: grid_size = sqrt(n_init)                   │
│    → Kodak: n_init=31 → 6×6 grid (rounded up)             │
│                                                             │
│ 2. For each grid point:                                    │
│    - Position: grid coordinate                             │
│    - Color: sample from image                              │
│    - Covariance: from structure tensor (edge-aligned)      │
│    - Size: clamped by geodesic EDT (anti-bleeding)         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ ADAPTIVE REFINEMENT (up to 10 passes)                      │
├─────────────────────────────────────────────────────────────┤
│ FOR pass = 0 to 9:                                         │
│                                                             │
│   ┌───────────────────────────────────────────────────────┐│
│   │ OPTIMIZATION (100 iterations)                         ││
│   ├───────────────────────────────────────────────────────┤│
│   │ FOR iter = 0 to 99:                                   ││
│   │   1. FORWARD: Render N Gaussians                      ││
│   │      - CPU: ~100ms for N=124, ~4s for N=616          ││
│   │      - GPU: ~20ms for N=124, ~1s for N=616           ││
│   │                                                        ││
│   │   2. LOSS: Compare rendered vs target                 ││
│   │      loss = avg((target - rendered)²)                 ││
│   │      - Should decrease each iteration                 ││
│   │      - Currently oscillating (needs tuning)           ││
│   │                                                        ││
│   │   3. BACKWARD: Compute gradients                      ││
│   │      ∂loss/∂(position, color, scale, rotation)        ││
│   │      - Analytical derivatives                         ││
│   │      - ~100-200ms for Kodak                           ││
│   │                                                        ││
│   │   4. UPDATE: Apply gradient descent                   ││
│   │      param -= learning_rate × gradient                ││
│   │      - Adaptive LR based on density                   ││
│   │      - Clamp to valid ranges                          ││
│   │                                                        ││
│   │   5. EARLY STOP: If no improvement for 20 iters       ││
│   │      - Restore best parameters                        ││
│   │      - Prevents divergence                            ││
│   └───────────────────────────────────────────────────────┘│
│                                                             │
│   ✓ Geodesic EDT clamping (anti-bleeding)                 │
│   ✓ Render final result                                   │
│   ✓ Compute PSNR                                           │
│                                                             │
│   IF loss < 0.001: CONVERGED ✓                            │
│   IF N >= n_max: MAX REACHED ⚠                            │
│                                                             │
│   ┌───────────────────────────────────────────────────────┐│
│   │ ERROR-DRIVEN SPLITTING                                ││
│   ├───────────────────────────────────────────────────────┤│
│   │ 1. Compute error map:                                 ││
│   │    error[pixel] = |target - rendered|                 ││
│   │                                                        ││
│   │ 2. Find hotspots (top 10% error pixels)              ││
│   │                                                        ││
│   │ 3. Add Gaussian at each hotspot:                     ││
│   │    - Position: hotspot location                       ││
│   │    - Color: target pixel color                        ││
│   │    - Size: σ=0.02 (fixed, needs tuning!)            ││
│   └───────────────────────────────────────────────────────┘│
│                                                             │
│ END FOR                                                     │
└─────────────────────────────────────────────────────────────┘
    ↓
Return: Vec<Gaussian2D> (final optimized Gaussians)
```

### 2. Single Iteration Detail (Optimizer)

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: Current Gaussians[N], Target Image                   │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│ 1. RENDER (Forward Pass)                                     │
├──────────────────────────────────────────────────────────────┤
│ rendered_image = zeros(width, height)                        │
│                                                              │
│ FOR each pixel (x, y):                                       │
│   FOR each gaussian in Gaussians:                            │
│     weight = gaussian.evaluate_2d(x, y)  // Gaussian PDF    │
│     rendered[x,y] += weight × gaussian.color                 │
│                                                              │
│ Complexity: O(width × height × N)                           │
│ Kodak: 393K × 124 = 48.7M evaluations (~100ms CPU)         │
│ 4K:    9.6M × 616 = 5.9B evaluations (~4s CPU)             │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. COMPUTE LOSS                                              │
├──────────────────────────────────────────────────────────────┤
│ mse = 0                                                      │
│ FOR each pixel:                                              │
│   diff_r = rendered.r - target.r                             │
│   diff_g = rendered.g - target.g                             │
│   diff_b = rendered.b - target.b                             │
│   mse += diff_r² + diff_g² + diff_b²                        │
│                                                              │
│ loss = mse / (pixels × 3)                                    │
│                                                              │
│ PSNR = -10 × log10(loss)  (if loss > 0)                    │
│                                                              │
│ Examples:                                                    │
│   loss=0.001 → PSNR=30 dB (excellent)                      │
│   loss=0.01  → PSNR=20 dB (good)                           │
│   loss=0.1   → PSNR=10 dB (poor)                           │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. COMPUTE GRADIENTS (Backward Pass)                         │
├──────────────────────────────────────────────────────────────┤
│ pixel_gradients = compute ∂loss/∂(rendered_pixel)           │
│   → For L2: pixel_grad = 2(rendered - target)              │
│   → For MS-SSIM: more complex (perceptual)                  │
│                                                              │
│ FOR each gaussian:                                           │
│   grad = GaussianGradient::zero()                           │
│                                                              │
│   FOR each pixel:                                            │
│     contribution = gaussian.evaluate_2d(pixel)              │
│     IF contribution > threshold:                             │
│       grad.color += pixel_grad × contribution               │
│       grad.position += pixel_grad × ∂contribution/∂pos      │
│       grad.scale += pixel_grad × ∂contribution/∂scale       │
│       grad.rotation += pixel_grad × ∂contribution/∂rotation │
│                                                              │
│ Complexity: O(width × height × N)  // Same as forward!     │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. UPDATE PARAMETERS                                         │
├──────────────────────────────────────────────────────────────┤
│ // Adaptive learning rate based on density                  │
│ density_factor = sqrt(100 / N).min(1.0)                     │
│ lr_color_adaptive = 0.6 × density_factor                    │
│ lr_position_adaptive = 0.1 × density_factor                 │
│ lr_scale_adaptive = 0.1 × density_factor                    │
│ lr_rotation_adaptive = 0.02 × density_factor                │
│                                                              │
│ FOR each gaussian:                                           │
│   gaussian.color -= lr_color × grad.color                   │
│   gaussian.position -= lr_position × grad.position          │
│   gaussian.scale -= lr_scale × grad.scale                   │
│   gaussian.rotation -= lr_rotation × grad.rotation          │
│                                                              │
│   // Clamp to valid ranges                                  │
│   gaussian.color = clamp(color, 0.0, 1.0)                   │
│   gaussian.position = clamp(position, 0.0, 1.0)             │
│   gaussian.scale = clamp(scale, 0.01, 0.25)                 │
│   gaussian.rotation = normalize_angle(rotation)             │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│ 5. EARLY STOPPING CHECK                                     │
├──────────────────────────────────────────────────────────────┤
│ IF loss < best_loss:                                         │
│   best_loss = loss                                           │
│   best_gaussians = current_gaussians.clone()                │
│   patience_counter = 0                                       │
│ ELSE:                                                        │
│   patience_counter++                                         │
│   IF patience_counter >= 20:                                │
│     gaussians = best_gaussians  // Restore                  │
│     BREAK  // Stop iterating                                │
└──────────────────────────────────────────────────────────────┘
    ↓
RETURN: final loss
```

---

## Foundational Image Analysis (CRITICAL - DO THIS FIRST!)

**Key Insight from Research**: The number of Gaussians should be **data-driven**, not formula-driven.

Before we can tune anything, we need to properly **characterize the image** to inform Gaussian allocation.

### What to Compute Before Gaussian Placement

#### 1. **Local Entropy Map** ✅ EXISTS (`lgi-core/src/entropy.rs`)
```rust
// Measures: "How unpredictable/complex is this region?"
entropy[tile] = variance_based_entropy(tile)
```
- **High entropy** = textures, noise, complex patterns → Need MORE Gaussians
- **Low entropy** = solid colors, smooth gradients → Need FEWER Gaussians
- **Currently**: Uses variance as entropy proxy (standard deviation)

#### 2. **Gradient Magnitude Map** ✅ EXISTS (Structure Tensor)
```rust
// Measures: "How fast is color changing?"
gradient[pixel] = ||∇I||
```
- **High gradient** = edges, boundaries → Need edge-aligned Gaussians
- **Low gradient** = smooth regions → Few large Gaussians sufficient
- **Currently**: Computed via structure tensor (σ_gradient=1.2)

#### 3. **Position Probability Map (PPM)** ❌ MISSING (IMPLEMENT THIS!)
```rust
// Combine entropy + gradient → placement probability
ppm[pixel] = α×entropy[pixel] + β×gradient[pixel]
```
- **Purpose**: Where to place each Gaussian?
- **Research-backed**: Floyd-Steinberg dithering or importance sampling
- **Priority**: HIGH - This is the missing link!

#### 4. **Spectral Entropy** ❌ MISSING (ADVANCED)
```rust
// Measures: "How much high-frequency content?"
spectral_entropy = entropy(eigenvalues(covariance))
```
- **Use**: Decide when to split vs clone Gaussians
- **Priority**: MEDIUM - Can add after basic PPM works

#### 5. **Laplacian Map** ⚠️ PARTIAL (Have structure tensor)
```rust
// Measures: "Where are the sharp transitions?"
laplacian = ∇²I
```
- **Use**: Gradient Domain Gaussian Splatting (10-100× fewer Gaussians!)
- **Priority**: HIGH - But requires Poisson solver

### Research-Backed Strategies for Gaussian Count

**Current (BROKEN):**
```rust
// Arbitrary formula - NO image analysis!
n_init = sqrt(pixels) / 20.0
n_max = n_init × 4
```

**Strategy 1: Entropy-Driven** ✅ **WE HAVE THIS!**
```rust
// From lgi-core/src/entropy.rs
fn adaptive_gaussian_count(image: &ImageBuffer) -> usize {
    let entropy = compute_image_entropy(image);  // Tile-based variance
    let pixels = image.width * image.height;

    // Adaptive: More entropy → More Gaussians
    let count = pixels × 0.005 × (1.0 + 3.0 × entropy);
    return count.clamp(50, 50000);
}
```
**Used in**: `EncoderV2::auto_gaussian_count()`

**Strategy 2: Position Probability Map** ❌ **IMPLEMENT NEXT!**
```rust
fn generate_ppm(image: &ImageBuffer) -> Vec<f32> {
    let entropy_map = compute_entropy_map(image, tile_size=8);
    let gradient_map = structure_tensor.gradient_magnitude();

    // Combine: 60% entropy, 40% gradient (TUNABLE!)
    let ppm = 0.6 × entropy_map + 0.4 × gradient_map;
    normalize(ppm);  // Sum to 1.0

    return ppm;
}

// Then sample Gaussian positions from PPM
let positions = importance_sample(ppm, n_gaussians);
```

**Strategy 3: Gradient Domain Splatting (GDGS)** ❌ **RESEARCH/FUTURE**
```rust
// Place Gaussians ONLY at Laplacian peaks (edges)
let laplacian = compute_laplacian(image);
let peaks = find_local_maxima(laplacian);
// Result: 10-100× fewer Gaussians for same quality!
```

### Immediate Action: Use Existing `auto_gaussian_count()`

**STOP using arbitrary formula!**

**Current bad code** (`real_world_benchmark.rs:94`):
```rust
let n_init = (pixels.sqrt() / 20.0) as usize;  // ARBITRARY!
```

**Change to**:
```rust
let n_init = encoder.auto_gaussian_count();  // DATA-DRIVEN!
```

### Initial Results (Initialization Quality Only, No Optimization)

**Test**: 4 Kodak images (768×512)

| Strategy | Avg N | Avg PSNR | Time | Gain |
|----------|-------|----------|------|------|
| Arbitrary | 31 | 17.03 dB | 78ms | baseline |
| Entropy | 2635 | **19.96 dB** | 4.7s | **+2.93 dB** ✅ |
| Hybrid | 1697 | 18.67 dB | 3.0s | +1.64 dB |

**Winner**: Entropy-based (3/4 wins)

**Critical Question**: Does this advantage hold AFTER optimization?
- Entropy uses 80× more Gaussians
- Optimization time scales with N² → **6400× slower!**
- Need to test with full Adam optimization next

---

## ALL Research-Backed Strategies (Comprehensive List)

Based on gaussian-count-and-merging-splitting-research-starters.md and latest papers.

### Category 1: N Determination (How Many Gaussians Total?)

#### Strategy A: **Content-Adaptive Density** ✅ IMPLEMENTED
```rust
// Current: lgi-core/src/entropy.rs
n = pixels × density × (1 + entropy_factor × normalized_entropy)
```
- **Input**: Tile-based variance (entropy proxy)
- **Output**: Single N for whole image
- **Pros**: Simple, data-driven
- **Cons**: Doesn't consider spatial distribution
- **Status**: Working, used in `auto_gaussian_count()`

#### Strategy B: **Gradient-Based Scaling** ✅ IMPLEMENTED
```rust
// Current: hybrid_gaussian_count()
n_hybrid = n_entropy × (0.6 + 0.4 × gradient_factor)
```
- **Input**: Mean gradient magnitude from structure tensor
- **Output**: N adjusted for edge density
- **Pros**: Accounts for detail vs smooth
- **Cons**: Still global, not per-region
- **Status**: Working, ~1.6 dB gain

#### Strategy C: **Neural Position Probability Map** ❌ FUTURE
```rust
// From research: Train CNN to predict PPM
let ppm = neural_network(image_features);  // [W×H] probabilities
let positions = floyd_steinberg_dithering(ppm, n);
```
- **Input**: Image patches → CNN features
- **Output**: Per-pixel placement probability
- **Pros**: Learned from data, optimal
- **Cons**: Requires training, heavy dependency
- **Priority**: LOW - Overkill for now

#### Strategy D: **K-Means Clustering** ❌ TO IMPLEMENT
```rust
// Cluster image patches, place Gaussian at centroids
let features = extract_features(image);  // Color, gradient, etc.
let clusters = kmeans(features, k=n);
let positions = cluster_centroids(clusters);
```
- **Input**: Pixel features (color, gradient, texture)
- **Output**: N cluster centers as Gaussian positions
- **Pros**: Distributes Gaussians based on similarity
- **Cons**: K-means slow for large images
- **Priority**: MEDIUM - Classic, proven

#### Strategy E: **SLIC Superpixels** ❌ TO IMPLEMENT
```rust
// Better clustering: SLIC (Simple Linear Iterative Clustering)
let superpixels = slic_segmentation(image, n_segments=n);
for segment in superpixels:
    position = segment.centroid
    color = segment.mean_color
    covariance = segment.covariance
    gaussians.push(Gaussian::from_superpixel(segment))
```
- **Input**: Image → SLIC segmentation
- **Output**: One Gaussian per superpixel
- **Pros**: Respects boundaries, perceptually meaningful
- **Cons**: Requires SLIC implementation (or Python call)
- **Priority**: HIGH - State-of-art initialization
- **Reference**: scikit-image.segmentation.slic

---

### Category 2: Position Determination (Where to Place Each Gaussian?)

#### Strategy F: **Uniform Grid** ✅ CURRENT DEFAULT
```rust
// Current: lib.rs initialize_gaussians
let grid_size = sqrt(n);
for y in 0..grid_size:
    for x in 0..grid_size:
        position = (x / grid_size, y / grid_size)
```
- **Pros**: Simple, fast, deterministic
- **Cons**: Wastes Gaussians in smooth regions
- **Status**: Default initialization

#### Strategy G: **Position Probability Map (PPM) Sampling** ❌ TO IMPLEMENT
```rust
// Combine entropy + gradient → probability map
let entropy_map = compute_entropy_map(image, tile_size=8);
let gradient_map = structure_tensor.gradient_magnitude();

// Per-pixel probability
let mut ppm = vec![0.0; pixels];
for i in 0..pixels:
    ppm[i] = alpha × entropy_map[i] + beta × gradient_map[i];

// Normalize
let sum: f32 = ppm.sum();
ppm.iter_mut().for_each(|p| *p /= sum);

// Sample N positions using Floyd-Steinberg dithering
let positions = floyd_steinberg_sample(ppm, n);
```
- **Input**: Entropy map + gradient map
- **Output**: Non-uniform Gaussian positions
- **Pros**: More Gaussians where needed, fewer in smooth areas
- **Cons**: Need to implement dithering/importance sampling
- **Priority**: HIGH - Core of Strategy 1
- **Papers**: arXiv:2506.23479v1, arXiv:2405.05446v1

#### Strategy H: **Gradient Peak Sampling** ❌ TO IMPLEMENT
```rust
// Place Gaussians at local gradient maxima (edges, corners)
let gradient_map = compute_gradient_magnitude(image);
let peaks = find_local_maxima(gradient_map, window=5);

// Take top N peaks
peaks.sort_by_gradient_descending();
let positions = peaks[0..n];
```
- **Input**: Gradient magnitude
- **Output**: Positions at edges/corners
- **Pros**: Focuses on details
- **Cons**: Ignores smooth regions entirely
- **Priority**: MEDIUM - Good for high-detail images

#### Strategy I: **Laplacian Peak Sampling (GDGS)** ❌ TO IMPLEMENT
```rust
// Gradient Domain Gaussian Splatting
let laplacian = compute_laplacian(image);  // ∇²I
let peaks = find_local_maxima(laplacian);   // Sharp transitions

// Place Gaussian at each peak
for peak in peaks:
    gaussians.push(Gaussian {
        position: peak,
        orientation: gradient_direction(image, peak),
        scale: laplacian_magnitude_to_scale(laplacian[peak]),
    });

// Reconstruct via Poisson solver
let reconstructed = poisson_solve(gaussians);
```
- **Input**: Laplacian (second derivative)
- **Output**: Sparse Gaussian placement (10-100× fewer!)
- **Pros**: Extremely efficient, mathematically elegant
- **Cons**: Requires Poisson solver implementation
- **Priority**: HIGH - Huge compression potential
- **Papers**: arXiv:2405.05446v1 (GDGS)

---

### Category 3: Split/Merge Strategies (Adaptive Refinement)

#### Strategy J: **Loss-Driven Splitting** ⚠️ PARTIALLY IMPLEMENTED
```rust
// Current: error_driven.rs adds Gaussians at high-error pixels
// Better: SPLIT existing Gaussians instead of just adding

for gaussian in gaussians:
    let region_loss = compute_gaussian_region_loss(gaussian, image);

    if region_loss > split_threshold:
        if gaussian.scale > max_scale:
            // Large Gaussian in high-error region → split into 2
            let (g1, g2) = split_gaussian_along_major_axis(gaussian);
            add_gaussians([g1, g2]);
            remove_gaussian(gaussian);
        else:
            // Small Gaussian → clone and perturb
            let g_clone = clone_and_perturb(gaussian);
            add_gaussian(g_clone);
```
- **Input**: Per-Gaussian reconstruction error
- **Output**: Split or clone decision
- **Pros**: Refines existing allocation
- **Cons**: Requires per-Gaussian loss computation
- **Priority**: HIGH - Core 3D splatting technique
- **Papers**: 3D Gaussian Splatting (Kerbl et al. 2023)

#### Strategy K: **Spectral Entropy Splitting** ❌ TO IMPLEMENT
```rust
// From Spectral-GS paper
for gaussian in gaussians:
    let cov = gaussian.covariance_matrix();
    let eigenvalues = cov.eigenvalues();

    // Spectral entropy = entropy of eigenvalue distribution
    let spectral_entropy = -sum(λ × log(λ)) for λ in eigenvalues;

    if spectral_entropy > THRESHOLD:
        // High-frequency content → split
        split_gaussian(gaussian);
```
- **Input**: Gaussian covariance eigenvalues
- **Output**: Split if representing high-frequency content
- **Pros**: Prevents elongated artifacts, shape-aware
- **Cons**: More complex, needs tuning
- **Priority**: MEDIUM - Quality improvement
- **Papers**: arXiv:2409.12771v1 (Spectral-GS)

#### Strategy L: **Redundancy-Based Merging** ❌ TO IMPLEMENT
```rust
// Merge overlapping Gaussians in low-error regions
for (i, j) in all_pairs(gaussians):
    let overlap = compute_overlap(gaussians[i], gaussians[j]);
    let region_error = compute_region_error(i, j);

    if overlap > 0.5 && region_error < merge_threshold:
        // Highly overlapping + low error → merge
        let merged = merge_gaussians(gaussians[i], gaussians[j]);
        remove_gaussians([i, j]);
        add_gaussian(merged);
```
- **Input**: Gaussian positions and reconstruction error
- **Output**: Merged Gaussians
- **Pros**: Reduces redundancy, compacts representation
- **Cons**: Need robust merging formula
- **Priority**: MEDIUM - Compression benefit
- **Papers**: Split-Merge EM (Ueda et al.)

#### Strategy M: **Opacity-Based Pruning** ❌ TO IMPLEMENT
```rust
// Periodic pruning of low-contribution Gaussians
if iteration % 1000 == 0:
    gaussians.retain(|g| g.alpha > min_opacity);
```
- **Input**: Per-Gaussian opacity
- **Output**: Pruned Gaussian set
- **Pros**: Removes useless Gaussians
- **Cons**: May remove Gaussians that could improve later
- **Priority**: LOW - Minor cleanup
- **Papers**: 3D Gaussian Splatting

---

### Category 4: Advanced Initialization Methods

#### Strategy N: **Hierarchical Multi-Scale** ❌ TO IMPLEMENT
```rust
// Laplacian pyramid: coarse → fine
let pyramid = build_laplacian_pyramid(image, levels=3);

let mut gaussians = Vec::new();

// Level 0 (coarse): Large Gaussians for base structure
let coarse_n = n / 4;
gaussians.extend(initialize_grid(pyramid[0], coarse_n));

// Level 1 (medium): Medium Gaussians for details
let medium_n = n / 2;
gaussians.extend(initialize_from_residual(pyramid[1], medium_n));

// Level 2 (fine): Small Gaussians for high-freq
let fine_n = n / 4;
gaussians.extend(initialize_from_residual(pyramid[2], fine_n));
```
- **Input**: Laplacian pyramid
- **Output**: Multi-scale Gaussian distribution
- **Pros**: Natural decomposition, matches image structure
- **Cons**: Complex, requires pyramid implementation
- **Priority**: MEDIUM - Interesting approach
- **Research**: Mentioned as "creative opportunity"

#### Strategy O: **Saliency-Weighted Allocation** ⚠️ PARTIAL (have saliency)
```rust
// Allocate more Gaussians to salient regions
let saliency_map = compute_saliency(image);  // We have this!

// Use saliency to weight PPM
let mut ppm = combine(entropy_map, gradient_map);
ppm = ppm × (1.0 + saliency_weight × saliency_map);

let positions = sample(ppm, n);
```
- **Input**: Saliency map (we have `SaliencyDetector`)
- **Output**: More Gaussians on faces, text, focal points
- **Pros**: Perceptually motivated
- **Cons**: Saliency detection may be imperfect
- **Priority**: MEDIUM - Perceptual quality
- **Research**: Mentioned as perceptual model approach

#### Strategy P: **Semantic Segmentation-Based** ❌ FUTURE
```rust
// Different Gaussian allocation per object type
let segments = segment_anything(image);  // External model

for segment in segments:
    let object_type = segment.label;  // "sky", "face", "grass", etc.

    let n_segment = match object_type:
        "sky" | "water" => segment.area × 0.0001,  // Very smooth
        "face" | "text" => segment.area × 0.01,    // High detail
        "foliage" | "texture" => segment.area × 0.005,  // Medium
        _ => segment.area × 0.003,

    gaussians.extend(initialize_segment(segment, n_segment));
```
- **Input**: Semantic segmentation (Segment Anything, Detectron2)
- **Output**: Per-object Gaussian allocation
- **Pros**: Content-aware, optimal resource use
- **Cons**: Requires external deep learning model
- **Priority**: LOW - Future research
- **Research**: Mentioned as "semantic partitioning"

---

### Category 5: Sampling Methods (From PPM to Positions)

#### Strategy Q: **Floyd-Steinberg Dithering** ❌ TO IMPLEMENT
```rust
// Convert continuous PPM to discrete positions
fn floyd_steinberg_sample(ppm: &[f32], n: usize) -> Vec<(u32, u32)> {
    let mut positions = Vec::new();
    let mut error = vec![0.0; ppm.len()];

    let threshold = 1.0 / n as f32;

    for y in 0..height:
        for x in 0..width:
            let value = ppm[y*width + x] + error[y*width + x];

            if value >= threshold:
                positions.push((x, y));
                error[y*width + x] = value - 1.0;
            else:
                error[y*width + x] = value;

            // Diffuse error to neighbors (Floyd-Steinberg weights)
            if x + 1 < width:
                error[y*width + (x+1)] += error[y*width + x] × 7.0/16.0;
            // ... (down-left, down, down-right)

    return positions;
}
```
- **Input**: PPM (continuous probabilities)
- **Output**: N discrete positions
- **Pros**: Standard technique, good distribution
- **Cons**: Sequential (not parallel)
- **Priority**: HIGH - Needed for PPM strategies
- **Papers**: Standard image processing, cited in research

#### Strategy R: **Importance Sampling** ❌ TO IMPLEMENT
```rust
// Sample positions from PPM using cumulative distribution
fn importance_sample(ppm: &[f32], n: usize) -> Vec<(u32, u32)> {
    // Build CDF
    let mut cdf = vec![0.0; ppm.len()];
    cdf[0] = ppm[0];
    for i in 1..ppm.len():
        cdf[i] = cdf[i-1] + ppm[i];

    // Sample N positions
    let mut positions = Vec::new();
    for _ in 0..n:
        let r = random(0.0, 1.0);
        let index = binary_search(cdf, r);
        let (x, y) = index_to_coords(index, width);
        positions.push((x, y));

    return positions;
}
```
- **Input**: PPM (probabilities)
- **Output**: N random positions
- **Pros**: Parallel-friendly, stochastic
- **Cons**: Random (not deterministic)
- **Priority**: HIGH - Alternative to Floyd-Steinberg

#### Strategy S: **Poisson Disk Sampling** ❌ TO IMPLEMENT
```rust
// Ensure minimum distance between Gaussians
fn poisson_disk_sample(ppm: &[f32], n: usize, min_distance: f32) -> Vec<(u32, u32)> {
    // Mitchell's algorithm
    let mut positions = Vec::new();
    let mut active = Vec::new();

    // Start with random high-probability point
    let start = sample_from_ppm(ppm, 1)[0];
    positions.push(start);
    active.push(start);

    while !active.is_empty() && positions.len() < n:
        let point = active.pop_random();

        // Try to place points around this one
        for _ in 0..30:  // k attempts
            let candidate = sample_annulus(point, min_distance, 2×min_distance, ppm);

            if far_enough(candidate, positions, min_distance):
                positions.push(candidate);
                active.push(candidate);

    return positions;
}
```
- **Input**: PPM + minimum distance constraint
- **Output**: Well-distributed positions (no clumping)
- **Pros**: Good coverage, no overlap
- **Cons**: Complex algorithm, slower
- **Priority**: MEDIUM - Quality benefit
- **Use Case**: Prevent Gaussian clumping

---

### Category 6: Iterative Refinement Strategies

#### Strategy T: **Split-Merge EM** ❌ TO IMPLEMENT
```rust
// Expectation-Maximization with split/merge operations
loop:
    // E-step: Assign pixels to nearest Gaussian
    let assignments = assign_pixels_to_gaussians(image, gaussians);

    // M-step: Update Gaussian parameters from assignments
    for (i, gaussian) in gaussians.iter_mut().enumerate():
        let pixels = get_assigned_pixels(assignments, i);
        gaussian.update_from_pixels(pixels);

    // Split: Low-likelihood Gaussians
    for gaussian in gaussians:
        if likelihood(gaussian) < split_threshold:
            split_gaussian(gaussian);

    // Merge: Overlapping Gaussians
    for (i, j) in find_overlapping_pairs():
        if similar_parameters(gaussians[i], gaussians[j]):
            merge_gaussians(i, j);

    // Check convergence
    if change < epsilon: break
```
- **Input**: Initial Gaussian set
- **Output**: Refined via EM iterations
- **Pros**: Principled statistical framework
- **Cons**: Can get stuck in local minima
- **Priority**: LOW - Complex, may not outperform gradient descent
- **Papers**: Ueda et al. (Split-Merge EM)

#### Strategy U: **Adaptive Densification (3D Splatting Style)** ❌ TO IMPLEMENT
```rust
// From 3D Gaussian Splatting - clone small, split large
every_100_iterations:
    for gaussian in gaussians:
        let grad_mag = gradient_magnitude(gaussian);

        if grad_mag > densify_threshold:
            if gaussian.scale > scale_threshold:
                // Large + high gradient → split
                let (g1, g2) = split_along_major_axis(gaussian);
                add([g1, g2]);
                remove(gaussian);
            else:
                // Small + high gradient → clone
                add(gaussian.clone());
```
- **Input**: Per-Gaussian gradients
- **Output**: Dynamic Gaussian allocation
- **Pros**: Proven in 3D splatting, adaptive
- **Cons**: Adds complexity to optimization loop
- **Priority**: HIGH - Core technique from SOTA
- **Papers**: 3D Gaussian Splatting (Kerbl et al.)

---

### Complete Strategy Matrix

| ID | Strategy | Type | Status | Priority | Expected Gain | Effort |
|----|----------|------|--------|----------|---------------|--------|
| A | Content-Adaptive Density | N-count | ✅ Done | - | - | - |
| B | Gradient Scaling | N-count | ✅ Done | - | +1.6 dB | - |
| C | Neural PPM | N-count | ❌ | LOW | Unknown | High |
| D | K-Means | N-count | ❌ | MEDIUM | +1-2 dB? | Medium |
| E | SLIC Superpixels | Position | ❌ | **HIGH** | **+3-5 dB?** | Medium |
| F | Uniform Grid | Position | ✅ Done | - | - | - |
| G | PPM + Dithering | Position | ❌ | **HIGH** | **+2-4 dB?** | Medium |
| H | Gradient Peaks | Position | ❌ | MEDIUM | +1-2 dB? | Low |
| I | GDGS (Laplacian) | Position | ❌ | **HIGH** | **10-100× fewer N!** | High |
| J | Loss-Driven Split | Refine | ⚠️ Partial | **HIGH** | **+2-3 dB?** | Medium |
| K | Spectral Entropy Split | Refine | ❌ | MEDIUM | +1-2 dB? | High |
| L | Redundancy Merge | Refine | ❌ | MEDIUM | -10% N | Medium |
| M | Opacity Pruning | Refine | ❌ | LOW | -5% N | Low |
| N | Hierarchical Multi-Scale | Init | ❌ | MEDIUM | +2-3 dB? | High |
| O | Saliency-Weighted | Position | ⚠️ Partial | MEDIUM | +1 dB? | Low |
| P | Semantic Segmentation | N-count | ❌ | LOW | +3-5 dB? | Very High |
| Q | Floyd-Steinberg | Sampling | ❌ | HIGH | - | Low |
| R | Importance Sampling | Sampling | ❌ | HIGH | - | Low |
| S | Poisson Disk | Sampling | ❌ | MEDIUM | +0.5 dB? | Medium |
| T | Split-Merge EM | Refine | ❌ | LOW | Unknown | High |
| U | Adaptive Densification | Refine | ❌ | **HIGH** | **+3-4 dB?** | Medium |

### Immediate Priorities (Next 1-2 Weeks)

**Week 1: Foundation**
1. ✅ Strategy E: SLIC Superpixels (best initialization in research)
2. ✅ Strategy G: PPM + Floyd-Steinberg dithering (non-uniform placement)
3. ✅ Strategy J: Loss-driven splitting (split large, clone small)

**Week 2: Refinement**
4. Strategy U: Adaptive densification (from 3D splatting)
5. Strategy I: GDGS (if time permits - requires Poisson solver)

**Later:**
- Strategy K: Spectral entropy splitting
- Strategy L: Gaussian merging
- Strategy M: Opacity pruning

---

## Key Tuning Questions

### Q1: How Many Gaussians (N)?

**Current Formula:**
```rust
n_init = sqrt(width × height) / 20.0
n_max = n_init × 4
```

**Examples:**
- Kodak (768×512): n_init=31, n_max=124
- 4K (4128×2322): n_init=154, n_max=616

**Questions to Answer:**
1. ❓ Is dividing by 20 optimal? Should it be 10? 30? 50?
2. ❓ Should n_max be 4× n_init? Or 2×? 8×?
3. ❓ Should N scale linearly with pixels? Or sqrt(pixels)?
4. ❓ Does optimal N depend on image complexity (edges, textures)?
5. ❓ Quality vs size tradeoff: How much PSNR per Gaussian?

**Where to Tune:**
- `real_world_benchmark.rs:94` (n_init formula)
- `real_world_benchmark.rs:95` (n_max multiplier)
- Or use `auto_gaussian_count()` based on image entropy

**Experiment Plan:**
- Sweep n_init divisor: 10, 15, 20, 25, 30
- Sweep n_max multiplier: 2×, 4×, 8×, 16×
- Measure PSNR vs N curve
- Find knee point (diminishing returns)

---

### Q2: Learning Rates

**Current Values:**
```rust
learning_rate_color = 0.6        // Color (RGB)
learning_rate_position = 0.1     // Position (x, y)
learning_rate_scale = 0.1        // Size (scale_x, scale_y)
learning_rate_rotation = 0.02    // Rotation angle

// Adaptive scaling for high N
density_factor = sqrt(100 / N).min(1.0)
lr_adaptive = lr_base × density_factor
```

**Questions to Answer:**
1. ❓ Why is color LR 6× higher than position? Is this optimal?
2. ❓ Why is rotation LR so low (0.02)?
3. ❓ Should LRs decrease over iterations (learning rate schedule)?
4. ❓ Is density scaling formula correct? Should it be linear instead of sqrt?
5. ❓ Different LRs for different image types (smooth vs textured)?

**Where Loss Oscillates:**
- Current: loss goes 0.018 → 0.065 → 0.071 → 0.021 (BAD)
- Expected: loss should monotonically decrease
- **Likely cause**: LR too high, overshooting minima

**Where to Tune:**
- `optimizer_v2.rs:52-55` (base learning rates)
- `optimizer_v2.rs:130` (density factor formula)
- `adam_optimizer.rs` (Adam optimizer has its own beta1, beta2 params)

**Experiment Plan:**
- Halve all LRs, check if oscillation stops
- Try learning rate schedules: `lr = lr_init × 0.95^iteration`
- Use Adam optimizer properly with momentum

---

### Q3: Optimization Iterations

**Current Values:**
```rust
max_iterations = 100          // Per refinement pass
patience_limit = 20           // Early stop if no improvement
max_passes = 10               // Refinement passes
```

**Questions to Answer:**
1. ❓ Is 100 iterations enough? Too many?
2. ❓ Should max_iterations increase for larger images?
3. ❓ Is patience=20 too aggressive? Missing slow improvements?
4. ❓ Is 10 refinement passes optimal?
5. ❓ Total time budget: What's acceptable for encoding?

**Current Performance:**
- Kodak (N=124): 100 iters × ~250ms = 25s per pass
- 4K (N=616): 100 iters × ~4s = 400s per pass (6.7 min!)

**Where to Tune:**
- `lib.rs:221` (max_iterations)
- `optimizer_v2.rs:56` (max_iterations default)
- `optimizer_v2.rs:138` (patience_limit)
- `lib.rs:226` (max refinement passes)

**Experiment Plan:**
- Test different iteration counts: 50, 100, 200, 500
- Measure PSNR vs time tradeoff
- Find optimal early stopping criteria

---

### Q4: Error-Driven Splitting

**Current Values:**
```rust
target_error = 0.001          // Stop when loss < this (30 dB)
split_percentile = 0.10       // Split top 10% error pixels
new_gaussian_sigma = 0.02     // Size of added Gaussians
```

**Questions to Answer:**
1. ❓ Is 10% the right percentile? Too aggressive? Too conservative?
2. ❓ Should σ=0.02 be fixed? Or adaptive based on error magnitude?
3. ❓ Should we add multiple Gaussians per hotspot?
4. ❓ Should hotspot size depend on local image complexity?
5. ❓ Is target_error=0.001 achievable? Or too ambitious?

**Current Behavior:**
- Always hits n_max before reaching target_error
- Suggests either:
  - Target is too ambitious
  - Or not enough Gaussians allowed

**Where to Tune:**
- `lib.rs:223` (target_error)
- `lib.rs:224` (split_percentile)
- `lib.rs:273` (new_gaussian_sigma)

**Experiment Plan:**
- Vary split_percentile: 5%, 10%, 20%, 30%
- Vary new sigma: 0.01, 0.02, 0.05, adaptive
- Track: How many Gaussians added per pass?

---

### Q5: Structure Tensor Parameters

**Current Values:**
```rust
σ_gradient = 1.2              // Gradient computation blur
σ_smooth = 1.0                // Smoothing blur
```

**Questions to Answer:**
1. ❓ Are these values optimal? Data-driven?
2. ❓ Should they vary by image resolution?
3. ❓ Should they vary by content (smooth vs detailed)?

**Where to Tune:**
- `lib.rs:53` (structure tensor computation)

**Reference:**
- Session 3 data: σ_smooth=1.0 gave +0.1 dB over 2.5

---

### Q6: Geodesic EDT Parameters

**Current Values:**
```rust
coherence_threshold = 0.7     // When to follow edges
edge_penalty = 50.0           // Cost of crossing edges
```

**Questions to Answer:**
1. ❓ How sensitive is anti-bleeding to these values?
2. ❓ Should edge_penalty scale with image size?
3. ❓ Does coherence threshold need per-image tuning?

**Where to Tune:**
- `lib.rs:57` (geodesic EDT computation)

---

### Q7: GPU vs CPU Tradeoff

**Current Performance:**
- Kodak (N=124): GPU 2-3× faster
- 4K (N=616): GPU 2-3× faster
- **Not 454× as advertised!**

**Questions to Answer:**
1. ❓ Why is GPU only 2-3× faster?
2. ❓ At what N does GPU become worthwhile?
3. ❓ Buffer transfer overhead vs compute time?
4. ❓ Can we batch operations better?

**Analysis:**
- Workload too small to saturate GPU (3072 cores)
- N=124 → only 1 workgroup = 256 threads (8% utilization)
- N=616 → only 3 workgroups = 768 threads (25% utilization)
- Need N>10,000 for full GPU saturation

**Where to Investigate:**
- `lgi-gpu/src/renderer.rs:155` (dispatch workgroups)
- `lgi-gpu/src/gradient.rs:260` (gradient workgroups)

---

## Parameter Reference

### Complete Parameter List

| Parameter | Location | Default | Unit | Affects |
|-----------|----------|---------|------|---------|
| **Gaussian Count** ||||
| `n_init` | `real_world_benchmark.rs:94` | `sqrt(pixels)/20` | count | Initial Gaussians |
| `n_max` | `real_world_benchmark.rs:95` | `n_init × 4` | count | Max Gaussians |
| **Learning Rates** ||||
| `lr_color` | `optimizer_v2.rs:54` | `0.6` | - | Color updates |
| `lr_position` | `optimizer_v2.rs:52` | `0.1` | - | Position updates |
| `lr_scale` | `optimizer_v2.rs:53` | `0.1` | - | Size updates |
| `lr_rotation` | `optimizer_v2.rs:55` | `0.02` | - | Rotation updates |
| **Optimization** ||||
| `max_iterations` | `optimizer_v2.rs:56` | `100` | iterations | Opt iterations per pass |
| `patience_limit` | `optimizer_v2.rs:138` | `20` | iterations | Early stopping threshold |
| `max_passes` | `lib.rs:226` | `10` | passes | Refinement passes |
| **Error-Driven** ||||
| `target_error` | `lib.rs:223` | `0.001` | MSE | Stop condition |
| `split_percentile` | `lib.rs:224` | `0.10` | fraction | Hotspot threshold |
| `new_sigma` | `lib.rs:273` | `0.02` | normalized | New Gaussian size |
| **Preprocessing** ||||
| `σ_gradient` | `lib.rs:53` | `1.2` | pixels | Structure tensor gradient |
| `σ_smooth` | `lib.rs:53` | `1.0` | pixels | Structure tensor smoothing |
| `coherence_threshold` | `lib.rs:57` | `0.7` | - | Geodesic EDT edge following |
| `edge_penalty` | `lib.rs:57` | `50.0` | - | Geodesic EDT edge crossing |
| **Adam Optimizer** ||||
| `beta1` | `adam_optimizer.rs` | `0.9` | - | Momentum |
| `beta2` | `adam_optimizer.rs` | `0.999` | - | RMSprop |
| `epsilon` | `adam_optimizer.rs` | `1e-8` | - | Numerical stability |

---

## Image Analysis for Technique Selection

### What We Need to Detect

Different image regions benefit from different techniques:

| Region Type | Characteristics | Best Technique | Why |
|-------------|----------------|----------------|-----|
| **Smooth areas** | Low gradient, uniform color | Few large Gaussians | Efficient, no detail needed |
| **Sharp edges** | High gradient, step change | Edge-aligned Gaussians | Structure tensor alignment |
| **Textures** | High-frequency patterns | Per-primitive textures | Capture repeating patterns |
| **Text/strokes** | Coherent thin structures | Specialized stroke Gaussians | Avoid bloating text |
| **Specular highlights** | Isolated bright spots | Small bright Gaussians | Capture highlights without bleeding |
| **Gradients** | Smooth color transitions | Overlapping Gaussians | Blend naturally |
| **Complex details** | High entropy | Many small Gaussians + residuals | Blue-noise for fine detail |

### Detection Methods (Already Implemented)

We have some analysis tools already:

1. **Structure Tensor** (`StructureTensorField`)
   - Detects: Edge orientation, coherence
   - Output: Eigenvalues λ1, λ2, eigenvectors
   - Use: λ1 >> λ2 → strong edge

2. **Geodesic EDT** (`GeodesicEDT`)
   - Detects: Distance to edges
   - Output: Distance field respecting edges
   - Use: Prevent Gaussian bleeding across edges

3. **Analytical Triggers** (`AnalyticalTriggers`)
   - Detects: Where to add textures/residuals
   - Methods:
     - ERR (Entropy-Residual Ratio)
     - LCC (Laplacian Consistency)
     - AGD (Anisotropy Gradient Divergence)
     - JCI (Jacobian Condition Index)
     - SPEC (Structure-Perceptual Error)

4. **Complexity Map** (`ComplexityMap`)
   - Detects: High-frequency regions
   - Output: Per-pixel complexity score

5. **Saliency Detection** (`SaliencyDetector`)
   - Detects: Important regions
   - Output: Saliency map

6. **Text/Stroke Detection** (`TextStrokeDetector`)
   - Detects: Coherent thin structures
   - Output: Boolean mask

7. **Specular Detection** (`SpecularDetector`)
   - Detects: Bright highlights
   - Output: Boolean mask

### What's Missing (Need to Research)

1. **Semantic Segmentation**
   - Classify regions: sky, face, foliage, water, etc.
   - Different content types need different handling

2. **Texture Classification**
   - Distinguish: random noise, periodic patterns, natural textures
   - Affects: Whether to use textures or residuals

3. **Perceptual Importance**
   - Which regions are visually important to humans?
   - Allocate more Gaussians to important regions

4. **Motion/Temporal Analysis** (if we do video)
   - Identify static vs dynamic regions
   - Different encoding strategies

---

## Debugging & Instrumentation

### What We Need to Add

**1. Per-Iteration Logging (Controlled by Flag)**

```rust
pub struct EncoderConfig {
    pub debug_mode: DebugLevel,
}

pub enum DebugLevel {
    None,           // No logging (production)
    Summary,        // Only pass-level summaries
    Detailed,       // Per-iteration metrics
    Verbose,        // Everything including per-Gaussian stats
}
```

**Example Output (Detailed Mode):**
```
Pass 0, Iter 0:
  N = 31
  Loss = 0.018032 (baseline)
  PSNR = 17.44 dB
  LR (adaptive): color=0.6, pos=0.1, scale=0.1, rot=0.02
  Gradient norms: color=1.23, pos=0.45, scale=0.67, rot=0.12

Pass 0, Iter 10:
  Loss = 0.065147 (+260% WORSE!)
  PSNR = 11.86 dB (-5.58 dB)
  Gaussians moved: 18/31 (58%)
  Gaussians out-of-bounds: 3 (clamped)
  ⚠️  Loss increased - possible divergence!
```

**2. Visual Debugging**

- Save rendered image every N iterations
- Save error map highlighting problem regions
- Visualize Gaussian placements (position + size)
- Heatmap of gradient magnitudes

**3. Technique Selection Trace**

```
Image: kodim02.png (768×512)
Region Analysis:
  Smooth areas: 45% of pixels → Use 20 large Gaussians
  Edges: 30% of pixels → Use 40 edge-aligned Gaussians
  Textures: 15% of pixels → Use 8×8 textures on 5 Gaussians
  High-freq: 10% of pixels → Use blue-noise residuals

Technique Selection:
  ✓ Structure tensor alignment: ENABLED (30% edges)
  ✓ Per-primitive textures: ENABLED (15% textured)
  ✓ Blue-noise residuals: ENABLED (10% high-freq)
  ✗ Guided filter: DISABLED (no smooth gradients)
  ✓ Geodesic EDT: ENABLED (anti-bleeding)
```

**4. Performance Profiling**

```
Encoding Profile:
  Preprocessing:         150ms (5%)
    - Structure tensor:   50ms
    - Geodesic EDT:      100ms

  Optimization:         2400ms (80%)
    - Rendering:        1500ms (50%)
    - Gradient comp:     800ms (27%)
    - Parameter update:  100ms (3%)

  Error-driven split:   450ms (15%)
    - Error map:         200ms
    - Hotspot finding:   150ms
    - Gaussian init:     100ms

  TOTAL:               3000ms

Bottleneck: Rendering (50% of time)
Recommendation: Use GPU or spatial acceleration
```

**Where to Add:**
- Create `EncoderConfig` struct with debug flags
- Add logging in optimizer loop
- Create visualization utilities
- Add profiling macros/timers

---

## Performance Bottlenecks

### Current Issues

**1. Rendering is O(width × height × N)**
- Kodak: 393K × 124 = 48.7M Gaussian evaluations
- 4K: 9.6M × 616 = 5.9B evaluations
- **Solution**: Spatial acceleration (BVH, octree, tiling)

**2. Gradient computation is also O(width × height × N)**
- Same complexity as forward pass
- No GPU acceleration for gradients yet
- **Solution**: GPU gradient computation

**3. No culling of distant Gaussians**
- Every pixel checks every Gaussian
- Most contributions are negligible (< 0.001)
- **Solution**: Pre-compute Gaussian bounding boxes, skip if outside

**4. Memory allocation in hot loops**
- Vec allocations every iteration
- Clone operations
- **Solution**: Reuse buffers, preallocate

**5. GPU underutilized**
- Only 2-3× speedup despite 3072 cores
- Workload too small (N=124-616)
- **Solution**: Batch multiple images, or use GPU for larger N

### Optimization Ideas to Research

1. **Spatial Indexing**
   - Build BVH or k-d tree for Gaussians
   - Only check nearby Gaussians per pixel
   - Expected: 10-100× speedup for large N

2. **Tile-Based Rendering**
   - Divide image into tiles
   - Assign Gaussians to tiles
   - Render tiles independently (parallel)
   - Expected: Near-linear scaling with CPU cores

3. **Importance Sampling**
   - Only evaluate Gaussians with weight > threshold
   - Adaptively determine evaluation order
   - Expected: 2-5× speedup

4. **Lazy Evaluation**
   - Don't re-render unchanged Gaussians
   - Track dirty flags per Gaussian
   - Expected: 2× speedup for late iterations

5. **SIMD Vectorization**
   - Process 4-8 pixels simultaneously
   - AVX2/AVX-512 on CPU
   - Expected: 2-4× speedup

---

## Next Steps for Tuning

### Phase 1: Fix Immediate Issues (Week 1)

1. ✅ Fix N scaling (DONE - but needs validation)
2. ⚠️  Fix loss oscillation:
   - Try halving all learning rates
   - Add learning rate schedule
   - Verify Adam optimizer implementation
3. Add basic debug logging
4. Document current results

### Phase 2: Parameter Sweeps (Week 2-3)

1. N scaling sweep (divisors: 10, 15, 20, 25, 30)
2. Learning rate sweep (0.5×, 1×, 2× current)
3. Iteration count sweep (50, 100, 200, 500)
4. Split percentile sweep (5%, 10%, 20%)

### Phase 3: Advanced Tuning (Week 4+)

1. Implement spatial acceleration
2. Add GPU gradient computation
3. Research semantic segmentation libraries
4. Implement adaptive technique selection
5. Add visual debugging tools

### Phase 4: Production Optimization (Later)

1. Tile-based parallel encoding
2. SIMD vectorization
3. Memory pool allocation
4. Full GPU pipeline

---

## References

**Code Locations:**
- Main encoder: `lgi-encoder-v2/src/lib.rs`
- Optimizer: `lgi-encoder-v2/src/optimizer_v2.rs`
- Adam optimizer: `lgi-encoder-v2/src/adam_optimizer.rs`
- GPU renderer: `lgi-gpu/src/renderer.rs`
- Benchmarks: `lgi-encoder-v2/examples/real_world_benchmark.rs`

**Session Logs:**
- Session 7: Major integrations, +8 dB claimed (needs verification)
- Session 8: Real-world testing, discovered issues

**Data to Review:**
- `SESSION_7_BENCHMARK_RESULTS.md` - Previous results
- `ROLLING_TECHNICAL_LOG_SESSION_7.md` - Detailed work log

---

*End of Tuning Guide - Update as we learn more!*
