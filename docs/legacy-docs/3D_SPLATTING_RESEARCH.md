# 3D Gaussian Splatting Research

**Question**: Can 3D Gaussian splatting techniques help with image/video compression?

**Created**: Session 8 (2025-10-07)

---

## Current Status

**LGI v2 uses 2D Gaussians:**
```rust
Gaussian2D<f32, Euler<f32>>
  - position: Vector2 (x, y in image space)
  - covariance: 2×2 matrix (2D ellipse shape)
  - color: Color4 (r, g, b, alpha)
```

**No 3D representation** - work directly in 2D image plane.

---

## What is 3D Gaussian Splatting?

**Original Paper**: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
**Authors**: Kerbl, Kopanas, Leimkühler, Drettakis (2023)
**Link**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

### Key Idea

Represent 3D scenes as collections of 3D Gaussian ellipsoids:

```python
Gaussian3D:
  - mean: (x, y, z) in 3D world space
  - covariance: 3×3 matrix (3D ellipsoid)
  - color: Spherical harmonics (view-dependent)
  - opacity: α ∈ [0,1]
```

### Rendering Pipeline

1. **Project** 3D Gaussians to 2D screen via camera matrix
2. **Sort** by depth (back-to-front or front-to-back)
3. **Rasterize** with alpha blending:
   ```
   C = Σ c_i × α_i × Π(1 - α_j)
   ```
4. **Differentiable** - can train with gradient descent

### Why It's Fast

- No neural network evaluation (unlike NeRF)
- GPU-friendly rasterization (like traditional graphics)
- Tile-based rendering (16×16 tiles, parallel)
- Achieves **real-time frame rates** (30-100 FPS)

---

## Relevance for Image Compression

### ❌ Not Directly Applicable

**For single images:**
- Don't have 3D scene
- Don't need view synthesis
- 2D Gaussians are simpler and sufficient

### ✅ Techniques We Can Borrow

#### 1. **Tile-Based Rasterization**

**What 3D Splatting Does:**
```cpp
// Divide screen into 16×16 tiles
for each tile:
    frustum_cull(tile_bounds)
    sort_gaussians_by_depth()
    render_tile_parallel()
```

**How We Could Use:**
- Divide image into tiles (e.g., 64×64)
- Assign Gaussians to overlapping tiles
- Render tiles in parallel (multi-threaded)
- **Expected**: Near-linear scaling with CPU cores

**Priority**: HIGH - Major performance win

---

#### 2. **Adaptive Density Control**

**What 3D Splatting Does:**
```python
# Every 100 iterations
for gaussian in gaussians:
    if gradient_magnitude(gaussian) > threshold:
        if gaussian.scale > max_scale:
            split_gaussian(gaussian)  # Large → 2 small
        else:
            clone_gaussian(gaussian)  # Duplicate

    if gaussian.opacity < min_opacity:
        remove_gaussian(gaussian)     # Prune
```

**How We Could Use:**
- Currently: Only add Gaussians, never remove
- Add: Split large Gaussians in high-error regions
- Add: Prune low-opacity Gaussians periodically
- **Expected**: Better N allocation, cleaner results

**Priority**: MEDIUM - Improves quality

---

#### 3. **Opacity-Based Culling**

**What 3D Splatting Does:**
```cpp
// Skip Gaussians with negligible contribution
if (gaussian.opacity * gaussian.weight < threshold) {
    continue;  // Skip evaluation
}
```

**How We Could Use:**
- Track per-Gaussian contribution to final image
- Skip rendering Gaussians with α < 0.01
- **Expected**: 10-30% speedup

**Priority**: LOW - Small gain, easy to add

---

#### 4. **Spherical Harmonics for Color**

**What 3D Splatting Does:**
```python
# View-dependent color (3 bands SH)
color = SH_0 + view_dir · SH_1 + (view_dir²) · SH_2
```

**How We Could Use:**
- NOT USEFUL for 2D (no view direction)
- But could use for multi-view images
- Or for view-dependent effects (iridescence, etc.)

**Priority**: NONE for now

---

## Relevance for Video Compression

### ✅ HIGHLY RELEVANT

Video is where 3D/4D techniques shine:

---

### Option 1: 2.5D Temporal Gaussians

**Concept**: 2D Gaussians with temporal persistence

```rust
struct TemporalGaussian2D {
    // Spatial (2D)
    position: Vector2<f32>,      // (x, y)
    covariance: Matrix2x2,       // 2D ellipse

    // Temporal
    velocity: Vector2<f32>,      // Motion vector (pixels/frame)
    birth_frame: u32,            // First appearance
    death_frame: u32,            // Last appearance

    // Appearance
    color: Color4,
    opacity_curve: Vec<f32>,     // Per-frame opacity
}
```

**Encoding Strategy:**
```
I-frame (keyframe):
  - Full set of Gaussians (like current image codec)

P-frame (predicted):
  - Update: motion vectors for existing Gaussians
  - Add: new Gaussians for uncovered regions
  - Remove: Gaussians that disappeared
  - Recolor: Gaussians with appearance change
```

**Benefits:**
- Temporal consistency (no flickering)
- Motion vectors = optical flow (useful for effects)
- Natural P-frame/B-frame structure
- Only encode changes per frame

**Challenges:**
- How to track Gaussian correspondence across frames?
- When to birth/death Gaussians?
- How to handle occlusions?

**Priority**: HIGH for video codec

---

### Option 2: 3D Scene + Camera Motion

**Concept**: Represent video as 3D scene + camera trajectory

```rust
struct VideoScene {
    // Static 3D representation
    gaussians_3d: Vec<Gaussian3D>,

    // Per-frame camera
    frames: Vec<CameraParams>,  // Pose, focal length, etc.
}
```

**Best For:**
- Static scenes with camera motion (drone footage, walkthrough)
- Videos where scene is mostly rigid
- Enables novel view synthesis

**Benefits:**
- Camera motion nearly free (just transform matrix)
- Extremely high compression for static scenes
- Can generate new viewpoints

**Challenges:**
- Not good for non-rigid motion (people, animals, fluids)
- Requires structure-from-motion (SfM) preprocessing
- More complex than 2.5D

**Priority**: MEDIUM for video codec (specific use cases)

---

### Option 3: Spatiotemporal 4D Gaussians

**Concept**: Gaussians in (x, y, t) or (x, y, z, t) space

```rust
struct Gaussian4D {
    position: Vector4<f32>,      // (x, y, z, time)
    covariance: Matrix4x4,       // 4D ellipsoid
    color: Color4,
    opacity: f32,
}
```

**Rendering:**
```python
# For frame at time t
rendered_gaussians = []
for gaussian in scene.gaussians_4d:
    # Slice 4D Gaussian at time t
    g_2d = gaussian.slice_at_time(t)
    rendered_gaussians.append(g_2d)

render(rendered_gaussians)
```

**Benefits:**
- Unified spatial and temporal representation
- Motion blur comes naturally (Gaussians "streak" through time)
- Good for smooth motion, deformation
- Enables time manipulation (slow-mo, reverse)

**Challenges:**
- High dimensional (4D optimization is hard)
- Memory intensive
- Not clear how to encode efficiently

**Priority**: LOW (research topic, not practical yet)

---

## Key Papers to Study

### 3D Gaussian Splatting (Original)

**"3D Gaussian Splatting for Real-Time Radiance Field Rendering"**
Kerbl et al., SIGGRAPH 2023
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

**Key Contributions:**
- Real-time rendering (vs NeRF's slow)
- Tile-based rasterization
- Adaptive density control
- Differentiable rendering

**Code**: Available (CUDA)

---

### Dynamic Scenes

**"Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis"**
Luiten et al., 3DV 2024
https://dynamic3dgaussians.github.io/

**Key Contributions:**
- Track Gaussians across time
- Deformable Gaussians (per-frame transforms)
- Handles non-rigid motion

**Relevance**: Direct application to video compression

---

**"4D Gaussian Splatting for Real-Time Dynamic Scene Rendering"**
Wu et al., CVPR 2024
https://guanjunwu.github.io/4dgs/

**Key Contributions:**
- Time as explicit dimension
- Gaussian parameters as functions of time
- Compact representation for dynamic scenes

**Relevance**: Spatiotemporal codec approach

---

### Compression-Specific

**"Compact 3D Gaussian Representation for Radiance Field"**
Lee et al., CVPR 2024
https://maincold2.github.io/c3dgs/

**Key Contributions:**
- Compact encoding of Gaussian parameters
- Quantization strategies
- Rate-distortion optimization

**Relevance**: Shows how to compress Gaussian parameters

---

**"LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction"**
Fan et al., 2023
https://lightgaussian.github.io/

**Key Contributions:**
- 15× compression of 3D Gaussian scenes
- Prune 92% of Gaussians
- Vector quantization of parameters

**Relevance**: Compression techniques we could adopt

---

## Techniques to Adopt for LGI v2

### Phase 1: Image Codec (Current)

**Immediately:**
1. ✅ Tile-based rendering (from 3D splatting paper)
2. ✅ Adaptive splitting (split large Gaussians in high-error regions)
3. ✅ Periodic pruning (remove low-opacity Gaussians)

**Later:**
4. Opacity reset strategy
5. Better densification heuristics

---

### Phase 2: Video Codec (Future)

**Research:**
1. Read Dynamic 3D Gaussians paper thoroughly
2. Read 4D Gaussian Splatting paper
3. Read compression papers (Compact 3D, LightGaussian)

**Prototype:**
1. Implement 2.5D temporal Gaussians
2. Test on simple video sequences
3. Measure temporal consistency (flicker metric)

**Optimize:**
1. Gaussian tracking across frames
2. Birth/death heuristics
3. Motion vector encoding
4. Keyframe strategy (I/P frames)

---

## Implementation Notes

### Tile-Based Rendering (High Priority)

**Current Rendering:**
```rust
// Naive: Check all Gaussians for every pixel
for pixel in image:
    for gaussian in gaussians:
        contribution = gaussian.evaluate_2d(pixel)
        color += contribution * gaussian.color
```
**Complexity**: O(width × height × N)

**Tile-Based Rendering:**
```rust
// Step 1: Assign Gaussians to tiles
let tiles = divide_into_tiles(image, tile_size=64);
let tile_gaussians = HashMap::new();

for gaussian in gaussians:
    let bounds = gaussian.bounding_box();
    for tile in overlapping_tiles(bounds, tiles):
        tile_gaussians[tile].push(gaussian);

// Step 2: Render tiles in parallel
tiles.par_iter_mut().for_each(|tile| {
    for pixel in tile:
        for gaussian in tile_gaussians[tile]:
            contribution = gaussian.evaluate_2d(pixel);
            color += contribution * gaussian.color;
});
```
**Complexity**: O((width × height × N_local) / num_cores)
**Expected**: 4-8× speedup on 8-core CPU

**To Add**: `lgi-encoder-v2/src/renderer_tiled.rs`

---

### Adaptive Split/Clone (Medium Priority)

```rust
impl ErrorDrivenEncoder {
    fn refine_gaussians(&mut self, gaussians: &mut Vec<Gaussian2D>) {
        let mut to_add = Vec::new();
        let mut to_remove = HashSet::new();

        for (i, gaussian) in gaussians.iter().enumerate() {
            let grad_mag = self.gradient_magnitude(gaussian);

            if grad_mag > self.split_threshold {
                // High gradient → needs refinement
                if gaussian.scale > self.max_scale {
                    // Large Gaussian → split into 2
                    let (g1, g2) = split_gaussian(gaussian);
                    to_add.push(g1);
                    to_add.push(g2);
                    to_remove.insert(i);
                } else {
                    // Small Gaussian → clone
                    to_add.push(gaussian.clone());
                }
            }

            if gaussian.alpha < self.min_opacity {
                // Low contribution → remove
                to_remove.insert(i);
            }
        }

        // Apply changes
        gaussians.retain_index(|i| !to_remove.contains(i));
        gaussians.extend(to_add);
    }
}
```

**To Add**: Methods in `error_driven.rs`

---

## Video Codec Prototype Plan

**When**: After image codec is stable (~1-2 months)

**Phase 1: Proof of Concept**
1. Load video as sequence of images
2. Encode frame 0 as I-frame (full Gaussian set)
3. For frames 1-N:
   - Track Gaussians via nearest-neighbor in feature space
   - Encode motion vectors
   - Add new Gaussians for uncovered regions
   - Remove Gaussians for disappeared regions
4. Decode and measure:
   - PSNR per frame
   - Temporal consistency (flicker)
   - Compression ratio vs encoding each frame independently

**Phase 2: Optimization**
1. Better tracking (optical flow, feature matching)
2. Rate-distortion optimization for I-frame interval
3. Motion compensation
4. Temporal filtering

**Phase 3: Comparison**
1. Benchmark vs H.264, H.265, VP9, AV1
2. Measure on standard video datasets
3. Publish results

---

## References

**Code Repositories:**
- 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
- Dynamic 3D Gaussians: https://github.com/JonathonLuiten/Dynamic3DGaussians
- 4D Gaussian Splatting: https://github.com/hustvl/4DGaussians
- Compact 3D Gaussians: https://github.com/maincold2/Compact-3DGS

**Academic Resources:**
- Papers With Code: https://paperswithcode.com/task/novel-view-synthesis
- 3D Gaussian Splatting Papers: https://github.com/MrNeRF/awesome-3D-gaussian-splatting

**Video Compression:**
- x264: https://www.videolan.org/developers/x264.html
- x265: https://x265.readthedocs.io/
- AV1: https://aomedia.org/av1/

---

*End of 3D Splatting Research - Update as we learn more!*
