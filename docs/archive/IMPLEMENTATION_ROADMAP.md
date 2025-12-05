# Implementation Roadmap: Theory to Working Code

**Date**: November 14, 2025
**Goal**: Build working Gaussian image codec incrementally from proven baseline
**Strategy**: Theory-first, validate each step, measure everything

---

## Phase 0: Research Complete ✅

**Duration**: Completed
**Status**: ✅ DONE

**Deliverables**:
- ✅ User research integrated (GAUSSIAN_CODEC_TECHNICAL_FRAMEWORK.md, GAUSSIAN_CODEC_RESEARCH_FRAMEWORK.md)
- ✅ Theory documents created (GAUSSIAN_REPRESENTATION.md, OPTIMIZATION_THEORY.md)
- ✅ Synthesis document (THEORY_SYNTHESIS_AND_V1_PLAN.md)
- ✅ Implementation spec (V1_IMPLEMENTATION_SPEC.md)
- ✅ This roadmap

**Key Insights**:
1. GaussianImage (ECCV 2024) achieves 2000 FPS with 7.375× compression
2. Content-aware initialization is 88% faster than random
3. Cholesky parameterization is standard in literature
4. Our current code is broken (-10 dB regression) due to inter-pass issues

**Decision**: Build minimal V1 without multi-pass complexity

---

## Phase 1: Baseline V1 (Single-Pass Optimization)

**Duration**: 1.5 hours (estimated)
**Goal**: Single 100-iteration pass achieving +6 dB improvement
**Branch**: `claude/continue-testing-work-01JCijpSxefyLBz8yiWVxMNa`

### 1.1 Code Changes (30 minutes)

**Priority 1: Simplify main loop**
- File: `packages/lgi-rs/lgi-encoder-v2/src/lib.rs`
- Action: Remove multi-pass loop, densification, geodesic clamping
- Acceptance: Single `optimize_iterations(100)` call

**Priority 2: Add iteration loop**
- File: `packages/lgi-rs/lgi-encoder-v2/src/adam_optimizer.rs`
- Action: Create `optimize_iterations()` method
- Acceptance: Logs every 10 iterations, returns final loss

**Priority 3: Add scale bounds**
- File: `packages/lgi-rs/lgi-encoder-v2/src/adam_optimizer.rs`
- Action: Clamp scales to MIN_SCALE = 0.001
- Acceptance: No scale collapse during optimization

**Priority 4: Add diagnostics**
- File: `packages/lgi-rs/lgi-encoder-v2/src/adam_optimizer.rs`
- Action: Check for NaN/Inf/scale collapse
- Acceptance: Early warning if numerical issues occur

### 1.2 Testing (30 minutes)

**Test 1: Unit test - Gradient verification**
- File: `tests/gradient_test.rs` (new)
- Method: Finite difference check
- Acceptance: Analytical gradient matches numerical within 1%

**Test 2: Integration test - Single-pass convergence**
- File: `tests/integration_test.rs`
- Method: Full pipeline on test image
- Acceptance: PSNR > 21.0 dB, no scale collapse

**Test 3: Regression test**
- Command: `cargo test --release -- --nocapture test_encoder_v2_basic`
- Acceptance: Matches Session 8 baseline (21.26 dB)

### 1.3 Expected Results

```
Iteration 10: loss = 0.280532
Iteration 20: loss = 0.251034
...
Iteration 100: loss = 0.148203
PSNR: 21.48 dB (+5.98 dB) ✅
W_median: 0.542 ✅
Min scale: 0.0058 ✅
```

### 1.4 Success Criteria

- ✅ PSNR > 21.0 dB (target: 21.26 dB)
- ✅ Loss decreases monotonically
- ✅ min_scale > 0.001 (no collapse)
- ✅ W_median > 0.5 (good coverage)
- ✅ Gradients verified correct

### 1.5 Deliverables

- ✅ Working single-pass optimization
- ✅ All tests passing
- ✅ BASELINE_FIX_STATUS.md updated with "V1 WORKING"
- ✅ Git commit with clear message

**BLOCKER**: None proceed immediately

---

## Phase 2: Multi-Pass Optimization (V2)

**Duration**: 2-3 hours (estimated)
**Goal**: Add back multi-pass loop WITHOUT breaking optimization
**Prerequisite**: Phase 1 complete

### 2.1 Code Changes

**Step 1: Re-enable multi-pass loop**
```rust
for pass in 0..num_passes {
    log::info!("Pass {}/{}", pass + 1, num_passes);
    optimizer.optimize_iterations(&mut gaussians, &target, 100);

    // NO densification yet
    // NO clamping yet
}
```

**Step 2: Verify each pass improves PSNR**
```rust
let psnr_before = compute_psnr(&renderer.render(&gaussians), &target);
optimizer.optimize_iterations(&mut gaussians, &target, 100);
let psnr_after = compute_psnr(&renderer.render(&gaussians), &target);

assert!(psnr_after >= psnr_before, "Pass regressed!");
```

### 2.2 Expected Results

```
Pass 1: 15.5 dB → 21.5 dB (+6.0 dB) ✅
Pass 2: 21.5 dB → 22.8 dB (+1.3 dB) ✅
Pass 3: 22.8 dB → 23.4 dB (+0.6 dB) ✅
...
Pass 10: 24.1 dB → 24.2 dB (+0.1 dB) ✅
Final: 24.2 dB (+8.7 dB total) ✅
```

### 2.3 Success Criteria

- ✅ Each pass improves or maintains PSNR
- ✅ No loss stuck at 0.375001
- ✅ Total improvement > +8 dB after 10 passes

### 2.4 Failure Handling

**If pass 2 breaks**: Debug with logging
```rust
log::info!("Before pass 2: min_scale = {:.6}, W_median = {:.4}", ...);
// ... run pass 2 ...
log::info!("After pass 2: min_scale = {:.6}, W_median = {:.4}", ...);
```

**Hypothesis**: Scales might need re-initialization between passes

---

## Phase 3: Adaptive Densification (V3)

**Duration**: 3-4 hours (estimated)
**Goal**: Add/split/prune Gaussians dynamically
**Prerequisite**: Phase 2 complete

### 3.1 Algorithm (from GaussianImage paper)

**Every 100 iterations**:
1. **Split**: Gaussians with high accumulated gradients (> 0.0002)
2. **Clone**: Gaussians in under-reconstructed regions
3. **Prune**: Gaussians with opacity < 0.01 (if using alpha compositing)

### 3.2 Code Changes

**Re-enable densification** (modified version):
```rust
if iteration % 100 == 0 && iteration > 0 {
    let num_before = gaussians.len();

    // Compute error map
    let error_map = compute_error_map(&rendered, &target);

    // Adaptive densification
    self.adaptive_densification(&mut gaussians, &error_map);

    let num_after = gaussians.len();
    log::info!("Densification: {} → {} Gaussians", num_before, num_after);
}
```

### 3.3 Expected Results

```
Initial: 100 Gaussians, PSNR = 21.5 dB
After densification 1: 150 Gaussians, PSNR = 23.2 dB ✅
After densification 2: 180 Gaussians, PSNR = 24.1 dB ✅
Final: 200 Gaussians, PSNR = 24.8 dB ✅
(Better quality with controlled Gaussian count)
```

### 3.4 Success Criteria

- ✅ Quality improves with densification
- ✅ Gaussian count controlled (not exploding to 10,000+)
- ✅ Pruning removes low-contribution Gaussians

---

## Phase 4: Content-Aware Initialization (V4)

**Duration**: 2-3 hours (estimated)
**Goal**: 10× faster encoding via superpixel initialization
**Prerequisite**: Phase 3 complete

### 4.1 Algorithm (from Neural Video Compression paper)

**Superpixel-based initialization**:
1. Run SLIC superpixel segmentation (OpenCV)
2. One Gaussian per superpixel:
   - μ = segment centroid
   - Σ = segment covariance
   - c = segment mean color
3. Fine-tune 100 iterations (instead of 1000)

### 4.2 Code Changes

**Add initialization method**:
```rust
fn initialize_from_superpixels(
    image: &RgbImage,
    num_segments: usize,
) -> Vec<Gaussian2D> {
    // Use OpenCV SLIC or custom implementation
    let segments = slic_segmentation(image, num_segments);

    segments.into_iter().map(|seg| {
        Gaussian2D {
            position: seg.centroid,
            shape: GaussianShape::from_covariance(seg.covariance),
            color: seg.mean_color,
        }
    }).collect()
}
```

### 4.3 Expected Results

```
Random init: 13-20 seconds to PSNR 30 ❌
Content-aware init: 1.5-2 seconds to PSNR 30 ✅
Speedup: 10× faster ✅
```

### 4.4 Success Criteria

- ✅ Encoding time < 2 seconds per image
- ✅ Same or better quality than random init
- ✅ Works on diverse images (not just gradients)

---

## Phase 5: GPU Acceleration (V5)

**Duration**: 1-2 weeks (estimated)
**Goal**: 1000+ FPS rendering via tile-based CUDA kernels
**Prerequisite**: Phase 4 complete, CPU version validated

### 5.1 Architecture (from user's technical framework)

**Tile-based rasterization**:
- 16×16 pixel tiles
- Each tile processed by thread block
- Shared memory for intermediate buffers
- Early termination when alpha ≈ 1.0

### 5.2 Implementation Path

**Option A: Custom CUDA kernels** (fastest, most control)
- Reference: 3D Gaussian Splatting CUDA code
- Tools: CUDA Toolkit, nvcc
- Time: 1-2 weeks

**Option B: Vulkan compute shaders** (cross-platform)
- Reference: Vulkan splatting examples
- Tools: vulkano or ash crate
- Time: 2-3 weeks

**Option C: wgpu compute shaders** (Rust-native)
- Reference: wgpu examples
- Tools: wgpu crate
- Time: 1-2 weeks

**Recommendation**: Start with custom CUDA for maximum performance

### 5.3 Expected Results

```
CPU (naive): 1-5 FPS ❌
GPU (tile-based): 1500-2000 FPS ✅
Speedup: 300-2000× ✅
```

### 5.4 Success Criteria

- ✅ Rendering speed > 1000 FPS
- ✅ Pixel-exact match with CPU version (or < 1% error)
- ✅ GPU memory < 500 MB for 10,000 Gaussians

---

## Phase 6: Advanced Loss Functions (V6)

**Duration**: 1-2 days (estimated)
**Goal**: Better perceptual quality via SSIM and LPIPS
**Prerequisite**: Phase 5 complete (or can do in parallel with Phase 5)

### 6.1 Multi-Component Loss

**Formula** (from GaussianImage):
```
L = λ₁ × L1 + λ₂ × L2 + λ₃ × (1 - SSIM)
  = 0.2 × L1 + 0.8 × L2 + 0.1 × (1 - SSIM)
```

### 6.2 Code Changes

**Add SSIM computation**:
```rust
fn compute_ssim(img1: &RgbImage, img2: &RgbImage) -> f32 {
    // Use existing library or implement
    // Reference: image-ssim crate
}

fn compute_loss(rendered: &RgbImage, target: &RgbImage) -> f32 {
    let l1 = l1_loss(rendered, target);
    let l2 = l2_loss(rendered, target);
    let ssim = compute_ssim(rendered, target);

    0.2 * l1 + 0.8 * l2 + 0.1 * (1.0 - ssim)
}
```

### 6.3 Expected Results

```
L2-only: PSNR = 24.2 dB, SSIM = 0.89 ❌
L1+L2+SSIM: PSNR = 24.5 dB, SSIM = 0.92 ✅
(Better perceptual quality)
```

---

## Phase 7: Compression Pipeline (V7)

**Duration**: 1-2 weeks (estimated)
**Goal**: Compress Gaussian attributes to bitstream
**Prerequisite**: Phase 6 complete

### 7.1 Quantization Strategy (from user's technical framework)

**Per-attribute quantization**:
- Position: FP16 (2 bytes per coord)
- Covariance (Euler): 6-bit per component (2.25 bytes)
- Color: Residual Vector Quantization (2-3 bytes)
- **Total**: ~6-8 bytes per Gaussian

### 7.2 Quantization-Aware Training (QAT)

**Algorithm**:
```rust
// Forward pass with fake quantization
let quantized_scale_x = quantize_to_6bit(gaussian.shape.scale_x);
let dequantized_scale_x = dequantize_from_6bit(quantized_scale_x);

// Render with quantized values
let rendered = render_with_quantized_gaussians(...);

// Backward pass (straight-through estimator)
// Treat quantization as identity for gradients
```

### 7.3 Entropy Coding

**Tools**:
- CompressAI library (PyTorch-compatible)
- ANS (Asymmetric Numeral Systems) coder
- Or: Arithmetic coding

### 7.4 Expected Results

```
Uncompressed: 8 × N × 4 bytes = 3.2 KB (for N=100)
Quantized: 6 × N bytes = 600 bytes ✅
Entropy-coded: ~400 bytes ✅
Compression ratio: 8× ✅
```

### 7.5 Success Criteria

- ✅ Bits per pixel (bpp) competitive with JPEG at same quality
- ✅ PSNR after quantization within 1 dB of full precision
- ✅ File format spec implemented (.ggc)

---

## Phase 8: Cholesky Parameterization (V8)

**Duration**: 2-3 days (estimated)
**Goal**: Switch from Euler to Cholesky for better stability
**Prerequisite**: Phase 7 complete (or earlier if needed)

### 8.1 Motivation

**Cholesky advantages** (from research):
- Guaranteed positive-definite covariance (no constraints needed)
- Standard in literature (easier to compare)
- Better numerical stability
- Same 3 parameters as Euler

### 8.2 Code Changes

**Update Gaussian representation**:
```rust
// OLD (Euler)
pub struct GaussianShape {
    pub scale_x: f32,
    pub scale_y: f32,
    pub rotation: f32,
}

// NEW (Cholesky)
pub struct GaussianShape {
    pub l1: f32, // Diagonal element 1
    pub l2: f32, // Diagonal element 2
    pub l3: f32, // Off-diagonal element
}

impl GaussianShape {
    fn covariance_matrix(&self) -> Matrix2x2 {
        let L = [[self.l1, 0.0],
                 [self.l3, self.l2]];
        // Σ = L × Lᵀ
        mat_mul(L, transpose(L))
    }
}
```

**Update gradients** (chain rule through Cholesky decomposition)

### 8.3 Expected Results

```
Euler: PSNR = 24.5 dB, occasionally fails on degenerate cases ❌
Cholesky: PSNR = 24.6 dB, always stable ✅
```

---

## Phase 9: Production Features (V9+)

### 9.1 Video Codec Extension
- I-frames (independent)
- P-frames (reference previous)
- 78% bitrate reduction vs image-only

### 9.2 Multi-Resolution/Progressive Rendering
- Coarse → medium → fine Gaussian levels
- Streaming support

### 9.3 Vector Graphics Input
- SVG → Gaussian conversion
- Bezier splatting integration

### 9.4 Quality Presets
- Fast (50 Gaussians, 1 pass)
- Balanced (100 Gaussians, 5 passes)
- High (500 Gaussians, 10 passes)
- Ultra (2000 Gaussians, 20 passes)

---

## Milestone Summary

| Phase | Duration | Deliverable | Success Metric |
|-------|----------|-------------|----------------|
| **0** | Done ✅ | Research complete | All theory docs created |
| **1** | 1.5 hrs | V1 baseline | PSNR > 21 dB, single pass |
| **2** | 2-3 hrs | Multi-pass | +8 dB total, 10 passes |
| **3** | 3-4 hrs | Densification | Better quality, controlled N |
| **4** | 2-3 hrs | Content-aware init | 10× faster encoding |
| **5** | 1-2 wks | GPU acceleration | 1000+ FPS rendering |
| **6** | 1-2 days | SSIM loss | Better perceptual quality |
| **7** | 1-2 wks | Compression | 8× compression ratio |
| **8** | 2-3 days | Cholesky | Better stability |
| **9+** | Variable | Production | Video, multi-res, etc. |

---

## Decision Points and Contingencies

### Decision Point 1: After Phase 1

**If V1 works**: Proceed to Phase 2
**If V1 fails**: Debug with diagnostics, may need to:
- Try Cholesky parameterization (skip to Phase 8)
- Try alpha compositing instead of weighted average
- Adjust learning rate or initialization

### Decision Point 2: After Phase 2

**If multi-pass works**: Proceed to Phase 3
**If pass 2 breaks**: Investigate further:
- May need to re-initialize optimizer state between passes
- May need to adjust learning rate schedule
- May need different loss function

### Decision Point 3: After Phase 5

**Path A**: Continue to compression (Phases 6-7)
**Path B**: Focus on quality improvements (advanced losses, better init)
**Decision criteria**: User's priority (speed vs quality vs compression)

---

## Testing Strategy Throughout

### Continuous Validation

**After each phase**:
1. Run full test suite
2. Measure PSNR on standard images (Kodak dataset)
3. Profile performance (encoding time, rendering FPS)
4. Check memory usage
5. Compare to baseline from previous phase

### Regression Prevention

**Always maintain**:
- Baseline commit hash for each phase
- PSNR measurements on reference images
- Can rollback if phase introduces regression

---

## Resource Requirements

### Development Environment

**Hardware**:
- CPU: Modern x64 (Ryzen/Intel)
- GPU: NVIDIA (for CUDA in Phase 5) - RTX 3060+ recommended
- RAM: 16 GB minimum
- Storage: 10 GB for datasets

**Software**:
- Rust toolchain (stable)
- CUDA Toolkit (for Phase 5)
- OpenCV (for Phase 4)
- Python (for validation scripts)

### Time Commitment

**Minimum viable product (Phases 0-3)**:
- ~8-12 hours total
- Can achieve competitive quality

**Production-ready (Phases 0-7)**:
- ~4-6 weeks total
- Includes GPU acceleration and compression

---

## Success Metrics

### Phase 1 (V1) Success

- ✅ PSNR: 15.5 → 21.5 dB (+6 dB)
- ✅ Time: < 5 seconds per image
- ✅ All tests pass

### Final Success (All Phases)

- ✅ PSNR: Competitive with JPEG2000 at same bitrate
- ✅ Rendering: 1000+ FPS
- ✅ Encoding: < 2 seconds per image
- ✅ Compression: 7-10× ratio
- ✅ Quality: SSIM > 0.92, LPIPS < 0.05

---

## Documentation Requirements

**Per phase**:
- Update BASELINE_FIX_STATUS.md with results
- Add test results to test documentation
- Update README.md with new features
- Commit with clear description

**Final deliverables**:
- Complete API documentation
- User guide (CLI usage)
- Technical deep-dive (architecture)
- Performance benchmarks
- Comparison with other codecs

---

## Risk Mitigation

**Risk 1**: GPU acceleration too complex
- **Mitigation**: CPU version is already useful (5 FPS sufficient for many use cases)
- **Fallback**: Use wgpu for simpler cross-platform GPU support

**Risk 2**: Compression doesn't achieve target ratio
- **Mitigation**: Research has proven 7-10× is achievable
- **Fallback**: Adjust quantization bits, use bits-back coding

**Risk 3**: Multi-pass optimization breaks again
- **Mitigation**: Thorough diagnostics in Phase 1
- **Fallback**: Single-pass with more iterations may be sufficient

---

## Next Immediate Actions

**RIGHT NOW**:
1. ✅ Research complete
2. ✅ Synthesis complete
3. ✅ V1 spec complete
4. ⏭️ **NEXT**: Implement Phase 1 (V1 baseline)

**Command to start**:
```bash
cd packages/lgi-rs/lgi-encoder-v2
# Open lib.rs and adam_optimizer.rs
# Follow V1_IMPLEMENTATION_SPEC.md checklist
```

---

**Status**: Theory complete, ready to implement V1
**Confidence**: High (based on comprehensive research)
**Timeline**: Phase 1 completion within this session (if proceeding)
