# LGI Project: Next Steps

**Date**: 2025-12-05
**Status**: Project reorganized, ready for focused work

---

## Fundamental Principles (Always Apply)

### 1. Layers, Never Tiles
- **NO tiling** or 2D geometric segmentation of images
- If techniques fail on whole image, use **semantic layers** (edges, textures, smooth)
- Gaussians are global primitives - preserve holistic representation

### 2. Optimization Toolkit, Not Single Winner
- Adam is what we've learned, NOT the mathematical answer
- Develop multiple methods: Adam, L-BFGS, Levenberg-Marquardt, hybrids
- Each needs its own hyperparameter tuning and initialization strategy

---

## Immediate Priorities (Do These First)

### 1. Validate Isotropic Edge Discovery on Full Dataset

**What**: The quantum research found that isotropic Gaussians work better for edges (+1.87 dB on 5 images). This needs validation on the full dataset.

**Action**:
```bash
cd packages/lgi-rs
cargo run --release --example test_isotropic_edges -- --all-kodak
```

**Expected outcome**: Confirm +1.5-2 dB improvement holds across all 24 Kodak images.

**If validated**: Make `encode_error_driven_adam_isotropic()` the default encoder.

### 2. Commit Untracked Code

**What**: Important code changes from the December 4 session are not committed.

**Files to commit**:
- `lgi-encoder-v2/src/gaussian_logger.rs` (data logging system)
- `lgi-encoder-v2/src/lib.rs` (isotropic encoder method)
- `lgi-encoder-v2/examples/test_isotropic_edges.rs`
- `lgi-core/src/structure_tensor.rs` (modified)
- `BREAKTHROUGH_ISOTROPIC_EDGES.md` (in packages/lgi-rs/)

**Command**:
```bash
git add packages/lgi-rs/lgi-encoder-v2/src/gaussian_logger.rs
git add packages/lgi-rs/lgi-encoder-v2/src/lib.rs
git add packages/lgi-rs/lgi-encoder-v2/examples/test_isotropic_edges.rs
git add packages/lgi-rs/lgi-core/src/structure_tensor.rs
git add packages/lgi-rs/BREAKTHROUGH_ISOTROPIC_EDGES.md
git commit -m "Feature: Isotropic edge Gaussians (+1.87 dB validated improvement)"
```

### 3. Run Full Quality Benchmark

**What**: Validate codec quality on real photos, not just synthetic images.

**Action**:
```bash
cd packages/lgi-rs
cargo run --release --example real_world_benchmark
```

**Expected**: 20-25 dB PSNR on complex real photos with isotropic edges.

---

## Short-term Goals (This Week)

### 4. Determine Optimal Gaussian Count Formula

**Problem**: Current N selection is heuristic. Entropy-based gives +2.93 dB but uses 80× more Gaussians.

**Research needed**: Find the right N(image) formula that balances quality and performance.

**Approaches to test**:
- Fixed N per megapixel
- Entropy-adaptive (current)
- Gradient-density-based
- Content-adaptive (edges get more)

### 5. Measure Actual Compression Ratio

**Problem**: We don't have real compression benchmarks vs JPEG/WebP.

**Action**: Encode Kodak images, measure:
- File size vs JPEG at same PSNR
- File size vs WebP at same PSNR
- Rate-distortion curve

### 6. Decide Quantum Research Future

**Options**:

**Option A: Abandon quantum** (recommended for now)
- Classical RBF clustering is 60× better
- Quantum found ONE useful thing (isotropic)
- That finding is already implemented
- Move on to engineering work

**Option B: Redesign Q2**
- Current Q2 experiment tests optimizers in Adam-tuned pipeline (flawed)
- Would need to design fair comparison
- Significant effort for uncertain gain

**Option C: Pursue Q3/Q4**
- Q3: Quantum for optimization iteration (very speculative)
- Q4: Alternative basis functions (could be done classically)
- Both require substantial work

**Recommendation**: Option A. The codec works. Ship it.

---

## Medium-term Goals (This Month)

### 7. GPU Gradient Computation

**Why**: Current encoding is CPU-bound (gradient computation is 99.7% of time).

**Expected speedup**: 100-1500× faster encoding

**Location**: `lgi-gpu/src/gradient.rs` (70% done, needs debugging)

**Shader**: `gradient_compute.wgsl` (236 lines, written)

### 8. Quantization Profiles

**What**: The format spec defines 4 quantization profiles (LGIQ-B/S/H/X).

**Status**: Specified but not implemented.

**Action**: Implement vector quantization + zstd compression pipeline.

### 9. Progressive Rendering

**What**: Render most important Gaussians first, refine progressively.

**Status**: Weight field exists in Gaussian struct, not used.

**Action**: Sort by importance, enable streaming decode.

---

## Open Research Questions

### Questions Worth Investigating

1. **Feature separation into layers**: Can we separate edges, textures, smooth regions and optimize each separately? (NOT TESTED)

2. **Learned initialization**: Can a neural network predict good initial Gaussian placement from image features? (NOT IMPLEMENTED)

3. **Optimal N discovery**: Is there a closed-form solution for how many Gaussians an image needs? (UNSOLVED)

4. **Video temporal coherence**: How do Gaussians track across frames? (SPEC EXISTS, NOT IMPLEMENTED)

### Questions NOT Worth Investigating (Based on Evidence)

1. **Quantum clustering**: Classical methods work 60× better for this problem.

2. **Per-channel optimizers**: The Q2 experiment design was flawed and incomplete.

3. **Elongated edge Gaussians**: Isotropic works better (validated).

---

## File Organization

### What to Read
- `GROUND_TRUTH.md` - Verified facts about the project
- `docs/research/PROJECT_HISTORY.md` - Full project journey
- `packages/lgi-rs/BREAKTHROUGH_ISOTROPIC_EDGES.md` - The validated finding

### Where Things Are
- **Core codec**: `packages/lgi-rs/lgi-encoder-v2/src/lib.rs`
- **Gaussian math**: `packages/lgi-rs/lgi-math/src/`
- **GPU code**: `packages/lgi-rs/lgi-gpu/src/`
- **Archived docs**: `docs/archive/` (don't need to read unless curious)

### Quantum Research Data (Archived)
- `quantum_research/kodak_gaussian_data/` - 682K real Gaussian configs (reusable)
- `quantum_research/*.pkl` - Processed feature datasets
- `docs/archive/quantum_exploration/` - Speculative quantum docs

---

## Commands Reference

### Build and Test
```bash
cd packages/lgi-rs
cargo build --release
cargo test
```

### Run Encoder
```bash
# Standard (Adam optimizer)
cargo run --release --bin lgi-encode -- input.png output.lgi

# With isotropic edges (recommended after validation)
cargo run --release --example test_isotropic_edges -- --image path/to/image.png
```

### View LGI File
```bash
cargo run --release --bin lgi-viewer -- output.lgi
```

### Benchmark
```bash
cargo run --release --example comprehensive_benchmark
cargo run --release --example real_world_benchmark
```

---

## Success Criteria

### For "Production Ready" Status

- [ ] Isotropic edges validated on full 24-image Kodak set
- [ ] PSNR > 22 dB average on real photos
- [ ] Compression ratio measured vs JPEG
- [ ] All code committed to git
- [ ] Basic documentation complete

### For "Release Candidate"

- [ ] GPU gradient computation working (fast encoding)
- [ ] Quantization profiles implemented
- [ ] File format finalized
- [ ] FFmpeg/ImageMagick integration tested
- [ ] Browser demo (WASM) working

---

*Focus on engineering, not speculation. The codec works - ship it.*
