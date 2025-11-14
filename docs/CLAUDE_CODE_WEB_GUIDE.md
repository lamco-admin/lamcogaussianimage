# Claude Code Web Development Guide

**Repository**: https://github.com/lamco-admin/lamcogaussianimage
**Last Updated**: November 14, 2025
**Purpose**: Quick-start guide for continuing LGI development with Claude Code Web

---

## Quick Start (First Session)

### 1. Clone the Repository

```bash
git clone https://github.com/lamco-admin/lamcogaussianimage.git
cd lamcogaussianimage
```

### 2. Open in VS Code / Claude Code Web

Simply open the `lamcogaussianimage/` directory in VS Code with Claude Code Web enabled.

### 3. Essential First Reads

**Read these IN ORDER for full context**:

1. **`README.md`** (5 min)
   - Project overview
   - Quick start commands
   - Current status

2. **`docs/research/PROJECT_HISTORY.md`** (15 min)
   - Complete journey from concept to +8dB quality
   - All 8 development sessions summarized
   - Critical decisions and lessons
   - **Most important**: Prevents repeating mistakes

3. **`docs/research/ROADMAP_CURRENT.md`** (5 min)
   - What's next (immediate priorities)
   - Track 1 progress: 32/150 techniques (21%)
   - Track 2 progress: 6/20 features (30%)
   - Current work: Session 8 validation

4. **`docs/research/EXPERIMENTS.md`** (optional, 10 min)
   - What experiments worked/failed
   - Includes FAIL-001 through FAIL-005 (critical learnings)

5. **`docs/research/DECISIONS.md`** (optional, 10 min)
   - Why we chose Rust, wgpu, two-track strategy
   - 16 major architectural decisions

---

## Current Work Status

### Where We Are (Nov 14, 2025)

**Last Active Session**: Session 8 (Oct 7, 2024)
**Focus**: Real-world validation on Kodak dataset
**Status**: Paused for reorganization, ready to continue

**Quality Achieved**:
- Sharp edges: **+9.69 dB** (14.67 → 24.36 dB)
- Complex patterns: **+6.46 dB** (15.50 → 21.96 dB)
- **Average**: +8.08 dB improvement

**Recommended Method**: `encode_error_driven_adam()`

---

### Immediate Next Tasks (Priority Order)

1. **Real Photo Benchmark** [P0 - Critical]
   - Run on 67 test images in `test-data/test_images/`
   - Validate +8dB holds on real photos (currently only tested on 2 synthetic)
   - Location: `packages/lgi-rs/lgi-benchmarks/`

2. **Empirical R-D Curve Fitting** [P0 - Critical]
   - Current formulas undershot 30 dB target by 6 dB
   - Need data-driven models: `PSNR = a×log(N) + b×complexity + c`
   - Location: `packages/lgi-rs/lgi-encoder-v2/src/rate_distortion.rs`

3. **Gaussian Count Strategy** [P1 - High]
   - Entropy-based: N=2635, +2.93 dB (but 80× more Gaussians)
   - Question: Does advantage persist after optimization?
   - Location: `packages/lgi-rs/lgi-encoder-v2/examples/gaussian_count_comparison.rs`

4. **Complete Track 1 P1** [P1 - High]
   - 5 remaining critical techniques
   - 10/15 complete (67%)
   - Follow Master Integration Plan methodology (Session 7)

---

## Building & Testing

### Build Everything
```bash
cd packages/lgi-rs
cargo build --release --all
```

### Run Tests
```bash
cargo test --workspace
```

### Quick Validation
```bash
# Test GPU detection
cargo run --release --example backend_detection

# Fast benchmark (2 test images)
cd lgi-encoder-v2
cargo run --release --example fast_benchmark
```

### Full Benchmark (Kodak dataset)
```bash
cd lgi-benchmarks
cargo run --release --bin real_world_benchmark
```

---

## Project Architecture

### Two-Track Strategy

**Track 1: Mathematical Techniques** (150 total)
- Goal: Comprehensive algorithmic toolkit
- Progress: 32/150 integrated (21%)
- Philosophy: ALL 150 matter (user mandate)
- Priority: Complete P1 (10/15 done), then P2, then P3

**Track 2: Format Features** (20 total)
- Goal: Production file format
- Progress: 6/20 complete (30%)
- Priority: Quantization profiles next
- Critical for compression ratio targets

**Why Two Tracks?**:
- Independent workstreams
- Faster parallel progress
- Can integrate later
- User-approved strategy (see `docs/research/DECISIONS.md` D-STRAT-001)

---

## Code Organization

### packages/lgi-rs/ (Production Rust - Main Work Here)

**Foundation** (stable):
- `lgi-math/` - Math primitives, Gaussian types (no changes needed)
- `lgi-format/` - File I/O (mostly stable)

**Active Development**:
- `lgi-encoder-v2/` - **PRIMARY FOCUS** (optimization, +8dB achieved)
- `lgi-core/` - Rendering, initialization (adding techniques)
- `lgi-gpu/` - GPU acceleration (gradient compute 70% done)

**Tools & Integration**:
- `lgi-cli/` - Command-line tools
- `lgi-viewer/` - GUI (functional)
- `lgi-benchmarks/` - Testing (use frequently)
- `lgi-ffi/` - C FFI (FFmpeg/ImageMagick)

**Future**:
- `lgi-pyramid/` - Multi-level zoom (implemented, needs integration)
- `lgi-wasm/` - WebAssembly (builds, needs testing)

---

## Key Files to Know

### Production Encoder
**`packages/lgi-rs/lgi-encoder-v2/src/lib.rs`**
- Main encoding API
- 17 public methods
- TOP 10 integrations from Session 7
- **This is where most work happens**

### Techniques Modules
**`packages/lgi-rs/lgi-core/src/`**
- `entropy.rs` - Auto Gaussian count
- `better_color_init.rs` - Improved initialization
- `error_driven.rs` - Adaptive refinement
- `adam_optimizer.rs` - Fast convergence
- `rate_distortion.rs` - Quality/size targeting
- `geodesic_edt.rs` - Anti-bleeding
- `ewa_splatting_v2.rs` - Alias-free rendering
- ...and 100+ more

### GPU
**`packages/lgi-rs/lgi-gpu/src/`**
- `manager.rs` - Global GPU singleton
- `gradient.rs` - GPU gradients (70% done, NEEDS DEBUGGING)
- `shaders/gradient_compute.wgsl` - Compute shader (complete)

### Benchmarks
**`packages/lgi-rs/lgi-benchmarks/`**
- `examples/fast_benchmark.rs` - Quick validation
- `examples/comprehensive_benchmark.rs` - Full suite
- `bin/real_world_benchmark.rs` - Kodak + 4K photos

---

## Common Development Tasks

### Adding a New Technique (Track 1)

1. **Check if already implemented**:
   ```bash
   grep -r "technique_name" packages/lgi-rs/lgi-core/src/
   ```

2. **If exists but not integrated**:
   - See Session 7 approach (Master Integration Plan)
   - Add method to `lgi-encoder-v2/src/lib.rs`
   - Create example to validate
   - Benchmark before/after

3. **If needs implementation**:
   - Create module in `lgi-core/src/technique_name.rs`
   - Write tests
   - Add to `lgi-core/src/lib.rs`
   - Then integrate (step 2)

### Running Benchmarks

**Quick test** (2 images, ~30 seconds):
```bash
cd packages/lgi-rs/lgi-encoder-v2
cargo run --release --example fast_benchmark
```

**Full Kodak** (24 images, ~10 minutes):
```bash
cd packages/lgi-rs/lgi-benchmarks
cargo run --release --bin kodak_benchmark
```

**4K Photos** (67 images, ~2 hours):
```bash
cargo run --release --bin real_world_benchmark
```

### Debugging GPU Issues

**Check GPU availability**:
```bash
cd packages/lgi-rs/lgi-gpu
cargo run --release --example backend_detection
```

**Test CPU vs GPU rendering**:
```bash
cargo run --release --example gpu_minimal_debug
# Should show: ✅ GPU matches CPU
```

---

## Critical Knowledge

### The +8 dB Breakthrough (Session 7)

**What happened**: All TOP 10 techniques integrated in one 4-hour session
**Result**: +8.08 dB average improvement
**Method**: Master Integration Plan (systematic approach)

**Lesson**: Integration > Implementation
- 39% of code was sitting unused
- Integration is the bottleneck
- Systematic planning works

### The 29 dB Regression Bug (Sessions 4-5)

**Problem**: Geodesic EDT over-clamping
**Impact**: Quality stuck at 5-7 dB (should be 30+ dB)
**Root cause**: Gaussians clamped to 1 pixel → zero coverage → zero gradients

**Fix**: Use coverage-based scales PRIMARY, geodesic as guidance only

**Lesson**: Always check W_median (coverage) > 0.5

**See**: `docs/research/EXPERIMENTS.md` FAIL-001 for details

### Two-Track Philosophy

**User Mandate**:
> "Even if tests don't show benefit, implement ALL 150 techniques"

**Rationale**:
- Mathematical foundation matters
- May help specific content types
- Future-proofing
- Comprehensive toolkit

**Approach**: Systematic, prioritized (P1 → P2 → P3)

---

## Testing Strategy

### Always Benchmark

**Before making changes**:
```bash
cargo run --release --example fast_benchmark > before.txt
```

**After changes**:
```bash
cargo run --release --example fast_benchmark > after.txt
diff before.txt after.txt
```

**Validate**: No regressions, measure improvement

### Test on Multiple Content Types

Don't just test on synthetic images!
- Sharp edges (HF_checkerboard, etc.)
- Complex patterns (HF_multi_gratings, etc.)
- Real photos (test_images/)
- Kodak dataset (standard benchmark)

**Why**: Techniques may help some content, hurt others

---

## Common Issues & Solutions

### Issue 1: GPU Not Available
**Symptom**: "GPU initialization failed"
**Solution**: CPU fallback automatic, or install Vulkan drivers

### Issue 2: Slow Encoding
**Current**: CPU gradients (99.7% of time)
**Solution**: GPU gradients (70% done, needs debugging)
**Impact**: 1500× speedup when complete

### Issue 3: Low Quality (<20 dB)
**Check**: W_median (coverage) in debug output
**Likely**: Gaussian scales too small
**Fix**: See geodesic clamping bug (EXPERIMENTS.md FAIL-001)

---

## Development Workflow

### Typical Session
1. **Read context**: PROJECT_HISTORY + ROADMAP
2. **Pick priority**: From ROADMAP_CURRENT.md
3. **Implement**: Follow Session 7 methodology
4. **Benchmark**: Before/after validation
5. **Document**: Update ROADMAP with findings

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes, test
cargo test --workspace

# Commit
git add .
git commit -m "Descriptive message"

# Push
git push -u origin feature/your-feature

# Create PR via GitHub
```

---

## Test Data Locations

### Synthetic Tests
**Path**: `test-data/test_images_new_synthetic/`
**Contents**:
- Sharp edges (HF_checkerboard, HF_hairlines)
- Complex patterns (HF_multi_gratings, HF_woven_texture)
- Band-limited (BN_blurred_discs, BN_lowfreq_field)
- Masks (MASK_variance_map)

**Use For**: Quick validation, specific technique testing

### Real Photos
**Path**: `test-data/test_images/`
**Contents**: 67 real photos (4K resolution, diverse content)
**Use For**: Real-world validation, final benchmarking

### Kodak Dataset
**Path**: `test-data/kodak-dataset/`
**Contents**: 24 standard benchmark images (768×512)
**Use For**: Industry comparison, published benchmarks

---

## Dependencies

### Build Requirements
- **Rust**: 1.75+ (`rustup update`)
- **Cargo**: Included with Rust

### Optional (GPU)
- **Vulkan**: Linux (mesa-vulkan-drivers)
- **DirectX 12**: Windows (built-in on Win10+)
- **Metal**: macOS (built-in)

### Optional (Viewer)
- **Slint**: Auto-installed via Cargo

**Everything else**: Managed by Cargo

---

## Performance Expectations

### Encoding (128×128)
- Adam (recommended): **1.4s**
- With GPU gradients (when done): **~0.1s** (14× faster)

### Rendering (1920×1080)
- CPU: **52ms** (19 FPS)
- GPU: **0.85ms** (1,176 FPS)

### Quality
- Baseline: **14-16 dB**
- Current: **22-24 dB** (+8 dB)
- Target: **30-35 dB** (need empirical tuning)

---

## Troubleshooting

### Build Errors
```bash
# Update Rust
rustup update

# Clean rebuild
cargo clean
cargo build --release
```

### Test Failures
```bash
# Run specific test
cargo test --package lgi-core test_name

# Verbose output
cargo test -- --nocapture
```

### GPU Issues
```bash
# Check backend
cd packages/lgi-rs/lgi-gpu
cargo run --release --example backend_detection

# Should show Vulkan/DX12/Metal available
```

---

## Critical Files Reference

### Main Documentation
- `README.md` - Entry point
- `docs/research/PROJECT_HISTORY.md` - Complete context
- `docs/research/EXPERIMENTS.md` - What worked/failed
- `docs/research/DECISIONS.md` - Why we chose this
- `docs/research/ROADMAP_CURRENT.md` - What's next
- `ACKNOWLEDGMENTS.md` - Third-party attribution

### Main Code
- `packages/lgi-rs/lgi-encoder-v2/src/lib.rs` - Encoder API
- `packages/lgi-rs/lgi-core/src/` - Technique modules
- `packages/lgi-rs/lgi-gpu/src/gradient.rs` - GPU gradients (needs work)

### Workspace Root
- `packages/lgi-rs/Cargo.toml` - Workspace config

---

## Session Template

**For every new session**:

```markdown
## Session Start Checklist

1. [ ] Read PROJECT_HISTORY.md (if first time)
2. [ ] Read ROADMAP_CURRENT.md (always)
3. [ ] Check current branch: `git status`
4. [ ] Verify build: `cargo build --release`
5. [ ] Run quick benchmark: `fast_benchmark`
6. [ ] Pick priority from ROADMAP_CURRENT.md
7. [ ] Review relevant EXPERIMENTS.md section
8. [ ] Review relevant DECISIONS.md entry

## Session Work
[Your work here]

## Session End Checklist

1. [ ] Run benchmarks (before/after)
2. [ ] Document findings
3. [ ] Update ROADMAP_CURRENT.md if priorities changed
4. [ ] Commit changes
5. [ ] Push to GitHub
```

---

## Remember

### From PROJECT_HISTORY.md Lessons:

**L001**: Documentation ≠ Reality
- Always validate claims with tests
- Don't trust aspirational targets

**L002**: Instrumentation > Guessing
- Add logging first
- Track intermediate values
- Detailed instrumentation finds bugs in minutes

**L003**: Implementation ≠ Integration
- Writing code is easy
- Integrating code is hard
- Integration is the bottleneck

**L007**: Systematic > Heroic
- Session 7: 4 hours, +8 dB ✅
- Session 2: 18 hours, exhaustion ❌
- Planning beats marathons

---

## Contact & Help

### For Questions
- Check `docs/research/` first
- GitHub Issues: https://github.com/lamco-admin/lamcogaussianimage/issues

### For Context
- Complete history: `PROJECT_HISTORY.md`
- Why decisions made: `DECISIONS.md`
- What failed before: `EXPERIMENTS.md` (FAIL sections)

---

**You have everything you need. The foundation is rock solid. Continue with confidence.** 🚀

**Repository**: https://github.com/lamco-admin/lamcogaussianimage

**Last Updated**: November 14, 2025
