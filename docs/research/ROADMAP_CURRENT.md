# LGI (Lamco Gaussian Image) Project Roadmap - Current Priorities

**Last Updated**: November 14, 2025 (Project reorganization)
**Status**: Post-Session 8, Foundation complete
**Current Phase**: Real-world validation and empirical tuning

---

## Status Snapshot

### Achievements (Oct 2-7, 2025)

**Quality**: **+8.08 dB average improvement** achieved
- Sharp edges: 14.67 dB → 24.36 dB (+9.69 dB)
- Complex patterns: 15.50 dB → 21.96 dB (+6.46 dB)

**Track Progress**:
- **Track 1** (Mathematical Techniques): 32/150 integrated (21%)
  - P1 (Critical): 10/15 (67%) ✅
  - P2 (High Value): 0/20 (0%)
  - P3 (Enhancement): 0/65 (0%)
- **Track 2** (Format Features): 6/20 complete (30%)

**Production-Ready Methods**:
1. `encode_error_driven_adam()` - RECOMMENDED
2. `encode_error_driven_gpu()` - For large images
3. `encode_error_driven_gpu_msssim()` - Ultimate quality
4. `encode_for_psnr()` / `encode_for_bitrate()` - Target-based

---

## Priority Legend

- 🔴 **P0 (Critical)**: Blocks production, must do next
- 🟠 **P1 (High)**: Core functionality, significant value
- 🟡 **P2 (Medium)**: Enhancement, measurable benefit
- 🟢 **P3 (Low)**: Nice to have, minor benefit

**Track Labels**:
- **[T1]**: Track 1 - Mathematical Technique
- **[T2]**: Track 2 - Format Feature
- **[T1+T2]**: Both tracks

---

## Immediate Priorities (2-4 weeks)

### Real-World Validation & Tuning

#### 1. Real Photo Benchmark [T1] 🔴 P0
**Effort**: 1 day

**Goal**: Validate Session 7 results on real photographic content

**Tasks**:
- Run benchmarks on 64 test images (4K photos)
- Kodak dataset validation (24 industry-standard images)
- Collect: PSNR, MS-SSIM, time, N per method
- Analysis: Which photo types benefit from each technique

**Success Criteria**:
- Adam maintains +6-8 dB average on photos ✅
- No regressions vs synthetic tests
- Identify hard/easy content types

**Blockers**: None (ready to execute)

---

#### 2. Empirical R-D Curve Fitting [T1] 🔴 P0
**Effort**: 1 day

**Goal**: Replace heuristic formulas with data-driven models

**Problem**: Current formulas undershot 30 dB target by 6 dB

**Approach**:
```
Collect data: (N, PSNR, MS_SSIM, image_complexity)
Fit: PSNR = a×log(N) + b×complexity + c
Invert: N = exp((target_PSNR - b×complexity - c) / a)
```

**Success Criteria**:
- R² > 0.85 for fitted model
- Target 30 dB → achieve 29-31 dB (±1 dB accuracy)
- Generalizes across image types

**Impact**: Predictable quality/size targeting for production

---

#### 3. Gaussian Count Strategy [T1] 🟠 P1
**Effort**: 2 days

**Goal**: Determine optimal N selection strategy

**Current Issue**:
- Arbitrary: N=31, 17.03 dB (baseline)
- Entropy: N=2635, 19.96 dB (+2.93 dB but 80× more Gaussians)
- Hybrid: N=1697, 18.67 dB (+1.64 dB)

**Critical Question**: Does entropy advantage persist after full optimization?

**Experiments**:
1. Full optimization test (100 iters) on all 3 strategies
2. Measure: final PSNR, encode time, trade-offs
3. O(N²) scaling analysis
4. Content-adaptive threshold tuning

**Outcome**: Recommended strategy for production

---

### Track 1 P1 Completion (5 Remaining)

#### 4. Complete Track 1 P1 Integrations [T1] 🟠 P1
**Effort**: 1 week

**Remaining 5 Techniques**:
- [Details from TRACK_1_P1_AUDIT_REPORT in research archive]

**Approach**: Follow Master Integration Plan methodology (Session 7)
- One technique per session
- Integration + validation
- Benchmark before/after

**Expected Impact**: +2-3 dB additional improvement

---

## Near-Term Priorities (1-2 months)

### Track 2: Format Features

#### 5. Quantization Profiles [T2] 🟠 P1
**Effort**: 1 week

**Goal**: Implement 4 LGI quantization profiles

**Profiles**:
- **LGIQ-B** (Basic): 8-bit quantization
- **LGIQ-S** (Standard): 16-bit, recommended
- **LGIQ-H** (High): 24-bit, archival quality
- **LGIQ-X** (Extreme): 32-bit, lossless

**Tasks**:
1. Vector quantization (VQ) implementation
2. Quantization-aware (QA) training
3. zstd compression layer
4. Validation: measure compression ratios

**Target**: 7.5-10.7× compression (from spec)

---

#### 6. Progressive Rendering [T2] 🟡 P2
**Effort**: 1 week

**Goal**: Stream LGI files with progressive quality

**Features**:
- Coarse-to-fine Gaussian transmission
- Importance-based ordering
- Partial decode support
- Useful for web streaming

**Implementation**: Chunk-based format already supports this

---

### Performance Optimization

#### 7. GPU Gradient Computation [T1] 🔴 P0
**Effort**: 1 week
**Status**: ⏳ 70% complete, needs debugging

**Impact**: **1500× encoding speedup**

**Current Bottleneck**:
- CPU gradients: 99.7% of encode time
- 15 seconds/iteration → 25 minutes for 100 iterations

**With GPU Gradients**:
- ~10ms/iteration → 1 second total for 100 iterations

**Blockers**:
- Shader complete (236 lines) ✅
- Module has compilation errors (313 lines)
- Async/sync boundary issues

**Priority**: Critical for production encoder

---

#### 8. Learned Initialization [T1] 🟡 P2
**Effort**: 2 weeks

**Goal**: 10× faster encoding via neural initialization

**Approach**:
- Small network proposes coarse Gaussians
- Brief fine-tuning (10-20 iterations vs 100)
- Inspired by Instant-GI paper

**Trade-off**: Need training data, model deployment

---

## Medium-Term Priorities (3-6 months)

### Production Readiness

#### 9. Comprehensive Benchmarking [T1+T2] 🟠 P1
**Effort**: 1 week

**Datasets**:
- Kodak (24 images) - already available
- DIV2K (800 high-res images)
- CLIC (professional photos)

**Metrics**:
- PSNR, MS-SSIM (perceptual)
- Compression ratio
- Encode/decode time
- Comparisons: JPEG, WebP, AVIF, HEIF

**Deliverable**: Technical paper / benchmark report

---

#### 10. FFmpeg Encoder [T1+T2] 🟠 P1
**Effort**: 1 week
**Status**: Decoder working ✅, encoder untested

**Goal**: Production-quality FFmpeg integration

**Tasks**:
- Test encoder code (written but not validated)
- Fix bugs discovered
- Add encoding parameters (quality, speed presets)
- Integration testing

**Outcome**: `ffmpeg -i input.png output.lgi` working

---

### Ecosystem Integration

#### 11. WebAssembly Build [T1+T2] 🟡 P2
**Effort**: 1 week

**Goal**: Browser-based LGI decoder

**Features**:
- Decode LGI in browser (no server)
- Canvas rendering
- Interactive zoom
- Demo website

**Tech**: Rust → WASM via wasm-pack

---

#### 12. Python Bindings [T1+T2] 🟡 P2
**Effort**: 1 week

**Goal**: PyPI package for Python users

**API**:
```python
import lgi

# Encode
gaussians = lgi.encode("input.png", quality=30)
lgi.save(gaussians, "output.lgi")

# Decode
image = lgi.decode("output.lgi")
```

**Tech**: PyO3 (Rust ↔ Python)

---

## Long-Term Vision (6-12 months)

### LGIV Video Codec

#### 13. Temporal Prediction [T1+T2] 🟢 P3
**Effort**: 1 month

**Goal**: Extend LGI to video (LGIV format)

**Specification**: Already complete (1,027 lines) ✅

**Features**:
- GOP structure
- Temporal Gaussian tracking
- Keyframe encoding (reuse LGI)
- Streaming support (HLS/DASH)

**Approach**:
- Phase 1: Simple temporal prediction
- Phase 2: Motion compensation
- Phase 3: Production integration

---

### Advanced Features

#### 14. CUDA Backend [T1] 🟢 P3
**Effort**: 2 weeks

**Goal**: NVIDIA-specific optimization

**Rationale**:
- wgpu great for cross-platform
- CUDA can be faster for NVIDIA users
- Optional backend

**Impact**: 20-30% speedup on NVIDIA GPUs

---

#### 15. Track 1 P2/P3 Completion [T1] 🟡 P2
**Effort**: 3-6 months

**Goal**: Complete all 150 mathematical techniques

**Philosophy** (user mandate):
> "Even if tests don't show benefit, implement ALL 150 techniques"

**Approach**: Systematic integration following Master Plan methodology

**Progress**:
- P1: 10/15 (67%) ← finish first
- P2: 0/20 (0%) ← next phase
- P3: 0/65 (0%) ← long-term

---

## Success Metrics

### Phase 1 (Immediate - Complete)
- ✅ Production-ready image codec
- ✅ +8 dB quality improvement
- ✅ Cross-platform GPU rendering
- ✅ FFmpeg/ImageMagick decoders

### Phase 2 (Near-term - In Progress)
- ⏳ Real photo validation
- ⏳ Empirical tuning complete
- ⏳ Track 1 P1 100% (currently 67%)
- ⏳ Quantization profiles working
- ⏳ GPU gradients functional

### Phase 3 (Medium-term)
- ⬜ 30-35 dB PSNR achieved
- ⬜ 7.5-10.7× compression validated
- ⬜ FFmpeg encoder production-ready
- ⬜ Comprehensive benchmarks published

### Phase 4 (Long-term)
- ⬜ LGIV video codec functional
- ⬜ All 150 techniques integrated
- ⬜ Ecosystem widely adopted

---

## Risks & Mitigations

### Risk 1: GPU Gradient Implementation
**Impact**: High (1500× speedup blocked)
**Probability**: Medium (70% complete, needs debugging)
**Mitigation**: Allocate dedicated debugging session, simplify if needed

### Risk 2: Compression Ratio Claims
**Impact**: Medium (spec targets not validated)
**Probability**: High (not tested yet)
**Mitigation**: Conservative claims, thorough benchmarking before public release

### Risk 3: Inconsistent Session Quality
**Impact**: High (regression hunting wastes time)
**Probability**: Historical issue (Sessions 4-5)
**Mitigation**:
- This reorganization (distilled docs) ✅
- Automated benchmarks in CI
- Clear handoff procedures

---

## Dependencies

### External
- **Kodak dataset**: Downloaded ✅
- **wgpu v27**: Current, stable ✅
- **Rust 1.75+**: Available ✅

### Internal
- **lgi-math**: Stable ✅
- **lgi-core**: Needs GPU gradient integration ⏳
- **lgi-encoder-v2**: Active development

### Blockers
- None critical (all work can proceed)

---

## Next Actions

**Immediate** (this week):
1. Complete project reorganization ✅
2. Set up Git repository
3. Create initial commit
4. Resume Session 8 validation work

**Next Session**:
1. Run real photo benchmarks
2. Kodak dataset validation
3. Analyze results
4. Update roadmap based on findings

---

## Appendix: Two-Track Strategy

### Track 1: Mathematical Techniques (150 total)
**Goal**: Comprehensive algorithmic toolkit

**Philosophy**: All 150 matter (user mandate)

**Progress**: 32/150 (21%)
- P1: 10/15 techniques
- P2: 0/20 techniques
- P3: 0/65 techniques
- Implemented but not integrated: 22

**Next**: Complete P1, then systematic P2/P3 integration

---

### Track 2: Format Features (20 total)
**Goal**: Production file format

**Progress**: 6/20 (30%)
- Core format: 3/5 ✅
- Quantization: 1/4 ⏳
- Compression: 0/4 ⬜
- Progressive: 0/3 ⬜
- Metadata: 2/4 ⏳

**Next**: Quantization profiles (critical for deployment)

---

**Document Status**: Living roadmap, updated after each major session

**See Also**:
- `PROJECT_HISTORY.md` - How we got here
- `EXPERIMENTS.md` - What worked/failed
- `DECISIONS.md` - Why we chose this path
