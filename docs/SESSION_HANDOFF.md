# LGI Development Session Handoff - November 2025

**Repository**: https://github.com/lamco-admin/lamcogaussianimage
**Last Session**: Session 8 (October 7, 2025)
**Project Reorganization**: November 14, 2025
**Next Session**: Continuation of Session 8 validation work
**Status**: Production-ready foundation, ready for next development phase

---

## ⚠️ CRITICAL: READ BEFORE STARTING ANY WORK

This handoff provides **complete context** to prevent the session inconsistencies and quality issues experienced in October 2025. **Read this entire document before writing any code.**

---

## 1. PROJECT SCOPE & AMBITION

### What LGI Actually Is

**LGI** = **Lamco Gaussian Image** Codec
**LGIV** = **Lamco Gaussian Image Video** Codec (future)

**This is NOT**:
- A simple research prototype
- A proof-of-concept experiment
- A minimal viable product
- A "good enough" solution

**This IS**:
- Revolutionary image/video codec (comparable to developing AV1 or AVIF from scratch)
- Complete format specification (like WebP, HEIF standards)
- Production-quality reference implementation (encoder + decoder)
- Full ecosystem integration (FFmpeg, ImageMagick, browsers)
- Research-backed innovation (150 mathematical techniques catalogued)
- Multi-year vision (Image → Video → SDK → Ecosystem)

**Ambition level**: World-class codec that competes with JPEG, WebP, AVIF

**Timeline**: NO deadline - "Best quality, fullest implementation, most features, most flexibility"

**User's mandate**: NO shortcuts, NO simplifications, NO stub implementations

---

## 2. QUALITY STANDARDS (MANDATORY - NO EXCEPTIONS)

### Rule 1: NO SHORTCUTS, EVER

**NEVER**:
- Write TODO comments ("implement later")
- Create stub implementations ("simplified version for now")
- Skip tests ("we'll add tests later")
- Use placeholders ("good enough for now")
- Bypass problems ("let's just remove the problematic part")

**ALWAYS**:
- Complete implementations only
- Face problems directly
- Debug until root cause found
- Comprehensive error handling
- Full test coverage

**Rationale**: From user's requirements and lessons learned from October sessions

**Document**: See `docs/DEVELOPMENT_STANDARDS.md` for complete standards

---

### Rule 2: VALIDATE EVERYTHING

**From historical failures** (Oct 2025, Sessions 1-5):
- Documentation claimed "30-40 dB PSNR" ❌
- Reality was 5-7 dB
- **29 dB regression** went unnoticed for 3 days
- 3 days wasted hunting "regression" that was never actually fixed

**Lesson**: Documentation ≠ Reality

**MANDATORY**:
- Run benchmarks before claiming performance
- Measure quality improvements empirically
- Test on multiple image types (synthetic + real photos + Kodak)
- Validate before every claim
- No aspirational targets presented as achievements

**Every quality claim MUST include**:
```
Claim: +8 dB improvement
Evidence:
  - Baseline: 14.67 dB (measured, fast_benchmark.rs)
  - Improved: 24.36 dB (measured, same benchmark)
  - Delta: +9.69 dB
  - Test: Sharp edge 128×128
  - Method: encode_error_driven_adam()
  - Date: Oct 6, 2025 (Session 7)
  - Reproducible: cargo run --release --example fast_benchmark
```

---

### Rule 3: ALL 150 TECHNIQUES MATTER

**User's philosophy**:
> "Even if tests don't show benefit, implement ALL 150 techniques. Mathematical foundation matters."

**What this means**:
- Technique X shows +0.1 dB? **Still implement it fully** ✅
- Might only help medical images? **Still implement it** ✅
- Not sure if useful? **Still implement it** ✅
- No immediate benefit? **Still implement it** ✅

**Rationale**:
- Future-proofing (unknown use cases)
- Comprehensive mathematical toolkit
- Research-backed approaches have value
- May matter for specific content types

**Priority system**:
- **P1** (Critical): 15 techniques - Implement first
- **P2** (High Value): 20 techniques - Next phase
- **P3** (Enhancement): 65 techniques - Later
- **ALL eventually get implemented** (150 total)

**Current Progress**: 32/150 integrated (21%)

---

### Rule 4: COMPLETE OR NOTHING

**Definition of "Complete"**:

**Code**:
- [ ] Full algorithm implementation (not simplified)
- [ ] All edge cases handled
- [ ] Comprehensive error handling
- [ ] No TODOs or unimplemented!() blocks
- [ ] No dead code or commented-out code

**Tests**:
- [ ] Unit tests (basic + edge cases + errors)
- [ ] Integration tests (with real data)
- [ ] Benchmarks (performance + quality)
- [ ] Minimum 80% code coverage

**Documentation**:
- [ ] Rustdoc comments (public API)
- [ ] Inline comments (complex algorithms)
- [ ] Example usage
- [ ] Performance characteristics
- [ ] References (paper/spec section)

**Validation**:
- [ ] Benchmarked on synthetic images
- [ ] Tested on real photos
- [ ] Kodak dataset (if major feature)
- [ ] PSNR + MS-SSIM measured
- [ ] No regressions

**If ANY checkbox unchecked**: FEATURE NOT COMPLETE

---

## 3. CURRENT STATUS (EXACT STATE)

### Quality Baseline (October 7, 2025)

```
Sharp Edge Images (128×128):
  Baseline:            14.67 dB
  Adam Optimizer:      24.36 dB  (+9.69 dB) ✅ BEST
  Average:             22-24 dB range

Complex Pattern Images (128×128):
  Baseline:            15.50 dB
  GPU Method:          21.96 dB  (+6.46 dB) ✅ BEST
  Average:             20-22 dB range

Overall Improvement:   +8.08 dB average ✅ VALIDATED
```

**Validation**: Measured, reproducible via `fast_benchmark.rs`

**NOT Yet Validated**:
- Real photo performance (Session 8 in progress)
- Kodak dataset results (benchmarks started)
- Compression ratios (Track 2 quantization not done)
- 30-35 dB target (gap: 6-13 dB remaining)

---

### Track Progress

**Track 1: Mathematical Techniques** (32/150 integrated = 21%)
- **P1** (Critical): 10/15 complete (67%)
- **P2** (High Value): 0/20 complete (0%)
- **P3** (Enhancement): 0/65 complete (0%)
- **Implemented but not integrated**: 22 techniques

**Track 2: Format Features** (6/20 complete = 30%)
- Core format: 3/5 ✅
- Quantization: 1/4 ⏳
- Compression: 0/4 ⬜
- Progressive: 0/3 ⬜
- Metadata: 2/4 ⏳

**Two-Track Strategy**: User-approved parallel development (see DECISIONS.md D-STRAT-001)

---

### Production-Ready Methods

**Four encoding methods available**:

1. **`encode_error_driven_adam()`** - **RECOMMENDED**
   - Quality: +9.69 dB on sharp edges
   - Speed: 1.38s for 128×128
   - Use: General purpose, best quality

2. **`encode_error_driven_gpu()`** - For Large Images
   - Quality: +6.46 dB on complex patterns
   - Speed: 1.47s for 128×128
   - Use: 512×512+ images

3. **`encode_error_driven_gpu_msssim()`** - Ultimate Quality
   - Quality: Best perceptual (MS-SSIM metric)
   - Speed: Similar to GPU method
   - Use: Highest visual quality needed

4. **`encode_for_psnr()` / `encode_for_bitrate()`** - Target-Based
   - Quality: Targets specific PSNR or file size
   - Status: ⚠️ Needs empirical tuning (undershot targets by 6 dB)
   - Use: When specific quality/size required

**Location**: `packages/lgi-rs/lgi-encoder-v2/src/lib.rs`

---

### What Works ✅

**Fully Functional**:
- Encoding (4 methods above)
- Decoding (LGI file → PNG)
- GPU rendering (1,176 FPS validated on RTX 4060)
- FFmpeg decoder (`ffmpeg -i file.lgi output.png`)
- ImageMagick decoder (`magick file.lgi output.png`)
- LGI Viewer (GUI with zoom, export, etc.)
- Benchmark suite

**Tests**: All passing (last validated Oct 7, 2025)

---

### What Needs Work ⏳

**Critical (Blocking)**:
1. **GPU Gradients** (70% complete)
   - Status: Shader complete, module has compilation errors
   - Impact: 1500× encoding speedup when done
   - Blocker: CPU gradients are 99.7% of encode time

2. **Empirical R-D Curve Fitting** (P0)
   - Current formulas undershot 30 dB target by 6 dB
   - Need: Data-driven models from real photo testing
   - Impact: Accurate quality/size targeting

3. **Real Photo Validation** (P0)
   - Status: Session 8 started, paused for reorganization
   - Need: Validate +8 dB holds on 67 test photos
   - Critical: All current measurements on synthetic only

**High Priority (Track 1 P1)**:
- 5 remaining critical techniques (67% → 100%)

**High Priority (Track 2)**:
- Quantization profiles (LGIQ-B/S/H/X)
- Compression layer (Vector Quantization + zstd)

---

## 4. IMMEDIATE NEXT TASKS (Priority Order)

### Task 1: Real Photo Benchmark [P0 - CRITICAL]

**Goal**: Validate Session 7 results on real photographic content

**Why Critical**:
- All current measurements on 2 synthetic images only
- Need to verify +8 dB holds on diverse real photos
- Establishes empirical foundation for all future work

**Action**:
```bash
cd packages/lgi-rs/lgi-benchmarks
cargo run --release --bin real_world_benchmark
```

**Test On**:
- 67 real photos in `test-data/test_images/` (4K resolution)
- 24 Kodak images in `test-data/kodak-dataset/` (768×512)

**Collect**:
- PSNR per image per method
- MS-SSIM per image per method
- Encoding time
- Gaussian count (N)
- Distribution analysis (easy/hard images)

**Success Criteria**:
- Adam maintains +6-8 dB average ✅
- No regressions vs synthetic (22-24 dB maintained)
- Identify which photo types benefit from each technique

**If Fails**:
- Debug immediately
- Don't simplify away
- Find root cause
- Document in EXPERIMENTS.md

---

### Task 2: Empirical R-D Curve Fitting [P0 - CRITICAL]

**Goal**: Replace heuristic formulas with data-driven models

**Problem**: `encode_for_psnr(30.0)` delivers ~24 dB (6 dB short)

**Approach**:
```python
# From Task 1 data:
data = [(N, PSNR, MS_SSIM, complexity, image_type), ...]

# Fit model:
PSNR = a × log(N) + b × complexity + c

# Invert for targeting:
N = exp((target_PSNR - b × complexity - c) / a)

# Update lgi-encoder-v2/src/rate_distortion.rs with fitted coefficients
```

**Success Criteria**:
- R² > 0.85 for fitted model
- Target 30 dB → achieve 29-31 dB (±1 dB accuracy)
- Generalizes across image types
- Validate on held-out images

**Documentation**: Add fitted formula to EXPERIMENTS.md

---

### Task 3: Gaussian Count Strategy [P1 - HIGH]

**Goal**: Determine optimal N selection strategy

**Current State** (from Oct 7, Session 8):
```
Strategy      | Avg N | Avg PSNR | Time |
--------------|-------|----------|------|
Arbitrary     |    31 | 17.03 dB | 78ms | (baseline)
Entropy-based |  2635 | 19.96 dB | 4.7s | (+2.93 dB, 80× more Gaussians)
Hybrid        |  1697 | 18.67 dB | 3.0s | (+1.64 dB)
```

**These were PRE-OPTIMIZATION only!**

**Critical Question**: Does +2.93 dB advantage persist after full 100-iteration optimization?

**Experiment Needed**:
```bash
# Full pipeline test
for strategy in [arbitrary, entropy, hybrid]:
    N = select_n(image, strategy)
    gaussians = initialize(image, N)
    optimized = optimize(gaussians, 100 iterations)  # FULL optimization
    measure(optimized)  # PSNR, MS-SSIM, time
```

**Trade-off Analysis**:
- Entropy uses 80× more Gaussians
- Optimization is O(N²)
- 2635 Gaussians → much longer encode
- Question: Is +2.93 dB worth 80× longer encoding?

**Success Criteria**:
- Measure quality vs encoding time trade-off
- Recommend strategy for production
- Document findings in EXPERIMENTS.md

---

### Task 4: Complete Track 1 P1 [P1 - HIGH]

**Goal**: Finish remaining 5 critical techniques

**Status**: 10/15 complete (67%)

**Approach**: Follow Session 7 methodology (Master Integration Plan)
- One technique per session (focused)
- Full implementation (no TODOs)
- Integration validation (benchmark before/after)
- No regressions

**Remaining 5**: [See ROADMAP_CURRENT.md for specifics]

**Expected Impact**: +2-3 dB additional improvement

---

## 5. MANDATORY READING (Before Coding)

### Read In This Order

**Session Start** (ALWAYS READ):

1. **This document** (SESSION_HANDOFF.md) - 15 min
   - Complete context
   - Quality standards
   - Current status
   - Next tasks

2. **docs/DEVELOPMENT_STANDARDS.md** - 10 min
   - Mandatory quality requirements
   - Anti-patterns to avoid
   - Testing requirements
   - Code quality standards

3. **docs/research/ROADMAP_CURRENT.md** - 5 min
   - Current priorities
   - Track progress
   - Success metrics

**First Time Only** (or if context lost):

4. **docs/research/PROJECT_HISTORY.md** - 20 min
   - Complete journey Sep-Oct 2025
   - All 8 sessions summarized
   - Critical decisions and lessons
   - **Prevents repeating mistakes**

5. **docs/research/EXPERIMENTS.md** - 15 min
   - What worked (18 experiments)
   - **What failed (5 major failures)**
   - Why things failed
   - **FAIL-001**: 29 dB regression bug (critical)

6. **docs/research/DECISIONS.md** - 10 min
   - 16 major architectural decisions
   - Why we chose Rust, wgpu, two-track strategy
   - Full rationale for each

**Before Implementing Specific Feature**:

7. **EXPERIMENTS.md** - Check if we tried this before
8. **DECISIONS.md** - Check if architectural choice already made
9. **ROADMAP_CURRENT.md** - Verify priority and approach

---

## 6. CRITICAL LESSONS (MUST REMEMBER)

### Lesson 1: Documentation ≠ Reality

**From Oct 2025, Sessions 4-5**:

**Problem**:
- Early docs claimed "30-40 dB PSNR, 7.5-10.7× compression, production-ready"
- Reality: 5-7 dB PSNR, compression not measured
- **29 dB gap** between claim and reality
- 3 days wasted hunting "regression"

**Root Cause**: Aspirational targets presented as achievements

**Prevention**:
- **NEVER** claim without evidence
- **ALWAYS** validate with benchmarks
- Clearly mark: `[TARGET]` vs `[ACHIEVED]`
- Automated benchmarks (don't trust memory)

---

### Lesson 2: Instrumentation > Guessing

**From Oct 2025, Session 5 (FAIL-001)**:

**Problem**: Quality stuck at 5-7 dB for days

**Breakthrough**: Added diagnostic logging
```rust
log::debug!("σ_base: {:.2}, σ_clamped: {:.2}, W_median: {:.4}",
    sigma_base, sigma_clamped, weight_median);
```

**Result**: Found bug in **minutes** (W_median=0.000 → zero coverage)

**Root Cause**: Geodesic EDT clamping Gaussians to 1 pixel → zero gradients

**Lesson**:
> "Detailed instrumentation finds bugs in minutes vs hours of guessing"

**Standard**: When debugging, add logging FIRST, guess second

---

### Lesson 3: Implementation ≠ Integration

**From Oct 2025, Sessions 1-7**:

**Discovery** (Oct 6, Session 7 prep):
- Audited all modules
- Found **22 production-ready modules (39%) sitting unused**
- Had all tools to achieve 24-27 dB
- Quality stuck at 14-16 dB because **not using them**!

**Quote**:
> "We have all these amazing tools, we're just not using them!"

**Solution** (Session 7):
- Created Master Integration Plan
- Prioritized TOP 10 integrations
- 4-hour focused session
- **Result**: +8 dB ✅

**Lesson**: Integration is the bottleneck, not implementation

**Standard**: After implementing, ALWAYS integrate and validate

---

### Lesson 4: Systematic > Heroic

**Comparison**:

**Session 2** (Oct 3, 2025):
- Duration: 18 hours continuous
- Result: FFmpeg/ImageMagick working ✅
- Cost: Exhaustion, bugs introduced late, poor decisions

**Session 7** (Oct 6, 2025):
- Duration: 4 hours focused
- Plan: Master Integration Plan (TOP 10)
- Result: All 10 integrated, +8 dB, no bugs ✅

**Lesson**: Planning > marathons

**Standard**: 4-5 hour sessions MAX, plan first, execute systematically

---

### Lesson 5: Both Techniques, Conditionally

**From Oct 2025, Sessions 4-5 (FAIL-006)**:

**Wrong Understanding** (Oct 4):
> "Isotropic is better than anisotropic"

**Corrected** (Oct 5):
> "BOTH are essential, applied conditionally based on content"

**Code**:
```rust
let (σ_x, σ_y) = if coherence < 0.2 {
    // Flat regions → isotropic
    (σ, σ)
} else {
    // Edges → anisotropic along gradient
    (σ * 0.5, σ * 2.0)
};
```

**Results**:
- Isotropic only: 20.6 dB
- Anisotropic only: 27.6 dB (+7.0 dB, but artifacts on flat)
- Conditional: 29.5 dB (+8.9 dB) ✅ WINNER

**Lesson**: Content-adaptive > one-size-fits-all

**Principle**: Build toolkit, use conditionally, no either/or thinking

---

## 7. DEVELOPMENT WORKFLOW (MANDATORY)

### Before Every Session (30 min)

**ALWAYS**:
1. Pull latest: `git pull origin main`
2. Read this handoff (SESSION_HANDOFF.md)
3. Read ROADMAP_CURRENT.md
4. Run baseline benchmark:
   ```bash
   cargo run --release --example fast_benchmark > baseline.txt
   ```
5. Pick priority task from ROADMAP
6. Check EXPERIMENTS.md (did we try this before?)
7. Check DECISIONS.md (architectural choice already made?)

**NEVER skip these steps** - prevents context loss

---

### During Session (3-4 hours MAX)

**Plan Phase** (30 min):
- Understand requirements fully
- Read relevant papers/specs
- Identify edge cases
- Plan test strategy
- Design algorithm

**Implementation Phase** (2-3 hours):
- Write tests FIRST (TDD where possible)
- Implement complete solution (no TODOs)
- Add comprehensive error handling
- Document as you go
- Test continuously

**Validation Phase** (30 min):
- Run all tests: `cargo test --workspace`
- Run benchmarks: `fast_benchmark` + relevant specific benchmarks
- Measure quality: PSNR + MS-SSIM
- Visual inspection (if applicable)
- Compare with baseline (no regressions)

**Total**: 4 hours focused work

**NEVER**: Marathon 18-hour sessions (causes mistakes)

---

### After Every Session (30 min)

**MANDATORY**:
1. Run final benchmark:
   ```bash
   cargo run --release --example fast_benchmark > session_end.txt
   diff baseline.txt session_end.txt  # Check for regressions
   ```

2. Update ROADMAP_CURRENT.md:
   - Mark completed tasks
   - Update progress percentages
   - Add findings/notes
   - Update priorities if changed

3. Document findings:
   - If experiment: Add to EXPERIMENTS.md
   - If decision: Add to DECISIONS.md
   - If significant: Update PROJECT_HISTORY.md

4. Commit and push:
   - Clean commit message
   - Include measurements
   - Reference task/issue
   - Push to GitHub

5. Write TODO for next session (if needed)

**Quality gate**: All tests passing, no regressions, documentation updated

---

## 8. ANTI-PATTERNS (NEVER DO THESE)

### Anti-Pattern 1: Stub Implementations

**❌ NEVER**:
```rust
pub fn advanced_feature() -> Result<Output> {
    // TODO: Implement full algorithm later
    // For now, just return simple version
    simple_fallback()
}
```

**✅ ALWAYS**:
```rust
pub fn advanced_feature() -> Result<Output> {
    // Full algorithm from [Paper] Section X
    let step1 = complete_step_1()?;
    let step2 = complete_step_2(step1)?;
    validate_and_return(step2)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_advanced_feature() { /* comprehensive tests */ }
}
```

---

### Anti-Pattern 2: Bypass Problems

**❌ NEVER** (from Oct 2025 lessons):
```rust
// Geodesic clamping causing issues?
// Let's just remove it instead of fixing it!
// let clamped = sigma;  // Commented out the problematic code
```

**✅ ALWAYS**:
```rust
// Geodesic clamping over-aggressive?
// Fix the formula, don't remove the feature!
let clamp_threshold = compute_proper_threshold(geodesic_dist);
let clamped = sigma.min(clamp_threshold);

#[cfg(debug_assertions)]
{
    // Validate coverage maintained
    assert!(compute_median_weight() > 0.5, "Zero coverage detected");
}
```

---

### Anti-Pattern 3: Trust Without Validation

**❌ NEVER**:
```markdown
## Performance

We achieve 30-40 dB PSNR! ✅
```
*(Without actually measuring)*

**✅ ALWAYS**:
```markdown
## Performance

Validated (Nov 2025):
- Sharp edges: 24.36 dB ✅ (measured, reproducible)
- Complex patterns: 21.96 dB ✅

Target (not yet achieved):
- 30-35 dB 🎯 (gap: 6-13 dB, work in progress)

Evidence: cargo run --release --example fast_benchmark
```

---

### Anti-Pattern 4: Single Iteration Testing

**❌ NEVER** (from Oct 2025 lessons):
```
Test once → looks good → ship it
```

**✅ ALWAYS**:
```
Test on:
- Synthetic (controlled): 4+ images
- Real photos: 10+ diverse images
- Kodak dataset: 24 industry-standard images
- Edge cases: extreme sizes, solid colors, high-frequency

Measure:
- PSNR (comparable metric)
- MS-SSIM (perceptual quality)
- Visual inspection (human validation)
- Encoding time (performance)
```

---

### Anti-Pattern 5: Implement Without Integrate

**❌ NEVER**:
```
Week 1: Implemented feature A ✅
Week 2: Implemented feature B ✅
Week 3: Implemented feature C ✅
Week 4: Quality still at baseline ❌
Reason: None integrated into encoder!
```

**✅ ALWAYS** (Session 7 Master Integration Plan approach):
```
Plan:
1. Audit: Find unused modules (found 22!)
2. Prioritize: TOP 10 by impact
3. Integrate: Systematically, one at a time
4. Validate: Benchmark each

Result: +8 dB in 4 hours ✅
```

---

## 9. QUALITY GATES (MUST PASS)

### Gate 1: Before Every Commit

```bash
# 1. Format
cargo fmt --all --check || exit 1

# 2. Lint
cargo clippy --workspace -- -D warnings || exit 1

# 3. Tests
cargo test --workspace || exit 1

# 4. Build (no warnings)
cargo build --release --all 2>&1 | grep -i warning && exit 1
```

**ALL must pass** - no exceptions

---

### Gate 2: Before Every PR

**Checklist**:
- [ ] All tests passing
- [ ] Benchmarks included (before/after measurements)
- [ ] No regressions (PSNR maintained or improved)
- [ ] Documentation complete (API docs + inline comments)
- [ ] ROADMAP_CURRENT.md updated
- [ ] Findings documented (EXPERIMENTS.md if significant)
- [ ] Code formatted (cargo fmt)
- [ ] No clippy warnings
- [ ] Tested on multiple image types
- [ ] Visual validation (if applicable)

**If ANY fails**: Fix before merge

---

### Gate 3: Feature Complete Definition

**Feature considered "complete" ONLY if**:

**Implementation**:
- [ ] Full algorithm (per paper/spec, not simplified)
- [ ] All edge cases handled
- [ ] Comprehensive error handling (Result<T, E>, no panics)
- [ ] No TODOs or unimplemented!() blocks
- [ ] Proper logging (debug/info/warn/error)

**Testing**:
- [ ] Unit tests (happy path + edge cases + errors)
- [ ] Integration tests (with real data)
- [ ] Benchmarks (performance + quality measured)
- [ ] 80%+ code coverage
- [ ] All tests passing

**Documentation**:
- [ ] Rustdoc complete (all public items)
- [ ] Inline comments (complex algorithms)
- [ ] Examples provided
- [ ] Performance characteristics documented
- [ ] References cited (papers, specs)

**Validation**:
- [ ] Synthetic test images ✅
- [ ] Real photos (10+ images) ✅
- [ ] Kodak dataset (if major) ✅
- [ ] PSNR + MS-SSIM measured ✅
- [ ] No regressions confirmed ✅

**If ANY unchecked**: NOT DONE - keep working

---

## 10. DEBUGGING STANDARDS

### When Something Goes Wrong

**NEVER**:
1. Guess at random
2. Try changes blindly
3. Simplify problem away
4. Remove failing feature
5. "Hope it will work"

**ALWAYS**:
1. **Add instrumentation FIRST**:
   ```rust
   log::debug!("intermediate_value: {:.6}", value);
   log::debug!("W_median: {:.4}, expected > 0.5", w_median);
   ```

2. **Run diagnostic test** (single iteration if needed):
   ```rust
   // Focus on ONE iteration to understand what's happening
   for iter in 0..1 {  // Not 100, just 1!
       log::debug!("=== ITERATION {} ===", iter);
       log::debug!("Loss: {:.6}", loss);
       log::debug!("|Δcolor|: {:.6}", delta_color.magnitude());
       log::debug!("|Δposition|: {:.6}", delta_position.magnitude());
       log::debug!("W_median: {:.4}", median_weight);
   }
   ```

3. **Analyze root cause**:
   - What's the intermediate values?
   - Where does it deviate from expected?
   - What assumption is violated?

4. **Fix properly**:
   - Address root cause
   - Don't remove feature
   - Add validation
   - Add test to prevent regression

5. **Document**:
   - Add to EXPERIMENTS.md (FAIL-XXX section)
   - Explain root cause
   - Document lesson learned

**Example from history**: FAIL-001 (Geodesic EDT Over-Clamping)
- Spent days guessing
- Added logging → found in minutes
- W_median=0.000 revealed zero coverage
- Fixed formula, didn't remove feature

---

## 11. TESTING PROTOCOL

### Test Every Change

**Before implementing**:
```bash
cargo run --release --example fast_benchmark > before.txt
```

**After implementing**:
```bash
cargo run --release --example fast_benchmark > after.txt
diff before.txt after.txt
```

**Expected**:
- PSNR improved (or maintained)
- No regressions on other tests
- Performance acceptable

**If regression**: Debug immediately, don't commit

---

### Test Coverage Requirements

**Every module MUST test**:

1. **Happy path**: Normal expected input
   ```rust
   #[test]
   fn test_normal_case() {
       let result = function(valid_input);
       assert!(result.is_ok());
       assert_eq!(result.unwrap().quality, expected_quality);
   }
   ```

2. **Edge cases**:
   ```rust
   #[test]
   fn test_empty_input() {
       let result = function(empty);
       assert!(result.is_err());
   }

   #[test]
   fn test_extreme_values() {
       let result = function(extreme_input);
       assert!(result.is_ok());
   }
   ```

3. **Integration**:
   ```rust
   #[test]
   fn test_with_real_data() {
       let image = load_test_image("kodak/kodim01.png");
       let result = encode(image);
       assert!(result.psnr > 20.0);  // Reasonable quality
   }
   ```

---

## 12. CURRENT BLOCKERS & HOW TO HANDLE

### Blocker 1: GPU Gradients (70% Complete)

**Status**: Shader complete, module has compilation errors

**Files**:
- `packages/lgi-rs/lgi-gpu/src/shaders/gradient_compute.wgsl` (236 lines) ✅
- `packages/lgi-rs/lgi-gpu/src/gradient.rs` (313 lines) - errors

**Impact**: 1500× encoding speedup blocked

**How to Handle**:
1. **DO NOT simplify** the shader
2. **DO NOT** "just use CPU for now"
3. **DEBUG** the compilation errors
4. **FIX** async/sync boundary issues
5. **TEST** thoroughly when fixed
6. **VALIDATE**: GPU gradients match CPU (bit-for-bit if possible)

**Expected Result**: Encoding time 15s → 0.01s per iteration

---

### Blocker 2: Empirical Formula Gaps

**Problem**: `encode_for_psnr(30.0)` delivers ~24 dB (6 dB short)

**Root Cause**: Heuristic formulas not data-driven

**How to Handle**:
1. **DO NOT** tweak formulas randomly
2. **COLLECT** data from Task 1 (real photo benchmarks)
3. **FIT** proper model: PSNR = a×log(N) + b×complexity + c
4. **VALIDATE** on held-out images
5. **UPDATE** `rate_distortion.rs` with fitted coefficients
6. **TEST** targeting again
7. **DOCUMENT** in EXPERIMENTS.md

---

## 13. FILE LOCATIONS (Quick Reference)

### Main Encoder
**`packages/lgi-rs/lgi-encoder-v2/src/lib.rs`**
- 17 public encoding methods
- TOP 10 integrations from Session 7
- Primary work location

### Technique Modules
**`packages/lgi-rs/lgi-core/src/`**
- 100+ technique modules
- entropy.rs, better_color_init.rs, error_driven.rs, etc.
- Add new techniques here

### Benchmarks
**`packages/lgi-rs/lgi-benchmarks/`**
- `examples/fast_benchmark.rs` - Quick validation (2 images, 30s)
- `bin/real_world_benchmark.rs` - Full validation (67 photos, hours)
- `bin/kodak_benchmark.rs` - Industry standard (24 images)

### GPU
**`packages/lgi-rs/lgi-gpu/src/`**
- `manager.rs` - Global GPU singleton (working ✅)
- `gradient.rs` - GPU gradients (70% done, needs debugging)
- `shaders/gradient_compute.wgsl` - Compute shader (complete ✅)

### Test Data
- `test-data/test_images/` - 67 real 4K photos
- `test-data/test_images_new_synthetic/` - Synthetic patterns
- `test-data/kodak-dataset/` - 24 industry benchmarks

---

## 14. BUILD & TEST COMMANDS

### Standard Development Cycle

```bash
# Pull latest
git pull origin main

# Create feature branch
git checkout -b feature/your-feature

# Build
cargo build --release --all

# Run tests
cargo test --workspace

# Benchmark (before changes)
cargo run --release --example fast_benchmark > before.txt

# [Make your changes]

# Test again
cargo test --workspace

# Benchmark (after changes)
cargo run --release --example fast_benchmark > after.txt

# Compare
diff before.txt after.txt

# If good, commit
git add .
git commit -m "Descriptive message with measurements"
git push -u origin feature/your-feature

# Create PR on GitHub
```

---

## 15. SESSION TEMPLATE

**Copy this for every new session**:

```markdown
## Session Start - [Date]

### Pre-Session Checklist
- [ ] Read SESSION_HANDOFF.md
- [ ] Read ROADMAP_CURRENT.md
- [ ] Git pull
- [ ] Run baseline benchmark
- [ ] Pick priority task
- [ ] Check EXPERIMENTS.md (tried before?)
- [ ] Check DECISIONS.md (choice made?)

### Planned Work
Task: [From ROADMAP_CURRENT.md]
Expected Impact: [Quality/performance gain]
Approach: [Algorithm/strategy]

### Session Log
[Document as you work]

### Results
Before: [PSNR, time, etc.]
After: [PSNR, time, etc.]
Delta: [Improvement]

### Validation
- [ ] Tests passing
- [ ] Benchmarks run
- [ ] No regressions
- [ ] Quality measured
- [ ] Documentation updated

### Session End Checklist
- [ ] Final benchmark saved
- [ ] ROADMAP_CURRENT.md updated
- [ ] Findings documented
- [ ] Committed and pushed
- [ ] Next TODO written
```

---

## 16. CRITICAL WARNINGS

### WARNING 1: Context Loss Prevention

**From October 2025 sessions**:
> "suffered repeatedly from inconsistent session behavior, quality, understanding and performance"

**WHY this happened**:
- Long gaps between sessions (context lost)
- Documentation diverged from code
- Fixes not properly committed
- Aspirational claims accepted as fact

**HOW we fixed it**:
- PROJECT_HISTORY.md (complete narrative)
- EXPERIMENTS.md (what worked/failed)
- DECISIONS.md (architectural rationale)
- This handoff (comprehensive context)

**PREVENTION**: Read docs EVERY session start

---

### WARNING 2: The Geodesic Clamping Bug

**Most Critical Bug in Project History** (FAIL-001):

**Impact**: 29 dB regression (37 dB → 7 dB)
**Duration**: 3 days of debugging
**Root Cause**: Gaussians clamped to 1 pixel → zero coverage → no gradients

**Code** (the bug):
```rust
let clamp_px = 0.5 + 0.3 * geod_dist_px;  // If geod_dist=0 → 0.5 pixels!
let sig_final = sig.min(clamp_px).clamp(1.0, 24.0);  // Clamped to 1 pixel
// Result: W_median = 0.000 (zero coverage)
```

**Prevention**: ALWAYS check W_median > 0.5 in debug builds

**If you see quality stuck at 5-7 dB**: Check coverage immediately!

---

### WARNING 3: Integration Bottleneck

**Discovery** (Oct 6, 2025):
- 22 production-ready modules (39%) unused
- Quality stuck because not integrated
- Implementation ≠ Integration

**Prevention**: After implementing, ALWAYS integrate and benchmark

---

## 17. SUCCESS PATTERN (Session 7 Model)

**The Best Session** (Oct 6, 2025, 4 hours):

**Approach**:
1. **Audit**: Found 22 unused modules
2. **Prioritize**: Selected TOP 10 by impact
3. **Plan**: Created Master Integration Plan
4. **Execute**: 4-hour focused session
5. **Validate**: Benchmark each integration

**Result**: +8.08 dB average improvement ✅

**Why it worked**:
- Planning before coding
- Systematic approach
- Clear priorities
- Validation at each step
- Focused (not marathon)

**Template**: Use this approach for future work

---

## 18. REPOSITORIES & LOCATIONS

### GitHub (Public, Synced)
**URL**: https://github.com/lamco-admin/lamcogaussianimage
**Contains**:
- All production code
- All test data
- Distilled documentation
- **Clone this for Claude Code Web**

### Local (Private, Not Synced)
**Path**: `/home/greg/gaussian-image-projects/local-research/`
**Contains**:
- 116 raw session markdown files
- 7,679 lines research notes
- All private development history
- **Reference only, not for Claude Code Web**

---

## 19. WHAT TO DO RIGHT NOW

### Immediate Action Items

**For Next Development Session**:

1. **Clone repository** (if using Claude Code Web):
   ```bash
   git clone https://github.com/lamco-admin/lamcogaussianimage.git
   cd lamcogaussianimage
   ```

2. **Read in order**:
   - This document (SESSION_HANDOFF.md) ← You are here
   - DEVELOPMENT_STANDARDS.md (quality requirements)
   - ROADMAP_CURRENT.md (what's next)
   - PROJECT_HISTORY.md (complete context, if first time)

3. **Validate build**:
   ```bash
   cd packages/lgi-rs
   cargo build --release --all
   cargo test --workspace
   ```

4. **Run baseline benchmark**:
   ```bash
   cargo run --release --example fast_benchmark
   ```
   Save output - compare after any changes

5. **Pick priority task**: From ROADMAP_CURRENT.md section "Immediate Priorities"

6. **Start work**: Following all quality standards

---

## 20. FOUNDATION QUALITY GUARANTEE

**What You Have**:
- ✅ Complete project history (nothing lost)
- ✅ All experiments documented (successes AND failures)
- ✅ All decisions explained (with full rationale)
- ✅ Clear roadmap (priorities defined)
- ✅ Quality standards (mandatory, comprehensive)
- ✅ Clean codebase (no artifacts, organized)
- ✅ Professional GitHub repo (public, documented)

**What You Don't Have**:
- ❌ Session confusion (PROJECT_HISTORY prevents)
- ❌ Lost context (distilled docs preserve)
- ❌ Unknown decisions (DECISIONS.md explains)
- ❌ Repeated mistakes (EXPERIMENTS.md FAIL sections)
- ❌ Unclear priorities (ROADMAP_CURRENT.md defines)

**Result**: Rock solid foundation for continuation to success ✅

---

## SUMMARY

**Repository**: https://github.com/lamco-admin/lamcogaussianimage

**Current Status** (Nov 14, 2025):
- Production-ready foundation ✅
- +8 dB quality improvement achieved ✅
- 32/150 techniques integrated (Track 1: 21%)
- 6/20 format features complete (Track 2: 30%)
- Ready for next development phase ✅

**Next Priorities**:
1. Real photo benchmark validation [P0]
2. Empirical R-D curve fitting [P0]
3. Gaussian count strategy [P1]
4. Complete Track 1 P1 (5 remaining) [P1]

**Quality Standards**: MANDATORY - no shortcuts, complete implementations only

**Documentation**: Comprehensive context always available

**Ready**: For systematic continuation to 30-35 dB target and production deployment

---

**Last Updated**: November 14, 2025
**Status**: COMPLETE HANDOFF - Ready for next session
**Next**: Read DEVELOPMENT_STANDARDS.md, then begin Task 1 (Real Photo Benchmark)
