# START HERE - Next LGI Development Session

**Date Created**: November 14, 2025
**Repository**: https://github.com/lamco-admin/lamcogaussianimage
**Purpose**: Quick-start prompt for continuing LGI codec development

---

## 🚀 QUICK START (5 Minutes)

### You Are Working On

**LGI (Lamco Gaussian Image) Codec** - A revolutionary cross-platform image codec based on 2D Gaussian splatting.

**Current Status** (Nov 14, 2025):
- Production-ready foundation ✅
- +8 dB quality improvement achieved (14.67 → 24.36 dB)
- 32/150 mathematical techniques integrated (Track 1: 21%)
- 6/20 format features complete (Track 2: 30%)
- Ready for real-world validation phase

**Your Goal**: Continue systematic development toward 30-35 dB target and production deployment.

---

## 📖 MANDATORY READING (Before Coding)

**Read IN THIS ORDER** (total: 30 min first time, 10 min thereafter):

### Every Session (ALWAYS):
1. **`docs/SESSION_HANDOFF.md`** (10 min)
   - Complete context
   - Quality standards
   - Current status
   - Next tasks

2. **`docs/DEVELOPMENT_STANDARDS.md`** (5 min)
   - NO shortcuts rule
   - Testing requirements
   - Anti-patterns
   - **Mandatory quality gates**

3. **`docs/research/ROADMAP_CURRENT.md`** (5 min)
   - Immediate priorities
   - Track progress
   - What to work on

### First Time Only (or if context lost):
4. **`docs/research/PROJECT_HISTORY.md`** (20 min)
   - Complete journey (Sept-Oct 2025)
   - 8 development sessions
   - Critical lessons
   - **Prevents repeating mistakes**

5. **`docs/research/EXPERIMENTS.md`** (15 min, skim FAIL sections)
   - What worked
   - **What failed (5 major failures)**
   - FAIL-001: 29 dB regression bug ⚠️
   - Why things failed

---

## ⚡ CRITICAL RULES (NO EXCEPTIONS)

### 🚫 NEVER DO:
- ❌ Write TODO comments or stub implementations
- ❌ Simplify problems away or bypass issues
- ❌ Skip tests ("we'll add later")
- ❌ Claim performance without measuring
- ❌ Trust documentation over validation
- ❌ Marathon coding sessions (18 hours)
- ❌ Implement without integrating
- ❌ Remove features when debugging (fix them instead)

### ✅ ALWAYS DO:
- ✅ Complete implementations only (full algorithms)
- ✅ Face problems directly (debug root cause)
- ✅ Write tests FIRST or alongside code
- ✅ Validate with benchmarks before claiming
- ✅ Measure PSNR + MS-SSIM empirically
- ✅ 4-5 hour focused sessions (systematic)
- ✅ Integrate after implementing (then benchmark)
- ✅ Add instrumentation when debugging

**Rationale**: These rules prevent all historical failures from October 2025.

**Document**: See `docs/DEVELOPMENT_STANDARDS.md` for full standards.

---

## 🎯 IMMEDIATE NEXT TASKS (Priority Order)

### Task 1: Real Photo Benchmark [P0 - CRITICAL]

**Status**: Started Session 8, paused for reorganization

**Goal**: Validate +8 dB on real photographic content

**Why Critical**:
- Current measurements ONLY on 2 synthetic images
- Need proof +8 dB generalizes to real photos
- Foundation for all empirical work

**Action**:
```bash
cd packages/lgi-rs/lgi-benchmarks
cargo run --release --bin real_world_benchmark
```

**Expected**: ~2-4 hours runtime (67 4K photos + 24 Kodak images)

**Collect**:
- PSNR per image per method
- MS-SSIM per image
- Encoding time
- Gaussian count
- Image type classification

**Success**: Adam maintains +6-8 dB average on photos

---

### Task 2: Empirical R-D Curve Fitting [P0 - CRITICAL]

**Problem**: `encode_for_psnr(30.0)` delivers ~24 dB (6 dB short)

**Approach**:
1. Use data from Task 1
2. Fit: PSNR = a×log(N) + b×complexity + c
3. Validate on held-out images (R² > 0.85)
4. Update `packages/lgi-rs/lgi-encoder-v2/src/rate_distortion.rs`
5. Test targeting again

**Success**: Target 30 dB → achieve 29-31 dB (±1 dB)

---

### Task 3: Gaussian Count Strategy [P1 - HIGH]

**Question**: Does entropy-based N advantage (+2.93 dB) persist after optimization?

**Current** (pre-optimization only):
- Arbitrary: N=31, 17.03 dB
- Entropy: N=2635, 19.96 dB (+2.93 dB, but 80× more Gaussians!)

**Experiment**: Full optimization test (100 iterations) on both strategies

**Decision needed**: Quality vs encoding time trade-off

---

## 🧪 BEFORE STARTING WORK

### Validation Commands

```bash
# 1. Build everything
cd packages/lgi-rs
cargo build --release --all

# 2. Run all tests
cargo test --workspace

# 3. Baseline benchmark (SAVE THIS)
cargo run --release --example fast_benchmark > baseline_$(date +%Y%m%d).txt

# 4. GPU check
cargo run --release --example backend_detection

# Expected output:
# ✅ Vulkan available
# ✅ GPU functional
```

**All should pass** - if not, debug before proceeding

---

## 📊 QUALITY METRICS (Reference)

### Current Achievements (Validated Oct 6, 2025)

```
Sharp Edge Images (128×128):
  Method: encode_error_driven_adam()
  Before: 14.67 dB
  After:  24.36 dB
  Delta:  +9.69 dB ✅

Complex Pattern Images (128×128):
  Method: encode_error_driven_gpu()
  Before: 15.50 dB
  After:  21.96 dB
  Delta:  +6.46 dB ✅

Average: +8.08 dB improvement ✅
```

**Validation**: Reproducible via `fast_benchmark.rs`

---

### Targets (Not Yet Achieved)

```
PSNR Target:       30-35 dB (currently 22-24 dB)
Gap:               6-13 dB remaining
Path:              Track 1 P1 completion + empirical tuning

Compression:       7.5-10.7× (not measured yet)
Blocker:           Track 2 quantization not implemented

GPU Gradients:     1500× speedup (70% done)
Blocker:           Compilation errors in gradient.rs
```

---

## 🛠️ DEVELOPMENT SETUP

### Repository Structure

```
lamcogaussianimage/
├── packages/lgi-rs/          ← Main Rust implementation (your primary workspace)
│   ├── lgi-encoder-v2/       ← Active development (optimization, +8dB)
│   ├── lgi-core/             ← Technique modules (150 total)
│   ├── lgi-gpu/              ← GPU acceleration (gradient.rs needs work)
│   ├── lgi-benchmarks/       ← Testing (use frequently)
│   └── ...8 more crates
│
├── test-data/                ← Test images (use for validation)
│   ├── test_images/          ← 67 real 4K photos
│   ├── test_images_new_synthetic/  ← Synthetic patterns
│   └── kodak-dataset/        ← 24 industry benchmarks
│
└── docs/                     ← Documentation (read before coding)
    ├── SESSION_HANDOFF.md    ← Start here (this file)
    ├── DEVELOPMENT_STANDARDS.md  ← Quality requirements
    ├── CLAUDE_CODE_WEB_GUIDE.md  ← Quick reference
    └── research/             ← Distilled knowledge
        ├── PROJECT_HISTORY.md    ← Complete context
        ├── EXPERIMENTS.md        ← What worked/failed
        ├── DECISIONS.md          ← Why we chose this
        └── ROADMAP_CURRENT.md    ← What's next
```

---

## ⚠️ CRITICAL WARNINGS

### Warning 1: Check Coverage (W_median > 0.5)

**From FAIL-001** (Most critical bug in project history):
- Gaussians clamped to 1 pixel
- Zero coverage → zero gradients → no optimization
- Quality stuck at 5-7 dB for 3 days

**Prevention**:
```rust
#[cfg(debug_assertions)]
{
    let w_median = compute_median_weight(&weights);
    assert!(w_median > 0.5,
        "Zero coverage! Check Gaussian scales. W_median={:.4}", w_median);
}
```

**If quality stuck <10 dB**: Check coverage IMMEDIATELY

---

### Warning 2: Validate Before Claiming

**Historical failure**: Documentation claimed 30-40 dB, reality was 5-7 dB

**Prevention**:
- Run benchmarks
- Save results
- Compare with previous
- Only claim what's measured

**Template**:
```
Achievement: +8 dB improvement ✅
Evidence: fast_benchmark.rs output (saved)
Date: Oct 6, 2025
Reproducible: Yes
```

---

### Warning 3: Integration ≠ Implementation

**Discovery** (Oct 6): 22 modules (39%) implemented but unused

**Prevention**:
- After implementing, INTEGRATE into encoder
- Benchmark to verify impact
- Don't let code sit unused

---

## 📝 SESSION START CHECKLIST

**Copy this at start of every session**:

```
## Pre-Session Checklist
- [ ] Read SESSION_HANDOFF.md (this file)
- [ ] Read DEVELOPMENT_STANDARDS.md
- [ ] Read ROADMAP_CURRENT.md
- [ ] Git pull latest
- [ ] Build: cargo build --release --all
- [ ] Test: cargo test --workspace
- [ ] Baseline: cargo run --example fast_benchmark > baseline.txt
- [ ] Pick task from ROADMAP priorities

## Work Plan
Task: [From ROADMAP]
Approach: [Strategy]
Expected Impact: [PSNR gain, speedup, etc.]

## Session Log
[Document as you work]

## Results
Before: [PSNR, time]
After: [PSNR, time]
Delta: [Improvement]

## Post-Session Checklist
- [ ] Tests passing
- [ ] Benchmarks run (no regressions)
- [ ] Quality measured (PSNR + MS-SSIM)
- [ ] ROADMAP updated
- [ ] Committed and pushed
- [ ] Next TODO written
```

---

## 🎓 KEY LEARNINGS (Remember)

### From Session 7 (Best Session)
**Approach**: Master Integration Plan
- 4 hours focused
- TOP 10 integrations
- Systematic validation
- **Result**: +8 dB ✅

**Lesson**: Planning > heroics

---

### From Sessions 4-5 (Regression Hunt)
**Problem**: 29 dB regression, 3 days wasted
**Cause**: No validation, documentation ≠ code
**Lesson**: Validate everything, trust measurements not docs

---

### From Session 2 (Marathon)
**Duration**: 18 hours continuous
**Result**: Working but exhausted, bugs late
**Lesson**: 4-hour sessions better

---

## 🔧 QUICK COMMANDS

```bash
# Build
cd packages/lgi-rs && cargo build --release --all

# Test
cargo test --workspace

# Quick benchmark (2 images, 30s)
cargo run --release --example fast_benchmark

# Full Kodak (24 images, 10min)
cd lgi-benchmarks
cargo run --release --bin kodak_benchmark

# Real photos (67 images, 2-4 hours)
cargo run --release --bin real_world_benchmark

# GPU check
cd lgi-gpu
cargo run --release --example backend_detection
```

---

## 🎯 SUMMARY

**Repository**: https://github.com/lamco-admin/lamcogaussianimage

**Status**: Rock solid foundation, ready for continuation

**Next**: Task 1 - Real photo benchmark validation

**Standards**: MANDATORY quality requirements (no shortcuts)

**Context**: Complete (PROJECT_HISTORY, EXPERIMENTS, DECISIONS, ROADMAP)

**Ready**: For systematic development to 30-35 dB target

---

## 🚦 GO/NO-GO

**You are GO for development if**:
- ✅ Read SESSION_HANDOFF.md
- ✅ Read DEVELOPMENT_STANDARDS.md
- ✅ Read ROADMAP_CURRENT.md
- ✅ Understand: NO shortcuts, complete implementations only
- ✅ Understand: Validate everything, measure don't assume
- ✅ Repository cloned and builds successfully

**You are NO-GO if**:
- ❌ Haven't read required docs
- ❌ Don't understand quality standards
- ❌ Planning to use TODOs or stubs
- ❌ Repository doesn't build

---

**When ready, begin with Task 1: Real Photo Benchmark**

**Remember**: Best quality, fullest implementation, systematic approach, validate everything.

**Foundation is solid. Let's build on it.** 🚀

---

**Last Updated**: November 14, 2025
**Next Session**: Continue Session 8 validation (Real photo benchmarks)
