# LGI Development Standards & Quality Requirements

**Project**: LGI (Lamco Gaussian Image) Codec
**Last Updated**: November 14, 2025
**Purpose**: Define uncompromising quality standards for all development work
**Status**: MANDATORY - All contributors must follow these standards

---

## Core Principles

### Principle 1: NO SHORTCUTS, EVER

**This project is building foundational technology comparable to AV1 or AVIF.**

**What this means**:
- No stub implementations ("TODO: implement later")
- No simplified versions ("let's just do the basic case")
- No skipped tests ("we'll add tests later")
- No "good enough for now" code
- No placeholders or incomplete features

**Rationale**:
> "This is NOT a simple codec experiment. This is a revolutionary image/video codec + SDK + ecosystem with a multi-year vision."

**User's mandate**: Best quality, fullest implementation, most features, most flexibility.

---

### Principle 2: COMPLETE IMPLEMENTATION ONLY

**Every feature must be**:
1. **Fully implemented** - No partial/stub code
2. **Thoroughly tested** - Unit tests + integration tests + benchmarks
3. **Properly documented** - API docs + inline comments + examples
4. **Validated empirically** - Measure impact with real data
5. **Production-ready** - Error handling, edge cases, performance

**Anti-pattern examples**:
```rust
// ❌ NEVER DO THIS
pub fn feature_x() {
    // TODO: Implement later
    unimplemented!()
}

// ❌ NEVER DO THIS
pub fn feature_y() {
    // Simplified version, full implementation later
    simple_approach()  // Missing 80% of actual algorithm
}

// ❌ NEVER DO THIS
#[cfg(test)]
mod tests {
    // TODO: Add tests
}
```

**Correct approach**:
```rust
// ✅ DO THIS
pub fn feature_x() -> Result<Output> {
    // Full algorithm implementation per paper/spec
    let step1 = complete_step_1()?;
    let step2 = complete_step_2(step1)?;
    validate_output(step2)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_feature_x_basic() { /* comprehensive test */ }

    #[test]
    fn test_feature_x_edge_cases() { /* edge cases */ }

    #[test]
    fn test_feature_x_performance() { /* benchmarks */ }
}
```

---

### Principle 3: FACE PROBLEMS DIRECTLY

**When encountering difficulties**:

**❌ NEVER**:
- Simplify the problem away
- Skip the hard parts
- "Leave for later"
- Implement workaround instead of solution
- Hope it won't matter

**✅ ALWAYS**:
- Debug until root cause found
- Implement the full solution
- Add comprehensive error handling
- Document the complexity
- Test the edge cases

**Example from history** (FAIL-001: Geodesic EDT Over-Clamping):
- **Wrong approach**: "Geodesic clamping causing issues? Let's just remove it."
- **Right approach**: "Why is it over-clamping? Fix the formula, not remove the feature."
- **Result**: Proper implementation of anti-bleeding without sacrificing coverage

---

### Principle 4: VALIDATE EVERYTHING

**Claims require evidence**:

**❌ Never claim**:
- "Should achieve 30 dB" (without testing)
- "Probably 10× compression" (without measuring)
- "This will work" (without validation)

**✅ Always validate**:
- Run benchmarks before claiming performance
- Measure quality improvements empirically
- Test on multiple image types
- Compare against baselines

**From lessons learned**:
> "Documentation ≠ Reality. Sessions 1-3 claimed 30-40 dB. Reality was 5-7 dB. 29 dB regression went unnoticed."

**Prevention**: Automated benchmarks, validation before claims, conservative estimates.

---

### Principle 5: ALL 150 TECHNIQUES MATTER

**User's philosophy**:
> "Even if tests don't show benefit, implement ALL 150 techniques. Mathematical foundation matters."

**What this means**:
- Technique shows +0.1 dB on test images? **Still implement it.**
- Might only help medical images? **Still implement it.**
- Not sure it's useful? **Still implement it.**

**Rationale**:
- Unknown unknowns (can't predict all use cases)
- Future-proofing the toolkit
- Research-backed approaches have value
- Comprehensive foundation > minimal product

**Priority system**:
- P1: Critical (15 techniques) - Implement first
- P2: High value (20 techniques) - Next
- P3: Enhancement (65 techniques) - Later
- **All eventually get implemented**

---

## Development Standards

### Standard 1: Testing Requirements

**Every module MUST have**:

1. **Unit Tests**:
   - Test each function independently
   - Cover edge cases (empty input, extreme values, errors)
   - Minimum 80% code coverage

2. **Integration Tests**:
   - Test module with real data
   - Validate against reference implementation (if exists)
   - Test interaction with other modules

3. **Benchmarks**:
   - Measure performance (time)
   - Measure quality (PSNR, MS-SSIM)
   - Compare with baseline
   - Document results

**Example**:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Basic case
        let result = function_under_test(valid_input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_edge_case_empty() {
        // Edge case: empty input
        let result = function_under_test(empty_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_case_extreme() {
        // Edge case: extreme values
        let result = function_under_test(extreme_input);
        assert!(result.is_ok());
        assert!(result.unwrap().is_valid());
    }

    #[test]
    fn test_reference_validation() {
        // Validate against known good output
        let result = function_under_test(test_input);
        assert_approx_eq!(result.unwrap(), expected_output, 1e-6);
    }
}

#[cfg(test)]
mod benchmarks {
    use criterion::{black_box, criterion_group, Criterion};

    fn bench_function(c: &mut Criterion) {
        c.bench_function("function_name", |b| {
            b.iter(|| function_under_test(black_box(input)))
        });
    }

    criterion_group!(benches, bench_function);
}
```

**No exceptions**: If code exists, tests must exist.

---

### Standard 2: Documentation Requirements

**Every public function MUST have**:

1. **Doc comment** explaining:
   - What it does
   - Parameters (type, meaning, constraints)
   - Return value (type, meaning, error conditions)
   - Example usage
   - Performance characteristics (if relevant)
   - References (paper, spec section)

2. **Inline comments** for:
   - Complex algorithms (explain the math)
   - Non-obvious optimizations
   - Numeric constants (where they come from)
   - Assumptions and invariants

**Example**:
```rust
/// Compute optimal Gaussian count based on image entropy.
///
/// Uses variance-based content analysis to adaptively determine
/// the number of Gaussians needed to represent the image at
/// target quality.
///
/// # Arguments
/// * `image` - Source image (RGB or grayscale)
///
/// # Returns
/// * Recommended Gaussian count (typically 100-5000 for 512×512 images)
///
/// # Performance
/// * O(W×H) - single pass over image
/// * ~5ms for 512×512 images
///
/// # References
/// * Based on entropy analysis from Session 7 integration
/// * See: docs/research/EXPERIMENTS.md EXP-002
///
/// # Examples
/// ```
/// let image = load_image("test.png")?;
/// let n = adaptive_gaussian_count(&image);
/// println!("Recommended N: {}", n);  // e.g., "Recommended N: 2635"
/// ```
pub fn adaptive_gaussian_count(image: &ColorSource) -> usize {
    // Compute local variance (content complexity indicator)
    let variance = compute_local_variance(image);

    // Higher variance → more Gaussians needed
    let base_count = (image.width() * image.height()) as f32).sqrt() / 10.0;
    let entropy_factor = variance.mean() / variance.global_mean();

    // Formula: N = base × (1 + entropy_factor)
    // Typical: base=500, entropy_factor=0.2-2.0 → N=600-1500
    (base_count * (1.0 + entropy_factor)) as usize
}
```

---

### Standard 3: Performance Requirements

**Every performance claim MUST**:

1. **Be measured** on reference hardware
2. **Show benchmarks** (code + results)
3. **Include variance** (not just single run)
4. **Compare baseline** (show improvement)
5. **Test at scale** (not just tiny test images)

**Benchmark template**:
```rust
// benchmarks/technique_name.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_technique(c: &mut Criterion) {
    let mut group = c.benchmark_group("technique_name");

    // Test on multiple scales
    for size in [128, 256, 512, 1024, 2048] {
        let image = generate_test_image(size, size);

        group.bench_with_input(
            BenchmarkId::new("baseline", size),
            &image,
            |b, img| b.iter(|| baseline_method(img))
        );

        group.bench_with_input(
            BenchmarkId::new("optimized", size),
            &image,
            |b, img| b.iter(|| optimized_method(img))
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_technique);
criterion_main!(benches);
```

**Report format**:
```
Technique: Feature X
Baseline:  145ms ± 12ms  (N=3 runs)
Optimized:  52ms ± 3ms   (N=3 runs)
Speedup:   2.79× ± 0.15×
Quality:   No regression (24.3 dB → 24.4 dB)
```

---

### Standard 4: Error Handling

**Every fallible operation MUST**:

1. **Return Result<T, E>** (not panic, not unwrap)
2. **Provide context** in error messages
3. **Handle gracefully** (fallback if possible)
4. **Log appropriately** (debug, warn, error levels)

**Examples**:

**❌ Never do this**:
```rust
pub fn load_image(path: &str) -> Image {
    let data = std::fs::read(path).unwrap();  // PANIC if file missing!
    decode_png(&data).unwrap()  // PANIC if corrupt!
}
```

**✅ Always do this**:
```rust
pub fn load_image(path: &str) -> Result<Image, LgiError> {
    let data = std::fs::read(path)
        .map_err(|e| LgiError::IoError {
            path: path.to_string(),
            source: e,
        })?;

    decode_png(&data)
        .map_err(|e| LgiError::InvalidFormat {
            path: path.to_string(),
            reason: format!("PNG decode failed: {}", e),
        })
}
```

**With fallback**:
```rust
pub fn render_with_fallback(gaussians: &[Gaussian2D]) -> Result<Image> {
    // Try GPU first
    match GpuManager::global().render(gaussians) {
        Ok(image) => {
            log::info!("GPU render successful");
            Ok(image)
        },
        Err(e) => {
            log::warn!("GPU render failed: {}, falling back to CPU", e);
            // Fallback to CPU
            CpuRenderer::new().render(gaussians)
        }
    }
}
```

---

### Standard 5: Code Quality

**Every commit MUST**:

1. **Compile without warnings**:
   ```bash
   cargo clippy --workspace -- -D warnings
   ```

2. **Format consistently**:
   ```bash
   cargo fmt --all
   ```

3. **Pass all tests**:
   ```bash
   cargo test --workspace
   ```

4. **No dead code** (unless explicitly marked for future use)

5. **No commented-out code** (use git history instead)

---

## Anti-Patterns (NEVER DO THESE)

### Anti-Pattern 1: Aspirational Documentation

**❌ WRONG**:
```markdown
## Performance

Our codec achieves:
- 30-40 dB PSNR ✅
- 7.5-10.7× compression ✅
- 1000+ FPS decode ✅
```
*(When none of these are actually measured)*

**✅ RIGHT**:
```markdown
## Performance (Validated)

Achieved (measured, Nov 2025):
- 22-24 dB PSNR ✅ (Sharp: 24.36, Complex: 21.96)
- Compression: Not yet measured ⏳
- 1,176 FPS GPU decode ✅ (RTX 4060, validated)

Targets (not yet achieved):
- 30-35 dB PSNR 🎯 (gap: 6-13 dB, work ongoing)
- 7.5-10.7× compression 🎯 (needs quantization profiles)
- 1000+ FPS 🎯 (achieved on NVIDIA, need AMD/Intel validation)
```

**Lesson from history**: Session 4-5 lost 3 days hunting "regression" that was never actually fixed.

---

### Anti-Pattern 2: Marathon Sessions

**❌ WRONG**: 18-hour coding marathon
- Exhaustion → mistakes
- Context overflow
- Poor decisions late in session

**✅ RIGHT**: 4-hour focused session
- Clear plan
- Systematic execution
- Fresh mind
- Better results

**Evidence from history**:
- Session 2: 18 hours → bugs, exhaustion
- Session 7: 4 hours → +8 dB, no bugs ✅

---

### Anti-Pattern 3: Implementation Without Integration

**❌ WRONG**:
```
Week 1: Implement technique A ✅
Week 2: Implement technique B ✅
Week 3: Implement technique C ✅
Week 4: Quality still at baseline ❌
Reason: None of them are actually USED!
```

**✅ RIGHT** (Session 7 approach):
```
Plan: Master Integration Plan
- Identify unused modules (found 22!)
- Prioritize by impact (TOP 10)
- Integrate systematically (one session)
- Validate each (benchmark before/after)
Result: +8 dB improvement ✅
```

**Lesson**: Implementation ≠ Integration. Integration is the bottleneck.

---

### Anti-Pattern 4: Single-Metric Optimization

**❌ WRONG**: Optimize for PSNR only
- Miss perceptual quality issues
- Sharp edges penalized
- Visually poor at "good" PSNR

**✅ RIGHT**: Multi-metric validation
- PSNR (baseline, comparable)
- MS-SSIM (perceptual structure)
- Visual inspection (human validation)
- Use-case specific (web vs archival)

**Implementation**: Always report both PSNR and MS-SSIM

---

### Anti-Pattern 5: Overconfident Assumptions

**❌ WRONG**: "Isotropic is better than anisotropic" (Session 4)
**✅ RIGHT**: "Both needed, conditionally applied" (Session 5 correction)

**Lesson**: Don't generalize from limited tests. Context matters.

**Prevention**:
- Test on diverse content
- Understand WHY something works
- Don't assume universal solutions
- Content-adaptive > one-size-fits-all

---

## Quality Gates

### Gate 1: Before Commit

**Every commit MUST pass**:
```bash
# 1. Format check
cargo fmt --all --check

# 2. Lint check
cargo clippy --workspace -- -D warnings

# 3. All tests pass
cargo test --workspace

# 4. No build warnings
cargo build --release --all 2>&1 | grep -i warning && exit 1

# 5. Documentation builds
cargo doc --no-deps
```

**Automated**: Set up git pre-commit hook

---

### Gate 2: Before PR/Merge

**Every PR MUST include**:

1. **Benchmark results**:
   - Before/after measurements
   - Multiple test images
   - PSNR + MS-SSIM
   - No regressions

2. **Test coverage**:
   - New code has tests
   - Edge cases covered
   - Integration tests pass

3. **Documentation**:
   - API docs complete
   - Inline comments for complex code
   - Updated ROADMAP if priorities changed

4. **Validation**:
   - On synthetic test images
   - On real photos
   - On Kodak dataset (if major change)

---

### Gate 3: Before Session End

**Every session MUST**:

1. **Validate work**:
   ```bash
   cargo run --release --example fast_benchmark
   ```

2. **Document findings**:
   - Update ROADMAP_CURRENT.md
   - Note any issues discovered
   - Record measurements

3. **Clean state**:
   - All tests passing
   - No uncommitted changes (or document why)
   - Clear TODO for next session

4. **Benchmark snapshot**:
   ```bash
   # Save baseline for next session
   cargo run --release --example comprehensive_benchmark > session_N_baseline.txt
   ```

---

## Critical Learnings from History

### Learning 1: Instrumentation First

**From FAIL-001 (Geodesic EDT bug)**:

**Problem**: Quality stuck at 5-7 dB, spent days guessing
**Solution**: Added diagnostic logging
```rust
log::debug!("σ_base: {:.2}, σ_clamped: {:.2}, W_median: {:.4}",
    sigma_base, sigma_clamped, weight_median);
```
**Result**: Found bug in minutes (W_median=0.000 → zero coverage)

**Lesson**:
> "Detailed instrumentation finds bugs in minutes vs hours of guessing"

**Standard**: Always add logging for intermediate values when debugging

---

### Learning 2: Both Techniques, Conditionally

**From FAIL-006 (Anisotropy misunderstanding)**:

**Early (wrong)**: "Isotropic better than anisotropic"
**Later (correct)**: "Both essential, conditionally applied"

**Code**:
```rust
let (σ_x, σ_y) = if coherence < 0.2 {
    // Flat regions → isotropic
    (σ, σ)
} else {
    // Edges → anisotropic
    (σ * 0.5, σ * 2.0)
};
```

**Lesson**: Beware either/or thinking. Build toolkit, use conditionally.

---

### Learning 3: Coverage is Critical

**From FAIL-001**:

**Requirement**: Gaussians must have coverage
**Metric**: W_median > 0.5 (at least half of pixels have weight)
**Check**: Always validate in debug builds

**Code**:
```rust
#[cfg(debug_assertions)]
{
    let w_median = compute_median_weight(&weights);
    assert!(w_median > 0.5,
        "Zero coverage! σ={:.4}, W_median={:.4}", sigma, w_median);
}
```

**Prevention**: Add coverage assertions during development

---

### Learning 4: Validation Prevents Regressions

**From Sessions 4-5**:

**Problem**: 29 dB regression unnoticed for days
**Cause**: No automated benchmarks
**Impact**: 3 days wasted

**Solution**: Automated validation
```bash
# Before any changes
cargo run --release --example fast_benchmark > baseline.txt

# After changes
cargo run --release --example fast_benchmark > current.txt

# Compare
diff baseline.txt current.txt
# If PSNR decreased → INVESTIGATE IMMEDIATELY
```

**Standard**: Always benchmark before/after significant changes

---

## Workflow Standards

### Standard Workflow

1. **Plan First**:
   - Read ROADMAP_CURRENT.md
   - Pick highest priority task
   - Understand requirements
   - Check EXPERIMENTS.md (did we try this before?)
   - Check DECISIONS.md (why did we choose current approach?)

2. **Design**:
   - Understand the math/algorithm fully
   - Read relevant papers
   - Identify edge cases
   - Plan test strategy

3. **Implement**:
   - Write tests FIRST (TDD where possible)
   - Implement complete solution (no TODOs)
   - Add comprehensive error handling
   - Document as you go

4. **Validate**:
   - Run unit tests
   - Run integration tests
   - Benchmark performance
   - Measure quality (PSNR, MS-SSIM)
   - Visual inspection

5. **Document**:
   - Update ROADMAP if needed
   - Note findings
   - Record measurements
   - Update EXPERIMENTS.md if significant

6. **Commit**:
   - Clean commit message
   - Reference issue/task
   - Include measurements in commit message

---

### Session Structure

**Start** (30 min):
- Read PROJECT_HISTORY.md (if first time, or refresh if needed)
- Read ROADMAP_CURRENT.md (always)
- Check git status, pull latest
- Run baseline benchmark
- Review previous session findings

**Work** (3-4 hours):
- Focused implementation
- Regular testing
- Document as you go

**End** (30 min):
- Run final benchmarks
- Update ROADMAP_CURRENT.md
- Commit and push
- Document TODO for next session

**Total**: 4-5 hours MAX (prevent exhaustion)

---

## Validation Checklist

### Before Claiming Success

**Feature implementation complete ONLY if**:
- [ ] Algorithm fully implemented (no simplifications)
- [ ] All edge cases handled
- [ ] Error handling comprehensive
- [ ] Unit tests written and passing
- [ ] Integration tests written and passing
- [ ] Benchmarks show expected performance
- [ ] Quality measured (PSNR, MS-SSIM)
- [ ] Tested on multiple image types
- [ ] No regressions in other tests
- [ ] Documentation complete
- [ ] Code review ready

**If ANY checkbox unchecked**: NOT DONE YET

---

## Testing Standards

### Test Coverage Requirements

**Every module MUST test**:

1. **Happy path**: Normal, expected inputs
2. **Edge cases**:
   - Empty input
   - Single element
   - Maximum size
   - Extreme values
3. **Error cases**:
   - Invalid input
   - Out of bounds
   - Numeric overflow
4. **Integration**:
   - Works with other modules
   - End-to-end flow

### Test Data Standards

**Always test on**:
1. **Synthetic images** (controlled):
   - Sharp edges (HF_checkerboard)
   - Complex patterns (HF_multi_gratings)
   - Smooth gradients (BN_radial_gradient)
   - Uniform (BN_uniform_gray)

2. **Real photos** (diverse):
   - At least 10 images
   - Different content types
   - Various resolutions

3. **Kodak dataset** (standard):
   - 24 industry-benchmark images
   - For comparison with other codecs

**Location**: All in `test-data/`

---

## Documentation Standards

### Required Documents

**For every significant feature**:

1. **API Documentation**:
   - Rustdoc comments (public API)
   - Examples in docs
   - Usage guide

2. **Implementation Notes**:
   - Algorithm explanation
   - Math derivation (if complex)
   - References to papers/specs

3. **Benchmark Results**:
   - Performance measurements
   - Quality measurements
   - Comparison with baseline

4. **Update Tracking Docs**:
   - ROADMAP_CURRENT.md (mark complete, update priorities)
   - EXPERIMENTS.md (if significant findings)
   - DECISIONS.md (if architectural choice made)

---

## Performance Standards

### Optimization Requirements

**Premature optimization is root of evil, BUT**:

1. **Algorithmic complexity** must be reasonable:
   - O(N²) acceptable for N < 5000
   - O(N³) needs justification
   - O(N⁴) unacceptable (unless N tiny)

2. **Use available parallelism**:
   - rayon for multi-threading (tiles, batches)
   - SIMD where applicable (hot paths)
   - GPU for suitable operations (rendering, gradients)

3. **Profile before optimizing**:
   - Measure where time spent
   - Optimize hot paths only
   - Validate speedup

**Example**:
```bash
# Profile
cargo build --release
perf record -g target/release/encoder input.png
perf report

# Optimize hot path

# Validate
cargo bench -- baseline vs optimized
```

---

## Research Standards

### When Implementing from Papers

**Process**:

1. **Read paper thoroughly**:
   - Understand math
   - Note assumptions
   - Identify hyperparameters

2. **Check existing implementations**:
   - Reference code (if available)
   - Validate understanding
   - Note differences

3. **Implement faithfully first**:
   - Match paper exactly
   - Use paper's notation
   - Reproduce results (if possible)

4. **Then optimize**:
   - Only after validation
   - Benchmark before/after
   - Document deviations

**Documentation**:
```rust
/// Implementation of [Algorithm Name] from [Paper].
///
/// # References
/// * [Author] et al., "[Title]", [Conference] [Year]
/// * arXiv: [link]
/// * Original implementation: [link if available]
///
/// # Differences from Paper
/// * We use f32 instead of f64 (negligible quality impact, 2× faster)
/// * We clamp iterations to 1000 max (paper used unlimited)
/// * We added early termination (our addition, +40% speedup)
```

---

## Git Standards

### Commit Message Format

```
Brief description (50 chars max)

Detailed explanation:
- What changed
- Why changed
- Impact (performance/quality measurements)
- References (issue, paper, spec section)

Validation:
- Tests: cargo test --workspace ✅
- Benchmarks: +2.3 dB on sharp edges, no regression on complex
- Clippy: No warnings ✅
```

**Example**:
```
Add entropy-based Gaussian count selection

Implements adaptive_gaussian_count() from Session 7 integration plan.
Uses variance-based content analysis to determine optimal N.

Changes:
- Added lgi-core/src/entropy.rs (250 lines)
- Integrated into lgi-encoder-v2/src/lib.rs
- Added comprehensive tests and benchmarks

Impact:
- Eliminates manual N tuning
- +2.93 dB on initialization (pre-optimization)
- Adapts to image complexity automatically
- Sharp: N=242, Complex: N=367

Validation:
- Tests: cargo test --package lgi-core ✅
- Benchmark: tested on 4 Kodak images
- No regressions on existing tests ✅

References:
- Session 7 Master Integration Plan
- EXPERIMENTS.md EXP-002
```

---

### Branch Strategy

**For features**:
- Branch name: `feature/descriptive-name`
- One feature per branch
- Keep up to date with main
- PR when complete (all quality gates passed)

**For bugs**:
- Branch name: `fix/bug-description`
- Include regression test
- Validate fix with test

**For experiments**:
- Branch name: `experiment/what-testing`
- Document results
- May not merge (if experiment fails)
- Preserve findings in EXPERIMENTS.md

---

## Maintenance Standards

### Code Maintenance

**Regularly**:
1. Update dependencies (`cargo update`)
2. Run full test suite
3. Check for deprecation warnings
4. Update documentation
5. Review TODO markers (if any - should be rare!)

### Documentation Maintenance

**After each session**:
- Update ROADMAP_CURRENT.md with progress
- Add findings to EXPERIMENTS.md (if significant)
- Update DECISIONS.md (if architectural choice made)

**Monthly**:
- Review and update README.md
- Check PROJECT_HISTORY.md for accuracy
- Validate all links in documentation

---

## Common Pitfalls (From History)

### Pitfall 1: Trust Documentation Over Code

**History**: Docs said "37 dB achieved", code delivered 7 dB
**Prevention**: Always validate, measure, confirm

### Pitfall 2: Optimize Before Understanding

**History**: Spent days on wrong problem (geodesic clamping)
**Prevention**: Instrument first, understand, then fix

### Pitfall 3: Implement Without Integrating

**History**: 39% of code unused, sitting idle
**Prevention**: Integration as important as implementation

### Pitfall 4: Simplify Away The Problem

**History**: Temptation to "just remove geodesic clamping"
**Prevention**: Fix the root cause, don't remove features

### Pitfall 5: Over-Generalize From Limited Data

**History**: "Isotropic better" from narrow testing
**Prevention**: Test diverse content, conditional application

---

## Success Patterns (From History)

### Success 1: Master Integration Plan (Session 7)

**Approach**:
1. Audit all modules (found 22 unused)
2. Prioritize by impact (TOP 10)
3. Systematic integration (4 hours)
4. Validate each integration

**Result**: +8 dB in one session ✅

**Lesson**: Planning > heroics

---

### Success 2: Diagnostic Instrumentation (Session 5)

**Approach**:
1. Add detailed logging (loss, PSNR, |Δc|, |Δμ|, W_median)
2. Single iteration test
3. Immediate diagnosis (W_median=0.000)

**Result**: Found 29 dB regression in minutes ✅

**Lesson**: Instrumentation > guessing

---

### Success 3: Empirical Validation (Session 8)

**Approach**:
1. Download Kodak dataset (industry standard)
2. Test at FULL resolution (caught downscaling bug)
3. Multiple strategies (arbitrary, entropy, hybrid)
4. Measure everything

**Result**: Data-driven decisions ✅

**Lesson**: Real data > assumptions

---

## Enforcement

### These Standards Are MANDATORY

**Not suggestions**: These are requirements
**No exceptions**: Quality cannot be compromised
**User's mandate**: "Best quality, fullest implementation"

### Code Review Checklist

**Every PR reviewed for**:
- [ ] No TODOs or stubs
- [ ] Complete implementation
- [ ] All tests passing
- [ ] Benchmarks included
- [ ] No regressions
- [ ] Documentation complete
- [ ] Error handling comprehensive
- [ ] Code formatted (cargo fmt)
- [ ] No clippy warnings

**If ANY item fails**: PR rejected, fix required

---

## Summary

**The Standard**: Production-quality, research-backed, comprehensively tested, fully documented code

**No compromises**: This is foundational technology with multi-year vision

**Quality over speed**: Better to do it right than do it fast

**Complete or nothing**: No partial implementations, no TODOs, no stubs

**Validate everything**: Measure, don't assume

**Learn from history**:
- PROJECT_HISTORY.md - What happened
- EXPERIMENTS.md - What worked/failed
- DECISIONS.md - Why we chose this

**These standards prevent**: Session confusion, lost context, regressions, wasted time

**These standards enable**: Systematic progress, high quality, successful completion

---

**Last Updated**: November 14, 2025
**Status**: MANDATORY for all LGI development
**Reference**: See PROJECT_HISTORY.md Lessons Learned section
