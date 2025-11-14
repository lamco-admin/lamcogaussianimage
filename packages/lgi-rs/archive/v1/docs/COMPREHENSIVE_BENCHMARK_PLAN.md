# Comprehensive LGI Codec Benchmark Plan

**Status**: Infrastructure Complete, Tests Running
**Date**: October 2, 2025
**Purpose**: Validate codec performance, identify bottlenecks, guide optimization

---

## 🎯 Benchmark Categories

### 1. Encoding Performance Benchmarks

**Goal**: Measure encoding speed under various conditions

**Test Dimensions**:
- **Gaussian Count**: 100, 200, 500, 1000, 2000, 5000, 10000
- **Image Size**: 128×128, 256×256, 512×512, 1024×1024
- **Quality Presets**: fast (500 iter), balanced (2000 iter), high (5000 iter), ultra (10000 iter)
- **Init Strategies**: Random, Grid, Gradient, Importance
- **Image Types**: See Pattern Benchmarks below

**Metrics Collected**:
- Encoding time (seconds)
- Iterations until convergence
- Gaussians optimized per second
- Final loss value
- Memory usage (peak RSS)

**Expected Insights**:
- Scaling: O(n × i × w × h) where n=Gaussians, i=iterations, w×h=resolution
- Bottlenecks: Rendering (forward pass) vs. gradient computation (backward pass)
- Init strategy impact on convergence speed
- Quality preset tradeoffs (time vs. quality)

### 2. Rendering/Decoding Performance Benchmarks

**Goal**: Measure decode speed and scaling behavior

**Test Dimensions**:
- **Gaussian Count**: 100, 500, 1000, 2000, 5000, 10000, 20000
- **Output Resolution**: 128×128, 256×256, 512×512, 1024×1024, 2048×2048, 4096×4096
- **Rendering Modes**: Sequential (1 thread), Parallel (4, 8, 16 threads)
- **Optimization Levels**: No culling, Bounding box only, Full optimization

**Metrics Collected**:
- Render time (milliseconds)
- FPS (frames per second)
- Megapixels per second
- Gaussians processed per second
- Gaussians culled (percentage)
- Early termination rate (percentage)
- CPU utilization (per-core)
- Memory bandwidth utilization

**Expected Insights**:
- Scaling with Gaussian count: Should be near-linear
- Scaling with resolution: Should be linear in pixels
- Parallel efficiency: Target 80-90% scaling to 16 cores
- Culling effectiveness: 70-95% depending on Gaussian distribution

### 3. Quality vs. Compression Benchmarks

**Goal**: Measure rate-distortion tradeoffs

**Test Dimensions**:
- **Gaussian Count**: 50, 100, 200, 500, 1000, 2000, 5000 (at fixed resolution)
- **Image Patterns**: All 10 test patterns
- **Compression Profiles** (when implemented): LGIQ-B, LGIQ-S, LGIQ-H, LGIQ-X

**Metrics Collected**:
- PSNR (dB)
- SSIM
- MS-SSIM (multi-scale)
- Perceptual loss (LPIPS, when available)
- Storage size (bytes, uncompressed)
- Compression ratio vs. PNG
- Compression ratio vs. JPEG (quality-matched)
- Bits per pixel (bpp)

**Expected Insights**:
- Diminishing returns curve: Where does quality plateau?
- Pattern difficulty: Which patterns need more Gaussians?
- Optimal Gaussian count per resolution
- Comparison with traditional codecs

### 4. Pattern-Specific Benchmarks

**Goal**: Identify difficult cases and failure modes

**Test Patterns**:
1. **Solid Color** (easiest)
   - Expected: Very few Gaussians needed, PSNR > 50 dB
   - Challenge: Ensures basic functionality

2. **Linear Gradient** (easy)
   - Expected: Smooth representation, PSNR > 40 dB
   - Challenge: Tests gradient handling

3. **Radial Gradient** (moderate)
   - Expected: Good representation, PSNR > 35 dB
   - Challenge: Tests non-axis-aligned structures

4. **Checkerboard** (hard)
   - Expected: Sharp edges difficult, PSNR 25-30 dB
   - Challenge: High-frequency content

5. **Concentric Circles** (hard)
   - Expected: Circular structures, PSNR 25-30 dB
   - Challenge: Periodic patterns

6. **Frequency Sweep** (very hard)
   - Expected: High-frequency regions difficult, PSNR 20-25 dB
   - Challenge: Variable frequency content

7. **Random Noise** (worst case)
   - Expected: Poor representation, PSNR 15-20 dB
   - Challenge: No spatial coherence (Gaussian strength is smoothness)

8. **Natural Scene** (realistic)
   - Expected: Balanced performance, PSNR 30-35 dB
   - Challenge: Representative of real-world use

9. **Geometric Shapes** (moderate)
   - Expected: Sharp edges, PSNR 28-33 dB
   - Challenge: Tests edge fidelity

10. **Text Pattern** (hard)
    - Expected: Thin lines difficult, PSNR 22-28 dB
    - Challenge: Critical for OCR, readability

**Expected Pattern Ranking** (Easiest to Hardest):
1. Solid Color (PSNR ~60 dB)
2. Linear Gradient (PSNR ~45 dB)
3. Radial Gradient (PSNR ~38 dB)
4. Natural Scene (PSNR ~32 dB)
5. Geometric (PSNR ~30 dB)
6. Concentric Circles (PSNR ~27 dB)
7. Text Pattern (PSNR ~25 dB)
8. Checkerboard (PSNR ~23 dB)
9. Frequency Sweep (PSNR ~20 dB)
10. Random Noise (PSNR ~15 dB)

### 5. Scaling & Parallelism Benchmarks

**Goal**: Validate multi-threading and memory behavior

**Tests**:
- **Thread Scaling**: 1, 2, 4, 8, 16, 32 threads
  - Measure: Speedup vs. single-thread, efficiency, Amdahl's law validation

- **Memory Scaling**: 100, 500, 1000, 5000, 10000, 50000 Gaussians
  - Measure: Memory usage, cache performance, TLB misses

- **Batch Size**: Process 1, 4, 8, 16 pixels in SIMD batches
  - Measure: Throughput improvement, vectorization efficiency

**Expected Results**:
- Thread scaling: ~13× speedup on 16 cores (Amdahl: 5% serial)
- Memory: ~48 bytes/Gaussian, linear scaling
- Batch processing: 4-8× improvement with AVX2 (when implemented)

### 6. Resolution Independence Benchmarks

**Goal**: Prove resolution-independent rendering

**Test**:
- Encode at 256×256 with 1000 Gaussians
- Decode at: 128×128, 256×256, 512×512, 1024×1024, 2048×2048, 4096×4096

**Metrics**:
- PSNR at each resolution (vs. ideal upscale of original)
- Rendering time scaling (should be O(pixels))
- Visual quality (subjective, artifacts?)
- Sharpness/blur metrics

**Expected**:
- Smooth scaling (no pixelation)
- Quality degrades gracefully at very high upscale factors
- Render time linear in output pixels

### 7. Stress Tests & Edge Cases

**Goal**: Find breaking points and failure modes

**Tests**:
- **Maximum Gaussian Count**: Increase until OOM or timeout
  - Expected limit: ~100K Gaussians on 16GB RAM

- **Maximum Resolution**: Increase until impractical
  - Expected limit: 4K encoding, 8K rendering

- **Degenerate Gaussians**: Tiny scales, huge scales, zero opacity
  - Test: Numerical stability, no crashes

- **Extreme Color Values**: HDR, out-of-gamut, negative
  - Test: Clamping behavior, no artifacts

- **Convergence Failure**: Images that don't optimize well
  - Identify: Patterns that confuse gradient descent

**Expected Failures**:
- Random noise: PSNR < 20 dB (Gaussians can't represent noise)
- Very high frequency: Aliasing artifacts
- Extreme aspect ratios: Gaussian footprint issues

### 8. Memory & Cache Benchmarks

**Goal**: Understand memory hierarchy performance

**Tests**:
- **Cache Hit Rate**: Profile L1, L2, L3 cache misses
- **Memory Bandwidth**: Measure bytes/second during rendering
- **Data Layout**: Compare AoS vs. SoA performance
- **Tiling**: Measure cache efficiency with different tile sizes (64, 128, 256, 512)

**Tools**:
- `perf` (Linux): cache-misses, cache-references
- `valgrind --tool=cachegrind`
- Custom instrumentation

**Expected**:
- L3 cache holds ~666K Gaussians (48 bytes × 666K = 32 MB)
- SoA layout: Better cache utilization for batch ops
- Tiling: Dramatic improvement for large images (10× speedup potential)

### 9. Convergence & Optimization Benchmarks

**Goal**: Understand optimizer behavior

**Tests**:
- **Learning Rate Sweep**: 0.001, 0.003, 0.01, 0.03, 0.1
  - Measure: Convergence speed, final quality, stability

- **Loss Function Weights**: L2 vs. SSIM (0:1, 0.2:0.8, 0.5:0.5, 0.8:0.2, 1:0)
  - Measure: Perceptual quality, convergence

- **Iteration Budget**: 100, 200, 500, 1000, 2000, 5000
  - Measure: Quality vs. time tradeoff

**Expected**:
- Optimal LR: ~0.01 for position, ~0.005 for scale
- Loss weights: 0.8×L2 + 0.2×SSIM performs well
- Diminishing returns after 2000-3000 iterations

### 10. Comparison Benchmarks

**Goal**: Compare with traditional codecs

**Comparisons** (when test images available):
- **vs. JPEG**: Quality-matched comparison (same PSNR)
  - Measure: File size, encoding time, decoding time

- **vs. PNG**: Lossless comparison
  - Measure: File size (expect LGI to be larger for lossless)

- **vs. WebP**: Modern codec comparison
  - Measure: Quality, file size, speed

- **vs. AVIF**: State-of-art comparison
  - Measure: All metrics

**Expected**:
- LGI decode: 10-100× faster than JPEG/PNG/WebP
- LGI encode: 10-100× slower than traditional (GPU will fix)
- LGI file size: 0.5-2× JPEG (depends on compression implementation)

---

## 📊 Benchmark Infrastructure

### Tools Created

1. **Test Image Generator** (`test_images.rs`)
   - 10 synthetic patterns
   - Configurable size, seed
   - Covers easy → hard spectrum

2. **Quality Metrics** (`metrics.rs`)
   - PSNR (overall + per-channel)
   - SSIM (structural similarity)
   - MSE, MAE
   - Extensible for LPIPS, MS-SSIM

3. **Benchmark Runner** (`benchmark_runner.rs`)
   - Automated test execution
   - CSV/JSON export
   - Configurable test matrix
   - Statistical summaries

4. **Stress Test** (`stress_test.rs`)
   - Maximum Gaussian count
   - Maximum resolution
   - Difficult patterns
   - Resolution independence

5. **Criterion Benches** (4 suites)
   - `encoding_suite`: Encoding performance
   - `rendering_suite`: Rendering performance
   - `quality_metrics`: Metric computation speed
   - `scaling_analysis`: Parallel vs. sequential

### Output Formats

**CSV**: Structured data for analysis
```csv
image_size,num_gaussians,quality_preset,encode_time_ms,decode_time_ms,psnr,ssim,...
256,500,fast,45230,145,32.45,0.9234,...
```

**JSON**: Complete structured results
```json
{
  "image_size": 256,
  "num_gaussians": 500,
  "psnr": 32.45,
  "encode_time_ms": 45230,
  ...
}
```

**Terminal**: Human-readable summaries
```
Test 1/30: size=256, gaussians=500, quality=fast
  ✓ Encode: 45.23s, Decode: 0.145s, PSNR: 32.45 dB
```

---

## 🔬 Expected Findings & Issues

### Known Issues to Watch For

1. **Low PSNR on First Demo** (5.73 dB observed)
   - **Cause**: Optimizer not fully implemented (scale/rotation not optimized)
   - **Fix**: Implement full backpropagation
   - **Target**: PSNR > 30 dB with full optimizer

2. **Slow Encoding** (17s for 256×256, 200 Gaussians, 170 iterations)
   - **Cause**: CPU-only rendering, simple gradients
   - **Fix**: GPU-accelerated differentiable rendering
   - **Target**: < 5 seconds for same scenario

3. **SSIM Computation Slow**
   - **Cause**: Window-based O(n²) computation
   - **Fix**: Use fused-ssim library or approximate
   - **Impact**: Slows training loop significantly

4. **Memory Usage** (not yet measured)
   - **Expected**: 48 bytes × N Gaussians + image buffers
   - **Watch**: Large Gaussian counts (10K+) may exhaust RAM
   - **Fix**: Streaming processing, tiling

### Potential Bottlenecks

**Encoding Bottlenecks** (in order of severity):
1. **Rendering (Forward Pass)**: 60-70% of time
   - Solution: GPU acceleration, tiling, LOD

2. **Gradient Computation**: 20-30% of time
   - Solution: Analytical gradients, GPU backprop

3. **SSIM Loss**: 10-20% of time (if enabled)
   - Solution: Fused-SSIM CUDA kernels or disable

4. **Memory Allocation**: 5-10% of time
   - Solution: Pre-allocate buffers, object pools

**Decoding Bottlenecks**:
1. **Gaussian Evaluation**: 50-60% of time
   - Solution: SIMD (AVX2/NEON), GPU compute shader

2. **Alpha Compositing**: 20-30% of time
   - Solution: SIMD, GPU

3. **Memory Bandwidth**: 10-20% of time
   - Solution: Cache blocking, prefetching

4. **Bounding Box Computation**: 5-10% of time
   - Solution: Precompute and cache

---

## 📈 Performance Targets & Validation

### Current Performance (Phase 1, CPU-only)

| Metric | Current (Measured) | Target (Phase 1) | Status |
|--------|-------------------|------------------|--------|
| **Math primitives** | 8.5 ns (Gaussian eval) | < 20 ns | ✅ **2.4× better** |
| **Rendering** | 14 FPS (256×256, 200 G) | 10 FPS | ✅ **Met** |
| **Encoding** | 17s (256×256, 200 G) | < 60s | ✅ **3.5× better** |
| **PSNR** | 5.73 dB* | > 30 dB | ❌ **Needs work** |

*Low PSNR due to partial optimizer implementation (expected)

### Phase 2 Targets (+ Full Optimizer)

| Metric | Target | How to Achieve |
|--------|--------|----------------|
| **PSNR** | > 35 dB | Full backprop (scale, rotation optimization) |
| **Encoding** | < 10s (256×256, 1000 G) | Better optimizer, warm start |
| **Rendering** | 30+ FPS (256×256) | Multi-threading, culling |

### Phase 3 Targets (+ SIMD)

| Metric | Target | How to Achieve |
|--------|--------|----------------|
| **Rendering** | 200+ FPS (1080p, 1000 G) | AVX2 vectorization |
| **Encoding** | < 5s (256×256) | SIMD rendering in forward pass |

### Phase 4 Targets (+ GPU)

| Metric | Target (Spec) | How to Achieve |
|--------|---------------|----------------|
| **Rendering** | 1000+ FPS (1080p, 1M G) | wgpu compute shader |
| **Encoding** | < 1s (256×256) | GPU diff. rendering + optimization |

---

## 🧪 Test Suite Organization

### Automated Tests (cargo test)

**Unit Tests** (31 total, 100% passing):
- Math library (24 tests)
- Core functionality (5 tests)
- Benchmarks library (1 test)
- Full pipeline integration test (pending)

**Integration Tests** (to create):
```
tests/
├── encode_decode_roundtrip.rs    # Full pipeline test
├── quality_validation.rs          # PSNR thresholds
├── performance_regression.rs      # Speed must not decrease
└── format_compliance.rs           # When format implemented
```

### Benchmarks (cargo bench)

**Criterion Benches** (4 suites):
```
benches/
├── encoding_suite.rs       # Encoding performance
├── rendering_suite.rs      # Rendering performance
├── quality_metrics.rs      # Metric computation speed
└── scaling_analysis.rs     # Parallelism, memory
```

### Examples (cargo run --example)

**Demo Programs**:
```
examples/
├── run_comprehensive_benchmark.rs  # Full test matrix
├── stress_test.rs                  # Edge cases, limits
└── compare_codecs.rs               # vs. JPEG/PNG (future)
```

---

## 📋 Benchmark Execution Plan

### Quick Validation (5-10 minutes)

```bash
# Test all 10 patterns at single configuration
cargo run --release --bin lgi-cli -- test -o test.png -s 256

for pattern in solid linear radial checkerboard circles sweep noise natural geometric text; do
  # Would need pattern flag, or use benchmark runner
  echo "Testing $pattern..."
done
```

### Comprehensive Suite (2-4 hours)

```bash
# Run full benchmark matrix
cargo run --release --example run_comprehensive_benchmark

# Generates:
# - benchmark_results/*.csv (per-pattern results)
# - benchmark_results/*.json (full data)
# - benchmark_results/*.png (test images)
```

**Test Matrix**:
- 10 patterns × 3 sizes × 5 Gaussian counts × 1 preset × 2 runs = **300 tests**
- Estimated time: ~3 hours (600s per test average)

### Stress Test (30-60 minutes)

```bash
# Push limits
cargo run --release --example stress_test

# Tests:
# - 1000, 2000, 5000, 10000 Gaussians
# - 256, 512, 1024 resolutions
# - Difficult patterns
# - Resolution independence (4 scales)
```

### Continuous Benchmarking (CI/CD)

```yaml
# GitHub Actions workflow
name: Benchmark
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmarks
        run: cargo bench --all
      - name: Store results
        uses: benchmark-action/github-action-benchmark@v1
```

---

## 📊 Analysis & Reporting

### Metrics to Track Over Time

1. **Performance Metrics**:
   - Encoding FPS (Gaussians/second)
   - Decoding FPS (frames/second)
   - Throughput (megapixels/second)

2. **Quality Metrics**:
   - PSNR (mean, min, max across test set)
   - SSIM (mean, std dev)
   - Visual inspection (subjective)

3. **Resource Metrics**:
   - Memory usage (peak, average)
   - CPU utilization (single-thread, multi-thread)
   - Cache hit rates (L1, L2, L3)

### Visualization & Reporting

**Plots to Generate**:
```
1. PSNR vs. Gaussian Count (per pattern)
2. Encoding Time vs. Gaussian Count (log-log)
3. Rendering FPS vs. Resolution (log-log)
4. Thread Scaling Efficiency (speedup vs. threads)
5. Rate-Distortion Curve (bits/pixel vs. PSNR)
6. Pattern Difficulty Ranking (bar chart)
```

**Reports**:
```
benchmark_report.md:
- Summary statistics
- Comparison with roadmap targets
- Identified bottlenecks
- Recommendations for optimization
```

---

## 🎯 Success Criteria

### Phase 1 Validation (Current)

- [x] Math library: < 10 ns operations ✅
- [ ] Encoding: PSNR > 30 dB ⚠️ (needs full optimizer)
- [x] Decoding: > 10 FPS (256×256) ✅
- [x] 10 test patterns working ✅
- [x] Benchmark infrastructure complete ✅

### Phase 2 Validation (Next)

- [ ] PSNR > 35 dB (all patterns except noise)
- [ ] Encoding < 30s (256×256, 1000 G)
- [ ] Compression ratio < 50% PNG
- [ ] Passes Kodak dataset quality threshold

### Long-Term Validation (Phases 3-4)

- [ ] Decoding > 1000 FPS (1080p, GPU)
- [ ] Encoding < 5s (1080p, GPU)
- [ ] Quality competitive with AVIF
- [ ] Production deployment ready

---

## 🔍 Analysis Framework

### Statistical Analysis

**For Each Benchmark Result**:
```rust
// Compute statistics over multiple runs
mean = Σx / n
std_dev = sqrt(Σ(x - mean)² / n)
min, max, median, p95, p99
```

**Regression Detection**:
```
if current_perf < baseline_perf * 0.95:
    alert("Performance regression detected!")
```

**Scaling Law Fitting**:
```
encoding_time = a * gaussians * iterations + b
rendering_time = c * gaussians * pixels + d

# Fit curves, extrapolate to larger sizes
```

### Bottleneck Identification

**Profiling Commands**:
```bash
# CPU profiling
cargo flamegraph --example stress_test

# Memory profiling
valgrind --tool=massif target/release/examples/stress_test

# Cache profiling
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
  target/release/examples/stress_test
```

**Analysis**:
- Flamegraph: Identify hot functions (>10% time)
- Massif: Memory allocation patterns
- Perf: Cache efficiency, branch prediction

---

## 🚀 Next Steps

### Immediate (This Session)

1. ✅ Create benchmark infrastructure
2. ✅ Implement test image generator (10 patterns)
3. ✅ Implement quality metrics (PSNR, SSIM)
4. ✅ Create stress test
5. ⏳ Run stress test and analyze results
6. 📋 Document findings

### Short-Term (Next Session)

7. Fix optimizer (full backprop) to achieve PSNR > 30 dB
8. Run comprehensive benchmark suite
9. Profile with flamegraph to find hotspots
10. Implement quick-win optimizations

### Medium-Term (Phase 2)

11. Add file format I/O and compression
12. Benchmark compression ratios
13. Compare with JPEG/PNG on real images
14. Publish benchmark results

---

**Document Version**: 1.0
**Status**: Infrastructure Complete, Tests Running
**Next**: Analyze stress test results, identify issues

---

**End of Benchmark Plan**
