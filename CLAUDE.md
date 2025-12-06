# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LGI (Lamco Gaussian Image) Codec - A cross-platform GPU-accelerated image codec based on 2D Gaussian splatting. Achieves +8 dB quality improvement over baseline with 1,176 FPS GPU decoding.

## Build Commands

```bash
# Build all packages
cd packages/lgi-rs
cargo build --release --all

# Build specific package
cargo build --release -p lgi-encoder-v2

# Run tests
cargo test --workspace

# Linting (required before commits)
cargo clippy --workspace -- -D warnings
cargo fmt --all --check

# Build documentation
cargo doc --no-deps
```

## Encoding/Decoding

```bash
# Encode with Adam optimizer (recommended)
cargo run --release --bin lgi-cli -- encode -i input.png -o output.lgi --method adam --iterations 100

# Target specific quality
cargo run --release --bin lgi-cli -- encode -i input.png -o output.lgi --target-psnr 30.0

# GPU-accelerated encoding
cargo run --release --bin lgi-cli -- encode -i input.png -o output.lgi --method gpu --iterations 50

# Decode
cargo run --release --bin lgi-cli -- decode -i output.lgi -o output.png
```

## Running Experiments (lgi-encoder-v2)

```bash
# Run optimizer experiments
cargo run --release --example test_multi_kodak -p lgi-encoder-v2
cargo run --release --example test_placement_strategies -p lgi-encoder-v2
cargo run --release --example comprehensive_optimizer_comparison -p lgi-encoder-v2

# Results saved to: lgi-encoder-v2/test-results/YYYY-MM-DD/
```

## Architecture

```
packages/lgi-rs/
├── lgi-math/        # Core math: Gaussian evaluation, compositing, color
├── lgi-core/        # Rendering pipeline, initialization, error metrics
├── lgi-encoder-v2/  # ACTIVE: Optimizers (Adam, V2, Hybrid, L-M)
├── lgi-format/      # File I/O, serialization, Zstandard compression
├── lgi-gpu/         # wgpu compute shaders (Vulkan/DX12/Metal/WebGPU)
├── lgi-cli/         # Command-line interface
├── lgi-viewer/      # GUI viewer with live encoding
├── lgi-ffi/         # C FFI for FFmpeg/ImageMagick integration
└── lgi-wasm/        # WebAssembly build
```

## Key Modules

**lgi-math**: Generic over f32/f64, 4 Gaussian parameterizations (Euler, Cholesky, LogRadius, InverseCovariance), SIMD-ready layouts.

**lgi-encoder-v2** (active development):
- `adam_optimizer.rs` - Recommended baseline (+2.25 dB avg)
- `optimizer_v2.rs` - Gradient descent with sign corrections
- `test_results.rs` - TestResult framework for experiments

## Testing Infrastructure Requirements

**All experiments MUST use TestResult framework with image saving:**
```rust
use lgi_encoder_v2::test_results::TestResult;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

// Create result at start of test
let mut result = TestResult::new("experiment_name", image_path);
result.optimizer = "Description".to_string();
result.n_gaussians = 256;
result.iterations = 200;
result.initial_psnr = init_psnr;
result.image_dimensions = (target.width, target.height);

// Record optimizer parameters using set_param()
result.set_param("lr_position", 0.0002);
result.set_param("lr_color", 0.02);
result.set_param("lr_scale", 0.005);
result.set_param("lr_opacity", 0.05);
result.set_param("beta1", 0.9);
result.set_param("beta2", 0.999);

// ... run optimization ...

// Complete and save with rendered image
result.final_psnr = final_psnr;
result.improvement_db = final_psnr - init_psnr;
result.elapsed_seconds = elapsed;
result.complete();
let _ = result.save_with_image(&rendered, RESULTS_DIR);
```

**Handling aborted tests (diverging, etc.):**
```rust
if psnr < init_psnr - 5.0 {
    result.abort("PSNR diverging - dropped 5dB below initial", iter, psnr);
    let _ = result.save_partial(RESULTS_DIR);  // Saves as *_aborted.json
    return;
}
```

**All tests MUST show verifiable progress:**
```rust
if iter % 10 == 0 {
    let direction = if psnr > last_psnr { "↑" } else { "↓" };
    println!("    [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
             iter, max_iterations, psnr, direction, psnr - last_psnr);
}
```

**Versioning for reproducibility:**
- TestResult auto-captures git hash + dirty status on creation
- Before important experiments, commit with `git add -A && git commit -m "Checkpoint before XYZ test"`
- Results JSON includes `code_version.git_hash` for reproduction

## Development Standards

**From `/docs/DEVELOPMENT_STANDARDS.md` - NON-NEGOTIABLE:**

1. **No shortcuts** - Complete implementations only, no stubs or TODOs
2. **Validate before claiming** - Run benchmarks, measure empirically
3. **Always return Result<T, E>** - Never panic/unwrap in library code
4. **Document public functions** - Algorithm explanation, parameters, performance characteristics

**Before every commit:**
```bash
cargo fmt --all --check
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

## Current Optimizer Status

| Optimizer | Improvement | Status |
|-----------|-------------|--------|
| Adam (per-param LR) | +2.25 dB avg | WORKING - Recommended |
| OptimizerV2 | +0.16 dB | WORKING - Slow |
| V2 + Gradient Norm | +0.77 dB avg | WORKING - Experimental |
| Hybrid (Adam→L-BFGS) | TBD | Needs investigation |
| L-M | TBD | Too slow |

## Research Priorities

1. **Placement strategies**: Jittered grid (+1.31 dB) beats uniform (+0.87 dB)
2. **Perceptual loss**: MS-SSIM, LPIPS implementation
3. **Optimizer improvements**: Per-parameter LR tuning critical

## Test Data

- Kodak dataset: `/test-data/kodak-dataset/` (24 benchmark images, 768×512)
- Use kodim03.png for quick single-image tests

## Key Documentation

- `/docs/DEVELOPMENT_STANDARDS.md` - Quality requirements
- `/docs/research/EXPERIMENTS.md` - What worked/failed
- `/packages/lgi-rs/lgi-encoder-v2/README.md` - Optimizer research
- `/packages/lgi-rs/lgi-encoder-v2/test-results/README.md` - TestResult framework documentation
- `/packages/lgi-rs/lgi-encoder-v2/test-results/COMPREHENSIVE_FINDINGS_2025-12-06.md` - Research findings
