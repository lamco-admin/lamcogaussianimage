# LGI Codec - Complete API Reference

**Version**: 0.1.0
**Date**: October 2, 2025
**Status**: Authoritative

---

## Quick Links

- [Core API](#core-api-lgi-core) - Rendering, initialization
- [Encoder API](#encoder-api-lgi-encoder) - Optimization, training
- [Format API](#format-api-lgi-format) - File I/O, compression
- [GPU API](#gpu-api-lgi-gpu) - GPU rendering
- [Pyramid API](#pyramid-api-lgi-pyramid) - Multi-level zoom
- [CLI Reference](#cli-reference) - Command-line tools

---

## Core API (lgi-core)

### Rendering

```rust
use lgi_core::{Renderer, RenderConfig, RenderMode};

// Basic rendering
let renderer = Renderer::new();
let output = renderer.render(&gaussians, 1920, 1080)?;

// Custom configuration
let config = RenderConfig::default()
    .with_accumulated_sum();  // Use GaussianImage mode
let renderer = Renderer::with_config(config);
let output = renderer.render(&gaussians, 1920, 1080)?;
```

**Key Types**:
- `Renderer` - Main rendering engine
- `RenderConfig` - Configuration (background, mode, thresholds)
- `RenderMode` - AlphaComposite | AccumulatedSum

---

### Initialization

```rust
use lgi_core::{Initializer, InitStrategy};

// Gradient-based initialization (recommended)
let initializer = Initializer::new(InitStrategy::Gradient)
    .with_scale(0.01);
let gaussians = initializer.initialize(&target_image, 1000)?;

// Adaptive count (auto-optimal)
use lgi_core::adaptive_gaussian_count;
let optimal_count = adaptive_gaussian_count(&image);
let gaussians = initializer.initialize(&image, optimal_count)?;
```

**Init Strategies**:
- `Random` - Random positions
- `Grid` - Regular grid
- `Gradient` - High-gradient regions (recommended)

---

## Encoder API (lgi-encoder)

### Basic Encoding

```rust
use lgi_encoder::{EncoderConfig, OptimizerV2};

// Quick encode with preset
let config = EncoderConfig::balanced();
let initializer = Initializer::new(config.init_strategy);
let mut gaussians = initializer.initialize(&target, 1000)?;

let optimizer = OptimizerV2::new(config);
let metrics = optimizer.optimize_with_metrics(&mut gaussians, &target)?;

// Check quality
println!("Final PSNR: {:.2} dB", metrics.final_psnr());
```

**Presets**:
- `fast()` - 500 iterations, quick
- `balanced()` - 2000 iterations (default)
- `high_quality()` - 5000 iterations
- `ultra()` - 10000 iterations

---

### Quantization-Aware Training

```rust
let mut config = EncoderConfig::balanced();
config.enable_qa_training = true;
config.qa_start_iteration = 1400;  // 70% of 2000

let optimizer = OptimizerV2::new(config);
let metrics = optimizer.optimize_with_metrics(&mut gaussians, &target)?;

// Gaussians now robust to quantization (<1 dB loss)
```

---

## Format API (lgi-format)

### Saving Files

```rust
use lgi_format::{LgiFile, LgiWriter, CompressionConfig};

// Balanced compression (recommended)
let file = LgiFile::with_compression(
    gaussians,
    1920,
    1080,
    CompressionConfig::balanced(),  // LGIQ-B + VQ + zstd
);

LgiWriter::write_file(&file, "output.lgi")?;
```

**Compression Presets**:
```rust
CompressionConfig::balanced()     // 7.5× compression, 30 dB
CompressionConfig::small()        // Max compression
CompressionConfig::high_quality() // Best quality
CompressionConfig::lossless()     // Bit-exact, 10.7× compression
```

---

### Loading Files

```rust
use lgi_format::LgiReader;

// Full read
let file = LgiReader::read_file("input.lgi")?;
let gaussians = file.gaussians();
let (width, height) = file.dimensions();

// Quick inspect (header only)
let header = LgiReader::read_header_file("input.lgi")?;
println!("{}×{}, {} Gaussians",
    header.width, header.height, header.gaussian_count);
```

---

### Custom Compression

```rust
use lgi_format::{CompressionConfig, QuantizationProfile};

// Custom configuration
let config = CompressionConfig::custom(
    QuantizationProfile::LGIQ_B,  // Balanced quantization
    true,    // Enable VQ
    256,     // VQ codebook size
    true,    // Enable zstd
    12,      // zstd level (0-22)
);

let file = LgiFile::with_compression(gaussians, width, height, config);
```

---

## GPU API (lgi-gpu)

### GPU Rendering

```rust
use lgi_gpu::{GpuRenderer, RenderMode};

// Auto-detect best GPU
let mut renderer = GpuRenderer::new().await?;

println!("Using: {} ({:?})",
    renderer.adapter_name(),
    renderer.backend()
);

// Render on GPU
let output = renderer.render(
    &gaussians,
    1920,
    1080,
    RenderMode::AlphaComposite
)?;

println!("Rendered in {:.2}ms ({:.1} FPS)",
    renderer.last_render_time_ms(),
    renderer.fps()
);
```

**Backends** (auto-detected):
- Vulkan (Linux, Windows)
- DX12 (Windows)
- Metal (macOS, iOS)
- WebGPU (browsers)

---

## Pyramid API (lgi-pyramid)

### Building Pyramids

```rust
use lgi_pyramid::{PyramidBuilder, Viewport};

// Build 4-level pyramid
let pyramid = PyramidBuilder::new()
    .num_levels(4)
    .gaussian_density(0.015)
    .build(&image)?;

pyramid.print_stats();
```

### Zoom Rendering

```rust
// Render at 4× zoom (O(1) complexity!)
let viewport = Viewport::new(0.25, 0.25, 0.5, 0.5);
let output = pyramid.render_at_zoom(
    4.0,           // Zoom factor
    viewport,
    1920,
    1080,
)?;

// Performance: Same speed at 1× or 100× zoom!
```

---

## CLI Reference

### encode - Encode Images

```bash
lgi-cli-v2 encode \
  -i input.png \
  -o output.png \
  -n 1000 \
  -q balanced \
  --qa-training \
  --save-lgi \
  --metrics-csv metrics.csv
```

**Arguments**:
- `-i, --input`: Input PNG file
- `-o, --output`: Output PNG file
- `-n, --gaussians`: Number of Gaussians (default: 1000)
- `-q, --quality`: Preset (fast/balanced/high/ultra)
- `--qa-training`: Enable QA training for better compression
- `--save-lgi`: Save as .lgi file
- `--adaptive`: Enable adaptive features (pruning/splitting)
- `--metrics-csv`: Export metrics to CSV
- `--metrics-json`: Export metrics to JSON

---

### decode - Decode .lgi Files

```bash
lgi-cli-v2 decode \
  -i compressed.lgi \
  -o output.png
```

**Arguments**:
- `-i, --input`: Input .lgi file
- `-o, --output`: Output PNG file

---

### info - Inspect .lgi Files

```bash
lgi-cli-v2 info -i file.lgi
```

Shows:
- Format version
- Dimensions
- Gaussian count
- Compression mode
- Quality metrics
- File size

---

## Error Handling

### Common Errors

```rust
use lgi_core::LgiError;

match result {
    Err(LgiError::InvalidDimensions(msg)) => {
        // Handle invalid dimensions
    }
    Err(LgiError::RenderingFailed(msg)) => {
        // Handle rendering failure
    }
    Ok(output) => {
        // Success
    }
}
```

**Error Types**:
- `LgiError` - Core errors
- `GpuError` - GPU rendering errors
- `LgiFormatError` - File format errors
- `PyramidError` - Pyramid errors

---

## Performance Tips

### 1. Choose Right Gaussian Count

```rust
// Auto-optimal (recommended)
let count = adaptive_gaussian_count(&image);

// Manual rules:
// - Solid colors: 50-200
// - Gradients: 200-500
// - Photos (simple): 1000-2000
// - Photos (complex): 5000-10000
```

### 2. Enable QA Training for Compression

```rust
config.enable_qa_training = true;  // <1 dB quality loss when compressed
```

### 3. Use GPU for Large Images

```rust
// CPU: Good for < 1000 Gaussians
// GPU: Essential for > 5000 Gaussians or > 1080p
```

### 4. Use Pyramid for Zoom Apps

```rust
// Without pyramid: O(N) Gaussians at all zoom levels
// With pyramid: O(1) Gaussians at each zoom level
```

---

## Examples

### Complete Workflow

```rust
use lgi_core::{ImageBuffer, Initializer, adaptive_gaussian_count};
use lgi_encoder::{EncoderConfig, OptimizerV2};
use lgi_format::{LgiFile, LgiWriter, CompressionConfig};

// 1. Load image
let target = ImageBuffer::load("photo.png")?;

// 2. Determine optimal Gaussian count
let count = adaptive_gaussian_count(&target);

// 3. Initialize
let config = EncoderConfig::balanced();
config.enable_qa_training = true;

let initializer = Initializer::new(config.init_strategy);
let mut gaussians = initializer.initialize(&target, count)?;

// 4. Optimize
let optimizer = OptimizerV2::new(config);
let metrics = optimizer.optimize_with_metrics(&mut gaussians, &target)?;

println!("Optimized to {:.2} dB PSNR", metrics.final_psnr());

// 5. Compress and save
let file = LgiFile::with_compression(
    gaussians,
    target.width,
    target.height,
    CompressionConfig::balanced(),
);

LgiWriter::write_file(&file, "output.lgi")?;
println!("Saved with {:.1}× compression", file.compression_ratio());

// 6. Load and render
let loaded = LgiReader::read_file("output.lgi")?;
let reconstructed = loaded.gaussians();

let renderer = Renderer::new();
let output = renderer.render(&reconstructed, target.width, target.height)?;
output.save("decoded.png")?;
```

---

**For detailed API documentation, run**: `cargo doc --open`

**This API is stable for v1.0 release.**
