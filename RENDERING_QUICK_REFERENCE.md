# RENDERING ENGINES - QUICK REFERENCE GUIDE

## Core Files at a Glance

### CPU Rendering (Rust)
| Implementation | File | Lines | Algorithm | Speed (512px) |
|---|---|---|---|---|
| **Primary (Alpha Composite)** | `/lgi-rs/lgi-core/src/renderer.rs` | 327 | Porter-Duff over | 5-50 ms |
| **EWA Splatting V1** | `/lgi-rs/lgi-core/src/ewa_splatting.rs` | 101 | Zwicker 2001 | 20-100 ms |
| **EWA Splatting V2** | `/lgi-rs/lgi-core/src/ewa_splatting_v2.rs` | 227 | Robust EWA | 40-150 ms |
| **Compositing (Math)** | `/lgi-rs/lgi-math/src/compositing.rs` | 266 | Alpha blend ops | N/A (utility) |
| **Renderer V2 (Encoder)** | `/lgi-rs/lgi-encoder-v2/src/renderer_v2.rs` | 114 | Weighted sum | 40-60 ms |
| **Renderer V3 (Textured)** | `/lgi-rs/lgi-encoder-v2/src/renderer_v3_textured.rs` | 205 | Texture support | 50-70 ms |

### GPU Rendering (WebGPU/WGPU)
| Component | File | Size | Purpose |
|---|---|---|---|
| **GPU Renderer** | `/lgi-rs/lgi-gpu/src/renderer.rs` | 279 lines | Vulkan/Metal/DX12 rendering |
| **Render Shader (WGSL)** | `/lgi-rs/lgi-gpu/src/shaders/gaussian_render.wgsl` | 138 lines | 2D Gaussian evaluation |
| **Gradient Shader (WGSL)** | `/lgi-rs/lgi-gpu/src/shaders/gradient_compute.wgsl` | 235 lines | Backprop for optimization |
| **Pipeline** | `/lgi-rs/lgi-gpu/src/pipeline.rs` | 112 lines | Shader compilation |
| **Buffers** | `/lgi-rs/lgi-gpu/src/buffer.rs` | 232 lines | Memory management |
| **Manager** | `/lgi-rs/lgi-gpu/src/manager.rs` | 104 lines | Global GPU instance |
| **Backend** | `/lgi-rs/lgi-gpu/src/backend.rs` | 103 lines | Adapter selection |

**Performance**: 1000+ FPS @ 1080p with 10K Gaussians

### Display & Visualization
| Component | File | Type | Purpose |
|---|---|---|---|
| **Desktop Viewer** | `/lgi-rs/lgi-viewer/src/main.rs` | Slint UI | Professional LGI viewer |
| **Web Viewer** | `/lgi-rs/lgi-wasm/src/lib.rs` | WASM | Browser-based renderer |
| **Debug Logger** | `/lgi-rs/lgi-encoder-v2/src/debug_logger.rs` | Debug util | Optimization visualization |
| **Visual Examples** | `/lgi-rs/lgi-encoder-v2/examples/` | Examples | Debug/test renderers |

### Legacy CUDA (3D Splatting)
| Component | File | Size | Purpose |
|---|---|---|---|
| **Forward Kernel** | `/lgi-legacy/image-gs/gsplat/cuda/csrc/forward.cu` | 20 KB | 3D projection |
| **Backward Kernel** | `/lgi-legacy/image-gs/gsplat/cuda/csrc/backward.cu` | 21 KB | Gradient computation |
| **Bindings** | `/lgi-legacy/image-gs/gsplat/cuda/csrc/bindings.cu` | 27 KB | PyTorch interface |
| **Python API** | `/lgi-legacy/image-gs/gsplat/gsplat/` | ~15 KB | High-level functions |

### Legacy CPU (PyTorch)
| File | Lines | Purpose |
|---|---|---|
| `/lgi-legacy/image-gs-cpu/gaussian_2d_cpu.py` | 325 | Pure PyTorch 2D renderer |

---

## Key Methods by Use Case

### Fast GPU Rendering
```rust
// Initialize GPU
let mut gpu = GpuRenderer::new().await?;

// Render
let image = gpu.render(&gaussians, width, height, RenderMode::AccumulatedSum)?;

// Check performance
println!("FPS: {}", gpu.fps());
```

### CPU Rendering with Alpha Compositing
```rust
let renderer = Renderer::new()
    .with_alpha_composite();
let image = renderer.render(&gaussians, width, height)?;
```

### Alias-Free Rendering (EWA)
```rust
let renderer = EWARendererV2::default();
let image = renderer.render(&gaussians, width, height, zoom);
```

### Professional Textured Rendering
```rust
let image = RendererV3::render(&textured_gaussians, width, height);

// With texture toggle
let image = RendererV3::render_adaptive(&textured_gaussians, width, height, use_textures);
```

### Debug Visualization
```rust
let mut debugger = DebugLogger::new(DebugConfig::default())?;
debugger.log_iteration(
    iteration, pass, &gaussians, 
    &target, loss, psnr, elapsed_ms
)?;
// Outputs: iteration images, error maps, metrics CSV
```

---

## Performance Characteristics

### CPU (sequential, 512×512, 10K Gaussians)
- Alpha Composite: 30-50 ms
- Accumulated Sum: 25-40 ms
- EWA V1: 40-80 ms
- EWA V2: 80-150 ms

### CPU (parallel, 8 cores)
- Alpha Composite: 5-10 ms (5-10× speedup)
- Accumulated Sum: 4-8 ms

### GPU (1080p, 10K Gaussians)
- Vulkan (NVIDIA): 1000+ FPS (<1 ms)
- Metal (Apple): 800+ FPS (1-1.25 ms)
- DX12 (Intel): 500+ FPS (2 ms)

### Gradients (GPU)
- Time: <1 ms for 10K Gaussians
- Speed: 100-1000× faster than CPU

---

## Configuration Options

### RenderConfig (renderer.rs)
```rust
RenderConfig {
    background: Color4::black(),      // Background color
    alpha_mode: AlphaMode::Straight,  // Straight or Premultiplied
    render_mode: RenderMode::AlphaComposite,  // or AccumulatedSum
    cutoff_threshold: 1e-5,           // Gaussian weight cutoff
    n_sigma: 3.5,                     // Bounding box radius (sigma)
    termination_threshold: 0.999,     // Early termination (99.9% opacity)
    parallel: true,                   // Use multi-threading
}
```

### EWARendererV2 (ewa_splatting_v2.rs)
```rust
EWARendererV2 {
    filter_bandwidth: 1.0,    // Pixel footprint
    cutoff_radius: 3.5,       // In sigma units
}
```

### DebugConfig (debug_logger.rs)
```rust
DebugConfig {
    output_dir: PathBuf::from("debug_output"),
    save_every_n_iters: 10,
    save_rendered: true,
    save_error_maps: true,
    save_gaussian_viz: true,
    save_comparison: true,
    save_metrics_csv: true,
}
```

---

## Algorithm Selection Guide

| Use Case | Algorithm | File | Why |
|---|---|---|---|
| Fast preview | Accumulated Sum | `renderer_v2.rs` | Simplest, fastest |
| Interactive viewing | Alpha Composite | `renderer.rs` | Best balance |
| High-quality zoom | EWA V2 | `ewa_splatting_v2.rs` | Alias-free |
| Professional/texture | V3 Textured | `renderer_v3_textured.rs` | Detail preservation |
| GPU acceleration | GpuRenderer | `lgi-gpu/renderer.rs` | 1000+ FPS |
| Gradient computation | GPU Gradient | `gradient_compute.wgsl` | 100-1000× speedup |

---

## Testing & Debugging

### Rendering Suite Benchmark
```bash
cargo test --release -- --nocapture rendering_suite
```

### GPU Minimal Debug
```bash
cargo run --example gpu_minimal_debug --release
```

### Visual Debug Demo
```bash
cargo run --example visual_debug_demo --release
```

### Viewer (Desktop)
```bash
cargo run --release -p lgi-viewer
```

---

## File Statistics

- **Total Rendering Code**: ~2000 lines
- **GPU Shaders**: 373 lines (WGSL)
- **Legacy CUDA**: ~71 KB kernels
- **CPU Implementations**: 4 distinct algorithms
- **GPU Implementations**: 2 (render + gradient)
- **Viewers/Display**: 2 (desktop + web)

---

## Key Technical Details

### Bounding Box Calculation
Standard: 3.5σ radius
- Covers 99.99% of Gaussian mass
- Early culling before per-pixel evaluation
- Configurable via `n_sigma` parameter

### Rotation Implementation
- Single angle θ (radians)
- Rotation matrix transformation:
  ```
  dx_rot = dx·cos(θ) + dy·sin(θ)
  dy_rot = -dx·sin(θ) + dy·cos(θ)
  ```
- Mahalanobis distance: (dx_rot/σx)² + (dy_rot/σy)²

### Early Termination
- Threshold: 0.999 (99.9% opacity)
- ~20-30% performance improvement
- Slight edge artifacts (imperceptible)

### Zero-Initialization (GPU)
**CRITICAL**: Output buffer must be explicitly zeroed (gaussian_render.wgsl:80)
- Prevents undefined memory contents
- Ensures deterministic results

---

## Where to Look for Specific Features

| Feature | File(s) |
|---|---|
| **Backward compatibility modes** | `renderer.rs`, `ewa_splatting.rs` |
| **Per-Gaussian culling** | `lod_system.rs` |
| **Texture sampling** | `textured_gaussian.rs`, `texture_map.rs` |
| **Multi-scale representation** | `pyramid.rs` |
| **GPU kernel info** | `gradient_compute.wgsl` |
| **Performance metrics** | `profiler.rs`, `debug_logger.rs` |
| **UI integration** | `lgi-viewer/main.rs` |
| **Error visualization** | `debug_logger.rs` examples |

---

## Performance Tips

1. **Use GPU rendering** for interactive applications (1000+ FPS)
2. **Enable parallelism** for CPU (5-10× speedup with 8 cores)
3. **Use AccumulatedSum** mode for faster CPU rendering (~20% speedup)
4. **Enable early termination** (default, ~25% speedup)
5. **Cull Gaussians** below visible threshold (LOD system)
6. **Batch rendering** with multiple images

---

## Known Issues & Limitations

1. CPU rendering slow with 50K+ Gaussians
2. WASM viewer is stub implementation
3. Texture support only in V3 renderer
4. Legacy CUDA limited to 3D scenes

## Future Optimization Opportunities

1. SIMD vectorization (AVX-512)
2. Tile-based rendering (cache optimization)
3. GPU-based sorting/culling
4. Adaptive resolution
5. Async GPU readback

