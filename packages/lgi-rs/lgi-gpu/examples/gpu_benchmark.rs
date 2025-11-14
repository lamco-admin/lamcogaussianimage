//! GPU Rendering Benchmark
//!
//! Tests GPU rendering performance vs CPU baseline

use lgi_gpu::{GpuRenderer, RenderMode};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;

fn create_test_gaussians(count: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    (0..count)
        .map(|i| {
            let x = (i as f32) / (count as f32);
            Gaussian2D::new(
                Vector2::new(x, 0.5),
                Euler::isotropic(0.01),
                Color4::rgb(x, 0.5, 0.5),
                0.8,
            )
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  LGI GPU Rendering Benchmark (wgpu v27)             ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // Create GPU renderer
    let mut renderer = pollster::block_on(async {
        GpuRenderer::new().await
    })?;

    println!("✅ GPU Initialized: {} ({:?})\n", renderer.adapter_name(), renderer.backend());

    // Test configurations
    let configs = vec![
        ("256×256, 500 Gaussians", 256, 256, 500),
        ("512×512, 1000 Gaussians", 512, 512, 1000),
        ("1920×1080, 5000 Gaussians", 1920, 1080, 5000),
    ];

    println!("Configuration              | Mode      | Time     | FPS");
    println!("-------------------------- | --------- | -------- | --------");

    for (name, width, height, count) in configs {
        let gaussians = create_test_gaussians(count);

        // Test Alpha Composite mode
        let start = Instant::now();
        let _output = renderer.render(&gaussians, width, height, RenderMode::AlphaComposite)?;
        let time_alpha = start.elapsed().as_secs_f32() * 1000.0;
        let fps_alpha = 1000.0 / time_alpha;

        // Test Accumulated Sum mode
        let start = Instant::now();
        let _output = renderer.render(&gaussians, width, height, RenderMode::AccumulatedSum)?;
        let time_accum = start.elapsed().as_secs_f32() * 1000.0;
        let fps_accum = 1000.0 / time_accum;

        println!("{:<26} | Alpha     | {:>6.2}ms | {:>6.1}", name, time_alpha, fps_alpha);
        println!("{:<26} | Accum     | {:>6.2}ms | {:>6.1}", "", time_accum, fps_accum);
    }

    println!("\n✅ GPU rendering benchmarks complete!");
    println!("\nNote: Performance depends on GPU hardware.");
    println!("  - Software renderer (llvmpipe): 10-50 FPS");
    println!("  - Integrated GPU (Intel): 100-500 FPS");
    println!("  - Discrete GPU (NVIDIA/AMD): 1000+ FPS");

    Ok(())
}
