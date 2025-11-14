//! Test GPU gradient computation
//!
//! This example tests GPU-accelerated gradient computation for encoding.
//! It should show "using GPU" for both render and gradients.

use lgi_core::{ImageBuffer, Initializer, InitStrategy};
use lgi_math::color::Color4;
use lgi_encoder::{EncoderConfig, OptimizerV2};
use std::time::Instant;

// Re-export pollster from wgpu crate (which lgi-gpu uses)
use wgpu::util::DeviceExt as _;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simple logging setup
    std::env::set_var("RUST_LOG", "info");

    println!("🧪 Testing GPU Gradient Computation");
    println!("==================================\n");

    // Load test image
    let img_path = "/tmp/test_small.png";
    println!("📁 Loading: {}", img_path);

    let img = image::open(img_path)?;
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    println!("📐 Image: {}×{}", width, height);

    // Convert to ImageBuffer
    let mut target = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_img.get_pixel(x, y);
            target.set_pixel(x, y, Color4::new(
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
                1.0,
            ));
        }
    }

    // Initialize GPU
    println!("\n🎮 Initializing GPU...");
    let start = Instant::now();
    futures::executor::block_on(async {
        lgi_gpu::GpuManager::global().initialize().await
    })?;
    println!("✅ GPU initialized with gradient computer ({:.2}ms)", start.elapsed().as_secs_f32() * 1000.0);

    // Initialize Gaussians
    let num_gaussians = 100; // Small number for quick test
    let initializer = Initializer::new(InitStrategy::KMeans);
    let mut gaussians = initializer.initialize(&target, num_gaussians)?;
    println!("🔵 Initialized {} Gaussians", gaussians.len());

    // Create encoder config with reduced iterations for testing
    let config = EncoderConfig {
        max_iterations: 50,  // Just test 50 iterations
        ..EncoderConfig::balanced()
    };

    println!("\n⚡ Starting encoding with GPU gradients...");
    println!("   Iterations: {}", config.max_iterations);
    println!("   Look for '📊 Iteration 0' logs showing 'using GPU'");
    println!();

    // Create optimizer with GPU enabled
    let encode_start = Instant::now();
    let optimizer = OptimizerV2::new(config).with_gpu();

    // Optimize (this will use GPU gradients!)
    let _metrics = optimizer.optimize_with_metrics(&mut gaussians, &target)?;

    let encode_time = encode_start.elapsed();

    println!("\n✅ Encoding complete in {:.2}s", encode_time.as_secs_f32());
    println!("   Average: {:.2}ms/iteration", encode_time.as_secs_f32() * 1000.0 / 50.0);

    Ok(())
}
