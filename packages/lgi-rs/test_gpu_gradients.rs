#!/usr/bin/env cargo +nightly -Zscript
//! Test GPU gradient computation
//!
//! ```cargo
//! [dependencies]
//! lgi-core = { path = "lgi-core" }
//! lgi-encoder = { path = "lgi-encoder" }
//! lgi-gpu = { path = "lgi-gpu" }
//! lgi-format = { path = "lgi-format" }
//! image = "0.25"
//! pollster = "0.3"
//! env_logger = "0.11"
//! ```

use lgi_core::{ImageBuffer, ColorSpace};
use lgi_encoder::{ EncoderConfig, OptimizerV2};
use lgi_format::EncodedImage;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

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
            target.set_pixel(x, y, lgi_core::Color4::new(
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
                1.0,
            ));
        }
    }

    // Initialize GPU
    println!("\n🎮 Initializing GPU...");
    pollster::block_on(async {
        lgi_gpu::GpuManager::global().initialize().await
    })?;
    println!("✅ GPU initialized with gradient computer");

    // Initialize Gaussians
    let num_gaussians = 100; // Small number for quick test
    let mut gaussians = lgi_encoder::initialize_gaussians_adaptive(&target, num_gaussians)?;
    println!("🔵 Initialized {} Gaussians", gaussians.len());

    // Create encoder config with reduced iterations for testing
    let config = EncoderConfig {
        max_iterations: 50,  // Just test 50 iterations
        ..EncoderConfig::balanced(width, height)
    };

    println!("\n⚡ Starting encoding with GPU gradients...");
    println!("   Iterations: {}", config.max_iterations);

    // Create optimizer with GPU enabled
    let optimizer = OptimizerV2::new(config).with_gpu();

    // Optimize (this will use GPU gradients!)
    let _metrics = optimizer.optimize_with_metrics(&mut gaussians, &target)?;

    println!("\n✅ Encoding complete!");
    println!("📊 Check logs above for 'using GPU' messages");

    Ok(())
}
