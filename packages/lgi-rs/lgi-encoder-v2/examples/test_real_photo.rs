//! EXP-015: Real photo test (Sam.jpg)
//! Test on actual difficult real-world content

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-015: Real Photo Test (Sam.jpg)         ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Load image
    println!("Loading Sam.jpg...");
    let target = match ImageBuffer::load("/media/nomachine/C on Player (NoMachine)/Sam.jpg") {
        Ok(img) => {
            println!("  Loaded: {}×{}", img.width, img.height);
            img
        }
        Err(e) => {
            println!("  Error loading image: {}", e);
            println!("  Trying to create test image instead...");

            // Fallback: create a complex synthetic image
            let mut img = ImageBuffer::new(256, 256);
            for y in 0..256 {
                for x in 0..256 {
                    let xf = x as f32 / 255.0;
                    let yf = y as f32 / 255.0;

                    // Complex pattern
                    let r = (xf * yf).sqrt();
                    let g = ((1.0 - xf) * yf).sqrt();
                    let b = (xf * (1.0 - yf)).sqrt();

                    img.set_pixel(x, y, lgi_math::color::Color4::new(r, g, b, 1.0));
                }
            }
            img
        }
    };

    // Test with multiple Gaussian counts
    println!("\nTesting different Gaussian counts:\n");

    for &grid_size in &[16, 20, 24, 32, 40] {
        let num_gaussians = grid_size * grid_size;

        println!("N={} ({}×{}):", num_gaussians, grid_size, grid_size);

        let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
        let mut gaussians = encoder.initialize_gaussians(grid_size);

        let init_rendered = RendererV2::render(&gaussians, target.width, target.height);
        let init_psnr = compute_psnr(&target, &init_rendered);

        // Use conservative settings for photos
        let mut optimizer = OptimizerV2::default();
        optimizer.learning_rate_color = 0.2;  // Lower for complex content
        optimizer.learning_rate_position = 0.03;
        optimizer.max_iterations = 200;

        let final_loss = optimizer.optimize(&mut gaussians, &target);

        let final_rendered = RendererV2::render(&gaussians, target.width, target.height);
        let final_psnr = compute_psnr(&target, &final_rendered);

        println!("  Init: {:.2} dB → Final: {:.2} dB (Δ: {:+.2} dB, loss: {:.6})",
            init_psnr, final_psnr, final_psnr - init_psnr, final_loss);

        // Check if optimization helped or hurt
        if final_psnr < init_psnr - 0.5 {
            println!("  ⚠️  WARNING: Optimization made quality WORSE!");
        } else if final_psnr > init_psnr + 1.0 {
            println!("  ✅ Good optimization improvement");
        }
    }

    println!("\n════════════════════════════════════════");
    println!("Target: 28-35 dB for photo quality");
    println!("════════════════════════════════════════");
}

fn compute_psnr(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (original.width * original.height * 3) as f32;

    for (p1, p2) in original.data.iter().zip(rendered.data.iter()) {
        mse += (p1.r - p2.r).powi(2);
        mse += (p1.g - p2.g).powi(2);
        mse += (p1.b - p2.b).powi(2);
    }

    mse /= count;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
