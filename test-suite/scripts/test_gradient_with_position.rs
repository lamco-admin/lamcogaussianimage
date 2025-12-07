//! Test: Gradient with Position Optimization
//!
//! Expected: With position updates, Gaussians move to optimal locations
//! Target: 20-28 dB PSNR

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   Gradient Test: Position + Color Optimization   ║");
    println!("║   Target: 20-28 dB PSNR                          ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    // Create linear gradient (blue → red)
    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let t = x as f32 / 255.0;
            target.set_pixel(x, y, Color4::new(t, 0.0, 1.0 - t, 1.0));
        }
    }

    // Test with different Gaussian counts
    let grid_sizes = [12, 16, 20, 24];  // 144, 256, 400, 576 Gaussians

    println!("Testing gradient with varying Gaussian counts:");
    println!("═══════════════════════════════════════════════════\n");

    for &grid_size in &grid_sizes {
        let num_gaussians = grid_size * grid_size;

        println!("───────────────────────────────────────────────────");
        println!("Grid {}×{} ({} Gaussians)", grid_size, grid_size, num_gaussians);
        println!("───────────────────────────────────────────────────");

        let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
        let mut gaussians = encoder.initialize_gaussians(grid_size);

        // Test without optimization first
        let init_rendered = RendererV2::render(&gaussians, 256, 256);
        let init_psnr = compute_psnr(&target, &init_rendered);
        println!("  Initial PSNR: {:.2} dB", init_psnr);

        // Optimize
        let mut optimizer = OptimizerV2::default();
        optimizer.learning_rate_color = 0.3;
        optimizer.learning_rate_position = 0.05;  // Moderate for gradient
        optimizer.max_iterations = 200;

        let final_loss = optimizer.optimize(&mut gaussians, &target);

        let final_rendered = RendererV2::render(&gaussians, 256, 256);
        let final_psnr = compute_psnr(&target, &final_rendered);

        println!("  Final PSNR:   {:.2} dB", final_psnr);
        println!("  Improvement:  {:+.2} dB", final_psnr - init_psnr);
        println!("  Final loss:   {:.6}", final_loss);

        if final_psnr >= 24.0 {
            println!("  Status: ✅ PASS (≥24 dB)");
        } else if final_psnr >= 20.0 {
            println!("  Status: ✓ Acceptable (20-24 dB)");
        } else {
            println!("  Status: ❌ FAIL (<20 dB)");
        }

        println!();
    }

    println!("═══════════════════════════════════════════════════");
    println!("CONCLUSION:");
    println!("Expected: Higher Gaussian counts → better PSNR");
    println!("Target: ≥24 dB with 400-576 Gaussians");
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

    if mse < 1e-10 {
        100.0
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}
