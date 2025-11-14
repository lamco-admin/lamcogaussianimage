//! EXP-014: Complex synthetic photo-like content
//! Combines: smooth gradients, edges, color variation, patterns

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};
use std::f32::consts::PI;

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-014: Complex Photo-Like Content        ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Create complex synthetic image
    println!("Creating complex synthetic image...");
    let mut target = ImageBuffer::new(256, 256);

    for y in 0..256 {
        for x in 0..256 {
            let xf = x as f32 / 255.0;
            let yf = y as f32 / 255.0;

            // Smooth background gradient
            let bg_r = xf * 0.3 + yf * 0.2;
            let bg_g = (1.0 - xf) * 0.4 + yf * 0.3;
            let bg_b = (1.0 - yf) * 0.5 + xf * 0.2;

            // Add circular feature (simulates face/object)
            let dx = xf - 0.5;
            let dy = yf - 0.5;
            let dist = (dx*dx + dy*dy).sqrt();
            let circle_intensity = if dist < 0.25 {
                1.0 - (dist / 0.25)
            } else {
                0.0
            };

            // Add some texture-like variation
            let texture = ((xf * 20.0).sin() * (yf * 20.0).cos() * 0.05).abs();

            // Combine
            let r = (bg_r + circle_intensity * 0.6 + texture).clamp(0.0, 1.0);
            let g = (bg_g + circle_intensity * 0.4 + texture).clamp(0.0, 1.0);
            let b = (bg_b + circle_intensity * 0.3 + texture).clamp(0.0, 1.0);

            target.set_pixel(x, y, Color4::new(r, g, b, 1.0));
        }
    }

    // Test with multiple Gaussian counts
    println!("Testing with different Gaussian counts:\n");

    for &grid_size in &[12, 16, 20, 24, 32] {
        let num_gaussians = grid_size * grid_size;

        println!("N={} ({}×{} grid):", num_gaussians, grid_size, grid_size);

        let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
        let mut gaussians = encoder.initialize_gaussians(grid_size);

        let init_rendered = RendererV2::render(&gaussians, 256, 256);
        let init_psnr = compute_psnr(&target, &init_rendered);

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 200;
        let final_loss = optimizer.optimize(&mut gaussians, &target);

        let final_rendered = RendererV2::render(&gaussians, 256, 256);
        let final_psnr = compute_psnr(&target, &final_rendered);

        println!("  Init: {:.2} dB → Final: {:.2} dB (Δ: {:+.2} dB, loss: {:.6})",
            init_psnr, final_psnr, final_psnr - init_psnr, final_loss);
    }

    println!("\n════════════════════════════════════════");
    println!("Analysis:");
    println!("  - Photo-like content complexity between gradient and edge");
    println!("  - Should achieve 28-35 dB for acceptable quality");
    println!("  - Identifies if current approach generalizes");
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
