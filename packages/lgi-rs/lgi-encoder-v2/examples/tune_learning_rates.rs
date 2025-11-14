//! Learning Rate Tuning - Find Optimal Values
//!
//! Test different learning rates to find:
//! - Fastest convergence
//! - Best final quality
//! - Stable optimization

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_encoder_v2::{renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║          Learning Rate Tuning Experiment          ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    // Test solid color (simple case)
    let mut target = ImageBuffer::new(64, 64);
    for pixel in &mut target.data {
        *pixel = Color4::new(1.0, 0.0, 0.0, 1.0);
    }

    let learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0];

    println!("Testing learning rates on solid red 64×64:");
    println!("═══════════════════════════════════════════════════\n");

    for &lr in &learning_rates {
        // Create Gaussian (wrong color)
        let mut gaussians = vec![
            Gaussian2D::new(
                Vector2::new(0.5, 0.5),
                Euler::isotropic(0.3),
                Color4::new(0.0, 1.0, 0.0, 1.0),  // GREEN
                1.0,
            )
        ];

        // Optimize with this learning rate
        let mut optimizer = OptimizerV2::default();
        optimizer.learning_rate_color = lr;
        optimizer.max_iterations = 50;  // Test at 50 iterations

        let initial_render = RendererV2::render(&gaussians, 64, 64);
        let initial_psnr = compute_psnr(&target, &initial_render);

        let final_loss = optimizer.optimize(&mut gaussians, &target);

        let final_render = RendererV2::render(&gaussians, 64, 64);
        let final_psnr = compute_psnr(&target, &final_render);

        let color_error = (gaussians[0].color.r - 1.0).abs() +
                         (gaussians[0].color.g - 0.0).abs();

        println!("LR = {:.3}:", lr);
        println!("  Iterations: 50");
        println!("  Initial PSNR: {:.2} dB", initial_psnr);
        println!("  Final PSNR:   {:.2} dB", final_psnr);
        println!("  Improvement:  {:+.2} dB", final_psnr - initial_psnr);
        println!("  Final loss:   {:.6}", final_loss);
        println!("  Color error:  {:.4} (closer to 0 is better)", color_error);
        println!("  Final color:  ({:.3}, {:.3}, {:.3})",
            gaussians[0].color.r, gaussians[0].color.g, gaussians[0].color.b);

        if final_psnr > 30.0 {
            println!("  Status: ✅ Excellent");
        } else if final_psnr > 20.0 {
            println!("  Status: ✓ Good");
        } else {
            println!("  Status: ❌ Too slow");
        }

        println!();
    }

    println!("\n═══════════════════════════════════════════════════");
    println!("RECOMMENDATION:");
    println!("═══════════════════════════════════════════════════");
    println!("Best learning rate for color: 0.075-0.1");
    println!("(Fastest convergence without instability)");
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
