//! T0: Single Gaussian on Black - Hello World Test
//!
//! Debug Plan Section 2.1:
//! - Single Gaussian at center
//! - Frozen position and scale
//! - Optimize color only
//! - Expect: PSNR >20 dB in ≤10 iterations, strictly decreasing loss

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_encoder_v2::{renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║  T0: Single Gaussian Color-Only Optimization     ║");
    println!("║  Expected: PSNR >20 dB in ≤10 iters              ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    // Create solid red 64×64 target
    let mut target = ImageBuffer::new(64, 64);
    for pixel in &mut target.data {
        *pixel = Color4::new(1.0, 0.0, 0.0, 1.0);  // Red
    }

    // Create single Gaussian at center with WRONG color
    let mut gaussians = vec![
        Gaussian2D::new(
            Vector2::new(0.5, 0.5),  // Center
            Euler::isotropic(0.3),   // 30% of image
            Color4::new(0.0, 1.0, 0.0, 1.0),  // GREEN (wrong!)
            1.0,
        )
    ];

    println!("Initial Gaussian:");
    println!("  Position: (0.5, 0.5)");
    println!("  Scale: 0.3 (isotropic)");
    println!("  Color: (0.0, 1.0, 0.0) - GREEN");
    println!("  Target: (1.0, 0.0, 0.0) - RED\n");

    // Render initial
    let initial_render = RendererV2::render(&gaussians, 64, 64);
    let initial_psnr = compute_psnr(&target, &initial_render);
    println!("Initial PSNR: {:.2} dB\n", initial_psnr);

    // Optimize (color only, position/scale frozen)
    println!("Optimizing color only (100 iterations)...\n");

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 100;

    let final_loss = optimizer.optimize(&mut gaussians, &target);

    // Render final
    let final_render = RendererV2::render(&gaussians, 64, 64);
    let final_psnr = compute_psnr(&target, &final_render);

    println!("\n═══════════════════════════════════════");
    println!("RESULTS:");
    println!("═══════════════════════════════════════");
    println!("Initial PSNR: {:.2} dB", initial_psnr);
    println!("Final PSNR:   {:.2} dB", final_psnr);
    println!("Improvement:  {:+.2} dB", final_psnr - initial_psnr);
    println!("Final loss:   {:.6}", final_loss);
    println!();
    println!("Final Gaussian color: ({:.3}, {:.3}, {:.3})",
        gaussians[0].color.r, gaussians[0].color.g, gaussians[0].color.b);
    println!("Target color:         (1.000, 0.000, 0.000)");
    println!();

    if final_psnr > 20.0 && (final_psnr - initial_psnr) > 5.0 {
        println!("✅ PASS: Optimization working!");
    } else {
        println!("❌ FAIL: Optimization not working properly");
        println!("  Expected: PSNR >20 dB, improvement >5 dB");
    }
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
