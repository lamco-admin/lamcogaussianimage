//! EXP-017: Error-Driven vs Uniform Placement
//! Data shows: Manual edge-aware placement gave +4.3 dB
//! Test if automatic error-driven placement achieves similar gains

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2, error_driven::ErrorDrivenEncoder};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-017: Error-Driven Placement Test       ║");
    println!("║  Expected: +2-4 dB vs uniform grid          ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Test on edge (where error-driven should help most)
    println!("Test 1: Vertical Edge");
    println!("═══════════════════════════════════════");

    let mut edge_target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let val = if x < 128 { 0.0 } else { 1.0 };
            edge_target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    // Uniform grid baseline (N=400)
    println!("\nBaseline: Uniform grid (20×20 = 400 Gaussians)");
    let encoder = EncoderV2::new(edge_target.clone()).expect("Encoder failed");
    let mut gaussians_uniform = encoder.initialize_gaussians(20);

    let init_psnr = {
        let rendered = RendererV2::render(&gaussians_uniform, 256, 256);
        compute_psnr(&edge_target, &rendered)
    };

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 100;
    let _loss = optimizer.optimize(&mut gaussians_uniform, &edge_target);

    let final_psnr_uniform = {
        let rendered = RendererV2::render(&gaussians_uniform, 256, 256);
        compute_psnr(&edge_target, &rendered)
    };

    println!("  Init: {:.2} dB → Final: {:.2} dB (Δ: {:+.2} dB)",
        init_psnr, final_psnr_uniform, final_psnr_uniform - init_psnr);

    // Error-driven placement
    println!("\nError-Driven: Adaptive placement (start N=100, add where needed)");
    let error_driven = ErrorDrivenEncoder::default();
    let gaussians_adaptive = error_driven.encode(&edge_target);

    let final_psnr_adaptive = {
        let rendered = RendererV2::render(&gaussians_adaptive, 256, 256);
        compute_psnr(&edge_target, &rendered)
    };

    println!("\nComparison:");
    println!("  Uniform (N=400):        {:.2} dB", final_psnr_uniform);
    println!("  Error-driven (N={}): {:.2} dB", gaussians_adaptive.len(), final_psnr_adaptive);
    println!("  Improvement:            {:+.2} dB", final_psnr_adaptive - final_psnr_uniform);

    if final_psnr_adaptive > final_psnr_uniform + 1.0 {
        println!("\n✅ Error-driven placement provides significant improvement!");
    } else if final_psnr_adaptive > final_psnr_uniform {
        println!("\n✓ Minor improvement from error-driven placement");
    } else {
        println!("\n⚠️  Error-driven not better than uniform (may need tuning)");
    }

    // Test on gradient
    println!("\n\nTest 2: Linear Gradient");
    println!("═══════════════════════════════════════");

    let mut grad_target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let t = x as f32 / 255.0;
            grad_target.set_pixel(x, y, Color4::new(t, 0.0, 1.0 - t, 1.0));
        }
    }

    // Uniform
    let encoder = EncoderV2::new(grad_target.clone()).expect("Encoder failed");
    let mut gaussians_uniform_grad = encoder.initialize_gaussians(20);
    let mut optimizer_grad = OptimizerV2::default();
    optimizer_grad.max_iterations = 100;
    optimizer_grad.optimize(&mut gaussians_uniform_grad, &grad_target);

    let psnr_uniform_grad = {
        let rendered = RendererV2::render(&gaussians_uniform_grad, 256, 256);
        compute_psnr(&grad_target, &rendered)
    };

    // Error-driven
    let error_driven_grad = ErrorDrivenEncoder::default();
    let gaussians_adaptive_grad = error_driven_grad.encode(&grad_target);

    let psnr_adaptive_grad = {
        let rendered = RendererV2::render(&gaussians_adaptive_grad, 256, 256);
        compute_psnr(&grad_target, &rendered)
    };

    println!("  Uniform (N=400):        {:.2} dB", psnr_uniform_grad);
    println!("  Error-driven (N={}): {:.2} dB", gaussians_adaptive_grad.len(), psnr_adaptive_grad);
    println!("  Difference:             {:+.2} dB", psnr_adaptive_grad - psnr_uniform_grad);
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
