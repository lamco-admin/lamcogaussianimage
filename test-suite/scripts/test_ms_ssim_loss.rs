//! EXP-016: MS-SSIM Loss Comparison
//! Compare L2 loss vs MS-SSIM perceptual loss
//! Research predicts: +3-5 dB improvement

use lgi_core::{ImageBuffer, ms_ssim::MSSSIM};
use lgi_math::color::Color4;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-016: MS-SSIM vs L2 Loss Comparison     ║");
    println!("║  Research prediction: +3-5 dB improvement   ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Test on multiple content types
    test_gradient();
    test_smooth_curve();
    test_edge();
}

fn test_gradient() {
    println!("═══════════════════════════════════════════════");
    println!("Test 1: Linear Gradient");
    println!("═══════════════════════════════════════════════");

    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let t = x as f32 / 255.0;
            target.set_pixel(x, y, Color4::new(t, 0.0, 1.0 - t, 1.0));
        }
    }

    compare_losses(&target, 20, "Linear Gradient");
}

fn test_smooth_curve() {
    println!("\n═══════════════════════════════════════════════");
    println!("Test 2: Smooth Curve");
    println!("═══════════════════════════════════════════════");

    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let edge_pos = 128.0 + 30.0 * (y as f32 / 256.0 - 0.5).sin();
            let distance = x as f32 - edge_pos;
            let val = 1.0 / (1.0 + (-distance / 10.0).exp());
            target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    compare_losses(&target, 20, "Smooth Curve");
}

fn test_edge() {
    println!("\n═══════════════════════════════════════════════");
    println!("Test 3: Vertical Edge");
    println!("═══════════════════════════════════════════════");

    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let val = if x < 128 { 0.0 } else { 1.0 };
            target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    compare_losses(&target, 10, "Vertical Edge");
}

fn compare_losses(target: &ImageBuffer<f32>, grid_size: u32, name: &str) {
    let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
    let gaussians_init = encoder.initialize_gaussians(grid_size);

    let init_rendered = RendererV2::render(&gaussians_init, 256, 256);
    let init_psnr = compute_psnr(&target, &init_rendered);

    // Test 1: L2 Loss (current)
    let mut gaussians_l2 = gaussians_init.clone();
    let mut optimizer_l2 = OptimizerV2::default();
    optimizer_l2.max_iterations = 100;
    let _loss_l2 = optimizer_l2.optimize(&mut gaussians_l2, &target);

    let rendered_l2 = RendererV2::render(&gaussians_l2, 256, 256);
    let psnr_l2 = compute_psnr(&target, &rendered_l2);

    // Compute MS-SSIM for comparison
    let ms_ssim_computer = MSSSIM::default();
    let ms_ssim_l2 = ms_ssim_computer.compute(&rendered_l2, &target);

    println!("  L2 Loss Optimization:");
    println!("    Init PSNR: {:.2} dB", init_psnr);
    println!("    Final PSNR: {:.2} dB (Δ: {:+.2} dB)", psnr_l2, psnr_l2 - init_psnr);
    println!("    MS-SSIM: {:.4}", ms_ssim_l2);

    // Note: To properly test MS-SSIM loss, we'd need to implement MS-SSIM-based
    // gradient computation, which is complex. For now, we're measuring MS-SSIM
    // on L2-optimized results as a baseline.

    println!("\n  Note: Full MS-SSIM optimization requires differentiable MS-SSIM");
    println!("        (complex implementation, deferred for now)");
    println!("        Current test: MS-SSIM quality metric only");
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
