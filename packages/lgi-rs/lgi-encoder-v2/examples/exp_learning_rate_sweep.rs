//! EXP-021: Learning Rate Optimization
//! Find optimal LR for different parameters

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-021: Learning Rate Sweep                ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Test on smooth curve (showed +4.6 dB potential with scale opt)
    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let edge_pos = 128.0 + 30.0 * (y as f32 / 256.0 - 0.5).sin();
            let distance = x as f32 - edge_pos;
            let val = 1.0 / (1.0 + (-distance / 10.0).exp());
            target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    println!("Test: Smooth Curve (N=400)");
    println!("═══════════════════════════════════════\n");

    let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
    let gaussians_init = encoder.initialize_gaussians(20);

    let init_psnr = {
        let rendered = RendererV2::render(&gaussians_init, 256, 256);
        compute_psnr(&target, &rendered)
    };

    println!("LR_scale sweep (LR_color=0.3, LR_position=0.05):");

    for &lr_scale in &[0.001, 0.003, 0.005, 0.01, 0.02, 0.05] {
        let mut gaussians = gaussians_init.clone();
        let mut optimizer = OptimizerV2::default();
        optimizer.learning_rate_scale = lr_scale;
        optimizer.max_iterations = 100;

        optimizer.optimize(&mut gaussians, &target);

        let final_rendered = RendererV2::render(&gaussians, 256, 256);
        let final_psnr = compute_psnr(&target, &final_rendered);

        println!("  LR_scale={:.3}: {:.2} → {:.2} dB (Δ: {:+.2} dB)",
            lr_scale, init_psnr, final_psnr, final_psnr - init_psnr);
    }

    println!("\nLR_rotation sweep:");

    for &lr_rot in &[0.001, 0.005, 0.01, 0.02, 0.05] {
        let mut gaussians = gaussians_init.clone();
        let mut optimizer = OptimizerV2::default();
        optimizer.learning_rate_rotation = lr_rot;
        optimizer.max_iterations = 100;

        optimizer.optimize(&mut gaussians, &target);

        let final_rendered = RendererV2::render(&gaussians, 256, 256);
        let final_psnr = compute_psnr(&target, &final_rendered);

        println!("  LR_rotation={:.3}: {:.2} → {:.2} dB (Δ: {:+.2} dB)",
            lr_rot, init_psnr, final_psnr, final_psnr - init_psnr);
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
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
