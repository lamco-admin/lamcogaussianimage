//! EXP-006: Coherence Threshold Sweep
//! Test different coherence thresholds for anisotropy decision

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-006: Coherence Threshold Sweep         ║");
    println!("║  Testing edge representation quality        ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Create vertical edge (black|white)
    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let color = if x < 128 { 0.0 } else { 1.0 };
            target.set_pixel(x, y, Color4::new(color, color, color, 1.0));
        }
    }

    // NOTE: This test requires modifying the coherence threshold in lib.rs
    // Current: coherence < 0.2
    // We need to test: [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]

    println!("Current coherence threshold: 0.2 (hardcoded in lib.rs)");
    println!("To run full sweep, modify lib.rs line ~105 and recompile\n");

    println!("Testing with current threshold (0.2):");
    let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
    let mut gaussians = encoder.initialize_gaussians(10);

    let init_rendered = RendererV2::render(&gaussians, 256, 256);
    let init_psnr = compute_psnr(&target, &init_rendered);
    println!("  Initial PSNR: {:.2} dB", init_psnr);

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 100;
    let _final_loss = optimizer.optimize(&mut gaussians, &target);

    let final_rendered = RendererV2::render(&gaussians, 256, 256);
    let final_psnr = compute_psnr(&target, &final_rendered);
    let edge_psnr = compute_psnr_region(&target, &final_rendered, 120, 136, 0, 256);

    println!("  Final PSNR:   {:.2} dB", final_psnr);
    println!("  Edge region:  {:.2} dB", edge_psnr);
    println!("  Improvement:  {:+.2} dB\n", final_psnr - init_psnr);

    println!("MANUAL SWEEP REQUIRED:");
    println!("1. Edit lib.rs line ~105: if tensor.coherence < THRESHOLD");
    println!("2. Test thresholds: [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]");
    println!("3. Record PSNR for each");
    println!("4. Document in rolling log");
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

fn compute_psnr_region(
    original: &ImageBuffer<f32>,
    rendered: &ImageBuffer<f32>,
    x_start: u32,
    x_end: u32,
    y_start: u32,
    y_end: u32,
) -> f32 {
    let mut mse = 0.0;
    let mut count = 0.0;

    for y in y_start..y_end.min(original.height) {
        for x in x_start..x_end.min(original.width) {
            if let (Some(p1), Some(p2)) = (original.get_pixel(x, y), rendered.get_pixel(x, y)) {
                mse += (p1.r - p2.r).powi(2);
                mse += (p1.g - p2.g).powi(2);
                mse += (p1.b - p2.b).powi(2);
                count += 3.0;
            }
        }
    }

    if count > 0.0 {
        mse /= count;
    }

    if mse < 1e-10 {
        100.0
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}
