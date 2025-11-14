//! EXP-018: High-Density Edge Testing
//! Test N=1000, 1600, 2500 to find edge quality limit

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-018: High-Density Edge Quality Test    ║");
    println!("║  Goal: Find quality limit for sharp edges   ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Vertical edge
    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let val = if x < 128 { 0.0 } else { 1.0 };
            target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    println!("Testing vertical edge with high Gaussian counts:\n");

    for &grid_size in &[32, 40, 50, 64] {  // 1024, 1600, 2500, 4096
        let num_gaussians = grid_size * grid_size;

        println!("N={} ({}×{} grid):", num_gaussians, grid_size, grid_size);

        let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
        let mut gaussians = encoder.initialize_gaussians(grid_size);

        let init_rendered = RendererV2::render(&gaussians, 256, 256);
        let init_psnr = compute_psnr(&target, &init_rendered);
        let init_edge_psnr = compute_psnr_region(&target, &init_rendered, 120, 136, 0, 256);

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 150;
        let final_loss = optimizer.optimize(&mut gaussians, &target);

        let final_rendered = RendererV2::render(&gaussians, 256, 256);
        let final_psnr = compute_psnr(&target, &final_rendered);
        let final_edge_psnr = compute_psnr_region(&target, &final_rendered, 120, 136, 0, 256);

        println!("  Init:  {:.2} dB overall, {:.2} dB edge", init_psnr, init_edge_psnr);
        println!("  Final: {:.2} dB overall, {:.2} dB edge", final_psnr, final_edge_psnr);
        println!("  Improvement: {:+.2} dB, loss: {:.6}\n", final_psnr - init_psnr, final_loss);
    }

    println!("═══════════════════════════════════════════════");
    println!("Analysis:");
    println!("  - Looking for: N where edge PSNR saturates");
    println!("  - Target: 27-30 dB overall, 20+ dB edge region");
    println!("  - If diminishing returns: Found optimal N");
    println!("═══════════════════════════════════════════════");
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

fn compute_psnr_region(
    original: &ImageBuffer<f32>,
    rendered: &ImageBuffer<f32>,
    x_start: u32, x_end: u32, y_start: u32, y_end: u32,
) -> f32 {
    let mut mse = 0.0;
    let mut count = 0.0;

    for y in y_start..y_end.min(original.height) {
        for x in x_start..x_end.min(original.width) {
            if let (Some(p1), Some(p2)) = (original.get_pixel(x, y), rendered.get_pixel(x, y)) {
                mse += (p1.r - p2.r).powi(2) + (p1.g - p2.g).powi(2) + (p1.b - p2.b).powi(2);
                count += 3.0;
            }
        }
    }

    if count > 0.0 { mse /= count; }
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
