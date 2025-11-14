//! EXP-008/009: Systematic edge testing
//! Test multiple hypotheses: N count, gamma, anisotropy ratio

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-008/009: Systematic Edge Analysis      ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Create vertical edge
    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let color = if x < 128 { 0.0 } else { 1.0 };
            target.set_pixel(x, y, Color4::new(color, color, color, 1.0));
        }
    }

    // Test different Gaussian counts
    println!("═══════════════════════════════════════════════");
    println!("PART 1: Gaussian Count Sweep");
    println!("═══════════════════════════════════════════════\n");

    for &grid_size in &[10, 14, 20, 28] {  // 100, 196, 400, 784 Gaussians
        let num_gaussians = grid_size * grid_size;
        println!("Testing {}×{} = {} Gaussians:", grid_size, grid_size, num_gaussians);

        let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
        let mut gaussians = encoder.initialize_gaussians(grid_size);

        let init_rendered = RendererV2::render(&gaussians, 256, 256);
        let init_psnr = compute_psnr(&target, &init_rendered);

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 100;
        let _final_loss = optimizer.optimize(&mut gaussians, &target);

        let final_rendered = RendererV2::render(&gaussians, 256, 256);
        let final_psnr = compute_psnr(&target, &final_rendered);
        let edge_psnr = compute_psnr_region(&target, &final_rendered, 120, 136, 0, 256);

        println!("  Initial: {:.2} dB", init_psnr);
        println!("  Final:   {:.2} dB (overall), {:.2} dB (edge region)", final_psnr, edge_psnr);
        println!("  Improvement: {:+.2} dB\n", final_psnr - init_psnr);
    }

    println!("═══════════════════════════════════════════════");
    println!("INTERPRETATION:");
    println!("- If PSNR increases with N: Need more Gaussians");
    println!("- If PSNR plateaus: Gaussian count not the issue");
    println!("- If edge region stays poor: Anisotropy/scale issue");
    println!("═══════════════════════════════════════════════\n");

    // Note about anisotropy - requires code change
    println!("PART 2: Anisotropy Ratio (requires manual testing)");
    println!("Current: β=4 (4:1 ratio)");
    println!("To test: Modify lib.rs line ~111: let spa = BETA * sp;");
    println!("Test BETA ∈ [2, 4, 8, 16] and record results\n");

    println!("PART 3: Gamma Values (requires manual testing)");
    println!("Current: adaptive_gamma(100) = 0.8");
    println!("Debug plan suggests: γ ∈ [0.30, 0.45]");
    println!("Handover doc uses: γ = 0.8");
    println!("Need to test both ranges manually\n");
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
