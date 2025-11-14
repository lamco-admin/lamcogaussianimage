//! EXP-011: Test on Different Content Types
//! Validate that our findings generalize OR identify content-specific needs

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-011: Content Type Validation           ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Test 1: Radial gradient
    println!("Test 1: Radial Gradient (smooth, circular structure)");
    test_radial_gradient();

    // Test 2: Checkerboard
    println!("\nTest 2: Checkerboard (multiple sharp edges)");
    test_checkerboard();

    // Test 3: Diagonal edge
    println!("\nTest 3: Diagonal Edge (non-axis-aligned)");
    test_diagonal_edge();

    // Test 4: Smooth curve
    println!("\nTest 4: Smooth Curve (test anisotropy hypothesis)");
    test_smooth_curve();
}

fn test_radial_gradient() {
    let mut target = ImageBuffer::new(256, 256);
    let center_x = 128.0_f32;
    let center_y = 128.0_f32;
    let max_dist = 128.0_f32 * 1.414;  // Corner distance

    for y in 0..256 {
        for x in 0..256 {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx*dx + dy*dy).sqrt();
            let t = (dist / max_dist).clamp(0.0, 1.0);
            target.set_pixel(x, y, Color4::new(1.0-t, 0.0, t, 1.0));
        }
    }

    test_image(&target, "Radial", 20);  // 20×20 grid
}

fn test_checkerboard() {
    let mut target = ImageBuffer::new(256, 256);
    let square_size = 32;

    for y in 0..256 {
        for x in 0..256 {
            let is_white = ((x / square_size) + (y / square_size)) % 2 == 0;
            let val = if is_white { 1.0 } else { 0.0 };
            target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    test_image(&target, "Checkerboard", 20);
}

fn test_diagonal_edge() {
    let mut target = ImageBuffer::new(256, 256);

    for y in 0..256 {
        for x in 0..256 {
            let val = if y > x { 0.0 } else { 1.0 };
            target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    test_image(&target, "Diagonal Edge", 20);
}

fn test_smooth_curve() {
    let mut target = ImageBuffer::new(256, 256);

    for y in 0..256 {
        for x in 0..256 {
            // Smooth sigmoid transition
            let edge_pos = 128.0 + 30.0 * (y as f32 / 256.0 - 0.5).sin();
            let distance = x as f32 - edge_pos;
            let val = 1.0 / (1.0 + (-distance / 10.0).exp());  // Smooth transition
            target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    test_image(&target, "Smooth Curve", 20);
}

fn test_image(target: &ImageBuffer<f32>, name: &str, grid_size: u32) {
    let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
    let mut gaussians = encoder.initialize_gaussians(grid_size);

    let init_rendered = RendererV2::render(&gaussians, 256, 256);
    let init_psnr = compute_psnr(&target, &init_rendered);

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 100;
    let final_loss = optimizer.optimize(&mut gaussians, &target);

    let final_rendered = RendererV2::render(&gaussians, 256, 256);
    let final_psnr = compute_psnr(&target, &final_rendered);

    println!("  N={}: Init {:.2} dB → Final {:.2} dB (Δ: {:+.2} dB, loss: {:.6})",
        grid_size * grid_size, init_psnr, final_psnr, final_psnr - init_psnr, final_loss);
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
