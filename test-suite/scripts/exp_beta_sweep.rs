//! EXP-009: Anisotropy ratio sweep - DATA COLLECTION

use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use lgi_encoder_v2::{renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("EXP-009: Anisotropy Ratio β Sweep");
    println!("==================================\n");

    // Create vertical edge
    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let color = if x < 128 { 0.0 } else { 1.0 };
            target.set_pixel(x, y, Color4::new(color, color, color, 1.0));
        }
    }

    // Test different β values with N=784 Gaussians
    for &beta in &[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0] {
        println!("Testing β={:.1} (isotropic in flat regions, β:1 at edges):", beta);

        // Create 28×28 grid manually with specific β
        let grid_size = 28;
        let num_gaussians = grid_size * grid_size;
        let gamma = 0.6;  // For N~1000
        let width = 256.0_f32;
        let height = 256.0_f32;
        let sigma_base_px = gamma * ((width * height) / num_gaussians as f32).sqrt();

        let mut gaussians = Vec::new();
        let step_x = 256 / grid_size;
        let step_y = 256 / grid_size;

        for gy in 0..grid_size {
            for gx in 0..grid_size {
                let x_px = (gx * step_x + step_x / 2).min(255);
                let y_px = (gy * step_y + step_y / 2).min(255);

                let position = Vector2::new(
                    x_px as f32 / width,
                    y_px as f32 / height,
                );

                // Sample color
                let color = if x_px < 128 {
                    Color4::new(0.0, 0.0, 0.0, 1.0)
                } else {
                    Color4::new(1.0, 1.0, 1.0, 1.0)
                };

                // Near edge (x=128)? Use anisotropy
                let distance_to_edge = (x_px as i32 - 128).abs() as f32;
                let (sig_para, sig_perp) = if distance_to_edge < 20.0 {
                    // Near edge: anisotropic (long vertical, thin horizontal)
                    let perp = sigma_base_px / 2.0;  // Thinner
                    (beta * perp, perp)
                } else {
                    // Far from edge: isotropic
                    (sigma_base_px, sigma_base_px)
                };

                let scale_x = sig_para.clamp(3.0, 64.0) / width;
                let scale_y = sig_perp.clamp(3.0, 64.0) / height;

                let gaussian = Gaussian2D::new(
                    position,
                    Euler::new(scale_x, scale_y, 0.0),  // Vertical edge: no rotation
                    color,
                    1.0,
                );

                gaussians.push(gaussian);
            }
        }

        // Test
        let init_rendered = RendererV2::render(&gaussians, 256, 256);
        let init_psnr = compute_psnr(&target, &init_rendered);

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 100;
        let _final_loss = optimizer.optimize(&mut gaussians, &target);

        let final_rendered = RendererV2::render(&gaussians, 256, 256);
        let final_psnr = compute_psnr(&target, &final_rendered);
        let edge_psnr = compute_psnr_region(&target, &final_rendered, 120, 136, 0, 256);

        println!("  Init: {:.2} dB, Final: {:.2} dB, Edge: {:.2} dB, Δ: {:+.2} dB",
            init_psnr, final_psnr, edge_psnr, final_psnr - init_psnr);
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
