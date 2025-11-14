//! EXP-020: Gamma Auto-Tuning
//! Test if we can automatically find optimal gamma for any content

use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use lgi_encoder_v2::{renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-020: Gamma Auto-Tuning Test            ║");
    println!("║  Can we auto-find optimal γ for content?    ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Test on multiple content types
    test_content("Linear Gradient", create_gradient());
    test_content("Vertical Edge", create_edge());
    test_content("Smooth Curve", create_curve());
}

fn test_content(name: &str, target: ImageBuffer<f32>) {
    println!("\n{}", name);
    println!("═══════════════════════════════════════");

    let grid_size = 20;  // Fixed N=400
    let num_gaussians = grid_size * grid_size;

    // Test different gamma values
    println!("Testing γ values:");

    for &gamma in &[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2] {
        let gaussians = initialize_with_gamma(&target, grid_size, gamma);

        let init_rendered = RendererV2::render(&gaussians, 256, 256);
        let init_psnr = compute_psnr(&target, &init_rendered);

        // Quick optimization (50 iters)
        let mut gaussians_opt = gaussians.clone();
        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 50;
        optimizer.optimize(&mut gaussians_opt, &target);

        let final_rendered = RendererV2::render(&gaussians_opt, 256, 256);
        let final_psnr = compute_psnr(&target, &final_rendered);

        println!("  γ={:.1}: Init {:.2} dB → Final {:.2} dB (Δ: {:+.2} dB)",
            gamma, init_psnr, final_psnr, final_psnr - init_psnr);
    }
}

fn initialize_with_gamma(target: &ImageBuffer<f32>, grid_size: u32, gamma: f32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut gaussians = Vec::new();
    let num_gaussians = grid_size * grid_size;
    let width = target.width as f32;
    let height = target.height as f32;

    let sigma_base_px = gamma * ((width * height) / num_gaussians as f32).sqrt();
    let sigma_norm = (sigma_base_px / width).clamp(0.01, 0.25);

    let step_x = target.width / grid_size;
    let step_y = target.height / grid_size;

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx * step_x + step_x / 2).min(target.width - 1);
            let y = (gy * step_y + step_y / 2).min(target.height - 1);

            let position = Vector2::new(
                x as f32 / target.width as f32,
                y as f32 / target.height as f32,
            );

            let color = target.get_pixel(x, y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            gaussians.push(Gaussian2D::new(
                position,
                Euler::isotropic(sigma_norm),
                color,
                1.0,
            ));
        }
    }

    gaussians
}

fn create_gradient() -> ImageBuffer<f32> {
    let mut img = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let t = x as f32 / 255.0;
            img.set_pixel(x, y, Color4::new(t, 0.0, 1.0 - t, 1.0));
        }
    }
    img
}

fn create_edge() -> ImageBuffer<f32> {
    let mut img = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let val = if x < 128 { 0.0 } else { 1.0 };
            img.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }
    img
}

fn create_curve() -> ImageBuffer<f32> {
    let mut img = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let edge_pos = 128.0 + 30.0 * (y as f32 / 256.0 - 0.5).sin();
            let distance = x as f32 - edge_pos;
            let val = 1.0 / (1.0 + (-distance / 10.0).exp());
            img.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }
    img
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
