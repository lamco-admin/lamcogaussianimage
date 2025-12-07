//! Gamma Parameter Sweep - EXP-4-002
//! Test different gamma values in coverage formula: σ = γ√(WH/N)

use lgi_core::ImageBuffer;
use lgi_encoder_v2::optimizer_v2::OptimizerV2;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

fn main() {
    // Use Tier 1 test image (from Session 3)
    let path = "/home/greg/gaussian-image-projects/test_images/133784383569199567.jpg";

    let target = match ImageBuffer::load(path) {
        Ok(img) => {
            // Resize to 512px (standard test size)
            let scale = 512.0 / img.width.max(img.height) as f32;
            let new_w = (img.width as f32 * scale) as u32;
            let new_h = (img.height as f32 * scale) as u32;
            resize_bilinear(&img, new_w, new_h)
        }
        Err(e) => {
            println!("Failed to load image: {}", e);
            return;
        }
    };

    println!("EXP-4-002: Gamma Parameter Sweep");
    println!("Image: {}×{}", target.width, target.height);
    println!();

    // Test N=400 (20×20 grid, like Session 3 baseline)
    let grid_size = 20;
    let n = grid_size * grid_size;

    // Current adaptive gamma for N=400 is 0.7
    // Test range: 0.5 to 1.0 in steps of 0.1
    let gamma_values = vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    println!("Testing N={} with {} iterations", n, 30);
    println!("Gamma values: {:?}", gamma_values);
    println!();

    for gamma in gamma_values {
        let mut gaussians = initialize_with_gamma(&target, grid_size, gamma);

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 30;
        // GPU used automatically if available

        optimizer.optimize(&mut gaussians, &target);

        let psnr = compute_psnr(&gaussians, &target);

        println!("γ={:.1} → {:.2} dB", gamma, psnr);
    }
}

fn initialize_with_gamma(target: &ImageBuffer<f32>, grid_size: usize, gamma: f32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let width_px = target.width as f32;
    let height_px = target.height as f32;
    let num_gaussians = (grid_size * grid_size) as f32;

    // Coverage formula: σ = γ√(WH/N)
    let sigma_base_px = gamma * ((width_px * height_px) / num_gaussians).sqrt();
    let sigma_base = sigma_base_px / width_px.max(height_px);

    let mut gaussians = Vec::new();

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx as f32 + 0.5) / grid_size as f32;
            let y = (gy as f32 + 0.5) / grid_size as f32;

            let px_x = (x * width_px) as u32;
            let px_y = (y * height_px) as u32;
            let color = target.get_pixel(px_x.min(target.width - 1), px_y.min(target.height - 1))
                .unwrap_or(lgi_math::color::Color4::new(0.5, 0.5, 0.5, 1.0));

            gaussians.push(Gaussian2D::new(
                Vector2::new(x, y),
                Euler::isotropic(sigma_base),
                Color4::new(color.r, color.g, color.b, 1.0),
                0.5,
            ));
        }
    }

    gaussians
}

fn compute_psnr(gaussians: &[Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) -> f32 {
    use lgi_encoder_v2::renderer_v2::RendererV2;

    let rendered = RendererV2::render(gaussians, target.width, target.height);

    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= (rendered.width * rendered.height * 3) as f32;

    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}

fn resize_bilinear(img: &ImageBuffer<f32>, new_width: u32, new_height: u32) -> ImageBuffer<f32> {
    let mut resized = ImageBuffer::new(new_width, new_height);
    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = x as f32 * (img.width as f32 / new_width as f32);
            let src_y = y as f32 * (img.height as f32 / new_height as f32);
            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(img.width - 1);
            let y1 = (y0 + 1).min(img.height - 1);
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;
            if let (Some(c00), Some(c10), Some(c01), Some(c11)) = (
                img.get_pixel(x0, y0), img.get_pixel(x1, y0),
                img.get_pixel(x0, y1), img.get_pixel(x1, y1),
            ) {
                let r = (1.0-fx)*(1.0-fy)*c00.r + fx*(1.0-fy)*c10.r + (1.0-fx)*fy*c01.r + fx*fy*c11.r;
                let g = (1.0-fx)*(1.0-fy)*c00.g + fx*(1.0-fy)*c10.g + (1.0-fx)*fy*c01.g + fx*fy*c11.g;
                let b = (1.0-fx)*(1.0-fy)*c00.b + fx*(1.0-fy)*c10.b + (1.0-fx)*fy*c01.b + fx*fy*c11.b;
                resized.set_pixel(x, y, lgi_math::color::Color4::new(r, g, b, 1.0));
            }
        }
    }
    resized
}
