//! EXP-038: Gamma Tuning for Photos at N=1600
//! Find optimal gamma to maximize photo quality

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_encoder_v2::{renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("EXP-038: Gamma Tuning for Photos (N=1600)\n");

    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133784383569199567.jpg";

    let target = match ImageBuffer::load(path) {
        Ok(img) => {
            let scale = 768.0 / img.width.max(img.height) as f32;
            let new_w = (img.width as f32 * scale) as u32;
            let new_h = (img.height as f32 * scale) as u32;
            resize_bilinear(&img, new_w, new_h)
        }
        Err(_) => return,
    };

    println!("Testing γ values at N=1600 (40×40 grid):\n");

    for &gamma in &[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5] {
        let gaussians = init_with_gamma(&target, 40, gamma);

        let init_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, target.width, target.height));

        let mut gaussians_opt = gaussians.clone();
        let mut opt = OptimizerV2::default();
        opt.max_iterations = 200;
        opt.optimize(&mut gaussians_opt, &target);

        let final_psnr = compute_psnr(&target, &RendererV2::render(&gaussians_opt, target.width, target.height));

        println!("γ={:.1}: Init {:.2} dB → Final {:.2} dB (Δ: {:+.2} dB)",
            gamma, init_psnr, final_psnr, final_psnr - init_psnr);
    }

    println!("\nLooking for: γ that maximizes final PSNR");
}

fn init_with_gamma(target: &ImageBuffer<f32>, grid_size: u32, gamma: f32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut gaussians = Vec::new();
    let n = grid_size * grid_size;
    let w = target.width as f32;
    let h = target.height as f32;

    let sigma_px = gamma * ((w * h) / n as f32).sqrt();
    let sigma_norm_x = (sigma_px / w).clamp(0.01, 0.25);
    let sigma_norm_y = (sigma_px / h).clamp(0.01, 0.25);

    let step_x = target.width / grid_size;
    let step_y = target.height / grid_size;

    // Use guided filter for photo colors
    let filter = lgi_core::guided_filter::GuidedFilter::default();
    let filtered = filter.filter(target);

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx * step_x + step_x / 2).min(target.width - 1);
            let y = (gy * step_y + step_y / 2).min(target.height - 1);

            let pos = Vector2::new(x as f32 / w, y as f32 / h);
            let color = filtered.get_pixel(x, y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            gaussians.push(Gaussian2D::new(pos, Euler::new(sigma_norm_x, sigma_norm_y, 0.0), color, 1.0));
        }
    }

    gaussians
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
                img.get_pixel(x0, y0),
                img.get_pixel(x1, y0),
                img.get_pixel(x0, y1),
                img.get_pixel(x1, y1),
            ) {
                let r = (1.0 - fx) * (1.0 - fy) * c00.r + fx * (1.0 - fy) * c10.r +
                        (1.0 - fx) * fy * c01.r + fx * fy * c11.r;
                let g = (1.0 - fx) * (1.0 - fy) * c00.g + fx * (1.0 - fy) * c10.g +
                        (1.0 - fx) * fy * c01.g + fx * fy * c11.g;
                let b = (1.0 - fx) * (1.0 - fy) * c00.b + fx * (1.0 - fy) * c10.b +
                        (1.0 - fx) * fy * c01.b + fx * fy * c11.b;

                resized.set_pixel(x, y, Color4::new(r, g, b, 1.0));
            }
        }
    }

    resized
}

fn compute_psnr(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (original.width * original.height * 3) as f32;
    for (p1, p2) in original.data.iter().zip(rendered.data.iter()) {
        mse += (p1.r - p2.r).powi(2) + (p1.g - p2.g).powi(2) + (p1.b - p2.b).powi(2);
    }
    mse /= count;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
