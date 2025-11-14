//! EXP-031: Photo Gamma Sweep - Find correct γ for photos

use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use lgi_encoder_v2::{renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("EXP-031: Photo Gamma Sweep\n");

    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/Lamco Head.jpg";

    let target = match ImageBuffer::load(path) {
        Ok(mut img) => {
            if img.width > 256 || img.height > 256 {
                img = resize_simple(&img, 256, 256);
            }
            img
        }
        Err(_) => return,
    };

    println!("Testing different gamma values (N=400, 20×20 grid):\n");

    for &gamma in &[0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0] {
        let gaussians = init_with_gamma(&target, 20, gamma);

        let init_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, 256, 256));

        let mut gaussians_opt = gaussians.clone();
        let mut opt = OptimizerV2::default();
        opt.max_iterations = 30;
        opt.optimize(&mut gaussians_opt, &target);

        let final_psnr = compute_psnr(&target, &RendererV2::render(&gaussians_opt, 256, 256));

        println!("γ={:.1}: Init {:.2} dB → Final {:.2} dB (Δ: {:+.2} dB)",
            gamma, init_psnr, final_psnr, final_psnr - init_psnr);
    }
}

fn init_with_gamma(target: &ImageBuffer<f32>, grid_size: u32, gamma: f32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut gaussians = Vec::new();
    let n = grid_size * grid_size;
    let w = target.width as f32;
    let h = target.height as f32;

    let sigma_px = gamma * ((w * h) / n as f32).sqrt();
    let sigma_norm = (sigma_px / w).clamp(0.01, 0.25);

    let step_x = target.width / grid_size;
    let step_y = target.height / grid_size;

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx * step_x + step_x / 2).min(target.width - 1);
            let y = (gy * step_y + step_y / 2).min(target.height - 1);

            let pos = Vector2::new(x as f32 / w, y as f32 / h);
            let color = target.get_pixel(x, y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            gaussians.push(Gaussian2D::new(pos, Euler::isotropic(sigma_norm), color, 1.0));
        }
    }

    gaussians
}

fn resize_simple(img: &ImageBuffer<f32>, w: u32, h: u32) -> ImageBuffer<f32> {
    let mut out = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sx = (x as f32 / w as f32 * img.width as f32) as u32;
            let sy = (y as f32 / h as f32 * img.height as f32) as u32;
            if let Some(c) = img.get_pixel(sx.min(img.width-1), sy.min(img.height-1)) {
                out.set_pixel(x, y, c);
            }
        }
    }
    out
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
