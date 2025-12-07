//! EXP-039: Max Iterations Tuning
//! Test if photos need more iterations (500, 1000, 2000)

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("EXP-039: Iteration Budget Tuning\n");

    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133784383569199567.jpg";

    let target = match ImageBuffer::load(path) {
        Ok(img) => {
            let scale = 768.0 / img.width.max(img.height) as f32;
            resize_bilinear(&img, (img.width as f32 * scale) as u32, (img.height as f32 * scale) as u32)
        }
        Err(_) => return,
    };

    let encoder = EncoderV2::new(target.clone()).unwrap();
    let gaussians_init = encoder.initialize_gaussians_guided(40); // N=1600

    println!("Testing different iteration budgets (N=1600):\n");

    for &max_iters in &[100, 200, 300, 500, 1000] {
        let mut gaussians = gaussians_init.clone();

        let mut opt = OptimizerV2::default();
        opt.max_iterations = max_iters;

        let loss = opt.optimize(&mut gaussians, &target);
        let psnr = compute_psnr(&target, &RendererV2::render(&gaussians, target.width, target.height));

        println!("Max iters {}: PSNR {:.2} dB, loss {:.6}", max_iters, psnr, loss);
    }
}

fn resize_bilinear(img: &ImageBuffer<f32>, w: u32, h: u32) -> ImageBuffer<f32> {
    let mut out = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sx = x as f32 * (img.width as f32 / w as f32);
            let sy = y as f32 * (img.height as f32 / h as f32);
            let x0 = sx.floor() as u32;
            let y0 = sy.floor() as u32;
            let x1 = (x0 + 1).min(img.width - 1);
            let y1 = (y0 + 1).min(img.height - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            if let (Some(c00), Some(c10), Some(c01), Some(c11)) = (
                img.get_pixel(x0, y0), img.get_pixel(x1, y0),
                img.get_pixel(x0, y1), img.get_pixel(x1, y1),
            ) {
                let r = (1.0-fx)*(1.0-fy)*c00.r + fx*(1.0-fy)*c10.r + (1.0-fx)*fy*c01.r + fx*fy*c11.r;
                let g = (1.0-fx)*(1.0-fy)*c00.g + fx*(1.0-fy)*c10.g + (1.0-fx)*fy*c01.g + fx*fy*c11.g;
                let b = (1.0-fx)*(1.0-fy)*c00.b + fx*(1.0-fy)*c10.b + (1.0-fx)*fy*c01.b + fx*fy*c11.b;
                out.set_pixel(x, y, lgi_math::color::Color4::new(r, g, b, 1.0));
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
