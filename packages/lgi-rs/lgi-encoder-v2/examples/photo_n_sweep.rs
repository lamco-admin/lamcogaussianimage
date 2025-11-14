//! EXP-030: Photo N Sweep - Find required Gaussian count

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("EXP-030: Photo Gaussian Count Sweep\n");

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

    println!("Photo N sweep (256×256):\n");

    // Test different N values, but limit iterations for speed
    for &grid_size in &[16, 20, 24, 28, 32] {
        let n = grid_size * grid_size;

        let encoder = EncoderV2::new(target.clone()).unwrap();
        let mut gaussians = encoder.initialize_gaussians_guided(grid_size);  // Use guided filter

        let init_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, 256, 256));

        let mut opt = OptimizerV2::default();
        opt.max_iterations = 30;  // Quick test
        opt.optimize(&mut gaussians, &target);

        let final_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, 256, 256));

        println!("N={:4}: Init {:.2} dB → Final {:.2} dB (Δ: {:+.2} dB)",
            n, init_psnr, final_psnr, final_psnr - init_psnr);
    }

    println!("\nLooking for: N where quality saturates or reaches 25+ dB");
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
