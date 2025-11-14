//! EXP-029: Guided Filter Impact on Real Photos
//! Compare direct pixel sampling vs guided filter

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-029: Guided Filter Impact Test         ║");
    println!("║  Research: +1-3 dB expected                  ║");
    println!("╚══════════════════════════════════════════════╝\n");

    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/Lamco Head.jpg";

    println!("Loading {}...", path.split('/').last().unwrap_or(path));
    let target = match ImageBuffer::load(path) {
        Ok(mut img) => {
            println!("Loaded: {}×{}", img.width, img.height);
            if img.width > 256 || img.height > 256 {
                img = resize_simple(&img, 256, 256);
                println!("Resized to 256×256");
            }
            img
        }
        Err(e) => {
            println!("Failed: {}", e);
            return;
        }
    };

    println!("\nTest 1: Direct Pixel Sampling (baseline)");
    println!("═══════════════════════════════════════");
    test_method(&target, 20, false, "Direct");

    println!("\nTest 2: Guided Filter Sampling");
    println!("═══════════════════════════════════════");
    test_method(&target, 20, true, "Guided");
}

fn test_method(target: &ImageBuffer<f32>, grid_size: u32, use_guided: bool, name: &str) {
    let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");

    let mut gaussians = if use_guided {
        encoder.initialize_gaussians_guided(grid_size)
    } else {
        encoder.initialize_gaussians(grid_size)
    };

    let init_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, target.width, target.height));
    println!("  Init PSNR: {:.2} dB", init_psnr);

    let mut opt = OptimizerV2::default();
    opt.max_iterations = 50;
    opt.optimize(&mut gaussians, &target);

    let final_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, target.width, target.height));
    println!("  Final PSNR (50 iters): {:.2} dB", final_psnr);
    println!("  Improvement: {:+.2} dB", final_psnr - init_psnr);
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
