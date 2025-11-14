//! Quick photo quality test - minimal optimization

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("Quick Real Photo Test\n");

    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/Lamco Head.jpg";

    println!("Loading {}...", path);
    let target = match ImageBuffer::load(path) {
        Ok(mut img) => {
            println!("Loaded: {}×{}", img.width, img.height);

            // Resize to 256×256 for speed
            if img.width != 256 || img.height != 256 {
                println!("Resizing to 256×256...");
                img = resize_simple(&img, 256, 256);
            }
            img
        }
        Err(e) => {
            println!("Load failed: {}", e);
            return;
        }
    };

    println!("\nTesting N=400:");
    let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
    let mut gaussians = encoder.initialize_gaussians(20);

    let init_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, 256, 256));
    println!("  Init PSNR: {:.2} dB", init_psnr);

    // Quick 20 iteration test
    let mut opt = OptimizerV2::default();
    opt.max_iterations = 20;
    opt.optimize(&mut gaussians, &target);

    let final_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, 256, 256));
    println!("  Final PSNR (20 iters): {:.2} dB", final_psnr);
    println!("  Improvement: {:+.2} dB", final_psnr - init_psnr);

    if final_psnr >= 28.0 {
        println!("\n✅ Photo quality GOOD - foundation works!");
    } else if final_psnr >= 23.0 {
        println!("\n⚠️  Photo quality ACCEPTABLE - may need guided filter");
    } else {
        println!("\n❌ Photo quality POOR - need major improvements");
    }
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
