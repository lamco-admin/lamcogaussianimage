//! Visual debugging - save original vs rendered to see what's wrong

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("Photo Visual Debug - Saving comparison images\n");

    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/Lamco Head.jpg";

    let target = match ImageBuffer::load(path) {
        Ok(mut img) => {
            if img.width > 256 || img.height > 256 {
                img = resize_simple(&img, 256, 256);
            }
            img
        }
        Err(e) => {
            println!("Load failed: {}", e);
            return;
        }
    };

    println!("Testing with N=400...");
    let encoder = EncoderV2::new(target.clone()).unwrap();
    let mut gaussians = encoder.initialize_gaussians_guided(20);

    let init_rendered = RendererV2::render(&gaussians, 256, 256);
    let init_psnr = compute_psnr(&target, &init_rendered);
    println!("Init PSNR: {:.2} dB", init_psnr);

    // Save init
    if let Err(e) = init_rendered.save("/tmp/photo_init.png") {
        println!("Save failed: {}", e);
    } else {
        println!("Saved: /tmp/photo_init.png");
    }

    // Optimize
    let mut opt = OptimizerV2::default();
    opt.max_iterations = 100;
    opt.optimize(&mut gaussians, &target);

    let final_rendered = RendererV2::render(&gaussians, 256, 256);
    let final_psnr = compute_psnr(&target, &final_rendered);
    println!("Final PSNR: {:.2} dB", final_psnr);

    // Save final
    if let Err(e) = final_rendered.save("/tmp/photo_final.png") {
        println!("Save failed: {}", e);
    } else {
        println!("Saved: /tmp/photo_final.png");
    }

    // Save target for comparison
    if let Err(e) = target.save("/tmp/photo_target.png") {
        println!("Save failed: {}", e);
    } else {
        println!("Saved: /tmp/photo_target.png");
    }

    println!("\nVisual comparison:");
    println!("  Original: /tmp/photo_target.png");
    println!("  Init:     /tmp/photo_init.png");
    println!("  Final:    /tmp/photo_final.png");
    println!("\nCheck visually to identify issues (blur, color shift, missing detail, etc.)");
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
