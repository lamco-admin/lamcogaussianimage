//! Simple N Sweep - No Error-Driven, Just Test Different N
//!
//! Disable ALL complexity:
//! - No error-driven growth
//! - No warmup
//! - No geodesic clamping during optimization
//! - Just: Initialize N Gaussians → Optimize → Measure
//!
//! Tests: N = 50, 100, 200, 500, 1000

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};
use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler};
use std::path::PathBuf;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("         SIMPLE N SWEEP - No Error-Driven Complexity           ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let image = load_test_image();
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    println!("Test: kodim02.png (768×512)");
    println!("Method: Initialize N Gaussians → Optimize 100 iters → Measure\n");

    let n_values = vec![50, 100, 200, 500, 1000];

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║ N     │ Init PSNR │ Final PSNR │ Gain    │ Time            ║");
    println!("╠════════════════════════════════════════════════════════════╣");

    for n in n_values {
        let grid_size = (n as f32).sqrt().ceil() as u32;

        // Initialize
        let mut gaussians = encoder.initialize_gaussians(grid_size);
        let init_psnr = compute_psnr(&image, &gaussians);

        // Optimize (simple, no growth)
        let start = std::time::Instant::now();
        let mut opt = OptimizerV2::default();
        opt.max_iterations = 100;
        let _loss = opt.optimize(&mut gaussians, &image);
        let time = start.elapsed();

        let final_psnr = compute_psnr(&image, &gaussians);
        let gain = final_psnr - init_psnr;

        let status = if gain > 0.3 { "✅" } else if gain > -0.5 { "⚠️ " } else { "❌" };

        println!("║ {:5} │   {:6.2}  │   {:7.2}  │ {:+6.2}  │ {:?} {} ║",
            n, init_psnr, final_psnr, gain, time, status);
    }

    println!("╚════════════════════════════════════════════════════════════╝");

    println!("\n📊 ANALYSIS:");
    println!("  If ALL show positive gain:");
    println!("    → Optimizer works fine, problem is error-driven growth");
    println!("  If gain decreases with larger N:");
    println!("    → Larger N makes optimization harder (need lower LR?)");
    println!("  If any show negative gain:");
    println!("    → Optimizer fundamentally broken at that N");
}

fn compute_psnr(target: &ImageBuffer<f32>, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> f32 {
    let rendered = RendererV2::render(gaussians, target.width, target.height);
    let mut mse = 0.0f32;
    let pixel_count = target.width * target.height;

    for y in 0..target.height {
        for x in 0..target.width {
            let t = target.get_pixel(x, y).unwrap();
            let r = rendered.get_pixel(x, y).unwrap();
            mse += (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
        }
    }

    mse /= (pixel_count * 3) as f32;
    if mse < 1e-10 { 100.0 } else { -10.0 * mse.log10() }
}

fn load_test_image() -> ImageBuffer<f32> {
    let path = PathBuf::from("../../kodak-dataset/kodim02.png");
    let img = image::open(&path).expect("Failed to load");
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut buffer = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            buffer.set_pixel(x, y, Color4::new(
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
                1.0
            ));
        }
    }
    buffer
}
