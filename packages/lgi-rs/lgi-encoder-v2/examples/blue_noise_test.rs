//! Blue-Noise Residual Standalone Test - EXP-3-009
//! Validate blue-noise implementation with parameter sweeps

use lgi_core::{ImageBuffer, blue_noise_residual::BlueNoiseGenerator};
use lgi_encoder_v2::{EncoderV2, optimizer_v2::OptimizerV2, renderer_v2::RendererV2};
use lgi_math::color::Color4;

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Blue-Noise Residual Test - EXP-3-009                   ║");
    println!("║  Parameter sweeps: amplitude × frequency                 ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Load test photo
    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133784383569199567.jpg";
    let target = match ImageBuffer::load(path) {
        Ok(img) => resize_bilinear(&img, 768, 432),
        Err(e) => {
            println!("Failed to load: {}", e);
            return;
        }
    };

    println!("Target: {}×{}\n", target.width, target.height);

    // Initialize and optimize base Gaussians
    let encoder = EncoderV2::new(target.clone()).unwrap();
    let mut gaussians = encoder.initialize_gaussians_guided(40);  // N=1600

    println!("Optimizing {} Gaussians (50 iterations)...", gaussians.len());
    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 50;
    optimizer.optimize(&mut gaussians, &target);

    let rendered_base = RendererV2::render(&gaussians, target.width, target.height);
    let psnr_base = compute_psnr(&rendered_base, &target);

    println!("  Baseline PSNR: {:.2} dB\n", psnr_base);

    // Parameter sweep
    let amplitudes = vec![0.01, 0.02, 0.05, 0.1];
    let frequencies = vec![0.5, 1.0, 2.0, 4.0];

    println!("═══════════════════════════════════════════");
    println!("PARAMETER SWEEP");
    println!("═══════════════════════════════════════════\n");

    let mut best_psnr = psnr_base;
    let mut best_amp = 0.0;
    let mut best_freq = 0.0;

    for &amplitude in &amplitudes {
        for &frequency in &frequencies {
            // Generate blue-noise
            let blue_noise = BlueNoiseGenerator::generate(
                target.width,
                target.height,
                amplitude,
                frequency * 10.0,  // Scale frequency
                42,  // seed
            );

            // Apply to rendered image
            let mut rendered_with_noise = rendered_base.clone();
            blue_noise.apply(&mut rendered_with_noise);

            let psnr = compute_psnr(&rendered_with_noise, &target);
            let delta = psnr - psnr_base;

            print!("  amp={:.2}, freq={:.1}: {:.2} dB ({:+.2} dB)",
                   amplitude, frequency, psnr, delta);

            if delta > 0.0 {
                println!(" ✅");
                if psnr > best_psnr {
                    best_psnr = psnr;
                    best_amp = amplitude;
                    best_freq = frequency;
                }
            } else {
                println!(" ❌");
            }
        }
    }

    println!("\n═══════════════════════════════════════════");
    println!("RESULTS");
    println!("═══════════════════════════════════════════");
    println!("  Baseline:     {:.2} dB", psnr_base);
    println!("  Best:         {:.2} dB (amp={:.2}, freq={:.1})", best_psnr, best_amp, best_freq);
    println!("  Improvement:  {:+.2} dB", best_psnr - psnr_base);

    if best_psnr > psnr_base {
        println!("\n  ✅ Blue-noise HELPS quality!");
    } else {
        println!("\n  ❌ Blue-noise does NOT help (all configs worse)");
    }
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
                resized.set_pixel(x, y, Color4::new(r, g, b, 1.0));
            }
        }
    }
    resized
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= (rendered.width * rendered.height * 3) as f32;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
