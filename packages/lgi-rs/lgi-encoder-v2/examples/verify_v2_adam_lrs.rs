//! Test OptimizerV2 with Adam-like learning rates
//!
//! OptimizerV2 default LRs (0.1, 0.6, 0.1) FAILED.
//! Testing with Adam-like LRs (0.0002, 0.02, 0.005).

use lgi_core::ImageBuffer;
use lgi_encoder_v2::optimizer_v2::OptimizerV2;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("=== TEST: OptimizerV2 with Adam-like LRs ===\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let img = image::open(image_path).expect("Failed to load");
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let mut target = ImageBuffer::new(width, height);
    for (x, y, pixel) in rgb.enumerate_pixels() {
        target.set_pixel(x, y, Color4::new(
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
            1.0,
        ));
    }

    let grid_size = 16;
    let n_gaussians = grid_size * grid_size;

    let mut gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = Vec::with_capacity(n_gaussians);
    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx as f32 + 0.5) / grid_size as f32;
            let y = (gy as f32 + 0.5) / grid_size as f32;
            let px = ((x * width as f32) as u32).min(width - 1);
            let py = ((y * height as f32) as u32).min(height - 1);
            let color = target.get_pixel(px, py).unwrap();
            let scale = 1.0 / grid_size as f32;
            gaussians.push(Gaussian2D::new(
                Vector2::new(x, y),
                Euler::new(scale, scale, 0.0),
                color,
                1.0,
            ));
        }
    }

    let initial_render = RendererV2::render(&gaussians, width, height);
    let initial_loss = compute_mse(&initial_render, &target);
    let initial_psnr = mse_to_psnr(initial_loss);

    println!("Initial: loss={:.6}, PSNR={:.2} dB", initial_loss, initial_psnr);

    // Test 1: Adam-like LRs
    println!("\n--- Test 1: Adam-like LRs (0.0002, 0.02, 0.005) ---");
    let mut gaussians_t1 = gaussians.clone();
    let mut opt1 = OptimizerV2::default();
    opt1.learning_rate_position = 0.0002;
    opt1.learning_rate_color = 0.02;
    opt1.learning_rate_scale = 0.005;
    opt1.max_iterations = 50;

    let start = Instant::now();
    let loss1 = opt1.optimize(&mut gaussians_t1, &target);
    let time1 = start.elapsed().as_secs_f32();
    let psnr1 = mse_to_psnr(loss1);
    println!("Result: loss={:.6}, PSNR={:.2} dB, time={:.1}s, improvement={:+.2} dB",
             loss1, psnr1, time1, psnr1 - initial_psnr);

    // Test 2: 10× higher than Adam
    println!("\n--- Test 2: 10× Adam LRs (0.002, 0.2, 0.05) ---");
    let mut gaussians_t2 = gaussians.clone();
    let mut opt2 = OptimizerV2::default();
    opt2.learning_rate_position = 0.002;
    opt2.learning_rate_color = 0.2;
    opt2.learning_rate_scale = 0.05;
    opt2.max_iterations = 50;

    let start = Instant::now();
    let loss2 = opt2.optimize(&mut gaussians_t2, &target);
    let time2 = start.elapsed().as_secs_f32();
    let psnr2 = mse_to_psnr(loss2);
    println!("Result: loss={:.6}, PSNR={:.2} dB, time={:.1}s, improvement={:+.2} dB",
             loss2, psnr2, time2, psnr2 - initial_psnr);

    // Test 3: Default V2 LRs (known to fail)
    println!("\n--- Test 3: Default V2 LRs (0.1, 0.6, 0.1) ---");
    let mut gaussians_t3 = gaussians.clone();
    let mut opt3 = OptimizerV2::default();
    opt3.max_iterations = 50;

    let start = Instant::now();
    let loss3 = opt3.optimize(&mut gaussians_t3, &target);
    let time3 = start.elapsed().as_secs_f32();
    let psnr3 = mse_to_psnr(loss3);
    println!("Result: loss={:.6}, PSNR={:.2} dB, time={:.1}s, improvement={:+.2} dB",
             loss3, psnr3, time3, psnr3 - initial_psnr);

    // Summary
    println!("\n=== SUMMARY ===");
    println!("Initial PSNR: {:.2} dB", initial_psnr);
    println!("Adam-like LRs:   {:+.2} dB ({})", psnr1 - initial_psnr,
             if psnr1 > initial_psnr { "PASS" } else { "FAIL" });
    println!("10× Adam LRs:    {:+.2} dB ({})", psnr2 - initial_psnr,
             if psnr2 > initial_psnr { "PASS" } else { "FAIL" });
    println!("Default V2 LRs:  {:+.2} dB ({})", psnr3 - initial_psnr,
             if psnr3 > initial_psnr { "PASS" } else { "FAIL" });

    // Save summary result
    let mut result = TestResult::new("v2_lr_sensitivity", image_path);
    result.optimizer = "OptimizerV2 (LR sweep)".to_string();
    result.notes = format!(
        "LR sensitivity test. Adam-like: {:+.2}dB, 10×: {:+.2}dB, Default: {:+.2}dB",
        psnr1 - initial_psnr, psnr2 - initial_psnr, psnr3 - initial_psnr
    );
    result.n_gaussians = n_gaussians;
    result.iterations = 50;
    result.initial_psnr = initial_psnr;
    result.final_psnr = psnr1;  // Best result
    result.improvement_db = psnr1 - initial_psnr;
    let _ = result.save(RESULTS_DIR);
}

fn compute_mse(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut sum = 0.0;
    let count = (rendered.width * rendered.height * 3) as f32;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        sum += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    sum / count
}

fn mse_to_psnr(mse: f32) -> f32 {
    if mse <= 0.0 { return 100.0; }
    10.0 * (1.0 / mse).log10()
}
