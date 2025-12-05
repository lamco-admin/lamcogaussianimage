//! VERIFICATION TEST: Does Hybrid optimizer work?
//!
//! Hybrid = Adam warm-start → L-BFGS refinement

use lgi_core::ImageBuffer;
use lgi_encoder_v2::hybrid_optimizer::{HybridOptimizer, HybridConfig};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("=== VERIFICATION: Hybrid Optimizer ===\n");

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
    println!("Image: {}x{}, Gaussians: {}", width, height, n_gaussians);

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

    println!("Initial: loss={:.6}, PSNR={:.2} dB\n", initial_loss, initial_psnr);

    // Configure shorter test (30 Adam + 20 L-BFGS)
    let config = HybridConfig {
        adam_iterations: 30,
        lbfgs_iterations: 20,
        lbfgs_history: 10,
        ..Default::default()
    };

    let mut optimizer = HybridOptimizer::new(config);

    let start = Instant::now();
    let final_loss = optimizer.optimize(&mut gaussians, &target);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(&gaussians, width, height);
    // Recompute with consistent MSE formula
    let final_loss_mse = compute_mse(&final_render, &target);
    let final_psnr = mse_to_psnr(final_loss_mse);

    println!("\n=== RESULTS ===");
    println!("Initial PSNR: {:.2} dB", initial_psnr);
    println!("Final PSNR:   {:.2} dB", final_psnr);
    println!("Improvement:  {:+.2} dB", final_psnr - initial_psnr);
    println!("Time: {:.1}s", elapsed.as_secs_f32());

    let passed = final_psnr > initial_psnr;
    println!("\nVERDICT: {}", if passed { "PASS" } else { "FAIL" });

    // Save result
    let mut result = TestResult::new("verify_hybrid", image_path);
    result.optimizer = "Hybrid (Adam→L-BFGS)".to_string();
    result.n_gaussians = n_gaussians;
    result.iterations = 50; // 30+20
    result.initial_psnr = initial_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - initial_psnr;
    result.final_loss = final_loss_mse;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.notes = format!("Adam(30) + L-BFGS(20). Passed: {}", passed);
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
