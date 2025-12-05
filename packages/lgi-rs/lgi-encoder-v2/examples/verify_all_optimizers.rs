//! Verify ALL optimizers are working
//!
//! Quick sanity check that each optimizer can run and improve PSNR.
//! Reports status: WORKING / BROKEN / SLOW / ERROR
//!
//! Run: cargo run --release --example verify_all_optimizers -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::optimizer_v2::OptimizerV2;
use lgi_encoder_v2::lm_optimizer::{LMOptimizer, LMConfig};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   ALL OPTIMIZER VERIFICATION                                 ║");
    println!("║   Quick sanity check - each optimizer must improve PSNR      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load");
    let (width, height) = (target.width, target.height);

    // Small problem for quick testing
    let grid_size = 8;  // 64 Gaussians for speed
    let iterations = 30;

    println!("Quick test: {}x{} image, {} Gaussians, {} iterations\n",
             width, height, grid_size * grid_size, iterations);

    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<20} {:>10} {:>10} {:>10} {:>8} {:>10}",
             "Optimizer", "Init", "Final", "Improve", "Time", "Status");
    println!("{:-<20} {:-<10} {:-<10} {:-<10} {:-<8} {:-<10}", "", "", "", "", "", "");

    // Test each optimizer
    test_adam(&target, grid_size, iterations, image_path);
    test_v2(&target, grid_size, iterations, image_path);
    test_lm(&target, grid_size, iterations, image_path);
    // test_hybrid(&target, grid_size, iterations, image_path);  // Too slow for quick test

    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Status meanings:");
    println!("  WORKING - Improves PSNR by >= 0.3 dB");
    println!("  SLOW    - Improves but slowly (< 0.3 dB)");
    println!("  BROKEN  - Loss increases or no improvement");
    println!("  ERROR   - Crashed or failed to run");
}

fn test_adam(target: &ImageBuffer<f32>, grid_size: usize, iterations: usize, image_path: &str) {
    let name = "Adam";
    let (width, height) = (target.width, target.height);
    let mut gaussians = create_gaussians(target, grid_size);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);

    let start = Instant::now();
    let mut adam = AdamOptimizer::new();
    adam.max_iterations = iterations;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        adam.optimize(&mut gaussians, target)
    }));

    match result {
        Ok(final_loss) => {
            let elapsed = start.elapsed();
            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = compute_psnr(&final_render, target);
            let improvement = final_psnr - init_psnr;

            let status = if improvement >= 0.3 { "WORKING" }
                        else if improvement > 0.0 { "SLOW" }
                        else { "BROKEN" };

            println!("{:<20} {:>10.2} {:>10.2} {:>+10.2} {:>7.1}s {:>10}",
                     name, init_psnr, final_psnr, improvement, elapsed.as_secs_f32(), status);

            save_result(name, image_path, (width, height), grid_size * grid_size, iterations,
                       init_psnr, final_psnr, improvement, final_loss, elapsed.as_secs_f32(), status);
        }
        Err(_) => {
            println!("{:<20} {:>10.2} {:>10} {:>10} {:>8} {:>10}",
                     name, init_psnr, "-", "-", "-", "ERROR");
        }
    }
}

fn test_v2(target: &ImageBuffer<f32>, grid_size: usize, iterations: usize, image_path: &str) {
    let name = "OptimizerV2";
    let (width, height) = (target.width, target.height);
    let mut gaussians = create_gaussians(target, grid_size);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);

    let start = Instant::now();
    let mut v2 = OptimizerV2::default();
    v2.max_iterations = iterations;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        v2.optimize(&mut gaussians, target)
    }));

    match result {
        Ok(final_loss) => {
            let elapsed = start.elapsed();
            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = compute_psnr(&final_render, target);
            let improvement = final_psnr - init_psnr;

            let status = if improvement >= 0.3 { "WORKING" }
                        else if improvement > 0.0 { "SLOW" }
                        else { "BROKEN" };

            println!("{:<20} {:>10.2} {:>10.2} {:>+10.2} {:>7.1}s {:>10}",
                     name, init_psnr, final_psnr, improvement, elapsed.as_secs_f32(), status);

            save_result(name, image_path, (width, height), grid_size * grid_size, iterations,
                       init_psnr, final_psnr, improvement, final_loss, elapsed.as_secs_f32(), status);
        }
        Err(_) => {
            println!("{:<20} {:>10.2} {:>10} {:>10} {:>8} {:>10}",
                     name, init_psnr, "-", "-", "-", "ERROR");
        }
    }
}

fn test_lm(target: &ImageBuffer<f32>, grid_size: usize, iterations: usize, image_path: &str) {
    let name = "L-M";
    let (width, height) = (target.width, target.height);
    let mut gaussians = create_gaussians(target, grid_size);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);

    let start = Instant::now();
    let mut config = LMConfig::default();
    config.max_iterations = iterations;
    let mut lm = LMOptimizer::new(config);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lm.optimize(&mut gaussians, target)
    }));

    match result {
        Ok(final_loss) => {
            let elapsed = start.elapsed();
            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = compute_psnr(&final_render, target);
            let improvement = final_psnr - init_psnr;

            let status = if improvement >= 0.3 { "WORKING" }
                        else if improvement > 0.0 { "SLOW" }
                        else { "BROKEN" };

            println!("{:<20} {:>10.2} {:>10.2} {:>+10.2} {:>7.1}s {:>10}",
                     name, init_psnr, final_psnr, improvement, elapsed.as_secs_f32(), status);

            save_result(name, image_path, (width, height), grid_size * grid_size, iterations,
                       init_psnr, final_psnr, improvement, final_loss, elapsed.as_secs_f32(), status);
        }
        Err(_) => {
            println!("{:<20} {:>10.2} {:>10} {:>10} {:>8} {:>10}",
                     name, init_psnr, "-", "-", "-", "ERROR");
        }
    }
}

fn create_gaussians(target: &ImageBuffer<f32>, grid_size: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(grid_size * grid_size);
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
    gaussians
}

fn save_result(name: &str, image_path: &str, dims: (u32, u32), n_gaussians: usize,
               iterations: usize, init_psnr: f32, final_psnr: f32, improvement: f32,
               final_loss: f32, elapsed: f32, status: &str) {
    let exp_name = format!("verify_all_{}", name.to_lowercase().replace(" ", "_").replace("-", "_"));
    let mut result = TestResult::new(&exp_name, image_path);
    result.optimizer = name.to_string();
    result.image_dimensions = dims;
    result.n_gaussians = n_gaussians;
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.final_loss = final_loss;
    result.elapsed_seconds = elapsed;
    result.extra_metrics = Some(json!({
        "status": status,
        "quick_test": true,
    }));
    let _ = result.save(RESULTS_DIR);
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height * 3) as f32;
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= count;
    if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}
