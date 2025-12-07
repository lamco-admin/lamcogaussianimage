//! Comprehensive Optimizer Comparison Experiment
//!
//! Tests all optimizer variants on the same image and saves results
//! using the TestResult framework for reproducible experiments.
//!
//! Run: cargo run --release --example comprehensive_optimizer_comparison -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{
    EncoderV2,
    renderer_v2::RendererV2,
    optimizer_v2::OptimizerV2,
    adam_optimizer::AdamOptimizer,
    test_results::{TestResult, save_rendered_image},
};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     COMPREHENSIVE OPTIMIZER COMPARISON EXPERIMENT            ║");
    println!("║     Testing all optimizer variants with saved results        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = std::env::args().nth(1)
        .unwrap_or_else(|| "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png".to_string());

    println!("Loading: {}", image_path);
    let target = match ImageBuffer::load(&image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            return;
        }
    };
    let (width, height) = (target.width, target.height);
    println!("Dimensions: {}x{}\n", width, height);

    // Test configurations
    let grid_size = 16;
    let n_gaussians = grid_size * grid_size;
    let iterations = 100;

    println!("═══════════════════════════════════════════════════════════════");
    println!("Test Configuration:");
    println!("  Grid: {}x{} = {} Gaussians", grid_size, grid_size, n_gaussians);
    println!("  Max iterations: {}", iterations);
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut all_results = Vec::new();

    // Test 1: Adam optimizer (baseline - known to work)
    println!("\n[1/5] Testing ADAM Optimizer...");
    let result = test_adam(&target, &image_path, grid_size, iterations);
    all_results.push(result);

    // Test 2: OptimizerV2 with default config
    println!("\n[2/5] Testing OptimizerV2 (default)...");
    let result = test_v2_default(&target, &image_path, grid_size, iterations);
    all_results.push(result);

    // Test 3: OptimizerV2 with MS-SSIM loss
    println!("\n[3/5] Testing OptimizerV2 (MS-SSIM loss)...");
    let result = test_v2_msssim(&target, &image_path, grid_size, iterations);
    all_results.push(result);

    // Test 4: OptimizerV2 with edge-weighted loss
    println!("\n[4/5] Testing OptimizerV2 (edge-weighted)...");
    let result = test_v2_edge_weighted(&target, &image_path, grid_size, iterations);
    all_results.push(result);

    // Test 5: OptimizerV2 without rotation (simpler like Adam)
    println!("\n[5/5] Testing OptimizerV2 (no rotation LR)...");
    let result = test_v2_no_rotation(&target, &image_path, grid_size, iterations);
    all_results.push(result);

    // Print summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                      RESULTS SUMMARY                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("{:<35} {:>8} {:>8} {:>10} {:>8}",
             "Optimizer", "Init", "Final", "Improve", "Time");
    println!("{}", "-".repeat(75));

    for result in &all_results {
        println!("{:<35} {:>7.2} {:>7.2} {:>+9.2} {:>7.1}s",
                 result.optimizer,
                 result.initial_psnr,
                 result.final_psnr,
                 result.improvement_db,
                 result.elapsed_seconds);
    }
    println!("{}", "-".repeat(75));

    // Find best
    if let Some(best) = all_results.iter().max_by(|a, b|
        a.improvement_db.partial_cmp(&b.improvement_db).unwrap())
    {
        println!("\n🏆 BEST: {} with +{:.2} dB improvement",
                 best.optimizer, best.improvement_db);
    }

    println!("\n✅ All results saved to {}/", RESULTS_DIR);
}

fn create_gaussians(target: &ImageBuffer<f32>, grid_size: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let n_gaussians = grid_size * grid_size;
    let mut gaussians = Vec::with_capacity(n_gaussians);

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

fn compute_metrics(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> (f32, f32) {
    let mut sum = 0.0;
    let count = (rendered.width * rendered.height * 3) as f32;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        sum += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    let mse = sum / count;
    let psnr = if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() };
    (mse, psnr)
}

fn test_adam(target: &ImageBuffer<f32>, image_path: &str, grid_size: usize, iterations: usize) -> TestResult {
    let mut gaussians = create_gaussians(target, grid_size);
    let (width, height) = (target.width, target.height);

    let init_render = RendererV2::render(&gaussians, width, height);
    let (_, init_psnr) = compute_metrics(&init_render, target);

    let mut adam = AdamOptimizer::new();
    adam.max_iterations = iterations;

    let start = Instant::now();
    let _ = adam.optimize(&mut gaussians, target);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(&gaussians, width, height);
    let (final_mse, final_psnr) = compute_metrics(&final_render, target);

    println!("  Initial: {:.2} dB → Final: {:.2} dB ({:+.2} dB)", init_psnr, final_psnr, final_psnr - init_psnr);

    // Save result
    let mut result = TestResult::new("optimizer_comparison_adam", image_path);
    result.optimizer = "Adam (per-param LR)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = grid_size * grid_size;
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.final_loss = final_mse;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.extra_metrics = Some(json!({
        "lr_position_initial": 0.0002,
        "lr_color": 0.02,
        "lr_scale": 0.005
    }));

    let _ = result.save(RESULTS_DIR);
    let _ = save_rendered_image(&final_render, RESULTS_DIR, "adam", "final");

    result
}

fn test_v2_default(target: &ImageBuffer<f32>, image_path: &str, grid_size: usize, iterations: usize) -> TestResult {
    let mut gaussians = create_gaussians(target, grid_size);
    let (width, height) = (target.width, target.height);

    let init_render = RendererV2::render(&gaussians, width, height);
    let (_, init_psnr) = compute_metrics(&init_render, target);

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = iterations;

    let start = Instant::now();
    let _ = optimizer.optimize(&mut gaussians, target);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(&gaussians, width, height);
    let (final_mse, final_psnr) = compute_metrics(&final_render, target);

    println!("  Initial: {:.2} dB → Final: {:.2} dB ({:+.2} dB)", init_psnr, final_psnr, final_psnr - init_psnr);

    let mut result = TestResult::new("optimizer_comparison_v2_default", image_path);
    result.optimizer = "OptimizerV2 (default L2)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = grid_size * grid_size;
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.final_loss = final_mse;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.extra_metrics = Some(json!({
        "lr_position": optimizer.learning_rate_position,
        "lr_color": optimizer.learning_rate_color,
        "lr_scale": optimizer.learning_rate_scale,
        "lr_rotation": optimizer.learning_rate_rotation,
        "use_ms_ssim": false,
        "use_edge_weighted": false
    }));

    let _ = result.save(RESULTS_DIR);
    let _ = save_rendered_image(&final_render, RESULTS_DIR, "v2_default", "final");

    result
}

fn test_v2_msssim(target: &ImageBuffer<f32>, image_path: &str, grid_size: usize, iterations: usize) -> TestResult {
    let mut gaussians = create_gaussians(target, grid_size);
    let (width, height) = (target.width, target.height);

    let init_render = RendererV2::render(&gaussians, width, height);
    let (_, init_psnr) = compute_metrics(&init_render, target);

    let mut optimizer = OptimizerV2::new_with_ms_ssim();
    optimizer.max_iterations = iterations;

    let start = Instant::now();
    let _ = optimizer.optimize(&mut gaussians, target);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(&gaussians, width, height);
    let (final_mse, final_psnr) = compute_metrics(&final_render, target);

    println!("  Initial: {:.2} dB → Final: {:.2} dB ({:+.2} dB)", init_psnr, final_psnr, final_psnr - init_psnr);

    let mut result = TestResult::new("optimizer_comparison_v2_msssim", image_path);
    result.optimizer = "OptimizerV2 (MS-SSIM loss)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = grid_size * grid_size;
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.final_loss = final_mse;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.extra_metrics = Some(json!({
        "use_ms_ssim": true,
        "use_edge_weighted": false,
        "note": "Perceptual loss function"
    }));

    let _ = result.save(RESULTS_DIR);
    let _ = save_rendered_image(&final_render, RESULTS_DIR, "v2_msssim", "final");

    result
}

fn test_v2_edge_weighted(target: &ImageBuffer<f32>, image_path: &str, grid_size: usize, iterations: usize) -> TestResult {
    let mut gaussians = create_gaussians(target, grid_size);
    let (width, height) = (target.width, target.height);

    let init_render = RendererV2::render(&gaussians, width, height);
    let (_, init_psnr) = compute_metrics(&init_render, target);

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = iterations;
    optimizer.use_edge_weighted = true;

    let start = Instant::now();
    let _ = optimizer.optimize(&mut gaussians, target);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(&gaussians, width, height);
    let (final_mse, final_psnr) = compute_metrics(&final_render, target);

    println!("  Initial: {:.2} dB → Final: {:.2} dB ({:+.2} dB)", init_psnr, final_psnr, final_psnr - init_psnr);

    let mut result = TestResult::new("optimizer_comparison_v2_edge", image_path);
    result.optimizer = "OptimizerV2 (edge-weighted)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = grid_size * grid_size;
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.final_loss = final_mse;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.extra_metrics = Some(json!({
        "use_ms_ssim": false,
        "use_edge_weighted": true,
        "note": "Higher weight on edge regions"
    }));

    let _ = result.save(RESULTS_DIR);
    let _ = save_rendered_image(&final_render, RESULTS_DIR, "v2_edge", "final");

    result
}

fn test_v2_no_rotation(target: &ImageBuffer<f32>, image_path: &str, grid_size: usize, iterations: usize) -> TestResult {
    let mut gaussians = create_gaussians(target, grid_size);
    let (width, height) = (target.width, target.height);

    let init_render = RendererV2::render(&gaussians, width, height);
    let (_, init_psnr) = compute_metrics(&init_render, target);

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = iterations;
    optimizer.learning_rate_rotation = 0.0;  // Disable rotation optimization

    let start = Instant::now();
    let _ = optimizer.optimize(&mut gaussians, target);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(&gaussians, width, height);
    let (final_mse, final_psnr) = compute_metrics(&final_render, target);

    println!("  Initial: {:.2} dB → Final: {:.2} dB ({:+.2} dB)", init_psnr, final_psnr, final_psnr - init_psnr);

    let mut result = TestResult::new("optimizer_comparison_v2_no_rot", image_path);
    result.optimizer = "OptimizerV2 (no rotation)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = grid_size * grid_size;
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.final_loss = final_mse;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.extra_metrics = Some(json!({
        "lr_rotation": 0.0,
        "note": "Rotation optimization disabled (simpler like Adam)"
    }));

    let _ = result.save(RESULTS_DIR);
    let _ = save_rendered_image(&final_render, RESULTS_DIR, "v2_no_rot", "final");

    result
}
