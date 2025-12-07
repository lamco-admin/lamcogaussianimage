//! Test ErrorDrivenAdamEncoder vs original ErrorDrivenEncoder
//!
//! Compares:
//! 1. Original ErrorDrivenEncoder (OptimizerV2) - baseline
//! 2. ErrorDrivenAdamEncoder (Adam) - improved
//! 3. Uniform Grid + Adam - reference
//!
//! Hypothesis: Adam-based encoder should achieve:
//! - Similar or better PSNR than original
//! - 2-5× faster convergence
//! - Better handling of new Gaussian integration
//!
//! Run: cargo run --release --example test_error_driven_adam -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::error_driven::ErrorDrivenEncoder;
use lgi_encoder_v2::error_driven_adam::{ErrorDrivenAdamConfig, ErrorDrivenAdamEncoder};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use serde_json::json;
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("{}",
        "╔══════════════════════════════════════════════════════════════╗");
    println!("{}",
        "║   ERROR-DRIVEN ENCODER: ADAM vs OptimizerV2 COMPARISON       ║");
    println!("{}",
        "╚══════════════════════════════════════════════════════════════╝\n");

    let image_path =
        "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}×{}", width, height);
    println!();

    let mut results = Vec::new();

    // Test 1: Uniform Grid + Adam (Reference Baseline)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Uniform Grid + Adam (Reference Baseline)");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let start = Instant::now();
        let mut gaussians = create_uniform_grid(&target, 256);
        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);
        println!("  Initial: {} Gaussians, PSNR={:.2}dB", gaussians.len(), init_psnr);

        let mut adam = AdamOptimizer::new();
        adam.max_iterations = 150;

        let mut last_psnr = init_psnr;
        for iter in 0..150 {
            adam.max_iterations = 1;
            adam.optimize(&mut gaussians, &target);

            if iter % 30 == 0 || iter == 149 {
                let rendered = RendererV2::render(&gaussians, width, height);
                let psnr = compute_psnr(&rendered, &target);
                let direction = if psnr > last_psnr { "↑" } else { "↓" };
                println!(
                    "    [{:3}/150] PSNR={:.2}dB {} ({:+.3}dB)",
                    iter, psnr, direction, psnr - last_psnr
                );
                last_psnr = psnr;
            }
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);

        println!("  Final: {:.2}dB (N={}), {:.1}s", final_psnr, gaussians.len(), elapsed.as_secs_f32());

        results.push(("Uniform+Adam(256)", 256, gaussians.len(), final_psnr, elapsed.as_secs_f32()));

        let mut result = TestResult::new("error_driven_adam_baseline", image_path);
        result.optimizer = "Uniform Grid + Adam".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = gaussians.len();
        result.iterations = 150;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({"test_type": "uniform_adam_baseline"}));
        let _ = result.save(RESULTS_DIR);
    }

    // Test 2: Original ErrorDrivenEncoder (OptimizerV2)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: Original ErrorDrivenEncoder (OptimizerV2)");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let start = Instant::now();
        let encoder = ErrorDrivenEncoder {
            target_error: 0.001,
            initial_gaussians: 64,
            max_gaussians: 256,
            split_percentile: 0.10,
        };

        let gaussians = encoder.encode(&target);
        let elapsed = start.elapsed();

        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);

        println!("  Final: {:.2}dB (N={}), {:.1}s", final_psnr, gaussians.len(), elapsed.as_secs_f32());

        results.push(("ErrorDriven(V2)", 64, gaussians.len(), final_psnr, elapsed.as_secs_f32()));

        let mut result = TestResult::new("error_driven_v2_original", image_path);
        result.optimizer = "ErrorDrivenEncoder (OptimizerV2)".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = gaussians.len();
        result.final_psnr = final_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({
            "test_type": "error_driven_original",
            "initial_gaussians": 64,
            "max_gaussians": 256,
            "optimizer": "OptimizerV2"
        }));
        let _ = result.save(RESULTS_DIR);
    }

    // Test 3: ErrorDrivenAdamEncoder (NEW)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 3: ErrorDrivenAdamEncoder (Adam-based)");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let start = Instant::now();
        let config = ErrorDrivenAdamConfig {
            target_error: 0.001,
            initial_gaussians: 64,
            max_gaussians: 256,
            split_percentile: 0.05,
            iterations_per_pass: 50,
            min_improvement_db: 0.05,
            max_passes: 8,
            min_gaussian_spacing: 0.02,
            ..Default::default()
        };

        let encoder = ErrorDrivenAdamEncoder::new(config.clone());
        let gaussians = encoder.encode(&target);
        let elapsed = start.elapsed();

        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);

        println!("  Final: {:.2}dB (N={}), {:.1}s", final_psnr, gaussians.len(), elapsed.as_secs_f32());

        results.push(("ErrorDriven(Adam)", 64, gaussians.len(), final_psnr, elapsed.as_secs_f32()));

        let mut result = TestResult::new("error_driven_adam_new", image_path);
        result.optimizer = "ErrorDrivenAdamEncoder".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = gaussians.len();
        result.final_psnr = final_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({
            "test_type": "error_driven_adam",
            "initial_gaussians": config.initial_gaussians,
            "max_gaussians": config.max_gaussians,
            "iterations_per_pass": config.iterations_per_pass,
            "min_improvement_db": config.min_improvement_db,
            "optimizer": "Adam"
        }));
        let _ = result.save(RESULTS_DIR);
    }

    // Test 4: ErrorDrivenAdamEncoder with larger budget
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 4: ErrorDrivenAdamEncoder (64→512, larger budget)");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let start = Instant::now();
        let config = ErrorDrivenAdamConfig {
            target_error: 0.001,
            initial_gaussians: 64,
            max_gaussians: 512,
            split_percentile: 0.05,
            iterations_per_pass: 50,
            min_improvement_db: 0.03,  // Lower threshold for more passes
            max_passes: 12,
            min_gaussian_spacing: 0.015,  // Allow slightly denser placement
            ..Default::default()
        };

        let encoder = ErrorDrivenAdamEncoder::new(config.clone());
        let gaussians = encoder.encode(&target);
        let elapsed = start.elapsed();

        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);

        println!("  Final: {:.2}dB (N={}), {:.1}s", final_psnr, gaussians.len(), elapsed.as_secs_f32());

        results.push(("ErrorDriven(Adam,512)", 64, gaussians.len(), final_psnr, elapsed.as_secs_f32()));

        let mut result = TestResult::new("error_driven_adam_512", image_path);
        result.optimizer = "ErrorDrivenAdamEncoder (large)".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = gaussians.len();
        result.final_psnr = final_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({
            "test_type": "error_driven_adam_large",
            "initial_gaussians": config.initial_gaussians,
            "max_gaussians": config.max_gaussians,
            "iterations_per_pass": config.iterations_per_pass,
            "min_improvement_db": config.min_improvement_db,
            "optimizer": "Adam"
        }));
        let _ = result.save(RESULTS_DIR);
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "{:<25} {:>8} {:>8} {:>10} {:>8}",
        "Configuration", "Init N", "Final N", "PSNR", "Time"
    );
    println!(
        "{:-<25} {:-<8} {:-<8} {:-<10} {:-<8}",
        "", "", "", "", ""
    );

    for (name, init_n, final_n, psnr, time) in &results {
        println!(
            "{:<25} {:>8} {:>8} {:>9.2}dB {:>7.1}s",
            name, init_n, final_n, psnr, time
        );
    }

    // Analysis
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        ANALYSIS                               ");
    println!("═══════════════════════════════════════════════════════════════");

    let uniform_result = results.iter().find(|r| r.0 == "Uniform+Adam(256)");
    let v2_result = results.iter().find(|r| r.0 == "ErrorDriven(V2)");
    let adam_result = results.iter().find(|r| r.0 == "ErrorDriven(Adam)");

    if let (Some(uniform), Some(v2), Some(adam)) = (uniform_result, v2_result, adam_result) {
        println!("Adam vs OptimizerV2:");
        println!("  PSNR: {:.2}dB vs {:.2}dB ({:+.2}dB)", adam.3, v2.3, adam.3 - v2.3);
        println!("  Time: {:.1}s vs {:.1}s ({:.1}× speedup)", adam.4, v2.4, v2.4 / adam.4);
        println!("  Gaussians: {} vs {}", adam.2, v2.2);

        println!("\nAdam ErrorDriven vs Uniform+Adam:");
        println!("  PSNR: {:.2}dB vs {:.2}dB ({:+.2}dB)", adam.3, uniform.3, adam.3 - uniform.3);
        println!("  Gaussians: {} vs {} ({:.0}% of uniform)", adam.2, uniform.2, (adam.2 as f32 / uniform.2 as f32) * 100.0);
    }

    println!("\nHYPOTHESIS RESULTS:");
    println!("  If Adam PSNR > V2 PSNR: Adam optimizer is more effective");
    println!("  If Adam time < V2 time: Adam converges faster");
    println!("  If ErrorDriven > Uniform: Adaptive placement provides benefit");
}

fn create_uniform_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n);
    let cell_size = 1.0 / grid_size as f32;

    for i in 0..n {
        let gx = i % grid_size;
        let gy = i / grid_size;

        let x = (gx as f32 + 0.5) / grid_size as f32;
        let y = (gy as f32 + 0.5) / grid_size as f32;

        let px = ((x * width as f32) as u32).min(width - 1);
        let py = ((y * height as f32) as u32).min(height - 1);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(cell_size, cell_size, 0.0),
            color,
            1.0,
        ));
    }

    gaussians
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height * 3) as f32;
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= count;
    if mse <= 0.0 {
        100.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}
