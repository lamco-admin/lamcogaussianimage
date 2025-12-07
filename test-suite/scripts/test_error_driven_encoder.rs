//! Test Error-Driven Encoder
//!
//! The ErrorDrivenEncoder adaptively places Gaussians where reconstruction
//! error is high. Claims: +4.3 dB potential vs uniform grid.
//!
//! Compares:
//! - Uniform grid + Adam (baseline)
//! - Error-driven iterative placement
//! - Error-driven + Adam refinement
//!
//! Run: cargo run --release --example test_error_driven_encoder -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::error_driven::ErrorDrivenEncoder;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   ERROR-DRIVEN ENCODER TEST                                  ║");
    println!("║   Adaptive placement at high-error hotspots                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);

    let mut results = Vec::new();

    // Test 1: Uniform Grid + Adam (Baseline)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 1: Uniform Grid + Adam Baseline (256 Gaussians)");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let start = Instant::now();
        let mut gaussians = create_uniform_grid(&target, 256);

        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);
        println!("  Initial PSNR: {:.2}dB", init_psnr);

        let mut adam = AdamOptimizer::new();
        adam.max_iterations = 100;

        let mut last_psnr = init_psnr;
        for iter in 0..100 {
            if iter % 20 == 0 {
                let rendered = RendererV2::render(&gaussians, width, height);
                let psnr = compute_psnr(&rendered, &target);
                let direction = if psnr > last_psnr { "↑" } else { "↓" };
                println!("    [{:3}/100] PSNR={:.2}dB {}", iter, psnr, direction);
                last_psnr = psnr;
            }
            adam.max_iterations = 1;
            adam.optimize(&mut gaussians, &target);
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        println!("  Final PSNR: {:.2}dB (N={})", final_psnr, gaussians.len());

        results.push(("Uniform+Adam(256)".to_string(), 256, gaussians.len(), final_psnr, elapsed.as_secs_f32()));

        let mut result = TestResult::new("error_driven_baseline_256", image_path);
        result.optimizer = "Uniform Grid + Adam".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = gaussians.len();
        result.iterations = 100;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({"test_type": "baseline"}));
        let _ = result.save(RESULTS_DIR);
    }

    // Test 2: Error-Driven Encoder (starts with 100, grows adaptively)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: Error-Driven Encoder (starts 100, max 500)");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let start = Instant::now();

        let encoder = ErrorDrivenEncoder {
            target_error: 0.001,
            initial_gaussians: 100,
            max_gaussians: 500,
            split_percentile: 0.10,
        };

        let gaussians = encoder.encode(&target);

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        println!("  Final PSNR: {:.2}dB (N={})", final_psnr, gaussians.len());

        results.push(("ErrorDriven(100→500)".to_string(), 100, gaussians.len(), final_psnr, elapsed.as_secs_f32()));

        let mut result = TestResult::new("error_driven_adaptive", image_path);
        result.optimizer = "ErrorDrivenEncoder".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = gaussians.len();
        result.final_psnr = final_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({
            "test_type": "error_driven",
            "initial_gaussians": 100,
            "max_gaussians": 500,
            "split_percentile": 0.10,
        }));
        let _ = result.save(RESULTS_DIR);
    }

    // Test 3: Error-Driven + Adam Refinement
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 3: Error-Driven + Adam Refinement");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let start = Instant::now();

        let encoder = ErrorDrivenEncoder {
            target_error: 0.001,
            initial_gaussians: 100,
            max_gaussians: 300,  // Fewer, then refine with Adam
            split_percentile: 0.10,
        };

        let mut gaussians = encoder.encode(&target);
        let error_driven_n = gaussians.len();

        let mid_render = RendererV2::render(&gaussians, width, height);
        let mid_psnr = compute_psnr(&mid_render, &target);
        println!("  After error-driven: {:.2}dB (N={})", mid_psnr, error_driven_n);

        // Refine with Adam
        println!("  Refining with Adam (50 iterations)...");
        let mut adam = AdamOptimizer::new();
        adam.max_iterations = 50;

        let mut last_psnr = mid_psnr;
        for iter in 0..50 {
            if iter % 10 == 0 {
                let rendered = RendererV2::render(&gaussians, width, height);
                let psnr = compute_psnr(&rendered, &target);
                let direction = if psnr > last_psnr { "↑" } else { "↓" };
                println!("    [{:3}/50] PSNR={:.2}dB {}", iter, psnr, direction);
                last_psnr = psnr;
            }
            adam.max_iterations = 1;
            adam.optimize(&mut gaussians, &target);
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        println!("  Final PSNR: {:.2}dB (N={})", final_psnr, gaussians.len());

        results.push(("ErrorDriven+Adam".to_string(), 100, gaussians.len(), final_psnr, elapsed.as_secs_f32()));

        let mut result = TestResult::new("error_driven_plus_adam", image_path);
        result.optimizer = "ErrorDriven + Adam".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = gaussians.len();
        result.initial_psnr = mid_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - mid_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({
            "test_type": "error_driven_plus_adam",
            "error_driven_n": error_driven_n,
            "adam_iterations": 50,
        }));
        let _ = result.save(RESULTS_DIR);
    }

    // Test 4: Error-Driven with smaller initial set
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 4: Error-Driven (starts 64, max 256)");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let start = Instant::now();

        let encoder = ErrorDrivenEncoder {
            target_error: 0.001,
            initial_gaussians: 64,
            max_gaussians: 256,
            split_percentile: 0.10,
        };

        let mut gaussians = encoder.encode(&target);

        // Quick Adam polish
        let mut adam = AdamOptimizer::new();
        adam.max_iterations = 30;

        for _ in 0..30 {
            adam.max_iterations = 1;
            adam.optimize(&mut gaussians, &target);
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        println!("  Final PSNR: {:.2}dB (N={})", final_psnr, gaussians.len());

        results.push(("ErrorDriven(64→256)".to_string(), 64, gaussians.len(), final_psnr, elapsed.as_secs_f32()));

        let mut result = TestResult::new("error_driven_64_256", image_path);
        result.optimizer = "ErrorDriven (64→256) + Adam".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = gaussians.len();
        result.final_psnr = final_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({
            "test_type": "error_driven_small",
            "initial_gaussians": 64,
            "max_gaussians": 256,
        }));
        let _ = result.save(RESULTS_DIR);
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<25} {:>8} {:>8} {:>10} {:>8}",
             "Configuration", "Init N", "Final N", "PSNR", "Time");
    println!("{:-<25} {:-<8} {:-<8} {:-<10} {:-<8}", "", "", "", "", "");

    for (name, init_n, final_n, psnr, time) in &results {
        println!("{:<25} {:>8} {:>8} {:>9.2}dB {:>7.1}s",
                 name, init_n, final_n, psnr, time);
    }

    println!("\nHYPOTHESIS: Error-driven placement should achieve similar quality");
    println!("to uniform grid with fewer Gaussians, or better quality with same count.");
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
        let color = target.get_pixel(px, py).unwrap();

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
    if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}
