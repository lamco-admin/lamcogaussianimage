//! Test EncoderV3 content-adaptive features with Adam optimizer
//!
//! EncoderV3Adaptive uses OptimizerV2 internally. This test explores whether
//! using Adam optimizer with V3's content-adaptive initialization improves results.
//!
//! Compares:
//! - EncoderV3 with OptimizerV2 (original)
//! - V3 content-adaptive init + Adam optimizer (variation)
//! - Pure Adam baseline
//!
//! Run: cargo run --release --example test_encoder_v3_with_adam -p lgi-encoder-v2

use lgi_core::{ImageBuffer, StructureTensorField, content_detection};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::encoder_v3_adaptive::EncoderV3Adaptive;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   ENCODER V3 WITH ADAM OPTIMIZER TEST                        ║");
    println!("║   Testing content-adaptive init + Adam optimizer             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let test_images = vec![
        ("kodim03", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png"),
        ("kodim08", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim08.png"),
        ("kodim13", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim13.png"),
    ];

    let n_gaussians = 256;
    let iterations = 100;

    for (name, image_path) in &test_images {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║ Image: {:<54}║", name);
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let target = ImageBuffer::load(image_path).expect("Failed to load image");
        let (width, height) = (target.width, target.height);

        // Detect content type
        let tensor_field = StructureTensorField::compute(&target, 1.2, 1.0).unwrap();
        let content_type = content_detection::ContentAnalyzer::detect_content_type(&target, &tensor_field);
        println!("Content type detected: {:?}", content_type);

        // Test 1: Pure Adam baseline (jittered grid init)
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("Test 1: Adam Optimizer Baseline (Jittered Grid)");
        println!("═══════════════════════════════════════════════════════════════");
        let adam_baseline = test_adam_baseline(&target, n_gaussians, iterations, image_path);

        // Test 2: EncoderV3Adaptive (original with OptimizerV2)
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("Test 2: EncoderV3 with OptimizerV2 (Original)");
        println!("═══════════════════════════════════════════════════════════════");
        let v3_original = test_encoder_v3_original(&target, n_gaussians, image_path);

        // Test 3: V3 content-adaptive init + Adam optimizer
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("Test 3: V3 Content-Adaptive Init + Adam Optimizer");
        println!("═══════════════════════════════════════════════════════════════");
        let v3_adam = test_v3_init_with_adam(&target, &content_type, n_gaussians, iterations, image_path);

        // Test 4: V3 init + Adam + more iterations
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("Test 4: V3 Content-Adaptive Init + Adam (200 iterations)");
        println!("═══════════════════════════════════════════════════════════════");
        let v3_adam_200 = test_v3_init_with_adam(&target, &content_type, n_gaussians, 200, image_path);

        // Summary
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("                    SUMMARY for {}", name);
        println!("═══════════════════════════════════════════════════════════════");
        println!("{:<40} {:>10} {:>10}",
                 "Configuration", "Final PSNR", "Time");
        println!("{:-<40} {:-<10} {:-<10}", "", "", "");
        println!("{:<40} {:>9.2}dB {:>9.1}s", "Adam Baseline (jittered)", adam_baseline.0, adam_baseline.1);
        println!("{:<40} {:>9.2}dB {:>9.1}s", "V3 + OptimizerV2", v3_original.0, v3_original.1);
        println!("{:<40} {:>9.2}dB {:>9.1}s", "V3 Init + Adam (100 iter)", v3_adam.0, v3_adam.1);
        println!("{:<40} {:>9.2}dB {:>9.1}s", "V3 Init + Adam (200 iter)", v3_adam_200.0, v3_adam_200.1);
    }
}

fn test_adam_baseline(target: &ImageBuffer<f32>, n: usize, iterations: usize, image_path: &str) -> (f32, f32) {
    let (width, height) = (target.width, target.height);
    let start = Instant::now();

    // Jittered grid initialization
    let mut gaussians = create_jittered_grid(target, n);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);
    println!("  Initial PSNR: {:.2}dB", init_psnr);

    // Adam optimization
    let mut adam = AdamOptimizer::new();
    adam.max_iterations = iterations;

    let mut last_psnr = init_psnr;
    for iter in 0..iterations {
        if iter % 20 == 0 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = compute_psnr(&rendered, target);
            let direction = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {}", iter, iterations, psnr, direction);
            last_psnr = psnr;
        }
        adam.max_iterations = 1;
        adam.optimize(&mut gaussians, target);
    }

    let elapsed = start.elapsed();
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);
    println!("  Final PSNR: {:.2}dB ({:+.2}dB)", final_psnr, final_psnr - init_psnr);

    // Save result
    let mut result = TestResult::new("v3_adam_baseline", image_path);
    result.optimizer = "Adam Baseline (jittered grid)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = n;
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.extra_metrics = Some(json!({
        "test_type": "adam_baseline",
    }));
    let _ = result.save(RESULTS_DIR);

    (final_psnr, elapsed.as_secs_f32())
}

fn test_encoder_v3_original(target: &ImageBuffer<f32>, n: usize, image_path: &str) -> (f32, f32) {
    let (width, height) = (target.width, target.height);
    let start = Instant::now();

    let encoder = EncoderV3Adaptive::new(target.clone()).expect("Failed to create encoder");
    let gaussians = encoder.encode_adaptive(n);

    let elapsed = start.elapsed();
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);

    println!("  Final PSNR: {:.2}dB (N_gaussians={})", final_psnr, gaussians.len());

    // Save result
    let mut result = TestResult::new("v3_original", image_path);
    result.optimizer = "EncoderV3 + OptimizerV2".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.final_psnr = final_psnr;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.extra_metrics = Some(json!({
        "test_type": "v3_original",
    }));
    let _ = result.save(RESULTS_DIR);

    (final_psnr, elapsed.as_secs_f32())
}

fn test_v3_init_with_adam(
    target: &ImageBuffer<f32>,
    content_type: &content_detection::ContentType,
    n: usize,
    iterations: usize,
    image_path: &str,
) -> (f32, f32) {
    let (width, height) = (target.width, target.height);
    let start = Instant::now();

    // Use content-adaptive gamma (from V3)
    let gamma = match content_type {
        content_detection::ContentType::Smooth => 1.2,
        content_detection::ContentType::Sharp => 0.4,
        content_detection::ContentType::HighFrequency => 0.4,
        content_detection::ContentType::Photo => 0.4,
        _ => 0.6,
    };
    println!("  Using gamma={:.1} for {:?}", gamma, content_type);

    // Content-adaptive initialization (similar to V3)
    let mut gaussians = create_content_adaptive_init(target, n, gamma);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);
    println!("  Initial PSNR: {:.2}dB", init_psnr);

    // Adam optimization (instead of OptimizerV2)
    let mut adam = AdamOptimizer::new();
    adam.max_iterations = iterations;

    let mut last_psnr = init_psnr;
    for iter in 0..iterations {
        if iter % 20 == 0 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = compute_psnr(&rendered, target);
            let direction = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {}", iter, iterations, psnr, direction);
            last_psnr = psnr;
        }
        adam.max_iterations = 1;
        adam.optimize(&mut gaussians, target);
    }

    let elapsed = start.elapsed();
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);
    println!("  Final PSNR: {:.2}dB ({:+.2}dB)", final_psnr, final_psnr - init_psnr);

    // Save result
    let exp_name = format!("v3_init_adam_{}iter", iterations);
    let mut result = TestResult::new(&exp_name, image_path);
    result.optimizer = format!("V3 Init + Adam ({}iter)", iterations);
    result.image_dimensions = (width, height);
    result.n_gaussians = n;
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.extra_metrics = Some(json!({
        "test_type": "v3_init_adam",
        "content_type": format!("{:?}", content_type),
        "gamma": gamma,
    }));
    let _ = result.save(RESULTS_DIR);

    (final_psnr, elapsed.as_secs_f32())
}

/// Content-adaptive initialization (based on V3)
fn create_content_adaptive_init(target: &ImageBuffer<f32>, n: usize, gamma: f32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n);

    // Scale based on gamma - smaller gamma = smaller initial scales for sharp content
    let base_scale = 1.0 / grid_size as f32;
    let scale = base_scale * gamma.sqrt();

    for i in 0..n {
        let gx = i % grid_size;
        let gy = i / grid_size;

        // Jitter within cell
        let jx = ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5;
        let jy = ((i * 11 + 7) % 100) as f32 / 100.0 - 0.5;

        let x = ((gx as f32 + 0.5 + jx * 0.4) / grid_size as f32).clamp(0.01, 0.99);
        let y = ((gy as f32 + 0.5 + jy * 0.4) / grid_size as f32).clamp(0.01, 0.99);

        let px = ((x * width as f32) as u32).min(width - 1);
        let py = ((y * height as f32) as u32).min(height - 1);
        let color = target.get_pixel(px, py).unwrap();

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(scale, scale, 0.0),
            color,
            1.0,
        ));
    }

    gaussians
}

/// Jittered grid initialization
fn create_jittered_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n);
    let cell_size = 1.0 / grid_size as f32;

    for i in 0..n {
        let gx = i % grid_size;
        let gy = i / grid_size;

        let jx = ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5;
        let jy = ((i * 11 + 7) % 100) as f32 / 100.0 - 0.5;

        let x = ((gx as f32 + 0.5 + jx * 0.4) / grid_size as f32).clamp(0.01, 0.99);
        let y = ((gy as f32 + 0.5 + jy * 0.4) / grid_size as f32).clamp(0.01, 0.99);

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
