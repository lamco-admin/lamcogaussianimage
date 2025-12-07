//! Test Encoder V3 Adaptive (Content-Adaptive Encoding)
//!
//! Tests the content-adaptive encoder that:
//! - Detects content type (Smooth, Sharp, HighFrequency, Photo)
//! - Adjusts gamma based on content
//! - Uses analytical triggers for mitigation
//! - Compares to baseline Adam optimizer
//!
//! Run: cargo run --release --example test_encoder_v3_adaptive -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::encoder_v3_adaptive::EncoderV3Adaptive;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   ENCODER V3 ADAPTIVE TEST                                   ║");
    println!("║   Content-Adaptive Encoding vs Baseline                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Test on multiple images with different content types
    let test_images = vec![
        // Natural photo with gradients and detail
        ("/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png", "kodim03 (mandrill)"),
        // Building with sharp edges
        ("/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim08.png", "kodim08 (building)"),
        // Smooth gradients (lighthouse sky)
        ("/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim21.png", "kodim21 (lighthouse)"),
    ];

    let n_gaussians = 256;
    let iterations = 100;

    let mut all_results: Vec<(&str, f32, f32, f32, f32, String)> = Vec::new();

    for (image_path, image_name) in &test_images {
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("Testing: {}", image_name);
        println!("═══════════════════════════════════════════════════════════════\n");

        let target = match ImageBuffer::load(image_path) {
            Ok(img) => img,
            Err(e) => {
                println!("Failed to load {}: {:?}", image_path, e);
                continue;
            }
        };
        let (width, height) = (target.width, target.height);
        println!("Image: {}x{}", width, height);

        // Test 1: Baseline Adam Optimizer
        println!("\n--- Test 1: Baseline Adam Optimizer ---");
        let (baseline_psnr, baseline_time) = test_baseline_adam(&target, n_gaussians, iterations, image_path);

        // Test 2: Encoder V3 Adaptive
        println!("\n--- Test 2: Encoder V3 Adaptive ---");
        let (adaptive_psnr, adaptive_time, content_type) = test_encoder_v3(&target, n_gaussians, image_path);

        all_results.push((
            image_name,
            baseline_psnr,
            baseline_time,
            adaptive_psnr,
            adaptive_time,
            content_type,
        ));
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<25} {:>12} {:>12} {:>10} {:>12}",
             "Image", "Adam PSNR", "V3 PSNR", "Diff", "Content");
    println!("{:-<25} {:-<12} {:-<12} {:-<10} {:-<12}", "", "", "", "", "");

    let mut total_adam = 0.0;
    let mut total_v3 = 0.0;
    for (name, adam_psnr, _, v3_psnr, _, content) in &all_results {
        let diff = v3_psnr - adam_psnr;
        println!("{:<25} {:>11.2}dB {:>11.2}dB {:>+9.2}dB {:>12}",
                 name, adam_psnr, v3_psnr, diff, content);
        total_adam += adam_psnr;
        total_v3 += v3_psnr;
    }

    let n = all_results.len() as f32;
    let avg_adam = total_adam / n;
    let avg_v3 = total_v3 / n;
    println!("{:-<25} {:-<12} {:-<12} {:-<10} {:-<12}", "", "", "", "", "");
    println!("{:<25} {:>11.2}dB {:>11.2}dB {:>+9.2}dB",
             "AVERAGE", avg_adam, avg_v3, avg_v3 - avg_adam);
    println!("═══════════════════════════════════════════════════════════════");
}

/// Test baseline Adam optimizer
fn test_baseline_adam(
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    iterations: usize,
    image_path: &str,
) -> (f32, f32) {
    let (width, height) = (target.width, target.height);

    // Create jittered grid initialization (our proven baseline)
    let mut gaussians = create_jittered_grid(target, n_gaussians);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);
    println!("  Initial: {} Gaussians, PSNR={:.2}dB", gaussians.len(), init_psnr);

    let start = Instant::now();
    let mut adam = AdamOptimizer::new();
    adam.max_iterations = iterations;
    adam.optimize(&mut gaussians, target);
    let elapsed = start.elapsed().as_secs_f32();

    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);

    println!("  Final:   {} Gaussians, PSNR={:.2}dB ({:+.2}dB) in {:.1}s",
             gaussians.len(), final_psnr, final_psnr - init_psnr, elapsed);

    // Save result
    let mut result = TestResult::new("encoder_v3_baseline_adam", image_path);
    result.optimizer = "Adam (baseline)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;
    let _ = result.save(RESULTS_DIR);

    (final_psnr, elapsed)
}

/// Test Encoder V3 Adaptive
fn test_encoder_v3(
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    image_path: &str,
) -> (f32, f32, String) {
    let (width, height) = (target.width, target.height);

    let start = Instant::now();

    // Create the adaptive encoder
    let encoder = match EncoderV3Adaptive::new(target.clone()) {
        Ok(enc) => enc,
        Err(e) => {
            println!("  Failed to create encoder: {:?}", e);
            return (0.0, 0.0, "Error".to_string());
        }
    };

    // Get content type for reporting
    let content_type = detect_content_type_name(target);
    println!("  Detected content type: {}", content_type);

    // Run adaptive encoding
    let gaussians = encoder.encode_adaptive(n_gaussians);
    let elapsed = start.elapsed().as_secs_f32();

    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);

    println!("  Final:   {} Gaussians, PSNR={:.2}dB in {:.1}s",
             gaussians.len(), final_psnr, elapsed);

    // Save result
    let mut result = TestResult::new("encoder_v3_adaptive", image_path);
    result.optimizer = format!("EncoderV3Adaptive (content: {})", content_type);
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.iterations = 0;  // V3 handles iterations internally
    result.initial_psnr = 0.0;  // Not tracked separately
    result.final_psnr = final_psnr;
    result.improvement_db = 0.0;  // Not meaningful for adaptive encoder
    result.elapsed_seconds = elapsed;
    result.extra_metrics = Some(json!({
        "content_type": content_type,
        "requested_gaussians": n_gaussians,
        "actual_gaussians": gaussians.len(),
    }));
    let _ = result.save(RESULTS_DIR);

    (final_psnr, elapsed, content_type)
}

/// Detect content type name (for display)
fn detect_content_type_name(target: &ImageBuffer<f32>) -> String {
    use lgi_core::{StructureTensorField, content_detection::ContentAnalyzer};

    let tensor_field = match StructureTensorField::compute(target, 1.2, 1.0) {
        Ok(tf) => tf,
        Err(_) => return "Unknown".to_string(),
    };

    let content_type = ContentAnalyzer::detect_content_type(target, &tensor_field);
    format!("{:?}", content_type)
}

fn create_jittered_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let grid_size = (n as f32).sqrt() as usize;
    let mut gaussians = Vec::with_capacity(grid_size * grid_size);

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            // Deterministic jitter based on position
            let jitter_x = ((gx * 7 + gy * 13) % 10) as f32 / 100.0 - 0.05;
            let jitter_y = ((gx * 11 + gy * 17) % 10) as f32 / 100.0 - 0.05;

            let x = ((gx as f32 + 0.5) / grid_size as f32 + jitter_x).clamp(0.01, 0.99);
            let y = ((gy as f32 + 0.5) / grid_size as f32 + jitter_y).clamp(0.01, 0.99);

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

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height * 3) as f32;
    let mut mse = 0.0_f32;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= count;
    if mse <= 0.0 { 100.0 } else { 10.0_f32 * (1.0_f32 / mse).log10() }
}
