//! Test better color initialization
//!
//! Compares single-pixel color sampling vs Gaussian-weighted average color.
//! The hypothesis is that weighted averaging over the Gaussian footprint
//! reduces aliasing and provides better initial colors.
//!
//! Run: cargo run --release --example test_better_color_init -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::better_color_init::gaussian_weighted_color;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   BETTER COLOR INITIALIZATION TEST                           ║");
    println!("║   Single-pixel vs Gaussian-weighted color                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let test_images = vec![
        ("kodim03", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png"),
        ("kodim05", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim05.png"),  // Contains fine detail
        ("kodim08", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim08.png"),
    ];

    let n_gaussians = 256;
    let iterations = 100;

    for (name, image_path) in &test_images {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║ Image: {:<54}║", name);
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        let target = ImageBuffer::load(image_path).expect("Failed to load image");
        let (width, height) = (target.width, target.height);

        let mut results = Vec::new();

        // Test 1: Single-pixel color sampling (standard)
        println!("═══════════════════════════════════════════════════════════════");
        println!("Test 1: Single-Pixel Color Sampling (Standard)");
        println!("═══════════════════════════════════════════════════════════════");
        {
            let start = Instant::now();
            let mut gaussians = create_grid_single_pixel(&target, n_gaussians);

            let init_render = RendererV2::render(&gaussians, width, height);
            let init_psnr = compute_psnr(&init_render, &target);
            println!("  Initial PSNR: {:.2}dB", init_psnr);

            let mut adam = AdamOptimizer::new();
            adam.max_iterations = iterations;

            let mut last_psnr = init_psnr;
            for iter in 0..iterations {
                if iter % 20 == 0 {
                    let rendered = RendererV2::render(&gaussians, width, height);
                    let psnr = compute_psnr(&rendered, &target);
                    let direction = if psnr > last_psnr { "↑" } else { "↓" };
                    println!("    [{:3}/{:3}] PSNR={:.2}dB {}", iter, iterations, psnr, direction);
                    last_psnr = psnr;
                }
                adam.max_iterations = 1;
                adam.optimize(&mut gaussians, &target);
            }

            let elapsed = start.elapsed();
            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = compute_psnr(&final_render, &target);
            println!("  Final PSNR: {:.2}dB ({:+.2}dB)", final_psnr, final_psnr - init_psnr);

            results.push(("Single-pixel".to_string(), init_psnr, final_psnr, elapsed.as_secs_f32()));

            let mut result = TestResult::new(&format!("color_init_single_pixel_{}", name), image_path);
            result.optimizer = "Adam + Single-pixel color".to_string();
            result.image_dimensions = (width, height);
            result.n_gaussians = n_gaussians;
            result.iterations = iterations;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = final_psnr - init_psnr;
            result.elapsed_seconds = elapsed.as_secs_f32();
            result.extra_metrics = Some(json!({"test_type": "single_pixel_color", "image": name}));
            let _ = result.save(RESULTS_DIR);
        }

        // Test 2: Gaussian-weighted color averaging
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("Test 2: Gaussian-Weighted Color Averaging");
        println!("═══════════════════════════════════════════════════════════════");
        {
            let start = Instant::now();
            let mut gaussians = create_grid_weighted_color(&target, n_gaussians);

            let init_render = RendererV2::render(&gaussians, width, height);
            let init_psnr = compute_psnr(&init_render, &target);
            println!("  Initial PSNR: {:.2}dB", init_psnr);

            let mut adam = AdamOptimizer::new();
            adam.max_iterations = iterations;

            let mut last_psnr = init_psnr;
            for iter in 0..iterations {
                if iter % 20 == 0 {
                    let rendered = RendererV2::render(&gaussians, width, height);
                    let psnr = compute_psnr(&rendered, &target);
                    let direction = if psnr > last_psnr { "↑" } else { "↓" };
                    println!("    [{:3}/{:3}] PSNR={:.2}dB {}", iter, iterations, psnr, direction);
                    last_psnr = psnr;
                }
                adam.max_iterations = 1;
                adam.optimize(&mut gaussians, &target);
            }

            let elapsed = start.elapsed();
            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = compute_psnr(&final_render, &target);
            println!("  Final PSNR: {:.2}dB ({:+.2}dB)", final_psnr, final_psnr - init_psnr);

            results.push(("Weighted-avg".to_string(), init_psnr, final_psnr, elapsed.as_secs_f32()));

            let mut result = TestResult::new(&format!("color_init_weighted_{}", name), image_path);
            result.optimizer = "Adam + Weighted-avg color".to_string();
            result.image_dimensions = (width, height);
            result.n_gaussians = n_gaussians;
            result.iterations = iterations;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = final_psnr - init_psnr;
            result.elapsed_seconds = elapsed.as_secs_f32();
            result.extra_metrics = Some(json!({"test_type": "weighted_avg_color", "image": name}));
            let _ = result.save(RESULTS_DIR);
        }

        // Summary for this image
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("                  SUMMARY for {}", name);
        println!("═══════════════════════════════════════════════════════════════");
        println!("{:<20} {:>10} {:>10} {:>10} {:>8}",
                 "Method", "Init PSNR", "Final PSNR", "Improve", "Time");
        println!("{:-<20} {:-<10} {:-<10} {:-<10} {:-<8}", "", "", "", "", "");

        for (method, init, final_psnr, time) in &results {
            println!("{:<20} {:>9.2}dB {:>9.2}dB {:>9.2}dB {:>7.1}s",
                     method, init, final_psnr, final_psnr - init, time);
        }

        // Calculate difference
        if results.len() >= 2 {
            let init_diff = results[1].1 - results[0].1;
            let final_diff = results[1].2 - results[0].2;
            println!("\nWeighted vs Single-pixel:");
            println!("  Initial PSNR: {:+.3}dB", init_diff);
            println!("  Final PSNR: {:+.3}dB", final_diff);
        }
    }

    println!("\n\nHYPOTHESIS: Weighted color averaging should improve initial PSNR");
    println!("by reducing aliasing artifacts from point sampling, especially");
    println!("on images with fine detail or high-frequency content.");
}

/// Standard grid with single-pixel color sampling
fn create_grid_single_pixel(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
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

/// Grid with Gaussian-weighted average color
fn create_grid_weighted_color(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let mut gaussians = Vec::with_capacity(n);
    let cell_size = 1.0 / grid_size as f32;

    for i in 0..n {
        let gx = i % grid_size;
        let gy = i / grid_size;

        let jx = ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5;
        let jy = ((i * 11 + 7) % 100) as f32 / 100.0 - 0.5;

        let x = ((gx as f32 + 0.5 + jx * 0.4) / grid_size as f32).clamp(0.01, 0.99);
        let y = ((gy as f32 + 0.5 + jy * 0.4) / grid_size as f32).clamp(0.01, 0.99);

        // Use weighted color instead of single pixel
        let color = gaussian_weighted_color(
            target,
            Vector2::new(x, y),
            cell_size,  // sigma_x
            cell_size,  // sigma_y
            0.0,        // rotation
        );

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
