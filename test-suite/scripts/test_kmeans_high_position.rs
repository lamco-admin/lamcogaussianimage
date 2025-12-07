//! Test K-Means with very high position weights
//!
//! Previous tests showed position-heavy (p=2) performed closest to grid methods.
//! This test explores even higher position weights to see if the trend continues.
//!
//! Run: cargo run --release --example test_kmeans_high_position -p lgi-encoder-v2

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::kmeans_init::{initialize_kmeans, KMeansConfig};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   K-MEANS HIGH POSITION WEIGHT TEST                          ║");
    println!("║   Testing p=5, p=10 to find optimal position weighting       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);
    println!("Computing structure tensor...\n");

    let tensor_field = StructureTensorField::compute(&target, 1.0, 2.0)
        .expect("Failed to compute structure tensor");

    let n_gaussians = 256;
    let iterations = 100;

    // Configuration variations - higher position weights
    let configs = vec![
        ("jittered_grid", None),  // Baseline
        ("kmeans_p2", Some((1.0, 2.0, 0.3))),   // Previous best k-means
        ("kmeans_p5", Some((0.5, 5.0, 0.2))),   // High position weight
        ("kmeans_p10", Some((0.3, 10.0, 0.1))), // Very high position weight
        ("kmeans_position_only", Some((0.0, 10.0, 0.0))), // Position only (like grid)
        ("kmeans_balanced", Some((1.0, 1.0, 1.0))), // Equal weights
    ];

    let mut results = Vec::new();

    for (name, weights) in &configs {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Configuration: {}", name);
        println!("═══════════════════════════════════════════════════════════════");

        let start = Instant::now();

        let mut gaussians = if let Some((c, p, g)) = weights {
            println!("  Weights: color={:.1}, position={:.1}, gradient={:.1}", c, p, g);

            let config = KMeansConfig {
                k: n_gaussians,
                max_iterations: 20,
                weight_color: *c,
                weight_position: *p,
                weight_gradient: *g,
                ..Default::default()
            };
            initialize_kmeans(&target, &tensor_field, &config)
        } else {
            println!("  Using jittered grid (baseline)");
            create_jittered_grid(&target, n_gaussians)
        };

        let init_time = start.elapsed();
        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);

        println!("  Init time: {:.2}s", init_time.as_secs_f32());
        println!("  Initial PSNR: {:.2}dB", init_psnr);

        // Optimize with Adam
        let opt_start = Instant::now();
        let mut adam = AdamOptimizer::new();
        adam.max_iterations = iterations;

        let mut last_psnr = init_psnr;
        for iter in 0..iterations {
            if iter % 25 == 0 {
                let rendered = RendererV2::render(&gaussians, width, height);
                let psnr = compute_psnr(&rendered, &target);
                let direction = if psnr > last_psnr { "↑" } else { "↓" };
                println!("    [{:3}/{:3}] PSNR={:.2}dB {}", iter, iterations, psnr, direction);
                last_psnr = psnr;
            }
            adam.max_iterations = 1;
            adam.optimize(&mut gaussians, &target);
        }

        let opt_time = opt_start.elapsed();
        let total_time = (init_time + opt_time).as_secs_f32();

        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        let improvement = final_psnr - init_psnr;

        println!("  Final PSNR: {:.2}dB ({:+.2}dB)", final_psnr, improvement);
        println!();

        results.push((name.to_string(), init_psnr, final_psnr, total_time));

        // Save result
        let exp_name = format!("kmeans_high_pos_{}", name);
        let mut result = TestResult::new(&exp_name, image_path);
        result.optimizer = format!("Adam + {} init", name);
        result.image_dimensions = (width, height);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = improvement;
        result.elapsed_seconds = total_time;
        result.extra_metrics = Some(json!({
            "init_type": name,
            "init_time_seconds": init_time.as_secs_f32(),
            "optimization_time_seconds": opt_time.as_secs_f32(),
            "weights": weights,
        }));
        let _ = result.save(RESULTS_DIR);
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<25} {:>10} {:>10} {:>10} {:>8}",
             "Configuration", "Init PSNR", "Final PSNR", "Improve", "Time");
    println!("{:-<25} {:-<10} {:-<10} {:-<10} {:-<8}", "", "", "", "", "");

    for (name, init, final_psnr, time) in &results {
        println!("{:<25} {:>9.2}dB {:>9.2}dB {:>9.2}dB {:>7.1}s",
                 name, init, final_psnr, final_psnr - init, time);
    }

    println!("\nHYPOTHESIS: Higher position weights should approach grid-like");
    println!("distribution while still benefiting from color clustering.");
}

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
