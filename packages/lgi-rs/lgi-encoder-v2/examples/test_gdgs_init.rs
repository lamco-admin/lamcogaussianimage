//! Test GDGS (Gradient Domain Gaussian Splatting) Initialization
//!
//! GDGS places Gaussians only at Laplacian peaks (edges, corners).
//! Paper claims 10-100× fewer Gaussians for same quality!
//!
//! Compares:
//! - Jittered grid (baseline)
//! - GDGS with various thresholds
//! - GDGS with target N (auto-tuned threshold)
//!
//! Run: cargo run --release --example test_gdgs_init -p lgi-encoder-v2

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::gdgs_init::{initialize_gdgs, initialize_gdgs_with_target_n, GDGSConfig};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   GDGS INITIALIZATION TEST                                   ║");
    println!("║   Gradient Domain Gaussian Splatting (Laplacian Peaks)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);
    println!("Computing structure tensor...\n");

    let structure_tensor = StructureTensorField::compute(&target, 1.0, 2.0)
        .expect("Failed to compute structure tensor");

    let iterations = 150;  // More iterations since GDGS may start with fewer Gaussians

    // Test configurations
    let configs: Vec<(&str, Option<GDGSConfig>, Option<usize>)> = vec![
        // Baselines
        ("jittered_grid_256", None, Some(256)),
        ("jittered_grid_128", None, Some(128)),

        // GDGS with different thresholds
        ("gdgs_low_thresh", Some(GDGSConfig {
            lap_threshold: 0.005,  // Low = more peaks
            max_gaussians: 500,
            base_sigma: 0.02,
            ..Default::default()
        }), None),

        ("gdgs_default", Some(GDGSConfig::default()), None),

        ("gdgs_high_thresh", Some(GDGSConfig {
            lap_threshold: 0.05,  // High = fewer peaks
            max_gaussians: 500,
            base_sigma: 0.03,  // Larger base for sparse init
            ..Default::default()
        }), None),

        // GDGS with target N (auto-tuned threshold)
        ("gdgs_target_256", None, Some(256)),  // Will use special function
        ("gdgs_target_128", None, Some(128)),  // Will use special function
    ];

    let mut results = Vec::new();

    for (name, config, target_n) in &configs {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Configuration: {}", name);
        println!("═══════════════════════════════════════════════════════════════");

        let start = Instant::now();

        let mut gaussians = if name.starts_with("jittered_grid") {
            // Baseline: jittered grid
            let n = target_n.unwrap_or(256);
            println!("  Using jittered grid with {} Gaussians", n);
            create_jittered_grid(&target, n)
        } else if name.starts_with("gdgs_target") {
            // GDGS with target N
            let n = target_n.unwrap_or(256);
            println!("  Using GDGS with target N={} (auto-tuned threshold)", n);
            initialize_gdgs_with_target_n(&target, &structure_tensor, n)
        } else if let Some(cfg) = config {
            // GDGS with specific config
            println!("  Using GDGS with lap_threshold={:.4}", cfg.lap_threshold);
            initialize_gdgs(&target, &structure_tensor, cfg)
        } else {
            panic!("Invalid configuration");
        };

        let init_time = start.elapsed();
        let n_gaussians = gaussians.len();

        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);

        println!("  Gaussians: {}", n_gaussians);
        println!("  Init time: {:.2}s", init_time.as_secs_f32());
        println!("  Initial PSNR: {:.2}dB", init_psnr);

        // Skip optimization if no Gaussians
        if n_gaussians == 0 {
            println!("  ⚠️ No Gaussians created - skipping optimization");
            continue;
        }

        // Optimize with Adam
        let opt_start = Instant::now();
        let mut adam = AdamOptimizer::new();
        adam.max_iterations = iterations;

        let mut last_psnr = init_psnr;
        for iter in 0..iterations {
            if iter % 30 == 0 {
                let rendered = RendererV2::render(&gaussians, width, height);
                let psnr = compute_psnr(&rendered, &target);
                let direction = if psnr > last_psnr { "↑" } else if psnr < last_psnr { "↓" } else { "→" };
                println!("    [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                         iter, iterations, psnr, direction, psnr - last_psnr);
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

        println!("  Final PSNR: {:.2}dB ({:+.2}dB improvement)", final_psnr, improvement);
        println!("  Optimization time: {:.2}s\n", opt_time.as_secs_f32());

        results.push((name.to_string(), n_gaussians, init_psnr, final_psnr, total_time));

        // Save result
        let exp_name = format!("gdgs_{}", name);
        let mut result = TestResult::new(&exp_name, image_path);
        result.optimizer = format!("Adam with {} init", name);
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
            "config": if let Some(cfg) = config {
                json!({
                    "lap_threshold": cfg.lap_threshold,
                    "max_gaussians": cfg.max_gaussians,
                    "base_sigma": cfg.base_sigma,
                })
            } else {
                json!(null)
            },
        }));
        let _ = result.save(RESULTS_DIR);
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<25} {:>8} {:>10} {:>10} {:>8}",
             "Configuration", "N_gauss", "Init PSNR", "Final PSNR", "Time");
    println!("{:-<25} {:-<8} {:-<10} {:-<10} {:-<8}", "", "", "", "", "");

    for (name, n, init_psnr, final_psnr, time) in &results {
        println!("{:<25} {:>8} {:>9.2}dB {:>9.2}dB {:>7.1}s",
                 name, n, init_psnr, final_psnr, time);
    }

    // Efficiency analysis
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                  EFFICIENCY ANALYSIS                          ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("GDGS claims 10-100× fewer Gaussians for same quality.");
    println!("\nQuality per Gaussian (PSNR / sqrt(N)):");

    for (name, n, _, final_psnr, _) in &results {
        let efficiency = final_psnr / (*n as f32).sqrt();
        println!("  {:<25}: {:.3} dB/√N", name, efficiency);
    }

    println!("\nHYPOTHESIS: GDGS should place Gaussians more efficiently,");
    println!("achieving comparable quality with fewer Gaussians.");
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
