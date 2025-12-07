//! Test gradient peak initialization strategy
//!
//! Compares gradient peak initialization (with structure tensor) against uniform grid.
//! Gradient peak init places Gaussians at local gradient maxima (edges, corners).
//!
//! Run: cargo run --release --example test_gradient_peak_init -p lgi-encoder-v2

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::gradient_peak_init::{initialize_gradient_peaks, GradientPeakConfig};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   GRADIENT PEAK INITIALIZATION TEST                          ║");
    println!("║   Comparing peak-based vs grid-based placement               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);
    println!("Computing structure tensor...\n");

    let structure_tensor = StructureTensorField::compute(&target, 1.0, 2.0)
        .expect("Failed to compute structure tensor");

    let iterations = 100;

    // Test configurations
    let configs = vec![
        ("uniform_grid_256", None, 256),
        ("gradient_peak_default", Some(GradientPeakConfig::default()), 0), // Dynamic count
        ("gradient_peak_200", Some(GradientPeakConfig {
            target_peak_gaussians: 200,
            background_grid_size: 4,
            ..GradientPeakConfig::default()
        }), 0),
        ("gradient_peak_only", Some(GradientPeakConfig {
            target_peak_gaussians: 256,
            add_background_grid: false,  // No background grid
            ..GradientPeakConfig::default()
        }), 0),
        ("gradient_peak_low_thresh", Some(GradientPeakConfig {
            gradient_threshold: 0.005,  // Lower threshold = more peaks
            target_peak_gaussians: 200,
            ..GradientPeakConfig::default()
        }), 0),
    ];

    for (name, config, fixed_count) in &configs {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Configuration: {}", name);
        println!("═══════════════════════════════════════════════════════════════");

        let start = Instant::now();

        let mut gaussians = if let Some(cfg) = config {
            // Use gradient peak initialization
            initialize_gradient_peaks(&target, &structure_tensor, cfg)
        } else {
            // Use uniform grid baseline
            create_uniform_grid(&target, *fixed_count)
        };

        let init_time = start.elapsed();
        let n_gaussians = gaussians.len();

        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);

        println!("  Gaussians: {}", n_gaussians);
        println!("  Init time: {:.2}s", init_time.as_secs_f32());
        println!("  Initial PSNR: {:.2}dB", init_psnr);

        // Optimize with Adam
        let opt_start = Instant::now();
        let mut adam = AdamOptimizer::new();
        adam.max_iterations = iterations;

        let mut last_psnr = init_psnr;
        for iter in 0..iterations {
            let rendered = RendererV2::render(&gaussians, width, height);

            if iter % 20 == 0 {
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

        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        let improvement = final_psnr - init_psnr;

        println!("  Final PSNR: {:.2}dB ({:+.2}dB improvement)", final_psnr, improvement);
        println!("  Optimization time: {:.2}s\n", opt_time.as_secs_f32());

        // Save result
        let exp_name = format!("gradient_peak_{}", name);
        let mut result = TestResult::new(&exp_name, image_path);
        result.optimizer = format!("Adam with {} init", name);
        result.image_dimensions = (width, height);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = improvement;
        result.elapsed_seconds = (init_time + opt_time).as_secs_f32();
        result.extra_metrics = Some(json!({
            "init_type": name,
            "init_time_seconds": init_time.as_secs_f32(),
            "optimization_time_seconds": opt_time.as_secs_f32(),
        }));
        let _ = result.save(RESULTS_DIR);
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("HYPOTHESIS: Gradient peak initialization should provide better");
    println!("starting points by placing Gaussians at edges and corners.");
    println!("═══════════════════════════════════════════════════════════════");
}

/// Create uniform grid placement (baseline)
fn create_uniform_grid(target: &ImageBuffer<f32>, n_gaussians: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n_gaussians as f32).sqrt().ceil() as usize;
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n_gaussians);

    for i in 0..n_gaussians {
        let gx = i % grid_size;
        let gy = i / grid_size;
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
