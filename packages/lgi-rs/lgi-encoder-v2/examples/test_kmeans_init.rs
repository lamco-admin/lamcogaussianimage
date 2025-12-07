//! Test K-Means Initialization Strategy
//!
//! Compares:
//! - Uniform Grid (current baseline)
//! - Jittered Grid (previous best)
//! - K-Means Clustering (clusters by color, position, gradient)
//!
//! Run: cargo run --release --example test_kmeans_init -p lgi-encoder-v2

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
    println!("║   K-MEANS INITIALIZATION TEST                                ║");
    println!("║   Comparing Grid vs Jittered vs K-Means                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);
    println!();

    let n_gaussians = 256;
    let iterations = 100;

    // Test 1: Uniform Grid (baseline)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Uniform Grid Initialization (Baseline)");
    println!("═══════════════════════════════════════════════════════════════");
    let grid_result = test_with_init(&target, InitType::UniformGrid, n_gaussians, iterations, image_path);

    // Test 2: Jittered Grid (previous best)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: Jittered Grid Initialization");
    println!("═══════════════════════════════════════════════════════════════");
    let jittered_result = test_with_init(&target, InitType::JitteredGrid, n_gaussians, iterations, image_path);

    // Test 3: K-Means (default weights)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 3: K-Means Initialization (Default Weights)");
    println!("═══════════════════════════════════════════════════════════════");
    let kmeans_default = test_with_init(&target, InitType::KMeans {
        weight_color: 1.0,
        weight_position: 0.5,
        weight_gradient: 0.3
    }, n_gaussians, iterations, image_path);

    // Test 4: K-Means (color-heavy)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 4: K-Means Initialization (Color-Heavy)");
    println!("═══════════════════════════════════════════════════════════════");
    let kmeans_color = test_with_init(&target, InitType::KMeans {
        weight_color: 2.0,
        weight_position: 0.3,
        weight_gradient: 0.1
    }, n_gaussians, iterations, image_path);

    // Test 5: K-Means (position-heavy)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 5: K-Means Initialization (Position-Heavy)");
    println!("═══════════════════════════════════════════════════════════════");
    let kmeans_position = test_with_init(&target, InitType::KMeans {
        weight_color: 0.5,
        weight_position: 2.0,
        weight_gradient: 0.3
    }, n_gaussians, iterations, image_path);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<30} {:>10} {:>10} {:>10} {:>8}",
             "Initialization", "Init PSNR", "Final PSNR", "Improve", "Time");
    println!("{:-<30} {:-<10} {:-<10} {:-<10} {:-<8}", "", "", "", "", "");

    let results = vec![
        ("Uniform Grid", grid_result),
        ("Jittered Grid", jittered_result),
        ("K-Means (default)", kmeans_default),
        ("K-Means (color-heavy)", kmeans_color),
        ("K-Means (position-heavy)", kmeans_position),
    ];

    for (name, (init_psnr, final_psnr, elapsed)) in &results {
        println!("{:<30} {:>10.2} {:>10.2} {:>9.2}dB {:>7.1}s",
                 name, init_psnr, final_psnr, final_psnr - init_psnr, elapsed);
    }
    println!("{:-<30} {:-<10} {:-<10} {:-<10} {:-<8}", "", "", "", "", "");

    // Find best
    let best = results.iter().max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap()).unwrap();
    println!("\nBest Final PSNR: {} with {:.2} dB", best.0, best.1.1);

    // Also report best initial PSNR (quality of initialization)
    let best_init = results.iter().max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap()).unwrap();
    println!("Best Initial PSNR: {} with {:.2} dB", best_init.0, best_init.1.0);
    println!("═══════════════════════════════════════════════════════════════");
}

#[derive(Clone)]
enum InitType {
    UniformGrid,
    JitteredGrid,
    KMeans { weight_color: f32, weight_position: f32, weight_gradient: f32 },
}

impl InitType {
    fn name(&self) -> String {
        match self {
            InitType::UniformGrid => "uniform_grid".to_string(),
            InitType::JitteredGrid => "jittered_grid".to_string(),
            InitType::KMeans { weight_color, weight_position, weight_gradient } =>
                format!("kmeans_c{}_p{}_g{}", weight_color, weight_position, weight_gradient),
        }
    }
}

/// Test with a specific initialization strategy
/// Returns: (init_psnr, final_psnr, elapsed_seconds)
fn test_with_init(
    target: &ImageBuffer<f32>,
    init_type: InitType,
    n_gaussians: usize,
    iterations: usize,
    image_path: &str,
) -> (f32, f32, f32) {
    let (width, height) = (target.width, target.height);

    let start = Instant::now();

    // Initialize Gaussians based on type
    let mut gaussians = match &init_type {
        InitType::UniformGrid => create_uniform_grid(target, n_gaussians),
        InitType::JitteredGrid => create_jittered_grid(target, n_gaussians),
        InitType::KMeans { weight_color, weight_position, weight_gradient } => {
            let tensor = StructureTensorField::compute(target, 1.2, 1.0)
                .expect("Failed to compute structure tensor");
            let config = KMeansConfig {
                k: n_gaussians,
                max_iterations: 50,
                weight_color: *weight_color,
                weight_position: *weight_position,
                weight_gradient: *weight_gradient,
            };
            initialize_kmeans(target, &tensor, &config)
        }
    };

    let init_time = start.elapsed().as_secs_f32();

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);

    println!("  Initialization: {} Gaussians in {:.2}s, PSNR={:.2}dB",
             gaussians.len(), init_time, init_psnr);

    // Optimize with Adam
    let opt_start = Instant::now();
    let mut adam = AdamOptimizer::new();
    adam.max_iterations = iterations;
    adam.optimize(&mut gaussians, target);
    let opt_time = opt_start.elapsed().as_secs_f32();

    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);

    let total_elapsed = start.elapsed().as_secs_f32();

    println!("  Optimization: {:.1}s, Final PSNR={:.2}dB ({:+.2}dB)",
             opt_time, final_psnr, final_psnr - init_psnr);

    // Save result
    let mut result = TestResult::new(&format!("kmeans_init_{}", init_type.name()), image_path);
    result.optimizer = format!("Adam with {} init", init_type.name());
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = total_elapsed;
    result.extra_metrics = Some(json!({
        "init_type": init_type.name(),
        "init_time_seconds": init_time,
        "optimization_time_seconds": opt_time,
    }));
    let _ = result.save(RESULTS_DIR);

    (init_psnr, final_psnr, total_elapsed)
}

fn create_uniform_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let grid_size = (n as f32).sqrt() as usize;
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
