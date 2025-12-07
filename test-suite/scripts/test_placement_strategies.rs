//! Test different initial Gaussian placement strategies
//!
//! Explores mathematical variants for placing Gaussians:
//! - Uniform grid (baseline)
//! - Jittered grid (random offset within cell)
//! - Golden ratio spiral
//! - Edge-aware placement (more Gaussians near edges)
//!
//! Run: cargo run --release --example test_placement_strategies -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   GAUSSIAN PLACEMENT STRATEGY COMPARISON                     ║");
    println!("║   Testing mathematical variants for initial placement        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load");
    let (width, height) = (target.width, target.height);

    let n_gaussians = 256;  // Fixed count for fair comparison
    let iterations = 100;

    println!("Image: {}x{}, Gaussians: {}, Iterations: {}\n", width, height, n_gaussians, iterations);

    // Pre-compute edge map for edge-aware placement
    let edge_map = compute_edge_map(&target);

    let strategies = vec![
        "uniform_grid",
        "jittered_grid",
        "golden_spiral",
        "edge_aware",
        "halton_sequence",
    ];

    for strategy in &strategies {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Strategy: {}", strategy);
        println!("═══════════════════════════════════════════════════════════════");

        let mut gaussians = match *strategy {
            "uniform_grid" => create_uniform_grid(&target, n_gaussians),
            "jittered_grid" => create_jittered_grid(&target, n_gaussians),
            "golden_spiral" => create_golden_spiral(&target, n_gaussians),
            "edge_aware" => create_edge_aware(&target, n_gaussians, &edge_map),
            "halton_sequence" => create_halton_sequence(&target, n_gaussians),
            _ => panic!("Unknown strategy"),
        };

        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);
        let init_loss = compute_loss(&init_render, &target);

        println!("  Initial: PSNR={:.2}dB, loss={:.6}", init_psnr, init_loss);

        // Optimize using Adam
        let mut adam = AdamOptimizer::new();
        adam.max_iterations = iterations;

        let start = Instant::now();
        let mut last_psnr = init_psnr;

        // Manual optimization loop for progress display
        for iter in 0..iterations {
            // One iteration
            let rendered = RendererV2::render(&gaussians, width, height);

            if iter % 20 == 0 {
                let psnr = compute_psnr(&rendered, &target);
                let direction = if psnr > last_psnr { "↑" } else if psnr < last_psnr { "↓" } else { "→" };
                println!("    [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                         iter, iterations, psnr, direction, psnr - last_psnr);
                last_psnr = psnr;
            }

            // Let Adam do one step
            adam.max_iterations = 1;
            adam.optimize(&mut gaussians, &target);
        }

        let elapsed = start.elapsed();

        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        let final_loss = compute_loss(&final_render, &target);
        let improvement = final_psnr - init_psnr;

        println!("  Final:   PSNR={:.2}dB, loss={:.6}", final_psnr, final_loss);
        println!("  Result:  {:+.2}dB improvement in {:.1}s\n", improvement, elapsed.as_secs_f32());

        // Save result
        let exp_name = format!("placement_{}", strategy);
        let mut result = TestResult::new(&exp_name, image_path);
        result.optimizer = format!("Adam + {} placement", strategy);
        result.image_dimensions = (width, height);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = improvement;
        result.final_loss = final_loss;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({
            "placement_strategy": strategy,
            "init_loss": init_loss,
        }));
        let _ = result.save(RESULTS_DIR);
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("HYPOTHESIS: Content-aware and mathematical distributions should");
    println!("outperform uniform grid by providing better coverage.");
    println!("═══════════════════════════════════════════════════════════════");
}

/// Uniform grid placement (baseline)
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

/// Jittered grid - uniform grid with random offset within each cell
fn create_jittered_grid(target: &ImageBuffer<f32>, n_gaussians: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n_gaussians as f32).sqrt().ceil() as usize;
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n_gaussians);
    let cell_size = 1.0 / grid_size as f32;

    // Simple deterministic jitter using index
    for i in 0..n_gaussians {
        let gx = i % grid_size;
        let gy = i / grid_size;

        // Jitter within cell (deterministic pseudo-random)
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

/// Golden ratio spiral placement
fn create_golden_spiral(target: &ImageBuffer<f32>, n_gaussians: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n_gaussians);
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let golden_angle = 2.0 * std::f32::consts::PI / (golden_ratio * golden_ratio);

    for i in 0..n_gaussians {
        // Sunflower pattern
        let r = (i as f32 / n_gaussians as f32).sqrt() * 0.45 + 0.05; // Radius from 0.05 to 0.5
        let theta = i as f32 * golden_angle;

        let x = (0.5 + r * theta.cos()).clamp(0.01, 0.99);
        let y = (0.5 + r * theta.sin()).clamp(0.01, 0.99);

        let px = ((x * width as f32) as u32).min(width - 1);
        let py = ((y * height as f32) as u32).min(height - 1);
        let color = target.get_pixel(px, py).unwrap();
        let scale = 1.0 / (n_gaussians as f32).sqrt();

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(scale, scale, 0.0),
            color,
            1.0,
        ));
    }

    gaussians
}

/// Edge-aware placement - more Gaussians near edges
fn create_edge_aware(target: &ImageBuffer<f32>, n_gaussians: usize, edge_map: &Vec<f32>) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n_gaussians);

    // Create cumulative distribution from edge map
    let total: f32 = edge_map.iter().sum();
    let mut cumulative = Vec::with_capacity(edge_map.len());
    let mut sum = 0.0;
    for &e in edge_map {
        sum += e / total;
        cumulative.push(sum);
    }

    // Sample positions weighted by edge strength
    for i in 0..n_gaussians {
        // Deterministic sampling using index
        let target_cum = (i as f32 + 0.5) / n_gaussians as f32;

        // Binary search for position
        let idx = cumulative.iter().position(|&c| c >= target_cum).unwrap_or(edge_map.len() - 1);

        let x = (idx % width as usize) as f32 / width as f32;
        let y = (idx / width as usize) as f32 / height as f32;

        let px = ((x * width as f32) as u32).min(width - 1);
        let py = ((y * height as f32) as u32).min(height - 1);
        let color = target.get_pixel(px, py).unwrap();
        let scale = 1.0 / (n_gaussians as f32).sqrt();

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(scale, scale, 0.0),
            color,
            1.0,
        ));
    }

    gaussians
}

/// Halton sequence (low-discrepancy sequence)
fn create_halton_sequence(target: &ImageBuffer<f32>, n_gaussians: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n_gaussians);

    for i in 0..n_gaussians {
        let x = halton(i + 1, 2);
        let y = halton(i + 1, 3);

        let px = ((x * width as f32) as u32).min(width - 1);
        let py = ((y * height as f32) as u32).min(height - 1);
        let color = target.get_pixel(px, py).unwrap();
        let scale = 1.0 / (n_gaussians as f32).sqrt();

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(scale, scale, 0.0),
            color,
            1.0,
        ));
    }

    gaussians
}

/// Generate Halton sequence value
fn halton(mut index: usize, base: usize) -> f32 {
    let mut result = 0.0;
    let mut f = 1.0 / base as f32;

    while index > 0 {
        result += f * (index % base) as f32;
        index /= base;
        f /= base as f32;
    }

    result
}

/// Compute simple edge map using Sobel-like operator
fn compute_edge_map(target: &ImageBuffer<f32>) -> Vec<f32> {
    let (width, height) = (target.width, target.height);
    let mut edges = vec![0.0; (width * height) as usize];

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = (y * width + x) as usize;

            // Simple gradient magnitude
            let c = target.get_pixel(x, y).unwrap();
            let r = target.get_pixel(x + 1, y).unwrap();
            let b = target.get_pixel(x, y + 1).unwrap();

            let gx = ((r.r - c.r).powi(2) + (r.g - c.g).powi(2) + (r.b - c.b).powi(2)).sqrt();
            let gy = ((b.r - c.r).powi(2) + (b.g - c.g).powi(2) + (b.b - c.b).powi(2)).sqrt();

            edges[idx] = (gx * gx + gy * gy).sqrt() + 0.01; // Small base to avoid zero
        }
    }

    edges
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

fn compute_loss(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height) as f32;
    let mut loss = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        loss += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    loss / count
}
