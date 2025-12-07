//! Test Position Probability Map (PPM) Guided Placement
//!
//! PPM combines entropy + gradient + saliency into a probability distribution
//! for importance sampling of Gaussian positions. This tests whether
//! placing Gaussians where complexity is highest improves quality.
//!
//! Run: cargo run --release --example test_ppm_placement -p lgi-encoder-v2

use lgi_core::{
    ImageBuffer,
    entropy::compute_entropy_map,
    position_probability_map::{PositionProbabilityMap, PPMConfig},
    laplacian::compute_laplacian,
};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   POSITION PROBABILITY MAP (PPM) GUIDED PLACEMENT            ║");
    println!("║   Importance sampling for Gaussian positions                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);
    println!("");

    // Step 1: Compute entropy map
    println!("═══════════════════════════════════════════════════════════════");
    println!("Step 1: Computing Entropy and Gradient Maps");
    println!("═══════════════════════════════════════════════════════════════");

    let entropy_start = Instant::now();

    // Compute per-pixel entropy (using tile-based variance)
    let tile_entropy = compute_entropy_map(&target, 8);
    let mut entropy_map = vec![0.0f32; (width * height) as usize];

    // Expand tile entropy to per-pixel
    let tiles_x = (width as usize + 7) / 8;
    for y in 0..height {
        for x in 0..width {
            let tx = (x as usize) / 8;
            let ty = (y as usize) / 8;
            if ty < tile_entropy.len() && tx < tile_entropy[ty].len() {
                entropy_map[(y * width + x) as usize] = tile_entropy[ty][tx];
            }
        }
    }

    // Compute gradient map (using Laplacian as proxy)
    let gradient_map = compute_laplacian(&target);

    let entropy_time = entropy_start.elapsed().as_secs_f32();

    // Statistics
    let entropy_sum: f32 = entropy_map.iter().sum();
    let gradient_sum: f32 = gradient_map.iter().sum();
    let entropy_max = entropy_map.iter().cloned().fold(0.0f32, f32::max);
    let gradient_max = gradient_map.iter().cloned().fold(0.0f32, f32::max);

    println!("  Computation time: {:.3}s", entropy_time);
    println!("  Entropy: sum={:.2}, max={:.4}", entropy_sum, entropy_max);
    println!("  Gradient: sum={:.2}, max={:.4}", gradient_sum, gradient_max);

    // Step 2: Create PPM with different weightings
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 2: Testing Different PPM Weightings");
    println!("═══════════════════════════════════════════════════════════════");

    let n_gaussians = 256;
    let configs = vec![
        ("Entropy-only (0.9/0.1)", PPMConfig { weight_entropy: 0.9, weight_gradient: 0.1, weight_saliency: 0.0, entropy_tile_size: 8 }),
        ("Balanced (0.6/0.4)", PPMConfig { weight_entropy: 0.6, weight_gradient: 0.4, weight_saliency: 0.0, entropy_tile_size: 8 }),
        ("Gradient-heavy (0.3/0.7)", PPMConfig { weight_entropy: 0.3, weight_gradient: 0.7, weight_saliency: 0.0, entropy_tile_size: 8 }),
        ("Gradient-only (0.0/1.0)", PPMConfig { weight_entropy: 0.0, weight_gradient: 1.0, weight_saliency: 0.0, entropy_tile_size: 8 }),
    ];

    let mut results = Vec::new();

    for (name, config) in &configs {
        println!("\n--- {} ---", name);

        // Create PPM
        let ppm = PositionProbabilityMap::from_entropy_gradient(
            &entropy_map,
            &gradient_map,
            width,
            height,
            config,
        );

        // Sample positions using top-N (deterministic)
        let positions = ppm.sample_positions_unique(n_gaussians);

        // Create Gaussians at sampled positions
        let gaussians = create_gaussians_at_positions(&target, &positions, width, height);

        // Render and measure initial
        let render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&render, &target);
        println!("  Initial PSNR: {:.2} dB", init_psnr);

        // Optimize with Adam
        let mut opt_gaussians = gaussians.clone();
        let mut adam = AdamOptimizer::new();
        adam.max_iterations = 100;

        let opt_start = Instant::now();
        adam.optimize(&mut opt_gaussians, &target);
        let opt_time = opt_start.elapsed().as_secs_f32();

        let opt_render = RendererV2::render(&opt_gaussians, width, height);
        let final_psnr = compute_psnr(&opt_render, &target);

        println!("  Final PSNR: {:.2} dB (+{:.2} dB)", final_psnr, final_psnr - init_psnr);
        println!("  Time: {:.2}s", opt_time);

        results.push((name.to_string(), init_psnr, final_psnr, opt_time));
    }

    // Step 3: Compare with uniform grid (baseline)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 3: Comparing with Uniform Grid (Baseline)");
    println!("═══════════════════════════════════════════════════════════════");

    let grid_gaussians = create_uniform_grid(&target, n_gaussians);
    let grid_render = RendererV2::render(&grid_gaussians, width, height);
    let grid_init_psnr = compute_psnr(&grid_render, &target);
    println!("  Grid Initial PSNR: {:.2} dB", grid_init_psnr);

    let mut grid_opt = grid_gaussians.clone();
    let mut adam_grid = AdamOptimizer::new();
    adam_grid.max_iterations = 100;
    adam_grid.optimize(&mut grid_opt, &target);

    let grid_opt_render = RendererV2::render(&grid_opt, width, height);
    let grid_final_psnr = compute_psnr(&grid_opt_render, &target);
    println!("  Grid Final PSNR: {:.2} dB (+{:.2} dB)", grid_final_psnr, grid_final_psnr - grid_init_psnr);

    // Summary
    println!("\n\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY                                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("");
    println!("{:<30} {:>10} {:>10} {:>10}", "Method", "Init dB", "Final dB", "vs Grid");
    println!("{}", "─".repeat(65));

    for (name, init_psnr, final_psnr, _) in &results {
        let delta = final_psnr - grid_final_psnr;
        println!("{:<30} {:>10.2} {:>10.2} {:>+10.2}", name, init_psnr, final_psnr, delta);
    }
    println!("{:<30} {:>10.2} {:>10.2} {:>10}", "Uniform Grid (baseline)", grid_init_psnr, grid_final_psnr, "—");

    // Find best
    let best = results.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    if let Some((name, _, final_psnr, _)) = best {
        println!("\nBest PPM config: {} ({:.2} dB)", name, final_psnr);
        if *final_psnr > grid_final_psnr {
            println!("PPM beats Grid by {:+.2} dB!", final_psnr - grid_final_psnr);
        } else {
            println!("Grid is still better by {:+.2} dB", grid_final_psnr - final_psnr);
        }
    }

    // Save result for best PPM
    if let Some((name, init_psnr, final_psnr, time)) = best {
        let mut result = TestResult::new("ppm_placement", image_path);
        result.optimizer = format!("PPM Guided Placement - {}", name);
        result.n_gaussians = n_gaussians;
        result.iterations = 100;
        result.initial_psnr = *init_psnr;
        result.final_psnr = *final_psnr;
        result.improvement_db = *final_psnr - *init_psnr;
        result.elapsed_seconds = *time;
        result.image_dimensions = (width, height);
        result.set_param("grid_init_psnr", grid_init_psnr);
        result.set_param("grid_final_psnr", grid_final_psnr);
        result.set_param("ppm_vs_grid_delta", *final_psnr - grid_final_psnr);
        result.complete();

        let _ = result.save(RESULTS_DIR);
    }

    println!("\n✓ PPM placement test complete!");
}

fn create_gaussians_at_positions(
    target: &ImageBuffer<f32>,
    positions: &[(u32, u32)],
    width: u32,
    height: u32,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    positions
        .iter()
        .map(|&(px, py)| {
            let x = (px as f32 + 0.5) / width as f32;
            let y = (py as f32 + 0.5) / height as f32;

            let color = target.get_pixel(px.min(width - 1), py.min(height - 1))
                .cloned()
                .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            Gaussian2D {
                position: Vector2::new(x, y),
                shape: Euler {
                    scale_x: 0.02,
                    scale_y: 0.02,
                    rotation: 0.0,
                },
                color: Color4::rgb(color.r, color.g, color.b),
                opacity: 0.8,
            }
        })
        .collect()
}

fn create_uniform_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let mut gaussians = Vec::with_capacity(n);

    for i in 0..grid_size {
        for j in 0..grid_size {
            if gaussians.len() >= n {
                break;
            }

            let x = (i as f32 + 0.5) / grid_size as f32;
            let y = (j as f32 + 0.5) / grid_size as f32;

            let px = (x * width as f32) as u32;
            let py = (y * height as f32) as u32;

            let color = target.get_pixel(px.min(width - 1), py.min(height - 1))
                .cloned()
                .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            gaussians.push(Gaussian2D {
                position: Vector2::new(x, y),
                shape: Euler {
                    scale_x: 0.02,
                    scale_y: 0.02,
                    rotation: 0.0,
                },
                color: Color4::rgb(color.r, color.g, color.b),
                opacity: 0.8,
            });
        }
    }

    gaussians
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mse: f32 = rendered
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(r, t)| {
            let dr = r.r - t.r;
            let dg = r.g - t.g;
            let db = r.b - t.b;
            (dr * dr + dg * dg + db * db) / 3.0
        })
        .sum::<f32>()
        / rendered.data.len() as f32;

    if mse < 1e-10 {
        100.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}
