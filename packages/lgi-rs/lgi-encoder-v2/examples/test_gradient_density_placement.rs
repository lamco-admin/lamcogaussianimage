//! Test Gradient-Guided Placement Density
//!
//! Hypothesis: Concentrating Gaussians in high-gradient regions (edges, textures)
//! while using fewer in smooth regions improves encoding quality per Gaussian.
//!
//! The intuition is that smooth regions can be covered by fewer, larger Gaussians,
//! while detail-rich regions need many small Gaussians.
//!
//! This test compares:
//! 1. Uniform grid: Equal density everywhere
//! 2. Gradient-weighted: Higher density in high-gradient areas
//! 3. Inverse gradient (sanity check): Should perform worse

use lgi_core::ImageBuffer;
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_math::{
    gaussian::Gaussian2D,
    parameterization::Euler,
    vec::Vector2,
    color::Color4,
};
use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("   GRADIENT-GUIDED PLACEMENT DENSITY TEST");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let image_path = "test-data/kodak-dataset/kodim03.png";
    let full_path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), image_path);

    // Load image
    let img = image::open(&full_path).expect("Failed to load image").to_rgb32f();
    let width = img.width();
    let height = img.height();

    let mut target = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            target.set_pixel(x, y, Color4::new(pixel[0], pixel[1], pixel[2], 1.0));
        }
    }

    println!("Image: kodim03 ({}x{})\n", width, height);

    // Compute gradient magnitude map
    println!("Step 1: Computing gradient magnitude...");
    let gradient_map = compute_gradient_magnitude(&target);

    // Compute statistics
    let max_grad = gradient_map.iter().cloned().fold(0.0_f32, f32::max);
    let mean_grad: f32 = gradient_map.iter().sum::<f32>() / gradient_map.len() as f32;
    println!("  Gradient magnitude: max={:.4}, mean={:.4}\n", max_grad, mean_grad);

    let n_gaussians = 256;
    let iterations = 200;

    // Test 1: Uniform grid (baseline)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: UNIFORM GRID - Equal density everywhere");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let uniform_result = run_encoding(
        "uniform_grid",
        &target,
        || init_uniform_grid(width, height, n_gaussians, &target),
        iterations,
        image_path,
    );

    // Test 2: Gradient-weighted placement
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: GRADIENT-WEIGHTED - Higher density in detail areas");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let gradient_result = run_encoding(
        "gradient_weighted",
        &target,
        || init_gradient_weighted(width, height, n_gaussians, &target, &gradient_map, max_grad, false),
        iterations,
        image_path,
    );

    // Test 3: Inverse gradient (sanity check)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: INVERSE GRADIENT (sanity check) - Should be worse");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let inverse_result = run_encoding(
        "inverse_gradient",
        &target,
        || init_gradient_weighted(width, height, n_gaussians, &target, &gradient_map, max_grad, true),
        iterations,
        image_path,
    );

    // Test 4: Adaptive sigma based on local gradient
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 4: GRADIENT-ADAPTIVE SIGMA - Placement + size adaptation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let adaptive_result = run_encoding(
        "gradient_adaptive_sigma",
        &target,
        || init_gradient_adaptive(width, height, n_gaussians, &target, &gradient_map, max_grad),
        iterations,
        image_path,
    );

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    PLACEMENT DENSITY SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:25} {:>12} {:>12} {:>12}", "Strategy", "Initial", "Final", "Improvement");
    println!("  {:-<25} {:->12} {:->12} {:->12}", "", "", "", "");

    for (name, (init_psnr, final_psnr)) in [
        ("Uniform grid", uniform_result),
        ("Gradient-weighted", gradient_result),
        ("Inverse (sanity)", inverse_result),
        ("Gradient + adaptive σ", adaptive_result),
    ] {
        let improvement = final_psnr - init_psnr;
        println!("  {:25} {:>10.2}dB {:>10.2}dB {:>+10.2}dB", name, init_psnr, final_psnr, improvement);
    }

    // Analysis
    let best_final = [uniform_result.1, gradient_result.1, inverse_result.1, adaptive_result.1]
        .into_iter()
        .fold(f32::NEG_INFINITY, f32::max);

    println!("\n  Best final PSNR: {:.2}dB", best_final);

    // Verify sanity check
    if inverse_result.1 < gradient_result.1 {
        println!("  ✓ Sanity check passed: Inverse gradient performs worse");
    } else {
        println!("  ⚠ Sanity check failed: Inverse shouldn't beat gradient-weighted");
    }

    // Compare strategies
    let gradient_advantage = gradient_result.1 - uniform_result.1;
    let adaptive_advantage = adaptive_result.1 - uniform_result.1;

    println!("\n  Gradient-weighted advantage: {:+.2}dB vs uniform", gradient_advantage);
    println!("  Adaptive sigma advantage: {:+.2}dB vs uniform", adaptive_advantage);

    if gradient_advantage > 0.1 || adaptive_advantage > 0.1 {
        println!("\n  ✓ Gradient-guided placement improves quality!");
    } else if gradient_advantage < -0.1 {
        println!("\n  ~ Uniform placement competitive - gradient weighting adds complexity without benefit");
    } else {
        println!("\n  ~ Results comparable - placement density has minimal impact after optimization");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn run_encoding<F>(
    name: &str,
    target: &ImageBuffer<f32>,
    init_fn: F,
    iterations: usize,
    image_path: &str,
) -> (f32, f32)
where
    F: FnOnce() -> Vec<Gaussian2D<f32, Euler<f32>>>,
{
    let width = target.width;
    let height = target.height;

    let mut result = TestResult::new(&format!("gradient_density_{}", name), image_path);
    result.optimizer = format!("Gradient-guided placement: {}", name);
    result.image_dimensions = (width, height);

    // Initialize Gaussians
    let start = Instant::now();
    let mut gaussians = init_fn();
    result.n_gaussians = gaussians.len();

    // Compute initial PSNR
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
    result.initial_psnr = init_psnr;

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", gaussians.len(), init_psnr);

    // Optimize with Adam
    let mut optimizer = AdamOptimizer::new();
    optimizer.max_iterations = iterations;
    result.iterations = iterations;

    let mut last_psnr = init_psnr;
    let mut final_psnr = init_psnr;

    for iter in 0..iterations {
        optimizer.optimize(&mut gaussians, target);

        if iter % 40 == 0 || iter == iterations - 1 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
            let direction = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                     iter + 1, iterations, psnr, direction, psnr - last_psnr);
            last_psnr = psnr;
            final_psnr = psnr;
        }
    }

    let elapsed = start.elapsed();

    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f64() as f32;
    result.complete();

    // Save result with image
    let final_render = RendererV2::render(&gaussians, width, height);
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    println!("  Final: PSNR={:.2}dB (improvement: {:+.2}dB)", final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.1}s", elapsed.as_secs_f64());

    (init_psnr, final_psnr)
}

fn compute_gradient_magnitude(target: &ImageBuffer<f32>) -> Vec<f32> {
    let width = target.width as i32;
    let height = target.height as i32;
    let mut gradients = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            // Sobel-like gradient computation
            let get_lum = |px: i32, py: i32| -> f32 {
                let px = px.clamp(0, width - 1) as u32;
                let py = py.clamp(0, height - 1) as u32;
                let c = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
                0.299 * c.r + 0.587 * c.g + 0.114 * c.b
            };

            // Horizontal gradient
            let gx = get_lum(x + 1, y) - get_lum(x - 1, y);
            // Vertical gradient
            let gy = get_lum(x, y + 1) - get_lum(x, y - 1);

            let magnitude = (gx * gx + gy * gy).sqrt();
            gradients.push(magnitude);
        }
    }

    gradients
}

fn init_uniform_grid(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    let cols = (count as f32).sqrt().ceil() as u32;
    let rows = (count as f32 / cols as f32).ceil() as u32;
    let cell_w = width as f32 / cols as f32;
    let cell_h = height as f32 / rows as f32;

    for i in 0..count {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;

        // Jittered position
        let jitter_x: f32 = rng.gen_range(-0.3..0.3);
        let jitter_y: f32 = rng.gen_range(-0.3..0.3);
        let px = ((col as f32 + 0.5 + jitter_x) * cell_w) as i32;
        let py = ((row as f32 + 0.5 + jitter_y) * cell_h) as i32;
        let px = px.clamp(0, width as i32 - 1) as u32;
        let py = py.clamp(0, height as i32 - 1) as u32;

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        gaussians.push(Gaussian2D::new(
            position,
            Euler::isotropic(0.04),
            color,
            1.0,
        ));
    }

    gaussians
}

fn init_gradient_weighted(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    gradient_map: &[f32],
    max_gradient: f32,
    inverse: bool,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    // Build probability distribution based on gradient magnitude
    let mut weights: Vec<f32> = gradient_map.iter().map(|&g| {
        let normalized = (g / max_gradient.max(0.001)).clamp(0.0, 1.0);
        if inverse {
            // Inverse: prefer smooth areas
            1.0 - normalized + 0.1  // +0.1 to avoid zero probability
        } else {
            // Normal: prefer high-gradient areas
            normalized + 0.1  // +0.1 to ensure smooth areas get some coverage
        }
    }).collect();

    // Normalize to create CDF
    let total: f32 = weights.iter().sum();
    for w in &mut weights {
        *w /= total;
    }

    // Create cumulative distribution
    let mut cdf = Vec::with_capacity(weights.len());
    let mut cumsum = 0.0;
    for w in &weights {
        cumsum += w;
        cdf.push(cumsum);
    }

    // Sample positions using inverse CDF
    let mut selected_positions = Vec::with_capacity(count);
    let min_dist_sq = 0.02_f32.powi(2);  // Minimum distance between Gaussians

    for _ in 0..count * 10 {  // Try more times than needed to ensure count Gaussians
        if selected_positions.len() >= count {
            break;
        }

        let r: f32 = rng.gen();
        let idx = cdf.iter().position(|&c| c >= r).unwrap_or(cdf.len() - 1);

        let px = (idx as u32) % width;
        let py = (idx as u32) / width;
        let x = px as f32 / width as f32;
        let y = py as f32 / height as f32;

        // Check minimum distance from existing
        let too_close = selected_positions.iter().any(|&(ex, ey): &(f32, f32)| {
            (x - ex).powi(2) + (y - ey).powi(2) < min_dist_sq
        });

        if !too_close {
            selected_positions.push((x, y));

            let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
            gaussians.push(Gaussian2D::new(
                Vector2::new(x, y),
                Euler::isotropic(0.04),
                color,
                1.0,
            ));
        }
    }

    // Fill remaining with random positions if needed
    while gaussians.len() < count {
        let x: f32 = rng.gen();
        let y: f32 = rng.gen();
        let px = (x * width as f32) as u32;
        let py = (y * height as f32) as u32;
        let px = px.min(width - 1);
        let py = py.min(height - 1);

        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::isotropic(0.04),
            color,
            1.0,
        ));
    }

    gaussians
}

fn init_gradient_adaptive(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    gradient_map: &[f32],
    max_gradient: f32,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    // Build probability distribution (same as gradient-weighted)
    let mut weights: Vec<f32> = gradient_map.iter().map(|&g| {
        let normalized = (g / max_gradient.max(0.001)).clamp(0.0, 1.0);
        normalized + 0.1
    }).collect();

    let total: f32 = weights.iter().sum();
    for w in &mut weights {
        *w /= total;
    }

    let mut cdf = Vec::with_capacity(weights.len());
    let mut cumsum = 0.0;
    for w in &weights {
        cumsum += w;
        cdf.push(cumsum);
    }

    // Sigma range based on gradient
    let sigma_min = 0.02;  // High gradient = small sigma
    let sigma_max = 0.06;  // Low gradient = large sigma

    let mut selected_positions = Vec::with_capacity(count);
    let min_dist_sq = 0.02_f32.powi(2);

    for _ in 0..count * 10 {
        if selected_positions.len() >= count {
            break;
        }

        let r: f32 = rng.gen();
        let idx = cdf.iter().position(|&c| c >= r).unwrap_or(cdf.len() - 1);

        let px = (idx as u32) % width;
        let py = (idx as u32) / width;
        let x = px as f32 / width as f32;
        let y = py as f32 / height as f32;

        let too_close = selected_positions.iter().any(|&(ex, ey): &(f32, f32)| {
            (x - ex).powi(2) + (y - ey).powi(2) < min_dist_sq
        });

        if !too_close {
            selected_positions.push((x, y));

            let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            // Adaptive sigma: high gradient → small sigma
            let gradient = gradient_map[idx];
            let norm_grad = (gradient / max_gradient.max(0.001)).clamp(0.0, 1.0);
            let sigma = sigma_max - norm_grad * (sigma_max - sigma_min);

            gaussians.push(Gaussian2D::new(
                Vector2::new(x, y),
                Euler::isotropic(sigma),
                color,
                1.0,
            ));
        }
    }

    // Fill remaining
    while gaussians.len() < count {
        let x: f32 = rng.gen();
        let y: f32 = rng.gen();
        let px = (x * width as f32) as u32;
        let py = (y * height as f32) as u32;
        let px = px.min(width - 1);
        let py = py.min(height - 1);

        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::isotropic(0.04),
            color,
            1.0,
        ));
    }

    gaussians
}
