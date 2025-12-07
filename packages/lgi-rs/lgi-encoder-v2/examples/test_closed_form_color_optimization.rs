//! Test Closed-Form Color Optimization
//!
//! Hypothesis: RGB colors can be solved analytically via linear least squares,
//! eliminating the need for iterative color optimization.
//!
//! Given fixed Gaussian positions and shapes, the color that minimizes MSE
//! can be computed directly: c* = argmin_c ||Ac - b||^2
//!
//! This test compares:
//! 1. Baseline: Adam optimizes all parameters including color
//! 2. Closed-form: Solve colors analytically, Adam only optimizes position/shape

use lgi_core::{
    ImageBuffer,
    Renderer, RenderConfig,
    Initializer, InitStrategy,
};
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_math::{
    gaussian::Gaussian2D,
    parameterization::Euler,
    vec::Vector2,
    color::Color4,
};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("    CLOSED-FORM COLOR OPTIMIZATION TEST - Analytical vs Iterative");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let image_path = "test-data/kodak-dataset/kodim03.png";
    let full_path = format!("{}/../../../../{}", env!("CARGO_MANIFEST_DIR"), image_path);

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

    let n_gaussians = 256;
    let iterations = 200;

    // Test 1: Baseline (Adam optimizes everything)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: BASELINE - Adam optimizes all parameters");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let baseline_result = run_baseline(&target, n_gaussians, iterations, image_path);

    // Test 2: Closed-form colors
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: CLOSED-FORM - Analytical color solve every 20 iterations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let closedform_result = run_closedform(&target, n_gaussians, iterations, image_path);

    // Test 3: Hybrid (closed-form init, then Adam)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: HYBRID - Closed-form init, then full Adam");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let hybrid_result = run_hybrid(&target, n_gaussians, iterations, image_path);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                           COMPARISON SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:25} {:>12} {:>12} {:>12}", "Method", "Initial", "Final", "Improvement");
    println!("  {:->25} {:->12} {:->12} {:->12}", "", "", "", "");

    for (name, (init_psnr, final_psnr)) in [
        ("Baseline (full Adam)", baseline_result),
        ("Closed-form colors", closedform_result),
        ("Hybrid", hybrid_result),
    ] {
        let improvement = final_psnr - init_psnr;
        println!("  {:25} {:>10.2}dB {:>10.2}dB {:>+10.2}dB", name, init_psnr, final_psnr, improvement);
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn run_baseline(
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    iterations: usize,
    image_path: &str,
) -> (f32, f32) {
    let width = target.width;
    let height = target.height;

    let renderer = Renderer::new(RenderConfig {
        width,
        height,
        background: Color4::new(0.0, 0.0, 0.0, 1.0),
        ..Default::default()
    });

    let initializer = Initializer::new(InitStrategy::JitteredGrid { seed: 42 });
    let mut gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = initializer.initialize(
        n_gaussians, width, height, target
    );

    let rendered = renderer.render(&gaussians);
    let init_psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
    println!("  Initial: {} Gaussians, PSNR={:.2}dB", gaussians.len(), init_psnr);

    let mut optimizer = AdamOptimizer::new(0.0002, 0.02, 0.005, 0.05, 0.9, 0.999);
    let mut last_psnr = init_psnr;

    for iter in 0..iterations {
        optimizer.step(&mut gaussians, target, &renderer);

        if iter % 40 == 0 || iter == iterations - 1 {
            let rendered = renderer.render(&gaussians);
            let psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {}", iter + 1, iterations, psnr, dir);
            last_psnr = psnr;
        }
    }

    (init_psnr, last_psnr)
}

fn run_closedform(
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    iterations: usize,
    image_path: &str,
) -> (f32, f32) {
    let width = target.width;
    let height = target.height;

    let renderer = Renderer::new(RenderConfig {
        width,
        height,
        background: Color4::new(0.0, 0.0, 0.0, 1.0),
        ..Default::default()
    });

    let initializer = Initializer::new(InitStrategy::JitteredGrid { seed: 42 });
    let mut gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = initializer.initialize(
        n_gaussians, width, height, target
    );

    // Solve colors initially
    solve_colors_closedform(&mut gaussians, target, &renderer);

    let rendered = renderer.render(&gaussians);
    let init_psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
    println!("  Initial (with closed-form colors): PSNR={:.2}dB", init_psnr);

    // Adam with zero color learning rate (only optimize position/shape)
    let mut optimizer = AdamOptimizer::new(0.0002, 0.0, 0.005, 0.05, 0.9, 0.999);
    let mut last_psnr = init_psnr;

    for iter in 0..iterations {
        optimizer.step(&mut gaussians, target, &renderer);

        // Re-solve colors periodically
        if iter % 20 == 0 {
            solve_colors_closedform(&mut gaussians, target, &renderer);
        }

        if iter % 40 == 0 || iter == iterations - 1 {
            let rendered = renderer.render(&gaussians);
            let psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {}", iter + 1, iterations, psnr, dir);
            last_psnr = psnr;
        }
    }

    (init_psnr, last_psnr)
}

fn run_hybrid(
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    iterations: usize,
    image_path: &str,
) -> (f32, f32) {
    let width = target.width;
    let height = target.height;

    let renderer = Renderer::new(RenderConfig {
        width,
        height,
        background: Color4::new(0.0, 0.0, 0.0, 1.0),
        ..Default::default()
    });

    let initializer = Initializer::new(InitStrategy::JitteredGrid { seed: 42 });
    let mut gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = initializer.initialize(
        n_gaussians, width, height, target
    );

    // Solve colors initially
    solve_colors_closedform(&mut gaussians, target, &renderer);

    let rendered = renderer.render(&gaussians);
    let init_psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
    println!("  Initial (with closed-form colors): PSNR={:.2}dB", init_psnr);

    // Full Adam optimization
    let mut optimizer = AdamOptimizer::new(0.0002, 0.02, 0.005, 0.05, 0.9, 0.999);
    let mut last_psnr = init_psnr;

    for iter in 0..iterations {
        optimizer.step(&mut gaussians, target, &renderer);

        if iter % 40 == 0 || iter == iterations - 1 {
            let rendered = renderer.render(&gaussians);
            let psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {}", iter + 1, iterations, psnr, dir);
            last_psnr = psnr;
        }
    }

    (init_psnr, last_psnr)
}

/// Solve Gaussian colors using closed-form linear least squares
///
/// For each Gaussian, compute the optimal color that minimizes MSE
/// given its current position and shape.
///
/// Simplified approach: For each Gaussian, average the target colors
/// at pixels where the Gaussian has significant contribution.
fn solve_colors_closedform(
    gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    renderer: &Renderer,
) {
    let width = target.width;
    let height = target.height;

    for g in gaussians.iter_mut() {
        let pos = g.position();
        let scale = g.scale();

        // Compute bounding box (3 sigma)
        let half_extent = (scale.x.max(scale.y) * 3.0) as i32;
        let x_min = ((pos.x as i32) - half_extent).max(0) as u32;
        let x_max = ((pos.x as i32) + half_extent).min(width as i32 - 1) as u32;
        let y_min = ((pos.y as i32) - half_extent).max(0) as u32;
        let y_max = ((pos.y as i32) + half_extent).min(height as i32 - 1) as u32;

        // Weighted average of target colors
        let mut sum_r = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;
        let mut sum_w = 0.0f32;

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let dx = x as f32 - pos.x;
                let dy = y as f32 - pos.y;

                // Evaluate Gaussian weight at this pixel
                let weight = g.evaluate(dx, dy);

                if weight > 0.01 {
                    let target_color = target.get_pixel(x, y).unwrap();
                    sum_r += target_color.r * weight;
                    sum_g += target_color.g * weight;
                    sum_b += target_color.b * weight;
                    sum_w += weight;
                }
            }
        }

        // Set optimal color
        if sum_w > 0.01 {
            g.color = Color4::new(
                (sum_r / sum_w).clamp(0.0, 1.0),
                (sum_g / sum_w).clamp(0.0, 1.0),
                (sum_b / sum_w).clamp(0.0, 1.0),
                1.0,
            );
        }
    }
}
