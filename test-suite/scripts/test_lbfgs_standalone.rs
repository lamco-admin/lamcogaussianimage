//! L-BFGS Standalone Optimizer Test
//!
//! Tests L-BFGS as a standalone optimizer (not just as refinement after Adam).
//! Goal: Determine if L-BFGS can work from scratch with proper initialization.
//!
//! Key differences from Adam:
//! - L-BFGS uses second-order curvature information
//! - May need different initialization strategy
//! - May benefit from different parameterization
//!
//! Run: cargo run --release --example test_lbfgs_standalone -p lgi-encoder-v2

use lgi_core::{ImageBuffer, lbfgs::LBFGS};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       L-BFGS STANDALONE OPTIMIZER TEST                       ║");
    println!("║       Testing L-BFGS without Adam warm-start                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let test_images = vec![
        "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png",
    ];

    for image_path in test_images {
        let image_name = image_path.split('/').last().unwrap_or("unknown");
        println!("\n{}", "=".repeat(70));
        println!("Testing: {}", image_name);
        println!("{}\n", "=".repeat(70));

        let target = ImageBuffer::load(image_path).expect("Failed to load image");
        let (width, height) = (target.width, target.height);
        println!("Image: {}x{}", width, height);

        // Test 1: L-BFGS from scratch with jittered grid
        println!("\n--- Test 1: L-BFGS from scratch (jittered grid, 256 Gaussians) ---");
        let result1 = run_lbfgs_standalone(&target, 256, 100, image_path, "jittered_grid");

        // Test 2: L-BFGS with more iterations
        println!("\n--- Test 2: L-BFGS with 200 iterations ---");
        let result2 = run_lbfgs_standalone(&target, 256, 200, image_path, "more_iters");

        // Test 3: L-BFGS with smaller history (faster per-iteration)
        println!("\n--- Test 3: L-BFGS with history=5 ---");
        let result3 = run_lbfgs_history(&target, 256, 100, 5, image_path);

        // Test 4: L-BFGS with larger history (better Hessian approximation)
        println!("\n--- Test 4: L-BFGS with history=20 ---");
        let result4 = run_lbfgs_history(&target, 256, 100, 20, image_path);

        // Summary
        println!("\n{}", "=".repeat(70));
        println!("SUMMARY for {}", image_name);
        println!("{}", "=".repeat(70));
        println!("{:<30} {:>10} {:>10} {:>12}", "Test", "PSNR", "Improve", "Time");
        println!("{:-<30} {:-<10} {:-<10} {:-<12}", "", "", "", "");
        println!("{:<30} {:>9.2}dB {:>+9.2}dB {:>10.1}s",
            "L-BFGS 100 iter", result1.0, result1.1, result1.2);
        println!("{:<30} {:>9.2}dB {:>+9.2}dB {:>10.1}s",
            "L-BFGS 200 iter", result2.0, result2.1, result2.2);
        println!("{:<30} {:>9.2}dB {:>+9.2}dB {:>10.1}s",
            "L-BFGS history=5", result3.0, result3.1, result3.2);
        println!("{:<30} {:>9.2}dB {:>+9.2}dB {:>10.1}s",
            "L-BFGS history=20", result4.0, result4.1, result4.2);
    }

    println!("\n\nAll tests complete!");
}

/// Run L-BFGS standalone with specified parameters
fn run_lbfgs_standalone(
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    max_iterations: usize,
    image_path: &str,
    variant: &str,
) -> (f32, f32, f32) {
    run_lbfgs_history(target, n_gaussians, max_iterations, 10, image_path)
}

/// Run L-BFGS with configurable history size
fn run_lbfgs_history(
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    max_iterations: usize,
    history_size: usize,
    image_path: &str,
) -> (f32, f32, f32) {
    let (width, height) = (target.width, target.height);

    // Initialize Gaussians with jittered grid
    let mut gaussians = create_jittered_grid(target, n_gaussians);

    // Initial render and PSNR
    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);
    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n_gaussians, init_psnr);

    let start = Instant::now();

    // Flatten to parameter vector (7 params per Gaussian: x, y, r, g, b, sx, sy)
    let mut x = flatten_gaussians(&gaussians);

    // Create L-BFGS optimizer
    let mut lbfgs = LBFGS::new(history_size);

    // Manual optimization loop with analytical gradients
    let mut gradient = compute_gradient_analytical(&gaussians, target);
    let mut rendered = RendererV2::render(&gaussians, width, height);
    let mut f_x = compute_mse(&rendered, target);
    let mut last_psnr = init_psnr;

    for iter in 0..max_iterations {
        // Check convergence
        let grad_norm: f32 = gradient.iter().map(|&g| g * g).sum::<f32>().sqrt();
        if grad_norm < 1e-8 {
            println!("    Converged at iter {} (grad_norm = {:.2e})", iter, grad_norm);
            break;
        }

        // Compute search direction
        let direction = lbfgs.compute_direction(&gradient);

        // Line search with Armijo condition
        let alpha = line_search(&x, &direction, &gradient, f_x, target);

        // Take step
        let x_prev = x.clone();
        let grad_prev = gradient.clone();

        for i in 0..x.len() {
            x[i] += alpha * direction[i];
        }

        // Apply constraints
        apply_constraints(&mut x, n_gaussians);

        // Update Gaussians from parameters
        unflatten_to_gaussians(&x, &mut gaussians);

        // Compute new gradient and loss
        gradient = compute_gradient_analytical(&gaussians, target);
        rendered = RendererV2::render(&gaussians, width, height);
        f_x = compute_mse(&rendered, target);

        // Update L-BFGS history
        lbfgs.update(&x_prev, &x, &grad_prev, &gradient);

        // Progress output
        if iter % 10 == 0 || iter == max_iterations - 1 {
            let psnr = compute_psnr(&rendered, target);
            let direction = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {} loss={:.6} α={:.2e}",
                iter, max_iterations, psnr, direction, f_x, alpha);
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);
    let improvement = final_psnr - init_psnr;

    println!("  Final: PSNR={:.2}dB ({:+.2}dB) in {:.1}s", final_psnr, improvement, elapsed);

    // Save result
    let mut result = TestResult::new(&format!("lbfgs_standalone_h{}", history_size), image_path);
    result.optimizer = format!("L-BFGS standalone (history={})", history_size);
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = max_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.elapsed_seconds = elapsed;
    result.set_param("history_size", history_size);
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (final_psnr, improvement, elapsed)
}

/// Line search with Armijo condition
fn line_search(
    x: &[f32],
    direction: &[f32],
    gradient: &[f32],
    f_x: f32,
    target: &ImageBuffer<f32>,
) -> f32 {
    let mut alpha = 1.0;
    let c1 = 1e-4;  // Armijo constant

    let directional_derivative: f32 = gradient.iter()
        .zip(direction.iter())
        .map(|(&g, &d)| g * d)
        .sum();

    if directional_derivative >= 0.0 {
        return 1e-6;  // Not a descent direction
    }

    let n_gaussians = x.len() / 7;

    for _ in 0..15 {
        let mut x_new: Vec<f32> = x.iter()
            .zip(direction.iter())
            .map(|(&xi, &di)| xi + alpha * di)
            .collect();

        apply_constraints(&mut x_new, n_gaussians);

        let mut temp_gaussians = vec![Gaussian2D::new(
            Vector2::zero(),
            Euler::new(0.01, 0.01, 0.0),
            Color4::new(0.5, 0.5, 0.5, 1.0),
            1.0,
        ); n_gaussians];
        unflatten_to_gaussians(&x_new, &mut temp_gaussians);

        let rendered = RendererV2::render(&temp_gaussians, target.width, target.height);
        let f_new = compute_mse(&rendered, target);

        if f_new <= f_x + c1 * alpha * directional_derivative {
            return alpha;
        }

        alpha *= 0.5;
    }

    alpha
}

/// Apply constraints to parameters
fn apply_constraints(x: &mut [f32], n_gaussians: usize) {
    for i in 0..n_gaussians {
        let base = i * 7;
        // Position: [0, 1]
        x[base] = x[base].clamp(0.0, 1.0);
        x[base + 1] = x[base + 1].clamp(0.0, 1.0);
        // Color: [0, 1]
        x[base + 2] = x[base + 2].clamp(0.0, 1.0);
        x[base + 3] = x[base + 3].clamp(0.0, 1.0);
        x[base + 4] = x[base + 4].clamp(0.0, 1.0);
        // Scale: [0.001, 0.5]
        x[base + 5] = x[base + 5].clamp(0.001, 0.5);
        x[base + 6] = x[base + 6].clamp(0.001, 0.5);
    }
}

/// Flatten Gaussians to parameter vector
fn flatten_gaussians(gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Vec<f32> {
    let mut x = Vec::with_capacity(gaussians.len() * 7);
    for g in gaussians {
        x.push(g.position.x);
        x.push(g.position.y);
        x.push(g.color.r);
        x.push(g.color.g);
        x.push(g.color.b);
        x.push(g.shape.scale_x);
        x.push(g.shape.scale_y);
    }
    x
}

/// Unflatten parameter vector to Gaussians
fn unflatten_to_gaussians(x: &[f32], gaussians: &mut [Gaussian2D<f32, Euler<f32>>]) {
    for (i, g) in gaussians.iter_mut().enumerate() {
        let base = i * 7;
        g.position.x = x[base];
        g.position.y = x[base + 1];
        g.color.r = x[base + 2];
        g.color.g = x[base + 3];
        g.color.b = x[base + 4];
        g.shape.scale_x = x[base + 5];
        g.shape.scale_y = x[base + 6];
    }
}

/// Compute gradient using analytical derivatives
fn compute_gradient_analytical(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
) -> Vec<f32> {
    let width = target.width;
    let height = target.height;
    let n = gaussians.len();

    let mut grads: Vec<[f32; 7]> = vec![[0.0; 7]; n];
    let rendered = RendererV2::render(gaussians, width, height);

    for y in 0..height {
        for x in 0..width {
            let px = (x as f32 + 0.5) / width as f32;
            let py = (y as f32 + 0.5) / height as f32;

            let rendered_color = rendered.get_pixel(x, y).unwrap();
            let target_color = target.get_pixel(x, y).unwrap();

            let error_r = 2.0 * (rendered_color.r - target_color.r);
            let error_g = 2.0 * (rendered_color.g - target_color.g);
            let error_b = 2.0 * (rendered_color.b - target_color.b);

            for (i, gaussian) in gaussians.iter().enumerate() {
                let dx = px - gaussian.position.x;
                let dy = py - gaussian.position.y;

                let sx = gaussian.shape.scale_x.max(1e-6);
                let sy = gaussian.shape.scale_y.max(1e-6);

                let dist_sq = (dx / sx).powi(2) + (dy / sy).powi(2);
                if dist_sq > 16.0 { continue; }

                let weight = (-0.5 * dist_sq).exp();
                if weight < 1e-6 { continue; }

                // Color gradients
                grads[i][2] += error_r * weight;
                grads[i][3] += error_g * weight;
                grads[i][4] += error_b * weight;

                // Position gradients
                let error_weighted = error_r * gaussian.color.r
                                   + error_g * gaussian.color.g
                                   + error_b * gaussian.color.b;

                grads[i][0] += error_weighted * weight * dx / (sx * sx);
                grads[i][1] += error_weighted * weight * dy / (sy * sy);

                // Scale gradients
                grads[i][5] += error_weighted * weight * (dx * dx) / (sx * sx * sx);
                grads[i][6] += error_weighted * weight * (dy * dy) / (sy * sy * sy);
            }
        }
    }

    // Flatten and normalize
    let pixel_count = (width * height) as f32;
    let mut flat = Vec::with_capacity(n * 7);
    for g in &grads {
        for &v in g {
            flat.push(v / pixel_count);
        }
    }
    flat
}

/// Create jittered grid initialization
fn create_jittered_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n);
    let cell_size = 1.0 / grid_size as f32;

    for i in 0..n {
        let gx = i % grid_size;
        let gy = i / grid_size;

        // Deterministic jitter
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

fn compute_mse(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse / (rendered.width * rendered.height * 3) as f32
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mse = compute_mse(rendered, target);
    if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}
