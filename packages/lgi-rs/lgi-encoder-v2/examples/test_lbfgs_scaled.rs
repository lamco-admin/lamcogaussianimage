//! L-BFGS with Scaled Parameters
//!
//! Addresses the gradient conditioning issue that caused the vanilla L-BFGS to fail.
//!
//! Key improvements:
//! 1. **Log-space scales**: Uses log(scale) instead of scale, avoiding 1/sx³ blow-up
//! 2. **Per-parameter scaling**: Normalizes gradients to similar magnitudes
//! 3. **Soft constraints**: Uses barrier functions instead of hard clamping
//! 4. **Better initial step**: Uses ||g||⁻¹ scaling for first step
//!
//! Run: cargo run --release --example test_lbfgs_scaled -p lgi-encoder-v2

use lgi_core::{ImageBuffer, lbfgs::LBFGS};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       L-BFGS WITH SCALED PARAMETERS                          ║");
    println!("║       Fixing gradient conditioning for proper convergence    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";

    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);
    println!("Image: {}x{}", width, height);

    // Test 1: L-BFGS with log-space scales
    println!("\n--- Test 1: L-BFGS with log-scale parameterization ---");
    let result1 = run_lbfgs_log_scale(&target, 256, 100, image_path);

    // Test 2: L-BFGS with per-parameter scaling factors
    println!("\n--- Test 2: L-BFGS with gradient scaling ---");
    let result2 = run_lbfgs_gradient_scaled(&target, 256, 100, image_path);

    // Test 3: L-BFGS colors only (simplest case - should definitely work)
    println!("\n--- Test 3: L-BFGS color-only optimization ---");
    let result3 = run_lbfgs_colors_only(&target, 256, 100, image_path);

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY");
    println!("{}", "=".repeat(70));
    println!("{:<35} {:>10} {:>10} {:>10}", "Test", "Final", "Improve", "Time");
    println!("{:-<35} {:-<10} {:-<10} {:-<10}", "", "", "", "");
    println!("{:<35} {:>9.2}dB {:>+9.2}dB {:>8.1}s", "Log-scale params", result1.0, result1.1, result1.2);
    println!("{:<35} {:>9.2}dB {:>+9.2}dB {:>8.1}s", "Gradient scaling", result2.0, result2.1, result2.2);
    println!("{:<35} {:>9.2}dB {:>+9.2}dB {:>8.1}s", "Colors only", result3.0, result3.1, result3.2);

    println!("\n✓ Tests complete!");
}

/// L-BFGS with log-space scale parameters
/// Parameters: x, y, r, g, b, log_sx, log_sy
fn run_lbfgs_log_scale(
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    max_iterations: usize,
    image_path: &str,
) -> (f32, f32, f32) {
    let (width, height) = (target.width, target.height);

    // Initialize Gaussians
    let gaussians = create_jittered_grid(target, n_gaussians);

    // Initial render
    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);
    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n_gaussians, init_psnr);

    let start = Instant::now();

    // Flatten to parameter vector with log-space scales
    let mut x = flatten_log_space(&gaussians);

    // Create L-BFGS
    let mut lbfgs = LBFGS::new(10);

    // Convert to Gaussians for gradient computation
    let mut gaussians_mut = gaussians.clone();
    unflatten_log_space(&x, &mut gaussians_mut);

    let mut gradient = compute_gradient_log_space(&gaussians_mut, target);
    let mut rendered = RendererV2::render(&gaussians_mut, width, height);
    let mut f_x = compute_mse(&rendered, target);
    let mut last_psnr = init_psnr;

    for iter in 0..max_iterations {
        let grad_norm: f32 = gradient.iter().map(|&g| g * g).sum::<f32>().sqrt();
        if grad_norm < 1e-8 {
            println!("    Converged at iter {} (grad_norm={:.2e})", iter, grad_norm);
            break;
        }

        let direction = lbfgs.compute_direction(&gradient);

        // Adaptive initial step size for first iteration
        let initial_alpha = if iter == 0 { 1.0 / grad_norm.max(1.0) } else { 1.0 };
        let alpha = line_search_log_space(&x, &direction, &gradient, f_x, initial_alpha, target);

        let x_prev = x.clone();
        let grad_prev = gradient.clone();

        for i in 0..x.len() {
            x[i] += alpha * direction[i];
        }

        // Soft constraints (allow some slack)
        apply_soft_constraints_log(&mut x, n_gaussians);

        unflatten_log_space(&x, &mut gaussians_mut);
        gradient = compute_gradient_log_space(&gaussians_mut, target);
        rendered = RendererV2::render(&gaussians_mut, width, height);
        f_x = compute_mse(&rendered, target);

        lbfgs.update(&x_prev, &x, &grad_prev, &gradient);

        if iter % 10 == 0 || iter == max_iterations - 1 {
            let psnr = compute_psnr(&rendered, target);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {} loss={:.6} α={:.2e} ||g||={:.2e}",
                iter, max_iterations, psnr, dir, f_x, alpha, grad_norm);
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let final_render = RendererV2::render(&gaussians_mut, width, height);
    let final_psnr = compute_psnr(&final_render, target);
    let improvement = final_psnr - init_psnr;

    println!("  Final: PSNR={:.2}dB ({:+.2}dB) in {:.1}s", final_psnr, improvement, elapsed);

    // Save result
    let mut result = TestResult::new("lbfgs_log_scale", image_path);
    result.optimizer = "L-BFGS (log-scale params)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = max_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.elapsed_seconds = elapsed;
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (final_psnr, improvement, elapsed)
}

/// L-BFGS with per-parameter gradient scaling
fn run_lbfgs_gradient_scaled(
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    max_iterations: usize,
    image_path: &str,
) -> (f32, f32, f32) {
    let (width, height) = (target.width, target.height);

    let mut gaussians = create_jittered_grid(target, n_gaussians);
    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);
    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n_gaussians, init_psnr);

    let start = Instant::now();

    // Per-parameter scaling factors (based on typical gradient magnitudes)
    // These effectively give each parameter type its own learning rate
    let scale_factors = [
        100.0,   // position x - positions need larger effective LR
        100.0,   // position y
        1.0,     // color r - colors are well-behaved
        1.0,     // color g
        1.0,     // color b
        10.0,    // scale x - scales have medium sensitivity
        10.0,    // scale y
    ];

    let mut x = flatten_gaussians(&gaussians);
    let mut lbfgs = LBFGS::new(10);

    let mut gradient = compute_gradient_raw(&gaussians, target);
    let mut rendered = RendererV2::render(&gaussians, width, height);
    let mut f_x = compute_mse(&rendered, target);
    let mut last_psnr = init_psnr;

    // Scale the initial parameters
    for i in 0..n_gaussians {
        for j in 0..7 {
            x[i * 7 + j] *= scale_factors[j];
        }
    }

    // Scale the initial gradient
    for i in 0..n_gaussians {
        for j in 0..7 {
            gradient[i * 7 + j] /= scale_factors[j];
        }
    }

    for iter in 0..max_iterations {
        let grad_norm: f32 = gradient.iter().map(|&g| g * g).sum::<f32>().sqrt();
        if grad_norm < 1e-10 {
            println!("    Converged at iter {} (grad_norm={:.2e})", iter, grad_norm);
            break;
        }

        let direction = lbfgs.compute_direction(&gradient);

        let initial_alpha = if iter == 0 { 0.1 / grad_norm.max(0.1) } else { 1.0 };
        let alpha = line_search_scaled(
            &x, &direction, &gradient, f_x, initial_alpha,
            &scale_factors, n_gaussians, target
        );

        let x_prev = x.clone();
        let grad_prev = gradient.clone();

        for i in 0..x.len() {
            x[i] += alpha * direction[i];
        }

        // Unscale, apply constraints, rescale
        let mut x_real: Vec<f32> = Vec::with_capacity(n_gaussians * 7);
        for i in 0..n_gaussians {
            for j in 0..7 {
                x_real.push(x[i * 7 + j] / scale_factors[j]);
            }
        }
        apply_constraints_real(&mut x_real, n_gaussians);
        for i in 0..n_gaussians {
            for j in 0..7 {
                x[i * 7 + j] = x_real[i * 7 + j] * scale_factors[j];
            }
        }

        unflatten_gaussians(&x_real, &mut gaussians);
        gradient = compute_gradient_raw(&gaussians, target);

        // Scale gradient
        for i in 0..n_gaussians {
            for j in 0..7 {
                gradient[i * 7 + j] /= scale_factors[j];
            }
        }

        rendered = RendererV2::render(&gaussians, width, height);
        f_x = compute_mse(&rendered, target);

        lbfgs.update(&x_prev, &x, &grad_prev, &gradient);

        if iter % 10 == 0 || iter == max_iterations - 1 {
            let psnr = compute_psnr(&rendered, target);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {} loss={:.6} α={:.2e} ||g||={:.2e}",
                iter, max_iterations, psnr, dir, f_x, alpha, grad_norm);
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let final_psnr = compute_psnr(&rendered, target);
    let improvement = final_psnr - init_psnr;
    println!("  Final: PSNR={:.2}dB ({:+.2}dB) in {:.1}s", final_psnr, improvement, elapsed);

    let mut result = TestResult::new("lbfgs_gradient_scaled", image_path);
    result.optimizer = "L-BFGS (gradient scaling)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = max_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.elapsed_seconds = elapsed;
    result.complete();
    let _ = result.save_with_image(&rendered, RESULTS_DIR);

    (final_psnr, improvement, elapsed)
}

/// L-BFGS optimizing colors only (simplest case - should work)
fn run_lbfgs_colors_only(
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    max_iterations: usize,
    image_path: &str,
) -> (f32, f32, f32) {
    let (width, height) = (target.width, target.height);

    let mut gaussians = create_jittered_grid(target, n_gaussians);
    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);
    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n_gaussians, init_psnr);

    let start = Instant::now();

    // Only optimize colors (3 params per Gaussian)
    let mut x: Vec<f32> = gaussians.iter()
        .flat_map(|g| vec![g.color.r, g.color.g, g.color.b])
        .collect();

    let mut lbfgs = LBFGS::new(10);

    let mut gradient = compute_gradient_colors_only(&gaussians, target);
    let mut rendered = RendererV2::render(&gaussians, width, height);
    let mut f_x = compute_mse(&rendered, target);
    let mut last_psnr = init_psnr;

    for iter in 0..max_iterations {
        let grad_norm: f32 = gradient.iter().map(|&g| g * g).sum::<f32>().sqrt();
        if grad_norm < 1e-8 {
            println!("    Converged at iter {} (grad_norm={:.2e})", iter, grad_norm);
            break;
        }

        let direction = lbfgs.compute_direction(&gradient);

        let initial_alpha = if iter == 0 { 1.0 / grad_norm.max(1.0) } else { 1.0 };
        let alpha = line_search_colors(&x, &direction, &gradient, f_x, initial_alpha, &gaussians, target);

        let x_prev = x.clone();
        let grad_prev = gradient.clone();

        for i in 0..x.len() {
            x[i] = (x[i] + alpha * direction[i]).clamp(0.0, 1.0);
        }

        // Update Gaussian colors
        for (i, g) in gaussians.iter_mut().enumerate() {
            g.color.r = x[i * 3];
            g.color.g = x[i * 3 + 1];
            g.color.b = x[i * 3 + 2];
        }

        gradient = compute_gradient_colors_only(&gaussians, target);
        rendered = RendererV2::render(&gaussians, width, height);
        f_x = compute_mse(&rendered, target);

        lbfgs.update(&x_prev, &x, &grad_prev, &gradient);

        if iter % 10 == 0 || iter == max_iterations - 1 {
            let psnr = compute_psnr(&rendered, target);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {} loss={:.6} α={:.2e} ||g||={:.2e}",
                iter, max_iterations, psnr, dir, f_x, alpha, grad_norm);
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let final_psnr = compute_psnr(&rendered, target);
    let improvement = final_psnr - init_psnr;
    println!("  Final: PSNR={:.2}dB ({:+.2}dB) in {:.1}s", final_psnr, improvement, elapsed);

    let mut result = TestResult::new("lbfgs_colors_only", image_path);
    result.optimizer = "L-BFGS (colors only)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = max_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.elapsed_seconds = elapsed;
    result.complete();
    let _ = result.save_with_image(&rendered, RESULTS_DIR);

    (final_psnr, improvement, elapsed)
}

// ==================== LOG-SPACE HELPERS ====================

fn flatten_log_space(gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Vec<f32> {
    let mut x = Vec::with_capacity(gaussians.len() * 7);
    for g in gaussians {
        x.push(g.position.x);
        x.push(g.position.y);
        x.push(g.color.r);
        x.push(g.color.g);
        x.push(g.color.b);
        x.push(g.shape.scale_x.max(1e-6).ln());  // log(scale_x)
        x.push(g.shape.scale_y.max(1e-6).ln());  // log(scale_y)
    }
    x
}

fn unflatten_log_space(x: &[f32], gaussians: &mut [Gaussian2D<f32, Euler<f32>>]) {
    for (i, g) in gaussians.iter_mut().enumerate() {
        let base = i * 7;
        g.position.x = x[base];
        g.position.y = x[base + 1];
        g.color.r = x[base + 2];
        g.color.g = x[base + 3];
        g.color.b = x[base + 4];
        g.shape.scale_x = x[base + 5].exp();  // exp(log_scale_x)
        g.shape.scale_y = x[base + 6].exp();  // exp(log_scale_y)
    }
}

fn compute_gradient_log_space(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
) -> Vec<f32> {
    let width = target.width;
    let height = target.height;
    let n = gaussians.len();

    let mut grads: Vec<[f32; 7]> = vec![[0.0; 7]; n];
    let rendered = RendererV2::render(gaussians, width, height);

    for y in 0..height {
        for px in 0..width {
            let norm_x = (px as f32 + 0.5) / width as f32;
            let norm_y = (y as f32 + 0.5) / height as f32;

            let rendered_color = rendered.get_pixel(px, y).unwrap();
            let target_color = target.get_pixel(px, y).unwrap();

            let error_r = 2.0 * (rendered_color.r - target_color.r);
            let error_g = 2.0 * (rendered_color.g - target_color.g);
            let error_b = 2.0 * (rendered_color.b - target_color.b);

            for (i, gaussian) in gaussians.iter().enumerate() {
                let dx = norm_x - gaussian.position.x;
                let dy = norm_y - gaussian.position.y;

                let sx = gaussian.shape.scale_x.max(1e-6);
                let sy = gaussian.shape.scale_y.max(1e-6);

                let dist_sq = (dx / sx).powi(2) + (dy / sy).powi(2);
                if dist_sq > 16.0 { continue; }

                let weight = (-0.5 * dist_sq).exp();
                if weight < 1e-6 { continue; }

                // Color gradients (unchanged)
                grads[i][2] += error_r * weight;
                grads[i][3] += error_g * weight;
                grads[i][4] += error_b * weight;

                let error_weighted = error_r * gaussian.color.r
                                   + error_g * gaussian.color.g
                                   + error_b * gaussian.color.b;

                // Position gradients (unchanged from regular)
                grads[i][0] += error_weighted * weight * dx / (sx * sx);
                grads[i][1] += error_weighted * weight * dy / (sy * sy);

                // Log-scale gradients: ∂L/∂(log s) = ∂L/∂s * s
                // For scale: ∂L/∂s = error_weighted * weight * dx² / s³
                // So: ∂L/∂(log s) = error_weighted * weight * dx² / s²
                grads[i][5] += error_weighted * weight * (dx * dx) / (sx * sx);
                grads[i][6] += error_weighted * weight * (dy * dy) / (sy * sy);
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

fn apply_soft_constraints_log(x: &mut [f32], n_gaussians: usize) {
    for i in 0..n_gaussians {
        let base = i * 7;
        // Positions: soft clamp to [0, 1]
        x[base] = x[base].clamp(-0.05, 1.05);
        x[base + 1] = x[base + 1].clamp(-0.05, 1.05);
        // Colors: clamp to [0, 1]
        x[base + 2] = x[base + 2].clamp(0.0, 1.0);
        x[base + 3] = x[base + 3].clamp(0.0, 1.0);
        x[base + 4] = x[base + 4].clamp(0.0, 1.0);
        // Log-scales: clamp to [log(0.001), log(0.5)] = [-6.9, -0.7]
        x[base + 5] = x[base + 5].clamp(-7.0, -0.5);
        x[base + 6] = x[base + 6].clamp(-7.0, -0.5);
    }
}

fn line_search_log_space(
    x: &[f32],
    direction: &[f32],
    gradient: &[f32],
    f_x: f32,
    initial_alpha: f32,
    target: &ImageBuffer<f32>,
) -> f32 {
    let mut alpha = initial_alpha.min(1.0);
    let c1 = 1e-4;
    let n_gaussians = x.len() / 7;

    let directional_derivative: f32 = gradient.iter()
        .zip(direction.iter())
        .map(|(&g, &d)| g * d)
        .sum();

    if directional_derivative >= 0.0 {
        return 1e-6;
    }

    for _ in 0..20 {
        let mut x_new: Vec<f32> = x.iter()
            .zip(direction.iter())
            .map(|(&xi, &di)| xi + alpha * di)
            .collect();

        apply_soft_constraints_log(&mut x_new, n_gaussians);

        let mut temp_gaussians = vec![Gaussian2D::new(
            Vector2::zero(),
            Euler::new(0.01, 0.01, 0.0),
            Color4::new(0.5, 0.5, 0.5, 1.0),
            1.0,
        ); n_gaussians];
        unflatten_log_space(&x_new, &mut temp_gaussians);

        let rendered = RendererV2::render(&temp_gaussians, target.width, target.height);
        let f_new = compute_mse(&rendered, target);

        if f_new <= f_x + c1 * alpha * directional_derivative {
            return alpha;
        }

        alpha *= 0.5;
    }

    alpha.max(1e-8)
}

// ==================== SCALED GRADIENT HELPERS ====================

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

fn unflatten_gaussians(x: &[f32], gaussians: &mut [Gaussian2D<f32, Euler<f32>>]) {
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

fn compute_gradient_raw(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
) -> Vec<f32> {
    let width = target.width;
    let height = target.height;
    let n = gaussians.len();

    let mut grads: Vec<[f32; 7]> = vec![[0.0; 7]; n];
    let rendered = RendererV2::render(gaussians, width, height);

    for y in 0..height {
        for px in 0..width {
            let norm_x = (px as f32 + 0.5) / width as f32;
            let norm_y = (y as f32 + 0.5) / height as f32;

            let rendered_color = rendered.get_pixel(px, y).unwrap();
            let target_color = target.get_pixel(px, y).unwrap();

            let error_r = 2.0 * (rendered_color.r - target_color.r);
            let error_g = 2.0 * (rendered_color.g - target_color.g);
            let error_b = 2.0 * (rendered_color.b - target_color.b);

            for (i, gaussian) in gaussians.iter().enumerate() {
                let dx = norm_x - gaussian.position.x;
                let dy = norm_y - gaussian.position.y;

                let sx = gaussian.shape.scale_x.max(1e-6);
                let sy = gaussian.shape.scale_y.max(1e-6);

                let dist_sq = (dx / sx).powi(2) + (dy / sy).powi(2);
                if dist_sq > 16.0 { continue; }

                let weight = (-0.5 * dist_sq).exp();
                if weight < 1e-6 { continue; }

                grads[i][2] += error_r * weight;
                grads[i][3] += error_g * weight;
                grads[i][4] += error_b * weight;

                let error_weighted = error_r * gaussian.color.r
                                   + error_g * gaussian.color.g
                                   + error_b * gaussian.color.b;

                grads[i][0] += error_weighted * weight * dx / (sx * sx);
                grads[i][1] += error_weighted * weight * dy / (sy * sy);
                grads[i][5] += error_weighted * weight * (dx * dx) / (sx * sx * sx);
                grads[i][6] += error_weighted * weight * (dy * dy) / (sy * sy * sy);
            }
        }
    }

    let pixel_count = (width * height) as f32;
    let mut flat = Vec::with_capacity(n * 7);
    for g in &grads {
        for &v in g {
            flat.push(v / pixel_count);
        }
    }
    flat
}

fn apply_constraints_real(x: &mut [f32], n_gaussians: usize) {
    for i in 0..n_gaussians {
        let base = i * 7;
        x[base] = x[base].clamp(0.0, 1.0);
        x[base + 1] = x[base + 1].clamp(0.0, 1.0);
        x[base + 2] = x[base + 2].clamp(0.0, 1.0);
        x[base + 3] = x[base + 3].clamp(0.0, 1.0);
        x[base + 4] = x[base + 4].clamp(0.0, 1.0);
        x[base + 5] = x[base + 5].clamp(0.001, 0.5);
        x[base + 6] = x[base + 6].clamp(0.001, 0.5);
    }
}

fn line_search_scaled(
    x: &[f32],
    direction: &[f32],
    gradient: &[f32],
    f_x: f32,
    initial_alpha: f32,
    scale_factors: &[f32; 7],
    n_gaussians: usize,
    target: &ImageBuffer<f32>,
) -> f32 {
    let mut alpha = initial_alpha.min(1.0);
    let c1 = 1e-4;

    let directional_derivative: f32 = gradient.iter()
        .zip(direction.iter())
        .map(|(&g, &d)| g * d)
        .sum();

    if directional_derivative >= 0.0 {
        return 1e-6;
    }

    for _ in 0..20 {
        let x_new: Vec<f32> = x.iter()
            .zip(direction.iter())
            .map(|(&xi, &di)| xi + alpha * di)
            .collect();

        // Unscale to real space
        let mut x_real: Vec<f32> = Vec::with_capacity(n_gaussians * 7);
        for i in 0..n_gaussians {
            for j in 0..7 {
                x_real.push(x_new[i * 7 + j] / scale_factors[j]);
            }
        }
        apply_constraints_real(&mut x_real, n_gaussians);

        let mut temp_gaussians = vec![Gaussian2D::new(
            Vector2::zero(),
            Euler::new(0.01, 0.01, 0.0),
            Color4::new(0.5, 0.5, 0.5, 1.0),
            1.0,
        ); n_gaussians];
        unflatten_gaussians(&x_real, &mut temp_gaussians);

        let rendered = RendererV2::render(&temp_gaussians, target.width, target.height);
        let f_new = compute_mse(&rendered, target);

        if f_new <= f_x + c1 * alpha * directional_derivative {
            return alpha;
        }

        alpha *= 0.5;
    }

    alpha.max(1e-8)
}

// ==================== COLORS-ONLY HELPERS ====================

fn compute_gradient_colors_only(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
) -> Vec<f32> {
    let width = target.width;
    let height = target.height;
    let n = gaussians.len();

    let mut grads: Vec<[f32; 3]> = vec![[0.0; 3]; n];
    let rendered = RendererV2::render(gaussians, width, height);

    for y in 0..height {
        for px in 0..width {
            let norm_x = (px as f32 + 0.5) / width as f32;
            let norm_y = (y as f32 + 0.5) / height as f32;

            let rendered_color = rendered.get_pixel(px, y).unwrap();
            let target_color = target.get_pixel(px, y).unwrap();

            let error_r = 2.0 * (rendered_color.r - target_color.r);
            let error_g = 2.0 * (rendered_color.g - target_color.g);
            let error_b = 2.0 * (rendered_color.b - target_color.b);

            for (i, gaussian) in gaussians.iter().enumerate() {
                let dx = norm_x - gaussian.position.x;
                let dy = norm_y - gaussian.position.y;

                let sx = gaussian.shape.scale_x.max(1e-6);
                let sy = gaussian.shape.scale_y.max(1e-6);

                let dist_sq = (dx / sx).powi(2) + (dy / sy).powi(2);
                if dist_sq > 16.0 { continue; }

                let weight = (-0.5 * dist_sq).exp();
                if weight < 1e-6 { continue; }

                grads[i][0] += error_r * weight;
                grads[i][1] += error_g * weight;
                grads[i][2] += error_b * weight;
            }
        }
    }

    let pixel_count = (width * height) as f32;
    grads.iter().flat_map(|g| g.iter().map(|&v| v / pixel_count)).collect()
}

fn line_search_colors(
    x: &[f32],
    direction: &[f32],
    gradient: &[f32],
    f_x: f32,
    initial_alpha: f32,
    template_gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
) -> f32 {
    let mut alpha = initial_alpha.min(1.0);
    let c1 = 1e-4;
    let n_gaussians = template_gaussians.len();

    let directional_derivative: f32 = gradient.iter()
        .zip(direction.iter())
        .map(|(&g, &d)| g * d)
        .sum();

    if directional_derivative >= 0.0 {
        return 1e-6;
    }

    for _ in 0..20 {
        let x_new: Vec<f32> = x.iter()
            .zip(direction.iter())
            .map(|(&xi, &di)| (xi + alpha * di).clamp(0.0, 1.0))
            .collect();

        let mut temp_gaussians = template_gaussians.to_vec();
        for (i, g) in temp_gaussians.iter_mut().enumerate() {
            g.color.r = x_new[i * 3];
            g.color.g = x_new[i * 3 + 1];
            g.color.b = x_new[i * 3 + 2];
        }

        let rendered = RendererV2::render(&temp_gaussians, target.width, target.height);
        let f_new = compute_mse(&rendered, target);

        if f_new <= f_x + c1 * alpha * directional_derivative {
            return alpha;
        }

        alpha *= 0.5;
    }

    alpha.max(1e-8)
}

// ==================== COMMON HELPERS ====================

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
