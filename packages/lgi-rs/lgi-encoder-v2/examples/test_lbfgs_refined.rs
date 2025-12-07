//! L-BFGS Refined Test - Three approaches to make L-BFGS work
//!
//! Based on our finding that colors-only L-BFGS achieved +2.44dB with α=1.0
//! but colors drifted out of bounds (63% OOB) and clamping destroyed quality.
//!
//! Test approaches:
//! 1. Colors-only with STRONGER barrier (0.1, 1.0 instead of 0.001)
//! 2. Block-diagonal L-BFGS (separate histories for colors vs geometry)
//! 3. Preconditioned gradients (scale gradients to similar magnitude)
//!
//! Run: cargo run --release --example test_lbfgs_refined -p lgi-encoder-v2

use lgi_core::{ImageBuffer, lbfgs::LBFGS};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       L-BFGS REFINED TEST                                    ║");
    println!("║       Three approaches to make L-BFGS work                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";

    let target = match ImageBuffer::load(image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            return;
        }
    };

    println!("Testing: {}", image_path.split('/').last().unwrap_or("unknown"));

    // Test 1a: Colors-only with barrier_weight = 0.1
    println!("\n═══ Test 1a: Colors-only L-BFGS with barrier_weight=0.1 ═══");
    let (_, imp1a, time1a) = test_colors_with_barrier(&target, image_path, 100, 0.1);

    // Test 1b: Colors-only with barrier_weight = 1.0
    println!("\n═══ Test 1b: Colors-only L-BFGS with barrier_weight=1.0 ═══");
    let (_, imp1b, time1b) = test_colors_with_barrier(&target, image_path, 100, 1.0);

    // Test 2: Block-diagonal L-BFGS
    println!("\n═══ Test 2: Block-diagonal L-BFGS (separate histories) ═══");
    let (_, imp2, time2) = test_block_diagonal(&target, image_path, 100);

    // Test 3: Preconditioned L-BFGS
    println!("\n═══ Test 3: Preconditioned L-BFGS ═══");
    let (_, imp3, time3) = test_preconditioned(&target, image_path, 100);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     SUMMARY                                  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Test 1a (colors, barrier=0.1): {:+.2}dB in {:.1}s", imp1a, time1a);
    println!("║ Test 1b (colors, barrier=1.0): {:+.2}dB in {:.1}s", imp1b, time1b);
    println!("║ Test 2  (block-diagonal):      {:+.2}dB in {:.1}s", imp2, time2);
    println!("║ Test 3  (preconditioned):      {:+.2}dB in {:.1}s", imp3, time3);
    println!("╚══════════════════════════════════════════════════════════════╝");
}

/// Test 1: Colors-only with stronger soft barrier
fn test_colors_with_barrier(
    target: &ImageBuffer<f32>,
    image_path: &str,
    max_iterations: usize,
    barrier_weight: f32,
) -> (f32, f32, f32) {
    let (width, height) = (target.width, target.height);
    let n_gaussians = 256;

    let mut gaussians = create_jittered_grid(target, n_gaussians);
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&rendered, target);

    println!("  Initial: {} Gaussians, PSNR={:.2}dB, barrier_weight={}",
             n_gaussians, init_psnr, barrier_weight);

    // Flatten colors only (3 params per Gaussian)
    let mut x: Vec<f32> = gaussians.iter()
        .flat_map(|g| [g.color.r, g.color.g, g.color.b])
        .collect();

    let mut gradient = compute_gradient_colors_with_barrier(&gaussians, target, &x, barrier_weight);
    let mut rendered = RendererV2::render(&gaussians, width, height);
    let mut f_x = compute_loss_colors_with_barrier(&rendered, target, &x, barrier_weight);
    let mut last_psnr = init_psnr;

    let mut lbfgs = LBFGS::new(10);
    let start = Instant::now();

    for iter in 0..max_iterations {
        let direction = lbfgs.compute_direction(&gradient);
        let grad_norm: f32 = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();

        // Line search with barrier penalty
        let alpha = line_search_colors_barrier(
            &x, &direction, &gradient, f_x,
            &gaussians, target, barrier_weight,
        );

        let x_prev = x.clone();
        let grad_prev = gradient.clone();

        // Update WITHOUT hard clamping
        for i in 0..x.len() {
            x[i] += alpha * direction[i];
        }

        // Update Gaussian colors (barrier keeps them mostly in bounds)
        for (i, g) in gaussians.iter_mut().enumerate() {
            g.color.r = x[i * 3];
            g.color.g = x[i * 3 + 1];
            g.color.b = x[i * 3 + 2];
        }

        gradient = compute_gradient_colors_with_barrier(&gaussians, target, &x, barrier_weight);
        rendered = RendererV2::render(&gaussians, width, height);
        f_x = compute_loss_colors_with_barrier(&rendered, target, &x, barrier_weight);

        lbfgs.update(&x_prev, &x, &grad_prev, &gradient);

        if iter % 10 == 0 || iter == max_iterations - 1 {
            let psnr = compute_psnr(&rendered, target);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };
            let oob = x.iter().filter(|&&v| v < 0.0 || v > 1.0).count();

            println!("    [{:3}/{:3}] PSNR={:.2}dB {} α={:.2e} ||g||={:.2e} OOB={}",
                iter, max_iterations, psnr, dir, alpha, grad_norm, oob);
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();

    // Soft clamp at the end (most should already be in bounds)
    for (i, g) in gaussians.iter_mut().enumerate() {
        g.color.r = x[i * 3].clamp(0.0, 1.0);
        g.color.g = x[i * 3 + 1].clamp(0.0, 1.0);
        g.color.b = x[i * 3 + 2].clamp(0.0, 1.0);
    }

    let final_rendered = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_rendered, target);
    let improvement = final_psnr - init_psnr;

    let oob_final = x.iter().filter(|&&v| v < 0.0 || v > 1.0).count();
    println!("  Final (after clamp): PSNR={:.2}dB ({:+.2}dB) in {:.1}s, OOB={}",
             final_psnr, improvement, elapsed, oob_final);

    let mut result = TestResult::new(&format!("lbfgs_colors_barrier_{}", barrier_weight), image_path);
    result.optimizer = format!("L-BFGS (colors, barrier={})", barrier_weight);
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = max_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.elapsed_seconds = elapsed;
    result.set_param("barrier_weight", barrier_weight);
    result.set_param("oob_count", oob_final);
    result.complete();
    let _ = result.save_with_image(&final_rendered, RESULTS_DIR);

    (final_psnr, improvement, elapsed)
}

/// Test 2: Block-diagonal L-BFGS
/// Separate L-BFGS instances for colors and geometry to avoid gradient scale mismatch
fn test_block_diagonal(
    target: &ImageBuffer<f32>,
    image_path: &str,
    max_iterations: usize,
) -> (f32, f32, f32) {
    let (width, height) = (target.width, target.height);
    let n_gaussians = 256;

    let mut gaussians = create_jittered_grid(target, n_gaussians);
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&rendered, target);

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n_gaussians, init_psnr);
    println!("  Using separate L-BFGS for colors vs positions/scales");

    // Separate parameter vectors
    // Colors: 3 params per Gaussian
    let mut x_colors: Vec<f32> = gaussians.iter()
        .flat_map(|g| [g.color.r, g.color.g, g.color.b])
        .collect();
    // Geometry: 4 params per Gaussian (x, y, sx, sy)
    let mut x_geom: Vec<f32> = gaussians.iter()
        .flat_map(|g| [g.position.x, g.position.y, g.shape.scale_x, g.shape.scale_y])
        .collect();

    // Separate L-BFGS instances
    let mut lbfgs_colors = LBFGS::new(10);
    let mut lbfgs_geom = LBFGS::new(10);

    let mut last_psnr = init_psnr;
    let start = Instant::now();

    for iter in 0..max_iterations {
        // Compute separate gradients
        let (grad_colors, grad_geom) = compute_gradients_split(&gaussians, target);

        let grad_norm_colors: f32 = grad_colors.iter().map(|g| g * g).sum::<f32>().sqrt();
        let grad_norm_geom: f32 = grad_geom.iter().map(|g| g * g).sum::<f32>().sqrt();

        // Compute separate directions
        let dir_colors = lbfgs_colors.compute_direction(&grad_colors);
        let dir_geom = lbfgs_geom.compute_direction(&grad_geom);

        // Line search for colors
        let alpha_colors = line_search_block_colors(
            &x_colors, &dir_colors, &grad_colors,
            &x_geom, &gaussians, target,
        );

        // Line search for geometry
        let alpha_geom = line_search_block_geom(
            &x_geom, &dir_geom, &grad_geom,
            &x_colors, &gaussians, target,
        );

        let x_colors_prev = x_colors.clone();
        let x_geom_prev = x_geom.clone();
        let grad_colors_prev = grad_colors.clone();
        let grad_geom_prev = grad_geom.clone();

        // Update colors (with soft barrier)
        for i in 0..x_colors.len() {
            x_colors[i] += alpha_colors * dir_colors[i];
            // Soft projection: if outside bounds, apply penalty but don't hard clamp
            if x_colors[i] < 0.0 {
                x_colors[i] = x_colors[i] * 0.5; // Dampen negative values
            } else if x_colors[i] > 1.0 {
                x_colors[i] = 1.0 + (x_colors[i] - 1.0) * 0.5; // Dampen excess
            }
        }

        // Update geometry (with soft constraints)
        for i in 0..n_gaussians {
            let base = i * 4;
            // Positions
            x_geom[base] += alpha_geom * dir_geom[base];
            x_geom[base + 1] += alpha_geom * dir_geom[base + 1];
            // Soft clamp positions
            x_geom[base] = x_geom[base].clamp(-0.1, 1.1);
            x_geom[base + 1] = x_geom[base + 1].clamp(-0.1, 1.1);
            // Scales (keep positive)
            x_geom[base + 2] += alpha_geom * dir_geom[base + 2];
            x_geom[base + 3] += alpha_geom * dir_geom[base + 3];
            x_geom[base + 2] = x_geom[base + 2].max(0.005);
            x_geom[base + 3] = x_geom[base + 3].max(0.005);
        }

        // Update Gaussians
        for (i, g) in gaussians.iter_mut().enumerate() {
            g.color.r = x_colors[i * 3];
            g.color.g = x_colors[i * 3 + 1];
            g.color.b = x_colors[i * 3 + 2];
            g.position.x = x_geom[i * 4];
            g.position.y = x_geom[i * 4 + 1];
            g.shape.scale_x = x_geom[i * 4 + 2];
            g.shape.scale_y = x_geom[i * 4 + 3];
        }

        // Update L-BFGS histories
        let (new_grad_colors, new_grad_geom) = compute_gradients_split(&gaussians, target);
        lbfgs_colors.update(&x_colors_prev, &x_colors, &grad_colors_prev, &new_grad_colors);
        lbfgs_geom.update(&x_geom_prev, &x_geom, &grad_geom_prev, &new_grad_geom);

        if iter % 10 == 0 || iter == max_iterations - 1 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = compute_psnr(&rendered, target);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };

            println!("    [{:3}/{:3}] PSNR={:.2}dB {} αc={:.2e} αg={:.2e} ||gc||={:.2e} ||gg||={:.2e}",
                iter, max_iterations, psnr, dir, alpha_colors, alpha_geom,
                grad_norm_colors, grad_norm_geom);
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();

    // Final clamp for visualization
    for (i, g) in gaussians.iter_mut().enumerate() {
        g.color.r = x_colors[i * 3].clamp(0.0, 1.0);
        g.color.g = x_colors[i * 3 + 1].clamp(0.0, 1.0);
        g.color.b = x_colors[i * 3 + 2].clamp(0.0, 1.0);
        g.position.x = x_geom[i * 4].clamp(0.0, 1.0);
        g.position.y = x_geom[i * 4 + 1].clamp(0.0, 1.0);
    }

    let final_rendered = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_rendered, target);
    let improvement = final_psnr - init_psnr;

    println!("  Final: PSNR={:.2}dB ({:+.2}dB) in {:.1}s", final_psnr, improvement, elapsed);

    let mut result = TestResult::new("lbfgs_block_diagonal", image_path);
    result.optimizer = "L-BFGS (block-diagonal: colors + geometry)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = max_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.elapsed_seconds = elapsed;
    result.complete();
    let _ = result.save_with_image(&final_rendered, RESULTS_DIR);

    (final_psnr, improvement, elapsed)
}

/// Test 3: Preconditioned L-BFGS
/// Scale gradients so all parameter types have similar magnitude
fn test_preconditioned(
    target: &ImageBuffer<f32>,
    image_path: &str,
    max_iterations: usize,
) -> (f32, f32, f32) {
    let (width, height) = (target.width, target.height);
    let n_gaussians = 256;

    let mut gaussians = create_jittered_grid(target, n_gaussians);
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&rendered, target);

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n_gaussians, init_psnr);

    // Preconditioning scales (empirically tuned from gradient analysis)
    // Position gradients have 1/sx² term → scale up
    // Color gradients are O(1) → keep as is
    // Scale gradients have 1/sx³ term → scale up even more
    let scale_position = 100.0;  // Positions are sensitive, scale up gradient
    let scale_color = 1.0;       // Colors are baseline
    let scale_scale = 500.0;     // Scales have largest gradient magnitude issue

    println!("  Preconditioner: pos={}, color={}, scale={}",
             scale_position, scale_color, scale_scale);

    // Flatten all 7 params per Gaussian: x, y, r, g, b, sx, sy
    let mut x = flatten_all(&gaussians);
    let mut gradient = compute_gradient_all(&gaussians, target);

    // Apply preconditioning to gradient
    apply_preconditioner(&mut gradient, n_gaussians, scale_position, scale_color, scale_scale);

    let mut rendered = RendererV2::render(&gaussians, width, height);
    let mut f_x = compute_mse(&rendered, target);
    let mut last_psnr = init_psnr;

    let mut lbfgs = LBFGS::new(10);
    let start = Instant::now();

    for iter in 0..max_iterations {
        let direction = lbfgs.compute_direction(&gradient);

        // Undo preconditioning on direction before applying to x
        let mut direction_unscaled = direction.clone();
        apply_preconditioner_inv(&mut direction_unscaled, n_gaussians,
                                  scale_position, scale_color, scale_scale);

        let grad_norm: f32 = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();

        // Line search with unscaled direction
        let alpha = line_search_preconditioned(
            &x, &direction_unscaled, &gradient, f_x,
            n_gaussians, target,
            scale_position, scale_color, scale_scale,
        );

        let x_prev = x.clone();
        let grad_prev = gradient.clone();

        // Update with unscaled direction (use soft constraints)
        for i in 0..n_gaussians {
            let base = i * 7;
            // Position
            x[base] += alpha * direction_unscaled[base];
            x[base + 1] += alpha * direction_unscaled[base + 1];
            // Colors
            x[base + 2] += alpha * direction_unscaled[base + 2];
            x[base + 3] += alpha * direction_unscaled[base + 3];
            x[base + 4] += alpha * direction_unscaled[base + 4];
            // Scales
            x[base + 5] += alpha * direction_unscaled[base + 5];
            x[base + 6] += alpha * direction_unscaled[base + 6];

            // Soft constraints (keep scales positive)
            x[base + 5] = x[base + 5].max(0.005);
            x[base + 6] = x[base + 6].max(0.005);
        }

        unflatten_all(&x, &mut gaussians);
        gradient = compute_gradient_all(&gaussians, target);
        apply_preconditioner(&mut gradient, n_gaussians, scale_position, scale_color, scale_scale);

        rendered = RendererV2::render(&gaussians, width, height);
        f_x = compute_mse(&rendered, target);

        lbfgs.update(&x_prev, &x, &grad_prev, &gradient);

        if iter % 10 == 0 || iter == max_iterations - 1 {
            let psnr = compute_psnr(&rendered, target);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };

            println!("    [{:3}/{:3}] PSNR={:.2}dB {} α={:.2e} ||g||={:.2e}",
                iter, max_iterations, psnr, dir, alpha, grad_norm);
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();

    // Final clamp
    clamp_all(&mut x, n_gaussians);
    unflatten_all(&x, &mut gaussians);

    let final_rendered = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_rendered, target);
    let improvement = final_psnr - init_psnr;

    println!("  Final: PSNR={:.2}dB ({:+.2}dB) in {:.1}s", final_psnr, improvement, elapsed);

    let mut result = TestResult::new("lbfgs_preconditioned", image_path);
    result.optimizer = "L-BFGS (preconditioned)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = max_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.elapsed_seconds = elapsed;
    result.set_param("scale_position", scale_position);
    result.set_param("scale_color", scale_color);
    result.set_param("scale_scale", scale_scale);
    result.complete();
    let _ = result.save_with_image(&final_rendered, RESULTS_DIR);

    (final_psnr, improvement, elapsed)
}

// ==================== HELPER FUNCTIONS ====================

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
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(cell_size, cell_size, 0.0),
            color,
            1.0,
        ));
    }
    gaussians
}

fn flatten_all(gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Vec<f32> {
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

fn unflatten_all(x: &[f32], gaussians: &mut [Gaussian2D<f32, Euler<f32>>]) {
    for (i, g) in gaussians.iter_mut().enumerate() {
        let base = i * 7;
        g.position.x = x[base];
        g.position.y = x[base + 1];
        g.color.r = x[base + 2];
        g.color.g = x[base + 3];
        g.color.b = x[base + 4];
        g.shape.scale_x = x[base + 5].max(1e-6);
        g.shape.scale_y = x[base + 6].max(1e-6);
    }
}

fn clamp_all(x: &mut [f32], n_gaussians: usize) {
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

/// Apply preconditioning: scale gradients by parameter type
fn apply_preconditioner(
    grad: &mut [f32],
    n_gaussians: usize,
    scale_pos: f32,
    scale_color: f32,
    scale_scale: f32,
) {
    for i in 0..n_gaussians {
        let base = i * 7;
        // Position
        grad[base] *= scale_pos;
        grad[base + 1] *= scale_pos;
        // Color
        grad[base + 2] *= scale_color;
        grad[base + 3] *= scale_color;
        grad[base + 4] *= scale_color;
        // Scale
        grad[base + 5] *= scale_scale;
        grad[base + 6] *= scale_scale;
    }
}

/// Inverse preconditioning: unscale direction before applying to x
fn apply_preconditioner_inv(
    direction: &mut [f32],
    n_gaussians: usize,
    scale_pos: f32,
    scale_color: f32,
    scale_scale: f32,
) {
    for i in 0..n_gaussians {
        let base = i * 7;
        // Position
        direction[base] /= scale_pos;
        direction[base + 1] /= scale_pos;
        // Color
        direction[base + 2] /= scale_color;
        direction[base + 3] /= scale_color;
        direction[base + 4] /= scale_color;
        // Scale
        direction[base + 5] /= scale_scale;
        direction[base + 6] /= scale_scale;
    }
}

/// Compute gradients for colors only with soft barrier penalty
fn compute_gradient_colors_with_barrier(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    x: &[f32],
    barrier_weight: f32,
) -> Vec<f32> {
    let mut grad = compute_gradient_colors(gaussians, target);

    // Add barrier gradient
    for i in 0..x.len() {
        let v = x[i];
        if v > 1.0 {
            grad[i] += barrier_weight * 2.0 * (v - 1.0);
        } else if v < 0.0 {
            grad[i] += barrier_weight * 2.0 * v;
        }
    }

    grad
}

fn compute_loss_colors_with_barrier(
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
    x: &[f32],
    barrier_weight: f32,
) -> f32 {
    let mse = compute_mse(rendered, target);

    let mut barrier = 0.0;
    for &v in x {
        if v > 1.0 {
            barrier += (v - 1.0).powi(2);
        } else if v < 0.0 {
            barrier += v.powi(2);
        }
    }

    mse + barrier_weight * barrier
}

/// Compute separate gradients for colors and geometry
fn compute_gradients_split(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let width = target.width;
    let height = target.height;
    let n = gaussians.len();

    let mut grad_colors: Vec<[f32; 3]> = vec![[0.0; 3]; n];
    let mut grad_geom: Vec<[f32; 4]> = vec![[0.0; 4]; n];
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

                // Color gradients
                grad_colors[i][0] += error_r * weight;
                grad_colors[i][1] += error_g * weight;
                grad_colors[i][2] += error_b * weight;

                let error_weighted = error_r * gaussian.color.r
                                   + error_g * gaussian.color.g
                                   + error_b * gaussian.color.b;

                // Geometry gradients
                grad_geom[i][0] += error_weighted * weight * dx / (sx * sx);  // x
                grad_geom[i][1] += error_weighted * weight * dy / (sy * sy);  // y
                grad_geom[i][2] += error_weighted * weight * (dx * dx) / (sx * sx * sx);  // sx
                grad_geom[i][3] += error_weighted * weight * (dy * dy) / (sy * sy * sy);  // sy
            }
        }
    }

    let pixel_count = (width * height) as f32;

    let colors_flat: Vec<f32> = grad_colors.iter()
        .flat_map(|g| g.iter().map(|&v| v / pixel_count))
        .collect();

    let geom_flat: Vec<f32> = grad_geom.iter()
        .flat_map(|g| g.iter().map(|&v| v / pixel_count))
        .collect();

    (colors_flat, geom_flat)
}

fn compute_gradient_colors(gaussians: &[Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) -> Vec<f32> {
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

fn compute_gradient_all(gaussians: &[Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) -> Vec<f32> {
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

                // Color gradients (indices 2,3,4)
                grads[i][2] += error_r * weight;
                grads[i][3] += error_g * weight;
                grads[i][4] += error_b * weight;

                let error_weighted = error_r * gaussian.color.r
                                   + error_g * gaussian.color.g
                                   + error_b * gaussian.color.b;

                // Position gradients (indices 0,1)
                grads[i][0] += error_weighted * weight * dx / (sx * sx);
                grads[i][1] += error_weighted * weight * dy / (sy * sy);

                // Scale gradients (indices 5,6)
                grads[i][5] += error_weighted * weight * (dx * dx) / (sx * sx * sx);
                grads[i][6] += error_weighted * weight * (dy * dy) / (sy * sy * sy);
            }
        }
    }

    let pixel_count = (width * height) as f32;
    grads.iter().flat_map(|g| g.iter().map(|&v| v / pixel_count)).collect()
}

// Line search functions

fn line_search_colors_barrier(
    x: &[f32],
    direction: &[f32],
    gradient: &[f32],
    f_x: f32,
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    barrier_weight: f32,
) -> f32 {
    let mut alpha = 1.0;
    let c1 = 1e-4;
    let width = target.width;
    let height = target.height;

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

        let mut temp_gaussians = gaussians.to_vec();
        for (i, g) in temp_gaussians.iter_mut().enumerate() {
            g.color.r = x_new[i * 3];
            g.color.g = x_new[i * 3 + 1];
            g.color.b = x_new[i * 3 + 2];
        }

        let rendered = RendererV2::render(&temp_gaussians, width, height);
        let f_new = compute_loss_colors_with_barrier(&rendered, target, &x_new, barrier_weight);

        if f_new <= f_x + c1 * alpha * directional_derivative {
            return alpha;
        }

        alpha *= 0.5;
        if alpha < 1e-10 {
            return 1e-10;
        }
    }

    alpha
}

fn line_search_block_colors(
    x_colors: &[f32],
    direction: &[f32],
    gradient: &[f32],
    x_geom: &[f32],
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
) -> f32 {
    let mut alpha = 1.0;
    let c1 = 1e-4;
    let width = target.width;
    let height = target.height;
    let n_gaussians = gaussians.len();

    let directional_derivative: f32 = gradient.iter()
        .zip(direction.iter())
        .map(|(&g, &d)| g * d)
        .sum();

    if directional_derivative >= 0.0 {
        return 1e-6;
    }

    for _ in 0..20 {
        let x_new: Vec<f32> = x_colors.iter()
            .zip(direction.iter())
            .map(|(&xi, &di)| xi + alpha * di)
            .collect();

        let mut temp_gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = Vec::with_capacity(n_gaussians);
        for i in 0..n_gaussians {
            temp_gaussians.push(Gaussian2D::new(
                Vector2::new(x_geom[i * 4], x_geom[i * 4 + 1]),
                Euler::new(x_geom[i * 4 + 2], x_geom[i * 4 + 3], 0.0),
                Color4::new(x_new[i * 3], x_new[i * 3 + 1], x_new[i * 3 + 2], 1.0),
                1.0,
            ));
        }

        let rendered = RendererV2::render(&temp_gaussians, width, height);
        let f_new = compute_mse(&rendered, target);

        if f_new <= f_x_colors(&rendered, target) + c1 * alpha * directional_derivative {
            return alpha;
        }

        alpha *= 0.5;
        if alpha < 1e-10 {
            return 1e-10;
        }
    }

    alpha
}

fn f_x_colors(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    compute_mse(rendered, target)
}

fn line_search_block_geom(
    x_geom: &[f32],
    direction: &[f32],
    gradient: &[f32],
    x_colors: &[f32],
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
) -> f32 {
    let mut alpha = 1.0;
    let c1 = 1e-4;
    let width = target.width;
    let height = target.height;
    let n_gaussians = gaussians.len();

    let directional_derivative: f32 = gradient.iter()
        .zip(direction.iter())
        .map(|(&g, &d)| g * d)
        .sum();

    if directional_derivative >= 0.0 {
        return 1e-6;
    }

    for _ in 0..20 {
        let x_new: Vec<f32> = x_geom.iter()
            .zip(direction.iter())
            .map(|(&xi, &di)| xi + alpha * di)
            .collect();

        let mut temp_gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = Vec::with_capacity(n_gaussians);
        for i in 0..n_gaussians {
            let base = i * 4;
            temp_gaussians.push(Gaussian2D::new(
                Vector2::new(x_new[base], x_new[base + 1]),
                Euler::new(x_new[base + 2].max(1e-6), x_new[base + 3].max(1e-6), 0.0),
                Color4::new(x_colors[i * 3], x_colors[i * 3 + 1], x_colors[i * 3 + 2], 1.0),
                1.0,
            ));
        }

        let rendered = RendererV2::render(&temp_gaussians, width, height);
        let f_new = compute_mse(&rendered, target);

        if f_new <= f_x_colors(&rendered, target) + c1 * alpha * directional_derivative {
            return alpha;
        }

        alpha *= 0.5;
        if alpha < 1e-10 {
            return 1e-10;
        }
    }

    alpha
}

fn line_search_preconditioned(
    x: &[f32],
    direction: &[f32],
    gradient: &[f32],
    f_x: f32,
    n_gaussians: usize,
    target: &ImageBuffer<f32>,
    _scale_pos: f32,
    _scale_color: f32,
    _scale_scale: f32,
) -> f32 {
    let mut alpha = 1.0;
    let c1 = 1e-4;
    let width = target.width;
    let height = target.height;

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

        let mut temp_gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = Vec::with_capacity(n_gaussians);
        for i in 0..n_gaussians {
            let base = i * 7;
            temp_gaussians.push(Gaussian2D::new(
                Vector2::new(x_new[base], x_new[base + 1]),
                Euler::new(x_new[base + 5].max(1e-6), x_new[base + 6].max(1e-6), 0.0),
                Color4::new(x_new[base + 2], x_new[base + 3], x_new[base + 4], 1.0),
                1.0,
            ));
        }

        let rendered = RendererV2::render(&temp_gaussians, width, height);
        let f_new = compute_mse(&rendered, target);

        if f_new <= f_x + c1 * alpha * directional_derivative {
            return alpha;
        }

        alpha *= 0.5;
        if alpha < 1e-10 {
            return 1e-10;
        }
    }

    alpha
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
