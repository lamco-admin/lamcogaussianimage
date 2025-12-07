//! L-BFGS Unconstrained Colors Test
//!
//! Tests L-BFGS without ANY constraints to verify the algorithm fundamentally works.
//! Colors are allowed to go outside [0,1] during optimization.
//! We clamp only at the end for visualization.
//!
//! Run: cargo run --release --example test_lbfgs_unconstrained -p lgi-encoder-v2

use lgi_core::{ImageBuffer, lbfgs::LBFGS};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       L-BFGS UNCONSTRAINED TEST                              ║");
    println!("║       No clamping during optimization - verify fundamentals  ║");
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

    // Test 1: Colors only, no constraints
    println!("\n--- Test 1: L-BFGS colors only (NO CONSTRAINTS) ---");
    let (psnr1, imp1, time1) = test_colors_unconstrained(&target, image_path, 100);

    // Test 2: All parameters, no constraints
    println!("\n--- Test 2: L-BFGS all parameters (NO CONSTRAINTS) ---");
    let (psnr2, imp2, time2) = test_all_unconstrained(&target, image_path, 100);

    // Test 3: All parameters with soft barrier penalty
    println!("\n--- Test 3: L-BFGS with soft barrier penalty ---");
    let (psnr3, imp3, time3) = test_soft_barrier(&target, image_path, 100);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     SUMMARY                                  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Test 1 (colors, no constraints): {:+.2}dB in {:.1}s              ", imp1, time1);
    println!("║ Test 2 (all, no constraints):    {:+.2}dB in {:.1}s              ", imp2, time2);
    println!("║ Test 3 (soft barrier):           {:+.2}dB in {:.1}s              ", imp3, time3);
    println!("╚══════════════════════════════════════════════════════════════╝");
}

/// Colors-only optimization with NO constraints
fn test_colors_unconstrained(target: &ImageBuffer<f32>, image_path: &str, max_iterations: usize) -> (f32, f32, f32) {
    let (width, height) = (target.width, target.height);
    let n_gaussians = 256;

    let mut gaussians = create_jittered_grid(target, n_gaussians);
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&rendered, target);

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n_gaussians, init_psnr);

    // Flatten colors only (3 params per Gaussian)
    let mut x: Vec<f32> = gaussians.iter()
        .flat_map(|g| [g.color.r, g.color.g, g.color.b])
        .collect();

    let mut gradient = compute_gradient_colors(&gaussians, target);
    let mut rendered = RendererV2::render(&gaussians, width, height);
    let mut f_x = compute_mse(&rendered, target);
    let mut last_psnr = init_psnr;

    let mut lbfgs = LBFGS::new(10);
    let start = Instant::now();

    for iter in 0..max_iterations {
        let direction = lbfgs.compute_direction(&gradient);
        let grad_norm: f32 = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();

        // Simple line search without constraints
        let alpha = line_search_colors(
            &x, &direction, &gradient, f_x,
            &gaussians, target,
        );

        let x_prev = x.clone();
        let grad_prev = gradient.clone();

        // Update WITHOUT clamping
        for i in 0..x.len() {
            x[i] += alpha * direction[i];
        }

        // Update Gaussian colors (may be outside [0,1])
        for (i, g) in gaussians.iter_mut().enumerate() {
            g.color.r = x[i * 3];
            g.color.g = x[i * 3 + 1];
            g.color.b = x[i * 3 + 2];
        }

        gradient = compute_gradient_colors(&gaussians, target);
        rendered = RendererV2::render(&gaussians, width, height);
        f_x = compute_mse(&rendered, target);

        lbfgs.update(&x_prev, &x, &grad_prev, &gradient);

        if iter % 10 == 0 || iter == max_iterations - 1 {
            let psnr = compute_psnr(&rendered, target);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };

            // Check how many colors are out of bounds
            let oob = x.iter().filter(|&&v| v < 0.0 || v > 1.0).count();

            println!("    [{:3}/{:3}] PSNR={:.2}dB {} α={:.2e} ||g||={:.2e} OOB={}",
                iter, max_iterations, psnr, dir, alpha, grad_norm, oob);
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();

    // Clamp colors only at the end for visualization
    for (i, g) in gaussians.iter_mut().enumerate() {
        g.color.r = x[i * 3].clamp(0.0, 1.0);
        g.color.g = x[i * 3 + 1].clamp(0.0, 1.0);
        g.color.b = x[i * 3 + 2].clamp(0.0, 1.0);
    }

    let final_rendered = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_rendered, target);
    let improvement = final_psnr - init_psnr;

    println!("  Final (after clamp): PSNR={:.2}dB ({:+.2}dB) in {:.1}s",
             final_psnr, improvement, elapsed);

    let mut result = TestResult::new("lbfgs_colors_unconstrained", image_path);
    result.optimizer = "L-BFGS (colors, unconstrained)".to_string();
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

/// All parameters, no constraints
fn test_all_unconstrained(target: &ImageBuffer<f32>, image_path: &str, max_iterations: usize) -> (f32, f32, f32) {
    let (width, height) = (target.width, target.height);
    let n_gaussians = 256;

    let mut gaussians = create_jittered_grid(target, n_gaussians);
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&rendered, target);

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n_gaussians, init_psnr);

    // Flatten all 7 params per Gaussian: x, y, r, g, b, sx, sy
    let mut x = flatten_all(&gaussians);
    let mut gradient = compute_gradient_all(&gaussians, target);
    let mut rendered = RendererV2::render(&gaussians, width, height);
    let mut f_x = compute_mse(&rendered, target);
    let mut last_psnr = init_psnr;

    let mut lbfgs = LBFGS::new(10);
    let start = Instant::now();

    for iter in 0..max_iterations {
        let direction = lbfgs.compute_direction(&gradient);
        let grad_norm: f32 = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();

        let alpha = line_search_all(
            &x, &direction, &gradient, f_x,
            n_gaussians, target,
        );

        let x_prev = x.clone();
        let grad_prev = gradient.clone();

        // Update WITHOUT clamping
        for i in 0..x.len() {
            x[i] += alpha * direction[i];
        }

        unflatten_all(&x, &mut gaussians);
        gradient = compute_gradient_all(&gaussians, target);
        rendered = RendererV2::render(&gaussians, width, height);
        f_x = compute_mse(&rendered, target);

        lbfgs.update(&x_prev, &x, &grad_prev, &gradient);

        if iter % 10 == 0 || iter == max_iterations - 1 {
            let psnr = compute_psnr(&rendered, target);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };

            // Count out of bounds for each parameter type
            let pos_oob = (0..n_gaussians).filter(|&i| {
                let base = i * 7;
                x[base] < -0.1 || x[base] > 1.1 || x[base+1] < -0.1 || x[base+1] > 1.1
            }).count();
            let scale_oob = (0..n_gaussians).filter(|&i| {
                let base = i * 7;
                x[base+5] < 0.001 || x[base+6] < 0.001
            }).count();

            println!("    [{:3}/{:3}] PSNR={:.2}dB {} α={:.2e} ||g||={:.2e} pos_oob={} scale_oob={}",
                iter, max_iterations, psnr, dir, alpha, grad_norm, pos_oob, scale_oob);
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();

    // Clamp only at the end
    clamp_all(&mut x, n_gaussians);
    unflatten_all(&x, &mut gaussians);

    let final_rendered = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_rendered, target);
    let improvement = final_psnr - init_psnr;

    println!("  Final (after clamp): PSNR={:.2}dB ({:+.2}dB) in {:.1}s",
             final_psnr, improvement, elapsed);

    let mut result = TestResult::new("lbfgs_all_unconstrained", image_path);
    result.optimizer = "L-BFGS (all params, unconstrained)".to_string();
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

/// All parameters with soft barrier penalty for constraints
fn test_soft_barrier(target: &ImageBuffer<f32>, image_path: &str, max_iterations: usize) -> (f32, f32, f32) {
    let (width, height) = (target.width, target.height);
    let n_gaussians = 256;

    let mut gaussians = create_jittered_grid(target, n_gaussians);
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&rendered, target);

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n_gaussians, init_psnr);

    let barrier_weight = 0.001; // Weight for barrier penalty

    let mut x = flatten_all(&gaussians);
    let mut gradient = compute_gradient_with_barrier(&gaussians, target, &x, n_gaussians, barrier_weight);
    let mut rendered = RendererV2::render(&gaussians, width, height);
    let mut f_x = compute_loss_with_barrier(&rendered, target, &x, n_gaussians, barrier_weight);
    let mut last_psnr = init_psnr;

    let mut lbfgs = LBFGS::new(10);
    let start = Instant::now();

    for iter in 0..max_iterations {
        let direction = lbfgs.compute_direction(&gradient);
        let grad_norm: f32 = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();

        let alpha = line_search_barrier(
            &x, &direction, &gradient, f_x,
            n_gaussians, target, barrier_weight,
        );

        let x_prev = x.clone();
        let grad_prev = gradient.clone();

        // Update WITHOUT clamping
        for i in 0..x.len() {
            x[i] += alpha * direction[i];
        }

        unflatten_all(&x, &mut gaussians);
        gradient = compute_gradient_with_barrier(&gaussians, target, &x, n_gaussians, barrier_weight);
        rendered = RendererV2::render(&gaussians, width, height);
        f_x = compute_loss_with_barrier(&rendered, target, &x, n_gaussians, barrier_weight);

        lbfgs.update(&x_prev, &x, &grad_prev, &gradient);

        if iter % 10 == 0 || iter == max_iterations - 1 {
            let psnr = compute_psnr(&rendered, target);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };
            let mse_only = compute_mse(&rendered, target);

            println!("    [{:3}/{:3}] PSNR={:.2}dB {} α={:.2e} ||g||={:.2e} mse={:.4e}",
                iter, max_iterations, psnr, dir, alpha, grad_norm, mse_only);
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();

    // Clamp only at the end
    clamp_all(&mut x, n_gaussians);
    unflatten_all(&x, &mut gaussians);

    let final_rendered = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_rendered, target);
    let improvement = final_psnr - init_psnr;

    println!("  Final (after clamp): PSNR={:.2}dB ({:+.2}dB) in {:.1}s",
             final_psnr, improvement, elapsed);

    let mut result = TestResult::new("lbfgs_soft_barrier", image_path);
    result.optimizer = "L-BFGS (soft barrier)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = max_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.elapsed_seconds = elapsed;
    result.set_param("barrier_weight", barrier_weight);
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
    let mut flat = Vec::with_capacity(n * 3);
    for g in &grads {
        for &v in g {
            flat.push(v / pixel_count);
        }
    }
    flat
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

                // Color gradients
                grads[i][2] += error_r * weight;
                grads[i][3] += error_g * weight;
                grads[i][4] += error_b * weight;

                let error_weighted = error_r * gaussian.color.r
                                   + error_g * gaussian.color.g
                                   + error_b * gaussian.color.b;

                // Position gradients
                grads[i][0] += error_weighted * weight * dx / (sx * sx);
                grads[i][1] += error_weighted * weight * dy / (sy * sy);

                // Scale gradients
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

/// Compute gradient including soft barrier penalty
fn compute_gradient_with_barrier(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    x: &[f32],
    n_gaussians: usize,
    barrier_weight: f32,
) -> Vec<f32> {
    let mut grad = compute_gradient_all(gaussians, target);

    // Add barrier gradient: penalize values near or outside bounds
    // Using quadratic penalty: barrier(v) = max(0, v - upper)² + max(0, lower - v)²
    // Gradient: 2*max(0, v - upper) or -2*max(0, lower - v)
    for i in 0..n_gaussians {
        let base = i * 7;

        // Position: [0, 1]
        for j in 0..2 {
            let v = x[base + j];
            if v > 1.0 {
                grad[base + j] += barrier_weight * 2.0 * (v - 1.0);
            } else if v < 0.0 {
                grad[base + j] += barrier_weight * 2.0 * v; // v is negative
            }
        }

        // Colors: [0, 1]
        for j in 2..5 {
            let v = x[base + j];
            if v > 1.0 {
                grad[base + j] += barrier_weight * 2.0 * (v - 1.0);
            } else if v < 0.0 {
                grad[base + j] += barrier_weight * 2.0 * v;
            }
        }

        // Scales: [0.001, 0.5]
        for j in 5..7 {
            let v = x[base + j];
            if v > 0.5 {
                grad[base + j] += barrier_weight * 2.0 * (v - 0.5);
            } else if v < 0.001 {
                grad[base + j] += barrier_weight * 2.0 * (v - 0.001);
            }
        }
    }

    grad
}

fn compute_loss_with_barrier(
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
    x: &[f32],
    n_gaussians: usize,
    barrier_weight: f32,
) -> f32 {
    let mse = compute_mse(rendered, target);

    let mut barrier = 0.0;
    for i in 0..n_gaussians {
        let base = i * 7;

        // Position: [0, 1]
        for j in 0..2 {
            let v = x[base + j];
            if v > 1.0 {
                barrier += (v - 1.0).powi(2);
            } else if v < 0.0 {
                barrier += v.powi(2);
            }
        }

        // Colors: [0, 1]
        for j in 2..5 {
            let v = x[base + j];
            if v > 1.0 {
                barrier += (v - 1.0).powi(2);
            } else if v < 0.0 {
                barrier += v.powi(2);
            }
        }

        // Scales: [0.001, 0.5]
        for j in 5..7 {
            let v = x[base + j];
            if v > 0.5 {
                barrier += (v - 0.5).powi(2);
            } else if v < 0.001 {
                barrier += (v - 0.001).powi(2);
            }
        }
    }

    mse + barrier_weight * barrier
}

fn line_search_colors(
    x: &[f32],
    direction: &[f32],
    gradient: &[f32],
    f_x: f32,
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
) -> f32 {
    let mut alpha = 1.0;
    let c1 = 1e-4;
    let n_gaussians = gaussians.len();
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

        // Create temporary Gaussians with updated colors (NO CLAMPING)
        let mut temp_gaussians = gaussians.to_vec();
        for (i, g) in temp_gaussians.iter_mut().enumerate() {
            g.color.r = x_new[i * 3];
            g.color.g = x_new[i * 3 + 1];
            g.color.b = x_new[i * 3 + 2];
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

fn line_search_all(
    x: &[f32],
    direction: &[f32],
    gradient: &[f32],
    f_x: f32,
    n_gaussians: usize,
    target: &ImageBuffer<f32>,
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

        // Create temporary Gaussians (NO CLAMPING)
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

fn line_search_barrier(
    x: &[f32],
    direction: &[f32],
    gradient: &[f32],
    f_x: f32,
    n_gaussians: usize,
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
        let f_new = compute_loss_with_barrier(&rendered, target, &x_new, n_gaussians, barrier_weight);

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
