//! Closed-Form Color Solver Test
//!
//! KEY INSIGHT from ADAM-considerations.txt:
//! Given positions + shapes fixed, rendering is LINEAR in colors:
//!   I_hat(x,y) = sum_i w_i(x,y) * c_i
//!
//! For a fixed set of per-pixel weights w_i, the best colors c_i
//! (in least-squares sense) solve a linear system: min_c ||I - W*c||^2
//!
//! This means we can:
//! 1. Use Adam ONLY on positions + shapes
//! 2. Every M iterations, solve for colors in closed form
//!
//! This is like doing a mini Gauss-Newton step in the "color" subspace
//! and potentially saves many iterations.
//!
//! Run: cargo run --release --example test_closed_form_colors -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

// Test images
const TEST_IMAGES: &[(&str, &str)] = &[
    ("kodim03", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png"),
    ("kodim05", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim05.png"),
    ("kodim08", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim08.png"),
];

/// Compute per-pixel Gaussian weights
/// Returns a matrix W where W[pixel_idx][gaussian_idx] = weight
fn compute_weight_matrix(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    width: u32,
    height: u32,
) -> Vec<Vec<f32>> {
    let num_pixels = (width * height) as usize;
    let num_gaussians = gaussians.len();

    // Weight matrix: W[pixel][gaussian] = contribution weight
    let mut weights = vec![vec![0.0f32; num_gaussians]; num_pixels];

    for (g_idx, g) in gaussians.iter().enumerate() {
        // Gaussian center in pixel coordinates
        let cx = g.position.x * width as f32;
        let cy = g.position.y * height as f32;

        // Compute effective radius (3 sigma covers ~99.7%)
        let sigma_x = (g.shape.scale_x * width as f32).max(1.0);
        let sigma_y = (g.shape.scale_y * height as f32).max(1.0);
        let radius = (sigma_x.max(sigma_y) * 3.0) as i32;

        // Only compute weights within radius
        let x_min = ((cx as i32 - radius).max(0)) as u32;
        let x_max = ((cx as i32 + radius).min(width as i32 - 1)) as u32;
        let y_min = ((cy as i32 - radius).max(0)) as u32;
        let y_max = ((cy as i32 + radius).min(height as i32 - 1)) as u32;

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let px = x as f32 / width as f32;
                let py = y as f32 / height as f32;

                let dx = px - g.position.x;
                let dy = py - g.position.y;
                let dist_sq = dx * dx + dy * dy;

                let scale_product = (g.shape.scale_x * g.shape.scale_y).max(1e-6);
                let weight = (-0.5 * dist_sq / scale_product).exp();

                if weight > 1e-6 {
                    let pixel_idx = (y * width + x) as usize;
                    weights[pixel_idx][g_idx] = weight * g.opacity;
                }
            }
        }
    }

    // Normalize weights so they sum to 1 at each pixel
    for pixel_weights in &mut weights {
        let sum: f32 = pixel_weights.iter().sum();
        if sum > 1e-6 {
            for w in pixel_weights.iter_mut() {
                *w /= sum;
            }
        }
    }

    weights
}

/// Solve for optimal colors using least-squares
/// Given: I = W * C (where I is target, W is weights, C is colors)
/// Solve: C = (W^T * W)^-1 * W^T * I
///
/// This is the CLOSED-FORM solution for colors!
fn solve_colors_least_squares(
    weights: &[Vec<f32>],
    target: &ImageBuffer<f32>,
    num_gaussians: usize,
) -> Vec<Color4<f32>> {
    let num_pixels = weights.len();

    // For each color channel, solve: minimize ||I_c - W * c_c||^2
    // Normal equations: (W^T * W) * c = W^T * I

    // Compute W^T * W (num_gaussians x num_gaussians matrix)
    let mut wtw = vec![vec![0.0f32; num_gaussians]; num_gaussians];
    for i in 0..num_gaussians {
        for j in 0..num_gaussians {
            let mut sum = 0.0f32;
            for p in 0..num_pixels {
                sum += weights[p][i] * weights[p][j];
            }
            wtw[i][j] = sum;
        }
        // Add regularization to prevent singular matrix
        wtw[i][i] += 1e-4;
    }

    // Compute W^T * I for each channel
    let mut wt_ir = vec![0.0f32; num_gaussians];
    let mut wt_ig = vec![0.0f32; num_gaussians];
    let mut wt_ib = vec![0.0f32; num_gaussians];

    for (p, pixel) in target.data.iter().enumerate() {
        for g in 0..num_gaussians {
            wt_ir[g] += weights[p][g] * pixel.r;
            wt_ig[g] += weights[p][g] * pixel.g;
            wt_ib[g] += weights[p][g] * pixel.b;
        }
    }

    // Solve using Gauss-Seidel iteration (simple, works for positive-definite)
    let colors_r = solve_linear_system(&wtw, &wt_ir);
    let colors_g = solve_linear_system(&wtw, &wt_ig);
    let colors_b = solve_linear_system(&wtw, &wt_ib);

    // Combine into Color4
    let mut colors = Vec::with_capacity(num_gaussians);
    for i in 0..num_gaussians {
        colors.push(Color4::new(
            colors_r[i].clamp(0.0, 1.0),
            colors_g[i].clamp(0.0, 1.0),
            colors_b[i].clamp(0.0, 1.0),
            1.0,
        ));
    }

    colors
}

/// Simple Gauss-Seidel solver for positive-definite systems
fn solve_linear_system(a: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
    let n = b.len();
    let mut x = b.to_vec();  // Start with b as initial guess

    for _iter in 0..50 {  // Fixed iterations
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..n {
                if i != j {
                    sum -= a[i][j] * x[j];
                }
            }
            x[i] = sum / a[i][i].max(1e-6);
        }
    }

    x
}

/// Adam optimizer that ONLY updates position and shape
struct PositionShapeOnlyAdam {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    lr_position: f32,
    lr_scale: f32,

    m_position: Vec<Vector2<f32>>,
    v_position: Vec<Vector2<f32>>,
    m_scale: Vec<(f32, f32)>,
    v_scale: Vec<(f32, f32)>,
    t: u32,
}

impl PositionShapeOnlyAdam {
    fn new() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            lr_position: 0.0002,
            lr_scale: 0.005,
            m_position: Vec::new(),
            v_position: Vec::new(),
            m_scale: Vec::new(),
            v_scale: Vec::new(),
            t: 0,
        }
    }

    fn ensure_state(&mut self, n: usize) {
        while self.m_position.len() < n {
            self.m_position.push(Vector2::zero());
            self.v_position.push(Vector2::zero());
            self.m_scale.push((0.0, 0.0));
            self.v_scale.push((0.0, 0.0));
        }
    }

    fn step(&mut self, gaussians: &mut [Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) {
        let n = gaussians.len();
        self.ensure_state(n);
        self.t += 1;

        let width = target.width;
        let height = target.height;

        let rendered = RendererV2::render(gaussians, width, height);

        // Compute gradients for position and scale only
        let grads = self.compute_position_scale_gradients(gaussians, &rendered, target);

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..n {
            // Position update
            self.m_position[i].x = self.beta1 * self.m_position[i].x + (1.0 - self.beta1) * grads[i].0.x;
            self.m_position[i].y = self.beta1 * self.m_position[i].y + (1.0 - self.beta1) * grads[i].0.y;
            self.v_position[i].x = self.beta2 * self.v_position[i].x + (1.0 - self.beta2) * grads[i].0.x.powi(2);
            self.v_position[i].y = self.beta2 * self.v_position[i].y + (1.0 - self.beta2) * grads[i].0.y.powi(2);

            let m_hat_px = self.m_position[i].x / bias_correction1;
            let m_hat_py = self.m_position[i].y / bias_correction1;
            let v_hat_px = self.v_position[i].x / bias_correction2;
            let v_hat_py = self.v_position[i].y / bias_correction2;

            gaussians[i].position.x -= self.lr_position * m_hat_px / (v_hat_px.sqrt() + self.epsilon);
            gaussians[i].position.y -= self.lr_position * m_hat_py / (v_hat_py.sqrt() + self.epsilon);
            gaussians[i].position.x = gaussians[i].position.x.clamp(0.01, 0.99);
            gaussians[i].position.y = gaussians[i].position.y.clamp(0.01, 0.99);

            // Scale update
            self.m_scale[i].0 = self.beta1 * self.m_scale[i].0 + (1.0 - self.beta1) * grads[i].1;
            self.m_scale[i].1 = self.beta1 * self.m_scale[i].1 + (1.0 - self.beta1) * grads[i].2;
            self.v_scale[i].0 = self.beta2 * self.v_scale[i].0 + (1.0 - self.beta2) * grads[i].1.powi(2);
            self.v_scale[i].1 = self.beta2 * self.v_scale[i].1 + (1.0 - self.beta2) * grads[i].2.powi(2);

            let m_hat_sx = self.m_scale[i].0 / bias_correction1;
            let m_hat_sy = self.m_scale[i].1 / bias_correction1;
            let v_hat_sx = self.v_scale[i].0 / bias_correction2;
            let v_hat_sy = self.v_scale[i].1 / bias_correction2;

            gaussians[i].shape.scale_x -= self.lr_scale * m_hat_sx / (v_hat_sx.sqrt() + self.epsilon);
            gaussians[i].shape.scale_y -= self.lr_scale * m_hat_sy / (v_hat_sy.sqrt() + self.epsilon);
            gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
            gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);
        }
    }

    fn compute_position_scale_gradients(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> Vec<(Vector2<f32>, f32, f32)> {
        let width = target.width;
        let height = target.height;
        let mut grads = vec![(Vector2::zero(), 0.0f32, 0.0f32); gaussians.len()];

        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / width as f32;
                let py = y as f32 / height as f32;
                let rc = rendered.get_pixel(x, y).unwrap();
                let tc = target.get_pixel(x, y).unwrap();

                let error = (rc.r - tc.r) + (rc.g - tc.g) + (rc.b - tc.b);

                for (i, g) in gaussians.iter().enumerate() {
                    let scale_x = g.shape.scale_x.max(0.001);
                    let scale_y = g.shape.scale_y.max(0.001);

                    let dx = px - g.position.x;
                    let dy = py - g.position.y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq > 0.1 { continue; }

                    let scale_product = (scale_x * scale_y).max(1e-6);
                    let weight = (-0.5 * dist_sq / scale_product).exp();
                    if weight < 1e-6 { continue; }

                    let error_weighted = error * weight;
                    grads[i].0.x += error_weighted * dx;
                    grads[i].0.y += error_weighted * dy;

                    let scale_x_sq = scale_x.powi(2).max(1e-6);
                    let scale_y_sq = scale_y.powi(2).max(1e-6);
                    grads[i].1 += error_weighted * dist_sq / scale_x_sq;
                    grads[i].2 += error_weighted * dist_sq / scale_y_sq;
                }
            }
        }

        let pixel_count = (width * height) as f32;
        for grad in &mut grads {
            grad.0.x /= pixel_count;
            grad.0.y /= pixel_count;
            grad.1 /= pixel_count;
            grad.2 /= pixel_count;
        }

        grads
    }
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

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height * 3) as f32;
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= count;
    if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}

fn run_baseline_adam(
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    iterations: usize,
) -> (f32, f32, f32) {
    let (width, height) = (target.width, target.height);
    let mut gaussians = create_jittered_grid(target, n_gaussians);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);

    // Use standard Adam optimizer
    let mut adam = lgi_encoder_v2::adam_optimizer::AdamOptimizer::new();
    adam.max_iterations = iterations;
    adam.optimize(&mut gaussians, target);

    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);

    (init_psnr, final_psnr, final_psnr - init_psnr)
}

fn run_hybrid_color_solver(
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    iterations: usize,
    color_solve_interval: usize,
) -> (f32, f32, f32, usize) {
    let (width, height) = (target.width, target.height);
    let mut gaussians = create_jittered_grid(target, n_gaussians);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);

    // Position/shape-only Adam
    let mut adam = PositionShapeOnlyAdam::new();
    let mut color_solves = 0;

    for iter in 0..iterations {
        // Adam step for positions and shapes
        adam.step(&mut gaussians, target);

        // Periodic closed-form color update
        if iter % color_solve_interval == 0 {
            let weights = compute_weight_matrix(&gaussians, width, height);
            let optimal_colors = solve_colors_least_squares(&weights, target, n_gaussians);

            // Apply optimal colors
            for (g, c) in gaussians.iter_mut().zip(optimal_colors.iter()) {
                g.color = *c;
            }
            color_solves += 1;
        }
    }

    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);

    (init_psnr, final_psnr, final_psnr - init_psnr, color_solves)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   CLOSED-FORM COLOR SOLVER TEST - Hybrid Adam Optimization   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let n_gaussians = 256;
    let iterations = 100;

    // Test configurations: (name, color_solve_interval)
    // Lower interval = more frequent color solves
    let configs = vec![
        ("solve_every_10", 10),
        ("solve_every_25", 25),
        ("solve_every_50", 50),
        ("solve_once_at_end", iterations + 1),  // Effectively never during iteration
    ];

    for (img_name, img_path) in TEST_IMAGES {
        println!("\n{}", "=".repeat(70));
        println!("IMAGE: {}", img_name);
        println!("{}\n", "=".repeat(70));

        let target = match ImageBuffer::load(img_path) {
            Ok(t) => t,
            Err(e) => {
                println!("Failed to load {}: {}", img_path, e);
                continue;
            }
        };
        let (width, height) = (target.width, target.height);

        // Baseline: Standard Adam
        println!("--- Baseline: Standard Adam ---");
        let start = Instant::now();
        let (init_psnr, baseline_psnr, baseline_improvement) = run_baseline_adam(&target, n_gaussians, iterations);
        let baseline_time = start.elapsed().as_secs_f32();
        println!("  {:.2}dB → {:.2}dB ({:+.2}dB) in {:.2}s",
                 init_psnr, baseline_psnr, baseline_improvement, baseline_time);

        // Save baseline result
        let exp_name = format!("closed_form_colors_baseline_{}", img_name);
        let mut result = TestResult::new(&exp_name, img_path);
        result.optimizer = "Adam (standard)".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = baseline_psnr;
        result.improvement_db = baseline_improvement;
        result.elapsed_seconds = baseline_time;
        result.extra_metrics = Some(json!({ "color_solve_interval": 0 }));
        let _ = result.save(RESULTS_DIR);

        // Test hybrid configurations
        for (config_name, interval) in &configs {
            println!("\n--- {} (interval: {}) ---", config_name, interval);
            let start = Instant::now();
            let (init_psnr, final_psnr, improvement, color_solves) =
                run_hybrid_color_solver(&target, n_gaussians, iterations, *interval);
            let elapsed = start.elapsed().as_secs_f32();

            let vs_baseline = final_psnr - baseline_psnr;
            println!("  {:.2}dB → {:.2}dB ({:+.2}dB) in {:.2}s ({} color solves)",
                     init_psnr, final_psnr, improvement, elapsed, color_solves);
            println!("  vs baseline: {:+.2}dB", vs_baseline);

            // Save result
            let exp_name = format!("closed_form_colors_{}_{}", config_name, img_name);
            let mut result = TestResult::new(&exp_name, img_path);
            result.optimizer = format!("Adam + ClosedFormColors (interval={})", interval);
            result.image_dimensions = (width, height);
            result.n_gaussians = n_gaussians;
            result.iterations = iterations;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = improvement;
            result.elapsed_seconds = elapsed;
            result.extra_metrics = Some(json!({
                "color_solve_interval": interval,
                "color_solves": color_solves,
                "vs_baseline_db": vs_baseline,
            }));
            let _ = result.save(RESULTS_DIR);
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("All tests complete! Results saved to {}", RESULTS_DIR);
    println!("{}", "=".repeat(70));
}
