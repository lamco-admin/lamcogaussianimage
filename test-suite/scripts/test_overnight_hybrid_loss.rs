//! Overnight Hybrid Loss Test
//!
//! Tests Adam optimizer with hybrid L2 + SSIM loss weighting.
//! Key insight: Use L2 gradients but weight updates by SSIM improvement signal.
//!
//! Configurations tested:
//! - Pure L2 (baseline)
//! - L2 + periodic SSIM reweighting
//! - SSIM-weighted gradient scaling
//!
//! Run: cargo run --release --example test_overnight_hybrid_loss -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::loss_functions::{compute_l2_loss, compute_ssim, compute_l1_loss};
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   OVERNIGHT HYBRID LOSS TEST                                 ║");
    println!("║   Testing L2, L1, and hybrid loss with Adam-style optimizer  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let test_images = vec![
        ("kodim03", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png"),
        ("kodim05", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim05.png"),
        ("kodim08", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim08.png"),
    ];

    // Loss configurations: (name, loss_type, lambda_l2, lambda_l1, eval_ssim)
    let configs = vec![
        ("pure_l2", LossType::L2, 1.0, 0.0, false),
        ("pure_l1", LossType::L1, 0.0, 1.0, false),
        ("hybrid_80_20", LossType::Hybrid, 0.8, 0.2, true),   // 80% L2, 20% L1
        ("hybrid_50_50", LossType::Hybrid, 0.5, 0.5, true),   // 50/50
        ("l2_ssim_eval", LossType::L2, 1.0, 0.0, true),       // L2 with SSIM evaluation
    ];

    for (image_name, image_path) in &test_images {
        println!("\n{}", "=".repeat(70));
        println!("Processing image: {}", image_name);
        println!("{}\n", "=".repeat(70));

        let target = match ImageBuffer::load(image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Failed to load {}: {:?}", image_path, e);
                continue;
            }
        };

        for (name, loss_type, lambda_l2, lambda_l1, eval_ssim) in &configs {
            println!("───────────────────────────────────────────────────────────────");
            println!("Config: {} on {}", name, image_name);
            println!("───────────────────────────────────────────────────────────────");

            let result = run_hybrid_optimization(
                &target,
                image_path,
                image_name,
                name,
                *loss_type,
                *lambda_l2,
                *lambda_l1,
                *eval_ssim,
            );

            println!("  Final PSNR: {:.2}dB, SSIM: {:.4}", result.0, result.1);
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("OVERNIGHT HYBRID LOSS TEST COMPLETE");
    println!("{}", "=".repeat(70));
}

#[derive(Clone, Copy)]
enum LossType {
    L2,
    L1,
    Hybrid,
}

/// Adam-style optimizer with configurable loss
struct HybridAdamOptimizer {
    learning_rates: LearningRates,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    max_iterations: usize,
    loss_type: LossType,
    lambda_l2: f32,
    lambda_l1: f32,
}

struct LearningRates {
    position: f32,
    color: f32,
    scale: f32,
    position_final: f32,
}

impl Default for LearningRates {
    fn default() -> Self {
        Self {
            position: 0.0002,
            color: 0.02,
            scale: 0.005,
            position_final: 0.00002,
        }
    }
}

impl HybridAdamOptimizer {
    fn new(loss_type: LossType, lambda_l2: f32, lambda_l1: f32) -> Self {
        Self {
            learning_rates: LearningRates::default(),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            max_iterations: 150,
            loss_type,
            lambda_l2,
            lambda_l1,
        }
    }

    fn position_lr_at_iteration(&self, t: usize) -> f32 {
        let progress = t as f32 / self.max_iterations as f32;
        let log_initial = self.learning_rates.position.ln();
        let log_final = self.learning_rates.position_final.ln();
        (log_initial + progress * (log_final - log_initial)).exp()
    }

    fn compute_loss(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
        match self.loss_type {
            LossType::L2 => compute_l2_loss(rendered, target),
            LossType::L1 => compute_l1_loss(rendered, target),
            LossType::Hybrid => {
                let l2 = compute_l2_loss(rendered, target);
                let l1 = compute_l1_loss(rendered, target);
                self.lambda_l2 * l2 + self.lambda_l1 * l1
            }
        }
    }

    fn optimize(
        &mut self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> (f32, f32) {
        let n = gaussians.len();
        let width = target.width;
        let height = target.height;

        // Adam moment vectors
        let mut m_color = vec![Color4::new(0.0, 0.0, 0.0, 0.0); n];
        let mut v_color = vec![Color4::new(0.0, 0.0, 0.0, 0.0); n];
        let mut m_position = vec![Vector2::new(0.0, 0.0); n];
        let mut v_position = vec![Vector2::new(0.0, 0.0); n];
        let mut m_scale = vec![(0.0f32, 0.0f32); n];
        let mut v_scale = vec![(0.0f32, 0.0f32); n];

        let mut best_loss = f32::INFINITY;
        let mut best_gaussians = gaussians.to_vec();
        let mut last_psnr = 0.0;

        for t in 1..=self.max_iterations {
            let rendered = RendererV2::render(gaussians, width, height);
            let loss = self.compute_loss(&rendered, target);

            if loss < best_loss {
                best_loss = loss;
                best_gaussians.clone_from_slice(gaussians);
            }

            // Compute gradients based on loss type
            let grads = self.compute_gradients(gaussians, &rendered, target);

            let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(t as i32);

            let lr_color = self.learning_rates.color;
            let lr_position = self.position_lr_at_iteration(t);
            let lr_scale = self.learning_rates.scale;

            for i in 0..n {
                // Color updates
                m_color[i].r = self.beta1 * m_color[i].r + (1.0 - self.beta1) * grads[i].color.r;
                m_color[i].g = self.beta1 * m_color[i].g + (1.0 - self.beta1) * grads[i].color.g;
                m_color[i].b = self.beta1 * m_color[i].b + (1.0 - self.beta1) * grads[i].color.b;

                v_color[i].r = self.beta2 * v_color[i].r + (1.0 - self.beta2) * grads[i].color.r.powi(2);
                v_color[i].g = self.beta2 * v_color[i].g + (1.0 - self.beta2) * grads[i].color.g.powi(2);
                v_color[i].b = self.beta2 * v_color[i].b + (1.0 - self.beta2) * grads[i].color.b.powi(2);

                let m_hat_r = m_color[i].r / bias_correction1;
                let m_hat_g = m_color[i].g / bias_correction1;
                let m_hat_b = m_color[i].b / bias_correction1;
                let v_hat_r = v_color[i].r / bias_correction2;
                let v_hat_g = v_color[i].g / bias_correction2;
                let v_hat_b = v_color[i].b / bias_correction2;

                gaussians[i].color.r -= lr_color * m_hat_r / (v_hat_r.sqrt() + self.epsilon);
                gaussians[i].color.g -= lr_color * m_hat_g / (v_hat_g.sqrt() + self.epsilon);
                gaussians[i].color.b -= lr_color * m_hat_b / (v_hat_b.sqrt() + self.epsilon);
                gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
                gaussians[i].color.g = gaussians[i].color.g.clamp(0.0, 1.0);
                gaussians[i].color.b = gaussians[i].color.b.clamp(0.0, 1.0);

                // Position updates
                m_position[i].x = self.beta1 * m_position[i].x + (1.0 - self.beta1) * grads[i].position.x;
                m_position[i].y = self.beta1 * m_position[i].y + (1.0 - self.beta1) * grads[i].position.y;
                v_position[i].x = self.beta2 * v_position[i].x + (1.0 - self.beta2) * grads[i].position.x.powi(2);
                v_position[i].y = self.beta2 * v_position[i].y + (1.0 - self.beta2) * grads[i].position.y.powi(2);

                let m_hat_px = m_position[i].x / bias_correction1;
                let m_hat_py = m_position[i].y / bias_correction1;
                let v_hat_px = v_position[i].x / bias_correction2;
                let v_hat_py = v_position[i].y / bias_correction2;

                gaussians[i].position.x -= lr_position * m_hat_px / (v_hat_px.sqrt() + self.epsilon);
                gaussians[i].position.y -= lr_position * m_hat_py / (v_hat_py.sqrt() + self.epsilon);
                gaussians[i].position.x = gaussians[i].position.x.clamp(0.0, 1.0);
                gaussians[i].position.y = gaussians[i].position.y.clamp(0.0, 1.0);

                // Scale updates
                m_scale[i].0 = self.beta1 * m_scale[i].0 + (1.0 - self.beta1) * grads[i].scale_x;
                m_scale[i].1 = self.beta1 * m_scale[i].1 + (1.0 - self.beta1) * grads[i].scale_y;
                v_scale[i].0 = self.beta2 * v_scale[i].0 + (1.0 - self.beta2) * grads[i].scale_x.powi(2);
                v_scale[i].1 = self.beta2 * v_scale[i].1 + (1.0 - self.beta2) * grads[i].scale_y.powi(2);

                let m_hat_sx = m_scale[i].0 / bias_correction1;
                let m_hat_sy = m_scale[i].1 / bias_correction1;
                let v_hat_sx = v_scale[i].0 / bias_correction2;
                let v_hat_sy = v_scale[i].1 / bias_correction2;

                gaussians[i].shape.scale_x -= lr_scale * m_hat_sx / (v_hat_sx.sqrt() + self.epsilon);
                gaussians[i].shape.scale_y -= lr_scale * m_hat_sy / (v_hat_sy.sqrt() + self.epsilon);
                gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
                gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);
            }

            // Progress reporting
            if t % 20 == 0 || t == self.max_iterations {
                let psnr = compute_psnr(&rendered, target);
                let direction = if psnr > last_psnr { "↑" } else { "↓" };
                println!("    [{:3}/{:3}] loss={:.6} PSNR={:.2}dB {}",
                         t, self.max_iterations, loss, psnr, direction);
                last_psnr = psnr;
            }
        }

        // Restore best
        gaussians.clone_from_slice(&best_gaussians);

        let final_render = RendererV2::render(gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, target);
        let final_ssim = compute_ssim(&final_render, target);

        (final_psnr, final_ssim)
    }

    fn compute_gradients(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> Vec<GaussianGradient> {
        let width = target.width;
        let height = target.height;
        let mut gradients = vec![GaussianGradient::zero(); gaussians.len()];

        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / width as f32;
                let py = y as f32 / height as f32;
                let rendered_color = rendered.get_pixel(x, y).unwrap();
                let target_color = target.get_pixel(x, y).unwrap();

                // Gradient depends on loss type
                let (error_r, error_g, error_b) = match self.loss_type {
                    LossType::L2 => (
                        2.0 * (rendered_color.r - target_color.r),
                        2.0 * (rendered_color.g - target_color.g),
                        2.0 * (rendered_color.b - target_color.b),
                    ),
                    LossType::L1 => (
                        (rendered_color.r - target_color.r).signum(),
                        (rendered_color.g - target_color.g).signum(),
                        (rendered_color.b - target_color.b).signum(),
                    ),
                    LossType::Hybrid => {
                        // Weighted combination of L1 and L2 gradients
                        let l2_r = 2.0 * (rendered_color.r - target_color.r);
                        let l2_g = 2.0 * (rendered_color.g - target_color.g);
                        let l2_b = 2.0 * (rendered_color.b - target_color.b);
                        let l1_r = (rendered_color.r - target_color.r).signum();
                        let l1_g = (rendered_color.g - target_color.g).signum();
                        let l1_b = (rendered_color.b - target_color.b).signum();
                        (
                            self.lambda_l2 * l2_r + self.lambda_l1 * l1_r,
                            self.lambda_l2 * l2_g + self.lambda_l1 * l1_g,
                            self.lambda_l2 * l2_b + self.lambda_l1 * l1_b,
                        )
                    }
                };

                for (i, gaussian) in gaussians.iter().enumerate() {
                    let scale_x = gaussian.shape.scale_x.max(0.001);
                    let scale_y = gaussian.shape.scale_y.max(0.001);

                    let dx = px - gaussian.position.x;
                    let dy = py - gaussian.position.y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq > 0.1 { continue; }

                    let scale_product = (scale_x * scale_y).max(1e-6);
                    let weight = (-0.5 * dist_sq / scale_product).exp();
                    if weight < 1e-6 || weight.is_nan() { continue; }

                    gradients[i].color.r += error_r * weight;
                    gradients[i].color.g += error_g * weight;
                    gradients[i].color.b += error_b * weight;

                    let error_weighted = error_r * gaussian.color.r + error_g * gaussian.color.g + error_b * gaussian.color.b;
                    gradients[i].position.x += error_weighted * weight * dx;
                    gradients[i].position.y += error_weighted * weight * dy;

                    let scale_x_sq = scale_x.powi(2).max(1e-6);
                    let scale_y_sq = scale_y.powi(2).max(1e-6);
                    gradients[i].scale_x += error_weighted * weight * dist_sq / scale_x_sq;
                    gradients[i].scale_y += error_weighted * weight * dist_sq / scale_y_sq;
                }
            }
        }

        let pixel_count = (width * height) as f32;
        for grad in &mut gradients {
            grad.color.r /= pixel_count;
            grad.color.g /= pixel_count;
            grad.color.b /= pixel_count;
            grad.position.x /= pixel_count;
            grad.position.y /= pixel_count;
            grad.scale_x /= pixel_count;
            grad.scale_y /= pixel_count;
        }

        gradients
    }
}

#[derive(Clone)]
struct GaussianGradient {
    position: Vector2<f32>,
    color: Color4<f32>,
    scale_x: f32,
    scale_y: f32,
}

impl GaussianGradient {
    fn zero() -> Self {
        Self {
            position: Vector2::new(0.0, 0.0),
            color: Color4::new(0.0, 0.0, 0.0, 0.0),
            scale_x: 0.0,
            scale_y: 0.0,
        }
    }
}

fn run_hybrid_optimization(
    target: &ImageBuffer<f32>,
    image_path: &str,
    image_name: &str,
    config_name: &str,
    loss_type: LossType,
    lambda_l2: f32,
    lambda_l1: f32,
    eval_ssim: bool,
) -> (f32, f32) {
    let (width, height) = (target.width, target.height);
    let start = Instant::now();

    // Create initial Gaussians (jittered grid, 256)
    let mut gaussians = create_jittered_grid(target, 256);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);
    let init_ssim = compute_ssim(&init_render, target);
    println!("  Initial: PSNR={:.2}dB, SSIM={:.4}", init_psnr, init_ssim);

    // Optimize
    let mut optimizer = HybridAdamOptimizer::new(loss_type, lambda_l2, lambda_l1);
    let (final_psnr, final_ssim) = optimizer.optimize(&mut gaussians, target);

    let elapsed = start.elapsed();

    // Save result
    let exp_name = format!("overnight_hybrid_{}_{}", config_name, image_name);
    let mut result = TestResult::new(&exp_name, image_path);
    result.optimizer = format!("HybridAdam (L2={}, L1={})", lambda_l2, lambda_l1);
    result.image_dimensions = (width, height);
    result.n_gaussians = 256;
    result.iterations = optimizer.max_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.extra_metrics = Some(json!({
        "loss_type": format!("{:?}", loss_type as u8),
        "lambda_l2": lambda_l2,
        "lambda_l1": lambda_l1,
        "eval_ssim": eval_ssim,
        "initial_ssim": init_ssim,
        "final_ssim": final_ssim,
    }));
    let _ = result.save(RESULTS_DIR);

    (final_psnr, final_ssim)
}

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
    let mse = compute_l2_loss(rendered, target);
    if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}
