//! Hybrid Adam → L-BFGS optimizer
//!
//! Based on 3DGS-LM research (ICCV 2025): Adam for warm-start, L-BFGS for refinement.
//! Key insight: L-BFGS from scratch provides NO benefit; hybrid is essential.
//!
//! Default strategy: 30 Adam iterations, then 20 L-BFGS iterations.
//!
//! **IMPORTANT**: Uses analytical gradients (not finite differences) for L-BFGS phase
//! to achieve practical performance. Finite differences would require O(2×7×n_gaussians)
//! renders per gradient computation, which is intractable.

use lgi_core::{ImageBuffer, lbfgs::LBFGS};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use crate::renderer_v2::RendererV2;
use crate::adam_optimizer::{AdamOptimizer, LearningRates};

/// Hybrid optimizer configuration
pub struct HybridConfig {
    /// Number of Adam warm-start iterations (default: 100)
    pub adam_iterations: usize,
    /// Number of L-BFGS refinement iterations (default: 50)
    pub lbfgs_iterations: usize,
    /// L-BFGS history size (default: 10)
    pub lbfgs_history: usize,
    /// Adam learning rates (uses per-parameter LRs by default)
    pub adam_lr: LearningRates,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            adam_iterations: 30,
            lbfgs_iterations: 20,
            lbfgs_history: 10,
            adam_lr: LearningRates::default(),
        }
    }
}

/// Hybrid Adam → L-BFGS optimizer
pub struct HybridOptimizer {
    pub config: HybridConfig,
}

impl Default for HybridOptimizer {
    fn default() -> Self {
        Self {
            config: HybridConfig::default(),
        }
    }
}

impl HybridOptimizer {
    pub fn new(config: HybridConfig) -> Self {
        Self { config }
    }

    /// Optimize Gaussians using Adam warm-start followed by L-BFGS refinement
    pub fn optimize(
        &mut self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> f32 {
        let n = gaussians.len();
        println!("=== Hybrid Adam→L-BFGS Optimizer ===");
        println!("  Gaussians: {}", n);
        println!("  Phase 1: Adam ({} iterations)", self.config.adam_iterations);
        println!("  Phase 2: L-BFGS ({} iterations, history={})",
                 self.config.lbfgs_iterations, self.config.lbfgs_history);

        // Phase 1: Adam warm-start
        println!("\n--- Phase 1: Adam Warm-Start ---");
        let mut adam = AdamOptimizer::new();
        adam.learning_rates = self.config.adam_lr.clone();
        adam.max_iterations = self.config.adam_iterations;

        let adam_loss = adam.optimize(gaussians, target);
        println!("  Adam final loss: {:.6}", adam_loss);

        // Phase 2: L-BFGS refinement
        println!("\n--- Phase 2: L-BFGS Refinement ---");
        let lbfgs_loss = self.run_lbfgs(gaussians, target);
        println!("  L-BFGS final loss: {:.6}", lbfgs_loss);

        println!("\n=== Hybrid Complete ===");
        println!("  Adam loss:  {:.6}", adam_loss);
        println!("  Final loss: {:.6}", lbfgs_loss);
        println!("  Improvement from L-BFGS: {:.2}%",
                 (adam_loss - lbfgs_loss) / adam_loss * 100.0);

        lbfgs_loss
    }

    /// Run L-BFGS on Gaussian parameters
    fn run_lbfgs(
        &self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> f32 {
        let n = gaussians.len();

        // Flatten Gaussians to parameter vector
        // Parameters per Gaussian: x, y, r, g, b, scale_x, scale_y (7 params)
        // We skip rotation and opacity for simplicity in this first version
        let params_per_gaussian = 7;
        let mut x: Vec<f32> = Vec::with_capacity(n * params_per_gaussian);

        for g in gaussians.iter() {
            x.push(g.position.x);
            x.push(g.position.y);
            x.push(g.color.r);
            x.push(g.color.g);
            x.push(g.color.b);
            x.push(g.shape.scale_x);
            x.push(g.shape.scale_y);
        }

        // Track best state
        let mut best_loss = f32::INFINITY;
        let mut best_x = x.clone();

        // We need to track iterations manually since we use analytical gradients
        // which require converting params back to Gaussians first
        let mut lbfgs = LBFGS::new(self.config.lbfgs_history);

        // Manual optimization loop to use analytical gradients
        let mut iteration = 0;
        let mut gradient = {
            let temp = self.params_to_gaussians(&x, gaussians);
            self.compute_gradient_analytical(&temp, target)
        };
        let mut f_x = {
            let temp = self.params_to_gaussians(&x, gaussians);
            let rendered = RendererV2::render(&temp, target.width, target.height);
            self.compute_loss(&rendered, target)
        };

        println!("  L-BFGS iter 0: loss = {:.6}", f_x);
        best_loss = f_x;
        best_x = x.clone();

        while iteration < self.config.lbfgs_iterations {
            // Check convergence
            let grad_norm: f32 = gradient.iter().map(|&g| g * g).sum::<f32>().sqrt();
            if grad_norm < 1e-7 {
                println!("  L-BFGS converged (grad_norm = {:.2e})", grad_norm);
                break;
            }

            // Compute search direction
            let direction = lbfgs.compute_direction(&gradient);

            // Line search
            let alpha = {
                let mut alpha = 1.0;
                let directional_derivative: f32 = gradient.iter()
                    .zip(direction.iter())
                    .map(|(&g, &d)| g * d)
                    .sum();

                if directional_derivative >= 0.0 {
                    alpha = 1e-6;  // Not a descent direction, use small step
                } else {
                    // Backtracking line search with Armijo condition
                    for _ in 0..10 {
                        let x_new: Vec<f32> = x.iter()
                            .zip(direction.iter())
                            .map(|(&xi, &di)| xi + alpha * di)
                            .collect();

                        let temp = self.params_to_gaussians(&x_new, gaussians);
                        let rendered = RendererV2::render(&temp, target.width, target.height);
                        let f_new = self.compute_loss(&rendered, target);

                        if f_new <= f_x + 1e-4 * alpha * directional_derivative {
                            break;
                        }
                        alpha *= 0.5;
                    }
                }
                alpha
            };

            // Take step
            let x_prev = x.clone();
            let grad_prev = gradient.clone();

            for i in 0..x.len() {
                x[i] += alpha * direction[i];
            }

            // Compute new function value and gradient
            let temp = self.params_to_gaussians(&x, gaussians);
            gradient = self.compute_gradient_analytical(&temp, target);
            let rendered = RendererV2::render(&temp, target.width, target.height);
            f_x = self.compute_loss(&rendered, target);

            // Update L-BFGS history
            lbfgs.update(&x_prev, &x, &grad_prev, &gradient);

            iteration += 1;

            // Track best
            if f_x < best_loss {
                best_loss = f_x;
                best_x = x.clone();
            }

            if iteration % 5 == 0 || iteration == self.config.lbfgs_iterations {
                println!("  L-BFGS iter {}: loss = {:.6} (α = {:.2e})", iteration, f_x, alpha);
            }
        }

        // Write best parameters back to Gaussians
        self.write_params_to_gaussians(&best_x, gaussians);

        println!("  L-BFGS completed {} iterations", iteration);

        best_loss
    }

    /// Convert flat parameter vector to temporary Gaussians
    fn params_to_gaussians(
        &self,
        params: &[f32],
        template: &[Gaussian2D<f32, Euler<f32>>],
    ) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        let n = template.len();
        let mut result = template.to_vec();

        for i in 0..n {
            let base = i * 7;
            result[i].position.x = params[base].clamp(0.0, 1.0);
            result[i].position.y = params[base + 1].clamp(0.0, 1.0);
            result[i].color.r = params[base + 2].clamp(0.0, 1.0);
            result[i].color.g = params[base + 3].clamp(0.0, 1.0);
            result[i].color.b = params[base + 4].clamp(0.0, 1.0);
            result[i].shape.scale_x = params[base + 5].clamp(0.001, 0.5);
            result[i].shape.scale_y = params[base + 6].clamp(0.001, 0.5);
        }

        result
    }

    /// Write parameters back to Gaussians
    fn write_params_to_gaussians(
        &self,
        params: &[f32],
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
    ) {
        let n = gaussians.len();

        for i in 0..n {
            let base = i * 7;
            gaussians[i].position.x = params[base].clamp(0.0, 1.0);
            gaussians[i].position.y = params[base + 1].clamp(0.0, 1.0);
            gaussians[i].color.r = params[base + 2].clamp(0.0, 1.0);
            gaussians[i].color.g = params[base + 3].clamp(0.0, 1.0);
            gaussians[i].color.b = params[base + 4].clamp(0.0, 1.0);
            gaussians[i].shape.scale_x = params[base + 5].clamp(0.001, 0.5);
            gaussians[i].shape.scale_y = params[base + 6].clamp(0.001, 0.5);
        }
    }

    /// Compute gradient using analytical derivatives
    ///
    /// Returns gradient vector in same order as params: [x, y, r, g, b, sx, sy] per Gaussian
    fn compute_gradient_analytical(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> Vec<f32> {
        let width = target.width;
        let height = target.height;
        let n = gaussians.len();

        // Per-Gaussian gradients
        let mut grads: Vec<[f32; 7]> = vec![[0.0; 7]; n];

        let rendered = RendererV2::render(gaussians, width, height);

        // Accumulate gradients from each pixel
        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / width as f32;
                let py = y as f32 / height as f32;

                let rendered_color = rendered.get_pixel(x, y).unwrap();
                let target_color = target.get_pixel(x, y).unwrap();

                // Error derivatives (2 * (rendered - target) for MSE)
                let error_r = 2.0 * (rendered_color.r - target_color.r);
                let error_g = 2.0 * (rendered_color.g - target_color.g);
                let error_b = 2.0 * (rendered_color.b - target_color.b);

                for (i, gaussian) in gaussians.iter().enumerate() {
                    let dx = px - gaussian.position.x;
                    let dy = py - gaussian.position.y;

                    let sx = gaussian.shape.scale_x.max(1e-6);
                    let sy = gaussian.shape.scale_y.max(1e-6);
                    let theta = gaussian.shape.rotation;
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();

                    // Rotated coordinates
                    let dx_rot = dx * cos_t + dy * sin_t;
                    let dy_rot = -dx * sin_t + dy * cos_t;

                    let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);
                    if dist_sq > 16.0 { continue; }

                    let weight = (-0.5 * dist_sq).exp();
                    if weight < 1e-6 { continue; }

                    // Color gradients: d_loss/d_color = error * weight
                    grads[i][2] += error_r * weight;  // r
                    grads[i][3] += error_g * weight;  // g
                    grads[i][4] += error_b * weight;  // b

                    // Position gradients: chain rule through Gaussian weight
                    let error_weighted = error_r * gaussian.color.r
                                       + error_g * gaussian.color.g
                                       + error_b * gaussian.color.b;

                    let grad_weight_x = weight * (dx_rot * cos_t / (sx * sx) - dy_rot * sin_t / (sy * sy));
                    let grad_weight_y = weight * (dx_rot * sin_t / (sx * sx) + dy_rot * cos_t / (sy * sy));

                    // Sign corrections from V2 optimizer research
                    grads[i][0] -= error_weighted * grad_weight_x;  // x (minus)
                    grads[i][1] += error_weighted * grad_weight_y;  // y (plus)

                    // Scale gradients
                    grads[i][5] += error_weighted * weight * (dx_rot * dx_rot) / (sx * sx * sx);  // scale_x
                    grads[i][6] += error_weighted * weight * (dy_rot * dy_rot) / (sy * sy * sy);  // scale_y
                }
            }
        }

        // Normalize by pixel count and flatten to parameter vector
        let pixel_count = (width * height) as f32;
        let mut flat_grad = Vec::with_capacity(n * 7);

        for g in &grads {
            for &v in g {
                flat_grad.push(v / pixel_count);
            }
        }

        flat_grad
    }

    fn compute_loss(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
        let mut loss = 0.0;
        for (r, t) in rendered.data.iter().zip(target.data.iter()) {
            loss += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
        }
        loss / (rendered.width * rendered.height * 3) as f32
    }
}
