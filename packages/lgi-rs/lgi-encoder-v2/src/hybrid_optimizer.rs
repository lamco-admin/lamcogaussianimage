//! Hybrid Adam → L-BFGS optimizer
//!
//! Based on 3DGS-LM research (ICCV 2025): Adam for warm-start, L-BFGS for refinement.
//! Key insight: L-BFGS from scratch provides NO benefit; hybrid is essential.
//!
//! Default strategy: 100 Adam iterations, then 50 L-BFGS iterations.

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
            adam_iterations: 100,
            lbfgs_iterations: 50,
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
        let mut best_iteration = 0;

        // Loss function closure
        let loss_fn = |params: &[f32]| -> f32 {
            let temp_gaussians = self.params_to_gaussians(params, gaussians);
            let rendered = RendererV2::render(&temp_gaussians, target.width, target.height);
            self.compute_loss(&rendered, target)
        };

        // Gradient function closure (finite differences for now)
        let grad_fn = |params: &[f32]| -> Vec<f32> {
            self.compute_gradient_fd(params, gaussians, target)
        };

        // Run L-BFGS
        let mut lbfgs = LBFGS::new(self.config.lbfgs_history);

        let (final_x, final_loss, iterations) = lbfgs.optimize(
            x,
            loss_fn,
            grad_fn,
            self.config.lbfgs_iterations,
            1e-7,  // tolerance
        );

        // Track best (L-BFGS should be monotonically improving, but just in case)
        if final_loss < best_loss {
            best_loss = final_loss;
            best_x = final_x.clone();
            best_iteration = iterations;
        }

        // Write best parameters back to Gaussians
        self.write_params_to_gaussians(&best_x, gaussians);

        println!("  L-BFGS converged in {} iterations", iterations);

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

    /// Compute gradient using finite differences
    fn compute_gradient_fd(
        &self,
        params: &[f32],
        template: &[Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> Vec<f32> {
        let eps = 1e-5;
        let n = params.len();
        let mut grad = vec![0.0f32; n];

        let f0 = {
            let temp_gaussians = self.params_to_gaussians(params, template);
            let rendered = RendererV2::render(&temp_gaussians, target.width, target.height);
            self.compute_loss(&rendered, target)
        };

        // Central difference for better accuracy
        let mut params_plus = params.to_vec();
        let mut params_minus = params.to_vec();

        for i in 0..n {
            params_plus[i] = params[i] + eps;
            params_minus[i] = params[i] - eps;

            let f_plus = {
                let temp = self.params_to_gaussians(&params_plus, template);
                let rendered = RendererV2::render(&temp, target.width, target.height);
                self.compute_loss(&rendered, target)
            };

            let f_minus = {
                let temp = self.params_to_gaussians(&params_minus, template);
                let rendered = RendererV2::render(&temp, target.width, target.height);
                self.compute_loss(&rendered, target)
            };

            grad[i] = (f_plus - f_minus) / (2.0 * eps);

            params_plus[i] = params[i];
            params_minus[i] = params[i];
        }

        grad
    }

    fn compute_loss(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
        let mut loss = 0.0;
        for (r, t) in rendered.data.iter().zip(target.data.iter()) {
            loss += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
        }
        loss / (rendered.width * rendered.height * 3) as f32
    }
}
