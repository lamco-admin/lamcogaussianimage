//! Levenberg-Marquardt optimizer for Gaussian splatting
//!
//! **Why L-M for this problem?**
//! Our loss IS nonlinear least-squares: ||rendered - target||²
//! L-M interpolates between gradient descent (safe) and Gauss-Newton (fast).
//!
//! Research shows 20-30% speedup over Adam for refinement (3DGS-LM, ICCV 2025).
//!
//! # Usage
//! Best used AFTER Adam warm-start (100+ iterations), not from scratch.
//! ```ignore
//! let mut lm = LMOptimizer::new(LMConfig::default());
//! let loss = lm.optimize(&mut gaussians, &target);
//! ```

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use crate::renderer_v2::RendererV2;

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DVector, DMatrix, Owned, Dyn};

/// Levenberg-Marquardt optimizer configuration
#[derive(Clone, Debug)]
pub struct LMConfig {
    /// Maximum iterations (default: 100)
    pub max_iterations: usize,
    /// Tolerance for convergence (default: 1e-8)
    pub tolerance: f64,
    /// Initial damping factor λ (default: 0.01)
    pub initial_lambda: f64,
    /// Whether to use finite differences for Jacobian (default: true)
    /// Set to false when analytical gradients are available
    pub use_finite_diff: bool,
    /// Finite difference step size (default: 1e-6)
    pub fd_epsilon: f64,
}

impl Default for LMConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
            initial_lambda: 0.01,
            use_finite_diff: true,
            fd_epsilon: 1e-6,
        }
    }
}

/// Levenberg-Marquardt optimizer
pub struct LMOptimizer {
    pub config: LMConfig,
}

impl Default for LMOptimizer {
    fn default() -> Self {
        Self {
            config: LMConfig::default(),
        }
    }
}

impl LMOptimizer {
    pub fn new(config: LMConfig) -> Self {
        Self { config }
    }

    /// Optimize Gaussians using Levenberg-Marquardt
    ///
    /// Returns final MSE loss
    pub fn optimize(
        &mut self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> f32 {
        let n = gaussians.len();
        println!("=== Levenberg-Marquardt Optimizer ===");
        println!("  Gaussians: {}", n);
        println!("  Max iterations: {}", self.config.max_iterations);
        println!("  Tolerance: {:e}", self.config.tolerance);
        println!("  Initial λ: {}", self.config.initial_lambda);

        // Flatten Gaussians to parameter vector
        // Parameters per Gaussian: x, y, r, g, b, scale_x, scale_y (7 params)
        let params_per_gaussian = 7;
        let mut params: Vec<f64> = Vec::with_capacity(n * params_per_gaussian);

        for g in gaussians.iter() {
            params.push(g.position.x as f64);
            params.push(g.position.y as f64);
            params.push(g.color.r as f64);
            params.push(g.color.g as f64);
            params.push(g.color.b as f64);
            params.push(g.shape.scale_x as f64);
            params.push(g.shape.scale_y as f64);
        }

        // Create problem instance
        let problem = GaussianLSProblem {
            target: target.clone(),
            n_gaussians: n,
            fd_epsilon: self.config.fd_epsilon,
            params: DVector::from_vec(params),
        };

        // Compute initial loss
        let init_residuals = problem.residuals();
        let init_loss = if let Some(r) = init_residuals {
            r.iter().map(|x| x * x).sum::<f64>() / r.len() as f64
        } else {
            f64::INFINITY
        };
        println!("  Initial loss: {:.6}", init_loss);

        // Run L-M optimization
        let (optimized_problem, report) = LevenbergMarquardt::new()
            .with_patience(self.config.max_iterations)
            .with_tol(self.config.tolerance)
            .minimize(problem);

        // Extract optimized parameters from the problem
        let optimized_params = optimized_problem.params();

        // Compute final loss
        let final_residuals = optimized_problem.residuals();
        let final_loss = if let Some(r) = final_residuals {
            r.iter().map(|x| x * x).sum::<f64>() / r.len() as f64
        } else {
            f64::INFINITY
        };

        // Write optimized parameters back to Gaussians (with clamping)
        for i in 0..n {
            let base = i * 7;
            gaussians[i].position.x = (optimized_params[base] as f32).clamp(0.0, 1.0);
            gaussians[i].position.y = (optimized_params[base + 1] as f32).clamp(0.0, 1.0);
            gaussians[i].color.r = (optimized_params[base + 2] as f32).clamp(0.0, 1.0);
            gaussians[i].color.g = (optimized_params[base + 3] as f32).clamp(0.0, 1.0);
            gaussians[i].color.b = (optimized_params[base + 4] as f32).clamp(0.0, 1.0);
            gaussians[i].shape.scale_x = (optimized_params[base + 5] as f32).clamp(0.001, 0.5);
            gaussians[i].shape.scale_y = (optimized_params[base + 6] as f32).clamp(0.001, 0.5);
        }

        println!("  Final loss: {:.6}", final_loss);
        println!("  Termination: {:?}", report.termination);
        println!("  Evaluations: {}", report.number_of_evaluations);
        println!("  Improvement: {:.2}%", (init_loss - final_loss) / init_loss * 100.0);

        final_loss as f32
    }
}

/// Gaussian splatting least-squares problem
///
/// Implements the `LeastSquaresProblem` trait for the L-M algorithm.
/// Residuals are per-pixel RGB differences between rendered and target.
struct GaussianLSProblem {
    target: ImageBuffer<f32>,
    n_gaussians: usize,
    fd_epsilon: f64,
    params: DVector<f64>,
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for GaussianLSProblem {
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    type ParameterStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, params: &DVector<f64>) {
        self.params = params.clone();
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        // Convert params to Gaussians
        let gaussians = self.params_to_gaussians();

        // Render
        let rendered = RendererV2::render(&gaussians, self.target.width, self.target.height);

        // Compute residuals (rendered - target for each RGB channel of each pixel)
        let n_pixels = (self.target.width * self.target.height) as usize;
        let n_residuals = n_pixels * 3; // RGB

        let mut residuals = DVector::zeros(n_residuals);

        for (i, (r, t)) in rendered.data.iter().zip(self.target.data.iter()).enumerate() {
            residuals[i * 3] = (r.r - t.r) as f64;
            residuals[i * 3 + 1] = (r.g - t.g) as f64;
            residuals[i * 3 + 2] = (r.b - t.b) as f64;
        }

        Some(residuals)
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        // Jacobian: d(residuals)/d(params)
        // Shape: (n_residuals, n_params)
        // This is expensive! Each column requires a render pass.

        let n_params = self.params.len();
        let n_pixels = (self.target.width * self.target.height) as usize;
        let n_residuals = n_pixels * 3;

        // For efficiency, we could use analytical gradients in the future
        // For now, use finite differences (correct but slow)

        let mut jacobian = DMatrix::zeros(n_residuals, n_params);

        // Get baseline residuals
        let r0 = self.residuals()?;

        // Finite difference for each parameter
        for j in 0..n_params {
            let mut params_plus = self.params.clone();
            params_plus[j] += self.fd_epsilon;

            // Create problem with perturbed params
            let problem_plus = GaussianLSProblem {
                target: self.target.clone(),
                n_gaussians: self.n_gaussians,
                fd_epsilon: self.fd_epsilon,
                params: params_plus,
            };

            let r_plus = problem_plus.residuals()?;

            // d(residuals)/d(param_j) ≈ (r_plus - r0) / epsilon
            for i in 0..n_residuals {
                jacobian[(i, j)] = (r_plus[i] - r0[i]) / self.fd_epsilon;
            }

            // Progress indicator every 100 params
            if j % 100 == 99 {
                print!(".");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
        }

        Some(jacobian)
    }
}

impl GaussianLSProblem {
    fn params_to_gaussians(&self) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        let mut gaussians = Vec::with_capacity(self.n_gaussians);

        for i in 0..self.n_gaussians {
            let base = i * 7;
            gaussians.push(Gaussian2D::new(
                Vector2::new(
                    (self.params[base] as f32).clamp(0.0, 1.0),
                    (self.params[base + 1] as f32).clamp(0.0, 1.0),
                ),
                Euler::new(
                    (self.params[base + 5] as f32).clamp(0.001, 0.5),
                    (self.params[base + 6] as f32).clamp(0.001, 0.5),
                    0.0, // Rotation not optimized in this version
                ),
                Color4::new(
                    (self.params[base + 2] as f32).clamp(0.0, 1.0),
                    (self.params[base + 3] as f32).clamp(0.0, 1.0),
                    (self.params[base + 4] as f32).clamp(0.0, 1.0),
                    1.0,
                ),
                1.0, // Opacity not optimized
            ));
        }

        gaussians
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lm_config_default() {
        let config = LMConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert!((config.tolerance - 1e-8).abs() < 1e-10);
    }

    #[test]
    fn test_lm_optimizer_creation() {
        let optimizer = LMOptimizer::default();
        assert_eq!(optimizer.config.max_iterations, 100);
    }
}
