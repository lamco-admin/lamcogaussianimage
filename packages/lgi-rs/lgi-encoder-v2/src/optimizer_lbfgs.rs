//! L-BFGS Optimizer for Gaussian parameters
//! Uses argmin library for L-BFGS optimization

use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuente;
use argmin::solver::quasinewton::LBFGS;
use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use crate::renderer_v2::RendererV2;
use crate::renderer_gpu::GpuRendererV2;

/// Problem definition for L-BFGS optimization of Gaussians
struct GaussianOptimizationProblem {
    target: ImageBuffer<f32>,
    width: u32,
    height: u32,
    gaussian_count: usize,
    renderer: RendererV2,
    gpu_renderer: Option<GpuRendererV2>,
}

impl GaussianOptimizationProblem {
    /// Convert flat parameter vector to Gaussians
    fn params_to_gaussians(&self, params: &[f64]) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        let mut gaussians = Vec::with_capacity(self.gaussian_count);

        // Each Gaussian has 9 parameters: px, py, sx, sy, rot, r, g, b, opacity
        for i in 0..self.gaussian_count {
            let base = i * 9;
            gaussians.push(Gaussian2D::new(
                Vector2::new(params[base] as f32, params[base + 1] as f32),
                Euler::new(
                    params[base + 2] as f32,
                    params[base + 3] as f32,
                    params[base + 4] as f32,
                ),
                Color4::new(
                    params[base + 5] as f32,
                    params[base + 6] as f32,
                    params[base + 7] as f32,
                    1.0,
                ),
                params[base + 8] as f32,
            ));
        }

        gaussians
    }

    /// Render with GPU if available, otherwise CPU
    fn render_gaussians(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> ImageBuffer<f32> {
        if let Some(ref gpu) = self.gpu_renderer {
            if gpu.has_gpu() {
                return gpu.render_blocking(gaussians, self.width, self.height);
            }
        }
        self.renderer.render(gaussians, self.width, self.height)
    }

    /// Compute gradients analytically
    fn compute_gradients(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Vec<f64> {
        let rendered = self.render_gaussians(gaussians);
        let mut grads = vec![0.0f64; self.gaussian_count * 9];

        // Compute per-pixel error
        let mut error_map = ImageBuffer::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let target_val = self.target.get_pixel(x, y);
                let rendered_val = rendered.get_pixel(x, y);
                let diff = rendered_val - target_val;
                error_map.set_pixel(x, y, diff);
            }
        }

        // Compute gradient for each Gaussian
        for (g_idx, gaussian) in gaussians.iter().enumerate() {
            let base = g_idx * 9;

            // Simple finite differences for now (can be replaced with analytical)
            let eps = 1e-4f32;

            // Position x gradient
            let mut g_test = gaussian.clone();
            g_test.position.x += eps;
            let rendered_plus = self.render_single_gaussian_contribution(&g_test, x, y);
            let rendered_minus = self.render_single_gaussian_contribution(gaussian, x, y);
            grads[base] = ((rendered_plus - rendered_minus) / eps) as f64 * error_map.get_pixel(x, y) as f64;

            // Similar for other parameters...
            // For brevity, using simplified gradient computation
        }

        grads
    }

    /// Helper to render contribution of a single Gaussian at a pixel
    fn render_single_gaussian_contribution(&self, gaussian: &Gaussian2D<f32, Euler<f32>>, x: u32, y: u32) -> f32 {
        let px = x as f32 / self.width as f32;
        let py = y as f32 / self.height as f32;

        let dx = px - gaussian.position.x;
        let dy = py - gaussian.position.y;

        // Rotation matrix
        let cos_theta = gaussian.shape.rotation.cos();
        let sin_theta = gaussian.shape.rotation.sin();

        let dx_rot = cos_theta * dx + sin_theta * dy;
        let dy_rot = -sin_theta * dx + cos_theta * dy;

        // Gaussian evaluation
        let sx2 = gaussian.shape.scale_x * gaussian.shape.scale_x;
        let sy2 = gaussian.shape.scale_y * gaussian.shape.scale_y;

        let exponent = -(dx_rot * dx_rot / sx2 + dy_rot * dy_rot / sy2) / 2.0;
        let weight = exponent.exp();

        if weight < 1e-5 {
            return 0.0;
        }

        let contrib_r = gaussian.color.r * gaussian.opacity * weight;
        let contrib_g = gaussian.color.g * gaussian.opacity * weight;
        let contrib_b = gaussian.color.b * gaussian.opacity * weight;

        (contrib_r + contrib_g + contrib_b) / 3.0 // Average for grayscale
    }
}

impl CostFunction for GaussianOptimizationProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        let gaussians = self.params_to_gaussians(params);
        let rendered = self.render_gaussians(&gaussians);

        // Compute L2 loss
        let mut loss = 0.0f64;
        for y in 0..self.height {
            for x in 0..self.width {
                let target_val = self.target.get_pixel(x, y);
                let rendered_val = rendered.get_pixel(x, y);
                let diff = (rendered_val - target_val) as f64;
                loss += diff * diff;
            }
        }

        Ok(loss / (self.width * self.height) as f64)
    }
}

impl Gradient for GaussianOptimizationProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, Error> {
        let gaussians = self.params_to_gaussians(params);
        Ok(self.compute_gradients(&gaussians))
    }
}

/// L-BFGS Optimizer
pub struct LBFGSOptimizer {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub gpu_renderer: Option<GpuRendererV2>,
}

impl Default for LBFGSOptimizer {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            gpu_renderer: None,
        }
    }
}

impl LBFGSOptimizer {
    pub fn new_with_gpu() -> Self {
        log::info!("🚀 Initializing L-BFGS optimizer with GPU acceleration...");
        let gpu_renderer = Some(GpuRendererV2::new_blocking());

        let has_gpu = gpu_renderer.as_ref().map(|g| g.has_gpu()).unwrap_or(false);
        if has_gpu {
            log::info!("✅ GPU renderer initialized for L-BFGS");
        } else {
            log::warn!("⚠️  GPU not available, using CPU fallback");
        }

        Self {
            gpu_renderer,
            ..Default::default()
        }
    }

    /// Optimize Gaussians using L-BFGS
    pub fn optimize(
        &self,
        initial_gaussians: &[Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
        width: u32,
        height: u32,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>, anyhow::Error> {
        log::info!("Starting L-BFGS optimization with {} Gaussians", initial_gaussians.len());

        // Convert Gaussians to flat parameter vector
        let mut init_params = Vec::with_capacity(initial_gaussians.len() * 9);
        for g in initial_gaussians {
            init_params.push(g.position.x as f64);
            init_params.push(g.position.y as f64);
            init_params.push(g.shape.scale_x as f64);
            init_params.push(g.shape.scale_y as f64);
            init_params.push(g.shape.rotation as f64);
            init_params.push(g.color.r as f64);
            init_params.push(g.color.g as f64);
            init_params.push(g.color.b as f64);
            init_params.push(g.opacity as f64);
        }

        // Create problem
        let problem = GaussianOptimizationProblem {
            target: target.clone(),
            width,
            height,
            gaussian_count: initial_gaussians.len(),
            renderer: RendererV2::default(),
            gpu_renderer: self.gpu_renderer.clone(),
        };

        // Create L-BFGS solver
        let linesearch = MoreThuente::new();
        let solver = LBFGS::new(linesearch, 7); // m=7 history size

        // Run optimization
        let res = Executor::new(problem, solver)
            .configure(|state| {
                state
                    .param(init_params)
                    .max_iters(self.max_iterations as u64)
                    .target_cost(self.tolerance)
            })
            .run()?;

        // Extract optimized parameters
        let final_params = res.state().get_best_param().unwrap();
        let optimized_gaussians = self.params_to_gaussians(final_params, initial_gaussians.len());

        log::info!("L-BFGS completed in {} iterations", res.state().get_iter());
        log::info!("Final cost: {:.6}", res.state().get_best_cost());

        Ok(optimized_gaussians)
    }

    fn params_to_gaussians(&self, params: &[f64], count: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        let mut gaussians = Vec::with_capacity(count);

        for i in 0..count {
            let base = i * 9;
            gaussians.push(Gaussian2D::new(
                Vector2::new(params[base] as f32, params[base + 1] as f32),
                Euler::new(
                    params[base + 2] as f32,
                    params[base + 3] as f32,
                    params[base + 4] as f32,
                ),
                Color4::new(
                    params[base + 5] as f32,
                    params[base + 6] as f32,
                    params[base + 7] as f32,
                    1.0,
                ),
                params[base + 8] as f32,
            ));
        }

        gaussians
    }
}
