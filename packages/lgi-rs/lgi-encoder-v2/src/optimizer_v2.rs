//! Optimizer v2 - Gradient-based parameter refinement
//!
//! Simple gradient descent to validate the encoding quality
//! (Will be replaced with L-BFGS + R-D optimization later)

use lgi_core::{ImageBuffer, ms_ssim_loss::MsssimLoss, ms_ssim_gradients::MsssimGradients, edge_weighted_loss::EdgeWeightedLoss, structure_tensor::StructureTensorField};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use crate::renderer_v2::RendererV2;
use crate::renderer_gpu::GpuRendererV2;

/// Gradient for a single Gaussian
#[derive(Clone)]
struct GaussianGradient {
    position: Vector2<f32>,
    color: Color4<f32>,
    scale_x: f32,
    scale_y: f32,
    rotation: f32,  // NEW: Rotation gradient
}

impl GaussianGradient {
    fn zero() -> Self {
        Self {
            position: Vector2::zero(),
            color: Color4::new(0.0, 0.0, 0.0, 0.0),
            scale_x: 0.0,
            scale_y: 0.0,
            rotation: 0.0,
        }
    }
}

/// Simple optimizer for validation
pub struct OptimizerV2 {
    pub learning_rate_position: f32,
    pub learning_rate_scale: f32,
    pub learning_rate_color: f32,
    pub learning_rate_rotation: f32,  // NEW
    pub max_iterations: usize,
    pub gpu_renderer: Option<GpuRendererV2>,  // GPU acceleration (454× faster!)
    pub use_ms_ssim: bool,  // Use MS-SSIM loss instead of L2
    pub use_edge_weighted: bool,  // NEW: Use edge-weighted gradients
    ms_ssim_loss: MsssimLoss,
    ms_ssim_grads: MsssimGradients,  // NEW: Analytical MS-SSIM gradients
    edge_weighted_loss: EdgeWeightedLoss,  // NEW: Edge-weighted loss
    tensor_field: Option<StructureTensorField>,  // NEW: Cached structure tensor
}

impl Default for OptimizerV2 {
    fn default() -> Self {
        Self {
            learning_rate_position: 0.10,  // 2× higher (Session 3 calibration: +0.58 dB)
            learning_rate_scale: 0.10,     // 2× higher
            learning_rate_color: 0.6,      // 2× higher
            learning_rate_rotation: 0.02,  // 2× higher
            max_iterations: 100,
            gpu_renderer: None,  // GPU created on-demand
            use_ms_ssim: false,  // Default to L2 (MS-SSIM optional)
            use_edge_weighted: false,  // NEW: Default disabled
            ms_ssim_loss: MsssimLoss::default(),
            ms_ssim_grads: MsssimGradients::default(),  // NEW
            edge_weighted_loss: EdgeWeightedLoss::default(),  // NEW
            tensor_field: None,  // NEW: Computed on-demand
        }
    }
}

impl OptimizerV2 {
    /// Create optimizer with GPU acceleration and MS-SSIM loss
    pub fn new_with_gpu() -> Self {
        log::info!("🚀 Initializing GPU-accelerated optimizer...");
        let gpu_renderer = Some(GpuRendererV2::new_blocking());

        let has_gpu = gpu_renderer.as_ref().map(|g| g.has_gpu()).unwrap_or(false);
        if has_gpu {
            log::info!("✅ GPU renderer initialized successfully");
        } else {
            log::warn!("⚠️  GPU not available, using CPU fallback");
        }

        Self {
            gpu_renderer,
            ..Default::default()
        }
    }

    /// Create optimizer with MS-SSIM perceptual loss
    pub fn new_with_ms_ssim() -> Self {
        Self {
            use_ms_ssim: true,
            ..Default::default()
        }
    }

    /// Create optimizer with GPU and MS-SSIM
    pub fn new_with_gpu_and_ms_ssim() -> Self {
        let mut opt = Self::new_with_gpu();
        opt.use_ms_ssim = true;
        opt
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        self.gpu_renderer.as_ref().map(|g| g.has_gpu()).unwrap_or(false)
    }

    /// Optimize Gaussian parameters to match target
    pub fn optimize(
        &mut self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> f32 {
        let width = target.width;
        let height = target.height;

        // Compute structure tensor if edge-weighted loss enabled
        if self.use_edge_weighted && self.tensor_field.is_none() {
            println!("  Computing structure tensor for edge-weighted loss...");
            if let Ok(tensor) = StructureTensorField::compute(target, 1.2, 1.0) {
                self.tensor_field = Some(tensor);
            } else {
                println!("  Warning: Failed to compute structure tensor, disabling edge-weighted loss");
                self.use_edge_weighted = false;
            }
        }

        // ISSUE-002 FIX: Adaptive learning rate based on Gaussian density
        // At high N, Gaussians overlap more → need lower LR to prevent divergence
        let n = gaussians.len() as f32;
        let density_factor = (100.0 / n).sqrt().min(1.0);  // Scale down for N>100
        let lr_color_adaptive = self.learning_rate_color * density_factor;
        let lr_position_adaptive = self.learning_rate_position * density_factor;
        let lr_rotation_adaptive = self.learning_rate_rotation * density_factor;

        let mut best_loss = f32::INFINITY;
        let mut best_gaussians = gaussians.to_vec();
        let mut patience_counter = 0;
        let patience_limit = 20;  // Stop if no improvement for 20 iterations

        for iter in 0..self.max_iterations {
            // Forward: render current Gaussians (GPU if available, CPU fallback)
            let rendered = if let Some(ref mut gpu) = self.gpu_renderer {
                // GPU path: 454× faster than CPU!
                gpu.render(gaussians, width, height)
            } else {
                // CPU fallback
                RendererV2::render(gaussians, width, height)
            };

            // Compute loss (MS-SSIM or L2)
            let loss = if self.use_ms_ssim {
                self.ms_ssim_loss.compute_loss(&rendered, target)
            } else {
                self.compute_loss(&rendered, target)
            };

            if iter % 10 == 0 {
                println!("  Iteration {}: loss = {:.6}", iter, loss);
            }

            // Track best loss and implement early stopping (ISSUE-002 fix)
            if loss < best_loss {
                best_loss = loss;
                best_gaussians.clone_from_slice(gaussians);
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= patience_limit {
                    // Restore best and stop
                    gaussians.copy_from_slice(&best_gaussians);
                    if iter % 10 != 0 {
                        println!("  Early stop at iteration {} (no improvement for {} iters)", iter, patience_limit);
                    }
                    break;
                }
            }

            // Backward: compute analytical gradients
            let grads = self.compute_gradients(gaussians, &rendered, target);

            // Update parameters with adaptive LR (ISSUE-002 fix)
            let lr_scale_adaptive = self.learning_rate_scale * density_factor;

            for (gaussian, grad) in gaussians.iter_mut().zip(grads.iter()) {
                // Update color
                gaussian.color.r -= lr_color_adaptive * grad.color.r;
                gaussian.color.g -= lr_color_adaptive * grad.color.g;
                gaussian.color.b -= lr_color_adaptive * grad.color.b;
                gaussian.color.r = gaussian.color.r.clamp(0.0, 1.0);
                gaussian.color.g = gaussian.color.g.clamp(0.0, 1.0);
                gaussian.color.b = gaussian.color.b.clamp(0.0, 1.0);

                // Update position
                gaussian.position.x -= lr_position_adaptive * grad.position.x;
                gaussian.position.y -= lr_position_adaptive * grad.position.y;
                gaussian.position.x = gaussian.position.x.clamp(0.0, 1.0);
                gaussian.position.y = gaussian.position.y.clamp(0.0, 1.0);

                // Update scales (scale optimization)
                gaussian.shape.scale_x -= lr_scale_adaptive * grad.scale_x;
                gaussian.shape.scale_y -= lr_scale_adaptive * grad.scale_y;
                gaussian.shape.scale_x = gaussian.shape.scale_x.clamp(0.01, 0.25);
                gaussian.shape.scale_y = gaussian.shape.scale_y.clamp(0.01, 0.25);

                // Update rotation (NEW: rotation optimization)
                gaussian.shape.rotation -= lr_rotation_adaptive * grad.rotation;
                // Normalize to [-π, π]
                use std::f32::consts::PI;
                while gaussian.shape.rotation > PI {
                    gaussian.shape.rotation -= 2.0 * PI;
                }
                while gaussian.shape.rotation < -PI {
                    gaussian.shape.rotation += 2.0 * PI;
                }
            }

            // Early stopping for perfect convergence
            if loss < 1e-4 {
                println!("  Converged at iteration {}", iter);
                break;
            }
        }

        best_loss
    }

    /// Compute gradients for all Gaussians (analytical)
    fn compute_gradients(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> Vec<GaussianGradient> {
        let width = target.width;
        let height = target.height;

        let mut gradients = vec![GaussianGradient::zero(); gaussians.len()];

        // Compute pixel-level gradients (MS-SSIM or L2)
        let mut pixel_grads = if self.use_ms_ssim {
            // MS-SSIM analytical gradients: ∂(1-MS-SSIM)/∂(rendered_pixel)
            self.ms_ssim_grads.compute_pixel_gradients(target, rendered)
        } else {
            // L2 gradients: ∂(MSE)/∂(rendered_pixel) = 2(rendered - target)
            let mut l2_grads = ImageBuffer::new(width, height);
            for (i, (r, t)) in rendered.data.iter().zip(target.data.iter()).enumerate() {
                l2_grads.data[i] = Color4::new(
                    2.0 * (r.r - t.r),
                    2.0 * (r.g - t.g),
                    2.0 * (r.b - t.b),
                    0.0,
                );
            }
            l2_grads
        };

        // Apply edge weighting if enabled
        if self.use_edge_weighted {
            if let Some(ref tensor_field) = self.tensor_field {
                let edge_weights = self.edge_weighted_loss.compute_weight_map(width, height, tensor_field);

                for (i, pixel_grad) in pixel_grads.data.iter_mut().enumerate() {
                    let weight = edge_weights[i];
                    pixel_grad.r *= weight;
                    pixel_grad.g *= weight;
                    pixel_grad.b *= weight;
                }
            }
        }

        // For each pixel, compute contribution to each Gaussian's gradient
        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / width as f32;
                let py = y as f32 / height as f32;

                // Get pixel-level gradient
                let pixel_grad = pixel_grads.get_pixel(x, y).unwrap();
                let error_r = pixel_grad.r;
                let error_g = pixel_grad.g;
                let error_b = pixel_grad.b;

                // Accumulate gradient for each Gaussian
                for (i, gaussian) in gaussians.iter().enumerate() {
                    let dx = px - gaussian.position.x;
                    let dy = py - gaussian.position.y;

                    // Rotate to Gaussian's frame
                    let cos_t = gaussian.shape.rotation.cos();
                    let sin_t = gaussian.shape.rotation.sin();
                    let dx_rot = dx * cos_t + dy * sin_t;
                    let dy_rot = -dx * sin_t + dy * cos_t;

                    // Mahalanobis distance
                    let sx = gaussian.shape.scale_x;
                    let sy = gaussian.shape.scale_y;
                    let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);

                    if dist_sq > 12.25 {
                        continue;  // Outside 3.5σ
                    }

                    // Gaussian value and weight
                    let gaussian_val = (-0.5 * dist_sq).exp();
                    let weight = gaussian.opacity * gaussian_val;

                    // Color gradient (simple)
                    gradients[i].color.r += error_r * weight;
                    gradients[i].color.g += error_g * weight;
                    gradients[i].color.b += error_b * weight;

                    // Position gradient: ∂weight/∂μ × error × color
                    // ∂weight/∂μ = weight × Σ^-1 × (x - μ)
                    let grad_weight_x = weight * (dx_rot * cos_t / (sx * sx) + dy_rot * (-sin_t) / (sy * sy));
                    let grad_weight_y = weight * (dx_rot * sin_t / (sx * sx) + dy_rot * cos_t / (sy * sy));

                    let error_weighted = error_r * gaussian.color.r +
                                        error_g * gaussian.color.g +
                                        error_b * gaussian.color.b;

                    gradients[i].position.x += error_weighted * grad_weight_x;
                    gradients[i].position.y += error_weighted * grad_weight_y;

                    // Scale gradients: ∂weight/∂σ × error × color
                    let grad_weight_sx = weight * (dx_rot / sx).powi(2) * (1.0 / sx);
                    let grad_weight_sy = weight * (dy_rot / sy).powi(2) * (1.0 / sy);

                    gradients[i].scale_x += error_weighted * grad_weight_sx;
                    gradients[i].scale_y += error_weighted * grad_weight_sy;

                    // Rotation gradient (NEW): ∂weight/∂θ × error × color
                    // ∂(x_rot)/∂θ = [-sin(θ), cos(θ)] · [dx, dy]
                    // ∂d²/∂θ involves rotating the coordinate system
                    let d_dx_rot_dtheta = -dx * sin_t + dy * cos_t;
                    let d_dy_rot_dtheta = -dx * cos_t - dy * sin_t;
                    let d_dist_sq_dtheta = 2.0 * (
                        (dx_rot / (sx * sx)) * d_dx_rot_dtheta +
                        (dy_rot / (sy * sy)) * d_dy_rot_dtheta
                    );
                    let grad_weight_theta = -0.5 * weight * d_dist_sq_dtheta;

                    gradients[i].rotation += error_weighted * grad_weight_theta;
                }
            }
        }

        // Normalize by pixel count
        let pixel_count = (width * height) as f32;
        for grad in &mut gradients {
            grad.color.r /= pixel_count;
            grad.color.g /= pixel_count;
            grad.color.b /= pixel_count;
            grad.position.x /= pixel_count;
            grad.position.y /= pixel_count;
            grad.scale_x /= pixel_count;
            grad.scale_y /= pixel_count;
            grad.rotation /= pixel_count;
        }

        gradients
    }

    fn compute_loss(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
        let mut loss = 0.0;
        let count = (rendered.width * rendered.height) as f32;

        for (r, t) in rendered.data.iter().zip(target.data.iter()) {
            loss += (r.r - t.r).powi(2);
            loss += (r.g - t.g).powi(2);
            loss += (r.b - t.b).powi(2);
        }

        loss / count
    }

}
