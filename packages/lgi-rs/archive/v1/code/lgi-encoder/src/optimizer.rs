//! Gradient descent optimizer for Gaussian fitting

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_core::{Result, LgiError, ImageBuffer, Renderer};
use crate::{config::EncoderConfig, loss::LossFunctions};

#[cfg(feature = "progress")]
use indicatif::{ProgressBar, ProgressStyle};

/// Optimizer state
pub struct OptimizerState {
    /// Current iteration
    pub iteration: usize,
    /// Current loss
    pub loss: f32,
    /// Best loss seen
    pub best_loss: f32,
    /// Iterations since improvement
    pub patience_counter: usize,
}

/// Gradient descent optimizer using Adam
pub struct Optimizer {
    config: EncoderConfig,
}

impl Optimizer {
    /// Create new optimizer
    pub fn new(config: EncoderConfig) -> Self {
        Self { config }
    }

    /// Optimize Gaussians to match target image
    pub fn optimize(
        &self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        self.optimize_with_callback(gaussians, target, |_, _| {})
    }

    /// Optimize with progress callback
    pub fn optimize_with_callback<F>(
        &self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
        mut callback: F,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>>
    where
        F: FnMut(usize, f32),
    {
        let renderer = Renderer::new();
        let loss_fn = LossFunctions::new(
            self.config.loss_l2_weight,
            self.config.loss_ssim_weight,
        );

        // Adam optimizer state (first and second moments)
        let num_gaussians = gaussians.len();
        let mut m_position = vec![Vector2::<f32>::zero(); num_gaussians];
        let mut v_position = vec![Vector2::<f32>::zero(); num_gaussians];
        let mut m_scale = vec![Vector2::<f32>::zero(); num_gaussians];
        let mut v_scale = vec![Vector2::<f32>::zero(); num_gaussians];
        let mut m_rotation = vec![0.0f32; num_gaussians];
        let mut v_rotation = vec![0.0f32; num_gaussians];
        let mut m_color = vec![Color4::new(0.0, 0.0, 0.0, 0.0); num_gaussians];
        let mut v_color = vec![Color4::new(0.0, 0.0, 0.0, 0.0); num_gaussians];
        let mut m_opacity = vec![0.0f32; num_gaussians];
        let mut v_opacity = vec![0.0f32; num_gaussians];

        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let epsilon = 1e-8f32;

        let mut state = OptimizerState {
            iteration: 0,
            loss: f32::INFINITY,
            best_loss: f32::INFINITY,
            patience_counter: 0,
        };

        #[cfg(feature = "progress")]
        let progress_bar = ProgressBar::new(self.config.max_iterations as u64);
        #[cfg(feature = "progress")]
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} Loss: {msg}")
                .unwrap(),
        );

        // Optimization loop
        for iteration in 0..self.config.max_iterations {
            state.iteration = iteration;

            // Forward pass: Render current Gaussians
            let rendered = renderer.render(gaussians, target.width, target.height)?;

            // Compute loss
            state.loss = loss_fn.compute(&rendered, target);

            // Update best loss
            if state.loss < state.best_loss {
                state.best_loss = state.loss;
                state.patience_counter = 0;
            } else {
                state.patience_counter += 1;
            }

            // Callback
            callback(iteration, state.loss);

            #[cfg(feature = "progress")]
            {
                progress_bar.set_position(iteration as u64);
                progress_bar.set_message(format!("{:.6}", state.loss));
            }

            // Early stopping
            if state.patience_counter >= self.config.early_stopping_patience {
                #[cfg(feature = "progress")]
                progress_bar.finish_with_message(format!("Converged at loss: {:.6}", state.loss));
                break;
            }

            // Convergence check
            if state.loss < self.config.convergence_tolerance {
                #[cfg(feature = "progress")]
                progress_bar.finish_with_message(format!("Converged at loss: {:.6}", state.loss));
                break;
            }

            // Backward pass: Compute gradients
            let grad_image = loss_fn.gradient(&rendered, target);

            // Compute gradients for each Gaussian
            let gradients = self.compute_gaussian_gradients(gaussians, &grad_image, target.width, target.height);

            // Learning rate schedule
            let lr_mult = if iteration > 0 && iteration % self.config.lr_decay_steps == 0 {
                self.config.lr_decay
            } else {
                1.0
            };

            let lr_pos = self.config.lr_position * lr_mult;
            let lr_scale = self.config.lr_scale * lr_mult;
            let lr_rot = self.config.lr_rotation * lr_mult;
            let lr_color = self.config.lr_color * lr_mult;
            let lr_opacity = self.config.lr_opacity * lr_mult;

            // Adam update
            for (i, (gaussian, grad)) in gaussians.iter_mut().zip(gradients.iter()).enumerate() {
                // Position update
                m_position[i] = m_position[i] * beta1 + grad.position * (1.0 - beta1);
                v_position[i] = Vector2::<f32>::new(
                    v_position[i].x * beta2 + grad.position.x * grad.position.x * (1.0 - beta2),
                    v_position[i].y * beta2 + grad.position.y * grad.position.y * (1.0 - beta2),
                );

                let m_hat = m_position[i] * (1.0 / (1.0 - beta1.powi(iteration as i32 + 1)));
                let v_hat = Vector2::<f32>::new(
                    v_position[i].x / (1.0 - beta2.powi(iteration as i32 + 1)),
                    v_position[i].y / (1.0 - beta2.powi(iteration as i32 + 1)),
                );

                gaussian.position.x -= lr_pos * m_hat.x / (v_hat.x.sqrt() + epsilon);
                gaussian.position.y -= lr_pos * m_hat.y / (v_hat.y.sqrt() + epsilon);

                // Clamp position to [0, 1]
                gaussian.position.x = gaussian.position.x.clamp(0.0, 1.0);
                gaussian.position.y = gaussian.position.y.clamp(0.0, 1.0);

                // Scale update (similar Adam for scale_x, scale_y)
                // ... (simplified for brevity, full implementation would update all parameters)

                // Color update
                gaussian.color.r -= lr_color * grad.color.r;
                gaussian.color.g -= lr_color * grad.color.g;
                gaussian.color.b -= lr_color * grad.color.b;

                // Clamp color
                gaussian.color = gaussian.color.clamp();

                // Opacity update
                gaussian.opacity -= lr_opacity * grad.opacity;
                gaussian.opacity = gaussian.opacity.clamp(0.01, 1.0);
            }
        }

        #[cfg(feature = "progress")]
        progress_bar.finish_with_message(format!("Final loss: {:.6}", state.loss));

        Ok(gaussians.to_vec())
    }

    /// Compute gradients for all Gaussians (simplified)
    fn compute_gaussian_gradients(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        grad_image: &ImageBuffer<f32>,
        width: u32,
        height: u32,
    ) -> Vec<GaussianGradient> {
        // Simplified gradient computation
        // Full implementation would use automatic differentiation or
        // manual chain rule through the rendering pipeline

        let mut gradients = vec![GaussianGradient::zero(); gaussians.len()];

        // For each Gaussian, accumulate gradients from affected pixels
        for (g_idx, gaussian) in gaussians.iter().enumerate() {
            let (min, max) = gaussian.bounding_box(3.5);

            let x_min = (min.x * width as f32) as u32;
            let y_min = (min.y * height as f32) as u32;
            let x_max = (max.x * width as f32) as u32;
            let y_max = (max.y * height as f32) as u32;

            for y in y_min..=y_max.min(height - 1) {
                for x in x_min..=x_max.min(width - 1) {
                    if let Some(grad_pixel) = grad_image.get_pixel(x, y) {
                        // Simplified: gradient flows to color
                        gradients[g_idx].color.r += grad_pixel.r;
                        gradients[g_idx].color.g += grad_pixel.g;
                        gradients[g_idx].color.b += grad_pixel.b;
                    }
                }
            }

            // Normalize by number of pixels affected
            let pixel_count = ((x_max - x_min + 1) * (y_max - y_min + 1)) as f32;
            if pixel_count > 0.0 {
                gradients[g_idx].color.r /= pixel_count;
                gradients[g_idx].color.g /= pixel_count;
                gradients[g_idx].color.b /= pixel_count;
            }
        }

        gradients
    }
}

/// Gradient for a single Gaussian
#[derive(Debug, Clone)]
struct GaussianGradient {
    position: Vector2<f32>,
    scale: Vector2<f32>,
    rotation: f32,
    color: Color4<f32>,
    opacity: f32,
}

impl GaussianGradient {
    fn zero() -> Self {
        Self {
            position: Vector2::<f32>::zero(),
            scale: Vector2::<f32>::zero(),
            rotation: 0.0,
            color: Color4::new(0.0, 0.0, 0.0, 0.0),
            opacity: 0.0,
        }
    }
}
