//! Optimizer v3 - Perceptual loss (MS-SSIM + Edge-weighted)

use lgi_core::{ImageBuffer, ms_ssim_loss::MsssimLoss, edge_weighted_loss::EdgeWeightedLoss, StructureTensorField};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use crate::renderer_v2::RendererV2;

pub struct OptimizerV3 {
    pub learning_rate_position: f32,
    pub learning_rate_scale: f32,
    pub learning_rate_color: f32,
    pub learning_rate_rotation: f32,
    pub max_iterations: usize,
    pub use_ms_ssim: bool,
    pub use_edge_weighted: bool,
    ms_ssim_loss: MsssimLoss,
    edge_weighted_loss: EdgeWeightedLoss,
}

impl Default for OptimizerV3 {
    fn default() -> Self {
        Self {
            learning_rate_position: 0.05,
            learning_rate_scale: 0.05,
            learning_rate_color: 0.3,
            learning_rate_rotation: 0.01,
            max_iterations: 100,
            use_ms_ssim: true,
            use_edge_weighted: true,
            ms_ssim_loss: MsssimLoss::default(),
            edge_weighted_loss: EdgeWeightedLoss::default(),
        }
    }
}

impl OptimizerV3 {
    pub fn optimize(
        &mut self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
        tensor_field: &StructureTensorField,
    ) -> f32 {
        let width = target.width;
        let height = target.height;
        let n = gaussians.len() as f32;
        let density_factor = (100.0 / n).sqrt().min(1.0);

        let mut best_loss = f32::INFINITY;
        let mut best_gaussians = gaussians.to_vec();
        let mut patience_counter = 0;

        for iter in 0..self.max_iterations {
            let rendered = RendererV2::render(gaussians, width, height);

            let loss = if self.use_ms_ssim {
                self.ms_ssim_loss.compute_loss(&rendered, target)
            } else if self.use_edge_weighted {
                self.edge_weighted_loss.compute_loss(&rendered, target, tensor_field)
            } else {
                self.compute_l2_loss(&rendered, target)
            };

            if iter % 10 == 0 {
                println!("  Iteration {}: loss = {:.6}", iter, loss);
            }

            if loss < best_loss {
                best_loss = loss;
                best_gaussians.clone_from_slice(gaussians);
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= 20 {
                    gaussians.copy_from_slice(&best_gaussians);
                    break;
                }
            }

            let grads = self.compute_gradients(gaussians, &rendered, target);
            let lr_color_adaptive = self.learning_rate_color * density_factor;
            let lr_position_adaptive = self.learning_rate_position * density_factor;
            let lr_scale_adaptive = self.learning_rate_scale * density_factor;

            for (gaussian, grad) in gaussians.iter_mut().zip(grads.iter()) {
                gaussian.color.r -= lr_color_adaptive * grad.color.r;
                gaussian.color.g -= lr_color_adaptive * grad.color.g;
                gaussian.color.b -= lr_color_adaptive * grad.color.b;
                gaussian.color.r = gaussian.color.r.clamp(0.0, 1.0);
                gaussian.color.g = gaussian.color.g.clamp(0.0, 1.0);
                gaussian.color.b = gaussian.color.b.clamp(0.0, 1.0);

                gaussian.position.x -= lr_position_adaptive * grad.position.x;
                gaussian.position.y -= lr_position_adaptive * grad.position.y;
                gaussian.position.x = gaussian.position.x.clamp(0.0, 1.0);
                gaussian.position.y = gaussian.position.y.clamp(0.0, 1.0);

                gaussian.shape.scale_x -= lr_scale_adaptive * grad.scale_x;
                gaussian.shape.scale_y -= lr_scale_adaptive * grad.scale_y;
                gaussian.shape.scale_x = gaussian.shape.scale_x.clamp(0.01, 0.25);
                gaussian.shape.scale_y = gaussian.shape.scale_y.clamp(0.01, 0.25);
            }

            if loss < 1e-4 { break; }
        }

        best_loss
    }

    fn compute_gradients(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>], rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> Vec<GaussianGradient> {
        let width = target.width;
        let height = target.height;
        let mut gradients = vec![GaussianGradient::zero(); gaussians.len()];

        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / width as f32;
                let py = y as f32 / height as f32;
                let rendered_color = rendered.get_pixel(x, y).unwrap();
                let target_color = target.get_pixel(x, y).unwrap();

                let error_r = 2.0 * (rendered_color.r - target_color.r);
                let error_g = 2.0 * (rendered_color.g - target_color.g);
                let error_b = 2.0 * (rendered_color.b - target_color.b);

                for (i, gaussian) in gaussians.iter().enumerate() {
                    let dx = px - gaussian.position.x;
                    let dy = py - gaussian.position.y;
                    let sx = gaussian.shape.scale_x;
                    let sy = gaussian.shape.scale_y;
                    let theta = gaussian.shape.rotation;
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();
                    let dx_rot = dx * cos_t + dy * sin_t;
                    let dy_rot = -dx * sin_t + dy * cos_t;
                    let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);

                    if dist_sq > 12.25 { continue; }

                    let gaussian_val = (-0.5 * dist_sq).exp();
                    let weight = gaussian.opacity * gaussian_val;

                    gradients[i].color.r += error_r * weight;
                    gradients[i].color.g += error_g * weight;
                    gradients[i].color.b += error_b * weight;

                    let grad_weight_x = weight * (dx_rot * cos_t / (sx * sx) + dy_rot * (-sin_t) / (sy * sy));
                    let grad_weight_y = weight * (dx_rot * sin_t / (sx * sx) + dy_rot * cos_t / (sy * sy));
                    let error_weighted = error_r * gaussian.color.r + error_g * gaussian.color.g + error_b * gaussian.color.b;

                    gradients[i].position.x += error_weighted * grad_weight_x;
                    gradients[i].position.y += error_weighted * grad_weight_y;

                    let grad_weight_sx = weight * (dx_rot / sx).powi(2) * (1.0 / sx);
                    let grad_weight_sy = weight * (dy_rot / sy).powi(2) * (1.0 / sy);

                    gradients[i].scale_x += error_weighted * grad_weight_sx;
                    gradients[i].scale_y += error_weighted * grad_weight_sy;
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

    fn compute_l2_loss(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
        let mut loss = 0.0;
        for (r, t) in rendered.data.iter().zip(target.data.iter()) {
            loss += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
        }
        loss / (rendered.width * rendered.height * 3) as f32
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
            position: Vector2::zero(),
            color: Color4::new(0.0, 0.0, 0.0, 0.0),
            scale_x: 0.0,
            scale_y: 0.0,
        }
    }
}
