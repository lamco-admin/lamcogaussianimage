//! Adam optimizer for Gaussian parameters

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use crate::renderer_v2::RendererV2;
use crate::gaussian_logger::GaussianLogger;

/// Per-parameter learning rates following 3DGS best practices
#[derive(Clone, Debug)]
pub struct LearningRates {
    pub position: f32,       // Default: 0.0002 (100× smaller than color)
    pub color: f32,          // Default: 0.02
    pub scale: f32,          // Default: 0.005
    pub opacity: f32,        // Default: 0.05
    pub position_final: f32, // Final position LR after decay (default: 0.00002)
}

impl Default for LearningRates {
    fn default() -> Self {
        Self {
            position: 0.0002,
            color: 0.02,
            scale: 0.005,
            opacity: 0.05,
            position_final: 0.00002,  // 10× smaller after decay
        }
    }
}

pub struct AdamOptimizer {
    /// Legacy single learning rate (deprecated, use learning_rates instead)
    pub learning_rate: f32,
    /// Per-parameter learning rates
    pub learning_rates: LearningRates,
    /// Enable per-parameter LR (set to false to use legacy single LR)
    pub use_per_param_lr: bool,
    pub beta1: f32,  // Momentum decay
    pub beta2: f32,  // RMSprop decay
    pub epsilon: f32,
    pub max_iterations: usize,
    m_color: Vec<Color4<f32>>,
    v_color: Vec<Color4<f32>>,
    m_position: Vec<Vector2<f32>>,
    v_position: Vec<Vector2<f32>>,
    m_scale: Vec<(f32, f32)>,
    v_scale: Vec<(f32, f32)>,
    m_opacity: Vec<f32>,
    v_opacity: Vec<f32>,
}

impl Default for AdamOptimizer {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,  // Legacy fallback
            learning_rates: LearningRates::default(),
            use_per_param_lr: true,  // Enable by default
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            max_iterations: 100,
            m_color: Vec::new(),
            v_color: Vec::new(),
            m_position: Vec::new(),
            v_position: Vec::new(),
            m_scale: Vec::new(),
            v_scale: Vec::new(),
            m_opacity: Vec::new(),
            v_opacity: Vec::new(),
        }
    }
}

impl AdamOptimizer {
    /// Create optimizer with default per-parameter learning rates
    pub fn new() -> Self {
        Self::default()
    }

    /// Create optimizer with legacy single learning rate
    pub fn with_single_lr(learning_rate: f32) -> Self {
        let mut opt = Self::default();
        opt.learning_rate = learning_rate;
        opt.use_per_param_lr = false;
        opt
    }

    /// Calculate exponentially decayed position learning rate
    /// Decay from initial to final over max_iterations
    fn position_lr_at_iteration(&self, t: usize) -> f32 {
        if !self.use_per_param_lr {
            return self.learning_rate;
        }

        let progress = t as f32 / self.max_iterations as f32;
        let log_initial = self.learning_rates.position.ln();
        let log_final = self.learning_rates.position_final.ln();
        (log_initial + progress * (log_final - log_initial)).exp()
    }

    pub fn optimize(&mut self, gaussians: &mut [Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) -> f32 {
        let n = gaussians.len();
        self.m_color.resize(n, Color4::new(0.0, 0.0, 0.0, 0.0));
        self.v_color.resize(n, Color4::new(0.0, 0.0, 0.0, 0.0));
        self.m_position.resize(n, Vector2::zero());
        self.v_position.resize(n, Vector2::zero());
        self.m_scale.resize(n, (0.0, 0.0));
        self.v_scale.resize(n, (0.0, 0.0));
        self.m_opacity.resize(n, 0.0);
        self.v_opacity.resize(n, 0.0);

        let mut best_loss = f32::INFINITY;
        let mut best_gaussians = gaussians.to_vec();  // Track best state
        let mut best_iteration = 0;

        for t in 1..=self.max_iterations {
            let rendered = RendererV2::render(gaussians, target.width, target.height);
            let loss = self.compute_loss(&rendered, target);

            if t % 10 == 0 {
                let pos_lr = self.position_lr_at_iteration(t);
                if self.use_per_param_lr {
                    println!("  Iteration {}/{}: loss = {:.6} (pos_lr = {:.6})", t, self.max_iterations, loss, pos_lr);
                } else {
                    println!("  Iteration {}/{}: loss = {:.6}", t, self.max_iterations, loss);
                }
            }

            if loss < best_loss {
                best_loss = loss;
                best_gaussians.clone_from_slice(gaussians);  // Save best state
                best_iteration = t;
            }

            let grads = self.compute_gradients(gaussians, &rendered, target);

            let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(t as i32);

            // Get learning rates for this iteration
            let lr_color = if self.use_per_param_lr { self.learning_rates.color } else { self.learning_rate };
            let lr_position = self.position_lr_at_iteration(t);
            let lr_scale = if self.use_per_param_lr { self.learning_rates.scale } else { self.learning_rate };
            let lr_opacity = if self.use_per_param_lr { self.learning_rates.opacity } else { self.learning_rate };

            for i in 0..n {
                // Color Adam update
                self.m_color[i].r = self.beta1 * self.m_color[i].r + (1.0 - self.beta1) * grads[i].color.r;
                self.m_color[i].g = self.beta1 * self.m_color[i].g + (1.0 - self.beta1) * grads[i].color.g;
                self.m_color[i].b = self.beta1 * self.m_color[i].b + (1.0 - self.beta1) * grads[i].color.b;

                self.v_color[i].r = self.beta2 * self.v_color[i].r + (1.0 - self.beta2) * grads[i].color.r.powi(2);
                self.v_color[i].g = self.beta2 * self.v_color[i].g + (1.0 - self.beta2) * grads[i].color.g.powi(2);
                self.v_color[i].b = self.beta2 * self.v_color[i].b + (1.0 - self.beta2) * grads[i].color.b.powi(2);

                let m_hat_r = self.m_color[i].r / bias_correction1;
                let m_hat_g = self.m_color[i].g / bias_correction1;
                let m_hat_b = self.m_color[i].b / bias_correction1;

                let v_hat_r = self.v_color[i].r / bias_correction2;
                let v_hat_g = self.v_color[i].g / bias_correction2;
                let v_hat_b = self.v_color[i].b / bias_correction2;

                gaussians[i].color.r -= lr_color * m_hat_r / (v_hat_r.sqrt() + self.epsilon);
                gaussians[i].color.g -= lr_color * m_hat_g / (v_hat_g.sqrt() + self.epsilon);
                gaussians[i].color.b -= lr_color * m_hat_b / (v_hat_b.sqrt() + self.epsilon);

                gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
                gaussians[i].color.g = gaussians[i].color.g.clamp(0.0, 1.0);
                gaussians[i].color.b = gaussians[i].color.b.clamp(0.0, 1.0);

                // Position Adam update (with exponential decay)
                self.m_position[i].x = self.beta1 * self.m_position[i].x + (1.0 - self.beta1) * grads[i].position.x;
                self.m_position[i].y = self.beta1 * self.m_position[i].y + (1.0 - self.beta1) * grads[i].position.y;

                self.v_position[i].x = self.beta2 * self.v_position[i].x + (1.0 - self.beta2) * grads[i].position.x.powi(2);
                self.v_position[i].y = self.beta2 * self.v_position[i].y + (1.0 - self.beta2) * grads[i].position.y.powi(2);

                let m_hat_px = self.m_position[i].x / bias_correction1;
                let m_hat_py = self.m_position[i].y / bias_correction1;
                let v_hat_px = self.v_position[i].x / bias_correction2;
                let v_hat_py = self.v_position[i].y / bias_correction2;

                gaussians[i].position.x -= lr_position * m_hat_px / (v_hat_px.sqrt() + self.epsilon);
                gaussians[i].position.y -= lr_position * m_hat_py / (v_hat_py.sqrt() + self.epsilon);

                gaussians[i].position.x = gaussians[i].position.x.clamp(0.0, 1.0);
                gaussians[i].position.y = gaussians[i].position.y.clamp(0.0, 1.0);

                // Scale Adam update
                self.m_scale[i].0 = self.beta1 * self.m_scale[i].0 + (1.0 - self.beta1) * grads[i].scale_x;
                self.m_scale[i].1 = self.beta1 * self.m_scale[i].1 + (1.0 - self.beta1) * grads[i].scale_y;

                self.v_scale[i].0 = self.beta2 * self.v_scale[i].0 + (1.0 - self.beta2) * grads[i].scale_x.powi(2);
                self.v_scale[i].1 = self.beta2 * self.v_scale[i].1 + (1.0 - self.beta2) * grads[i].scale_y.powi(2);

                let m_hat_sx = self.m_scale[i].0 / bias_correction1;
                let m_hat_sy = self.m_scale[i].1 / bias_correction1;
                let v_hat_sx = self.v_scale[i].0 / bias_correction2;
                let v_hat_sy = self.v_scale[i].1 / bias_correction2;

                gaussians[i].shape.scale_x -= lr_scale * m_hat_sx / (v_hat_sx.sqrt() + self.epsilon);
                gaussians[i].shape.scale_y -= lr_scale * m_hat_sy / (v_hat_sy.sqrt() + self.epsilon);

                gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
                gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);

                // Opacity Adam update (if gradient is computed)
                if grads[i].opacity != 0.0 {
                    self.m_opacity[i] = self.beta1 * self.m_opacity[i] + (1.0 - self.beta1) * grads[i].opacity;
                    self.v_opacity[i] = self.beta2 * self.v_opacity[i] + (1.0 - self.beta2) * grads[i].opacity.powi(2);

                    let m_hat_o = self.m_opacity[i] / bias_correction1;
                    let v_hat_o = self.v_opacity[i] / bias_correction2;

                    gaussians[i].color.a -= lr_opacity * m_hat_o / (v_hat_o.sqrt() + self.epsilon);
                    gaussians[i].color.a = gaussians[i].color.a.clamp(0.01, 1.0);
                }
            }

            if loss < 1e-4 { break; }
        }

        // Restore best parameters (imperative - don't return degraded state)
        if best_iteration > 0 && best_iteration < self.max_iterations {
            gaussians.clone_from_slice(&best_gaussians);
            println!("  Restored best from iteration {}/{} (loss: {:.6})",
                     best_iteration, self.max_iterations, best_loss);
        }

        best_loss
    }

    /// Optimize Gaussians with data logging callback
    ///
    /// This version accepts an optional logger to record Gaussian states during optimization.
    /// Used for quantum research data collection.
    pub fn optimize_with_logger(
        &mut self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
        mut logger: Option<&mut dyn GaussianLogger>,
    ) -> f32 {
        let n = gaussians.len();
        self.m_color.resize(n, Color4::new(0.0, 0.0, 0.0, 0.0));
        self.v_color.resize(n, Color4::new(0.0, 0.0, 0.0, 0.0));
        self.m_position.resize(n, Vector2::zero());
        self.v_position.resize(n, Vector2::zero());
        self.m_scale.resize(n, (0.0, 0.0));
        self.v_scale.resize(n, (0.0, 0.0));
        self.m_opacity.resize(n, 0.0);
        self.v_opacity.resize(n, 0.0);

        let mut best_loss = f32::INFINITY;
        let mut best_gaussians = gaussians.to_vec();  // Track best state
        let mut best_iteration = 0;

        for t in 1..=self.max_iterations {
            let rendered = RendererV2::render(gaussians, target.width, target.height);
            let loss = self.compute_loss(&rendered, target);

            if t % 10 == 0 {
                let pos_lr = self.position_lr_at_iteration(t);
                if self.use_per_param_lr {
                    println!("  Iteration {}/{}: loss = {:.6} (pos_lr = {:.6})", t, self.max_iterations, loss, pos_lr);
                } else {
                    println!("  Iteration {}/{}: loss = {:.6}", t, self.max_iterations, loss);
                }
            }

            if loss < best_loss {
                best_loss = loss;
                best_gaussians.clone_from_slice(gaussians);  // Save best state
                best_iteration = t;
            }

            // Log Gaussian states (every 10th iteration to reduce data volume)
            if let Some(ref mut log) = logger {
                if t % 10 == 0 {
                    log.log_iteration(gaussians, t as u32, loss);
                }
            }

            let grads = self.compute_gradients(gaussians, &rendered, target);

            let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(t as i32);

            // Get learning rates for this iteration
            let lr_color = if self.use_per_param_lr { self.learning_rates.color } else { self.learning_rate };
            let lr_position = self.position_lr_at_iteration(t);
            let lr_scale = if self.use_per_param_lr { self.learning_rates.scale } else { self.learning_rate };
            let lr_opacity = if self.use_per_param_lr { self.learning_rates.opacity } else { self.learning_rate };

            for i in 0..n {
                // Color Adam update
                self.m_color[i].r = self.beta1 * self.m_color[i].r + (1.0 - self.beta1) * grads[i].color.r;
                self.m_color[i].g = self.beta1 * self.m_color[i].g + (1.0 - self.beta1) * grads[i].color.g;
                self.m_color[i].b = self.beta1 * self.m_color[i].b + (1.0 - self.beta1) * grads[i].color.b;

                self.v_color[i].r = self.beta2 * self.v_color[i].r + (1.0 - self.beta2) * grads[i].color.r.powi(2);
                self.v_color[i].g = self.beta2 * self.v_color[i].g + (1.0 - self.beta2) * grads[i].color.g.powi(2);
                self.v_color[i].b = self.beta2 * self.v_color[i].b + (1.0 - self.beta2) * grads[i].color.b.powi(2);

                let m_hat_r = self.m_color[i].r / bias_correction1;
                let m_hat_g = self.m_color[i].g / bias_correction1;
                let m_hat_b = self.m_color[i].b / bias_correction1;

                let v_hat_r = self.v_color[i].r / bias_correction2;
                let v_hat_g = self.v_color[i].g / bias_correction2;
                let v_hat_b = self.v_color[i].b / bias_correction2;

                gaussians[i].color.r -= lr_color * m_hat_r / (v_hat_r.sqrt() + self.epsilon);
                gaussians[i].color.g -= lr_color * m_hat_g / (v_hat_g.sqrt() + self.epsilon);
                gaussians[i].color.b -= lr_color * m_hat_b / (v_hat_b.sqrt() + self.epsilon);

                gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
                gaussians[i].color.g = gaussians[i].color.g.clamp(0.0, 1.0);
                gaussians[i].color.b = gaussians[i].color.b.clamp(0.0, 1.0);

                // Position Adam update (with exponential decay)
                self.m_position[i].x = self.beta1 * self.m_position[i].x + (1.0 - self.beta1) * grads[i].position.x;
                self.m_position[i].y = self.beta1 * self.m_position[i].y + (1.0 - self.beta1) * grads[i].position.y;

                self.v_position[i].x = self.beta2 * self.v_position[i].x + (1.0 - self.beta2) * grads[i].position.x.powi(2);
                self.v_position[i].y = self.beta2 * self.v_position[i].y + (1.0 - self.beta2) * grads[i].position.y.powi(2);

                let m_hat_px = self.m_position[i].x / bias_correction1;
                let m_hat_py = self.m_position[i].y / bias_correction1;
                let v_hat_px = self.v_position[i].x / bias_correction2;
                let v_hat_py = self.v_position[i].y / bias_correction2;

                gaussians[i].position.x -= lr_position * m_hat_px / (v_hat_px.sqrt() + self.epsilon);
                gaussians[i].position.y -= lr_position * m_hat_py / (v_hat_py.sqrt() + self.epsilon);

                gaussians[i].position.x = gaussians[i].position.x.clamp(0.0, 1.0);
                gaussians[i].position.y = gaussians[i].position.y.clamp(0.0, 1.0);

                // Scale Adam update
                self.m_scale[i].0 = self.beta1 * self.m_scale[i].0 + (1.0 - self.beta1) * grads[i].scale_x;
                self.m_scale[i].1 = self.beta1 * self.m_scale[i].1 + (1.0 - self.beta1) * grads[i].scale_y;

                self.v_scale[i].0 = self.beta2 * self.v_scale[i].0 + (1.0 - self.beta2) * grads[i].scale_x.powi(2);
                self.v_scale[i].1 = self.beta2 * self.v_scale[i].1 + (1.0 - self.beta2) * grads[i].scale_y.powi(2);

                let m_hat_sx = self.m_scale[i].0 / bias_correction1;
                let m_hat_sy = self.m_scale[i].1 / bias_correction1;
                let v_hat_sx = self.v_scale[i].0 / bias_correction2;
                let v_hat_sy = self.v_scale[i].1 / bias_correction2;

                gaussians[i].shape.scale_x -= lr_scale * m_hat_sx / (v_hat_sx.sqrt() + self.epsilon);
                gaussians[i].shape.scale_y -= lr_scale * m_hat_sy / (v_hat_sy.sqrt() + self.epsilon);

                gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
                gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);

                // Opacity Adam update (if gradient is computed)
                if grads[i].opacity != 0.0 {
                    self.m_opacity[i] = self.beta1 * self.m_opacity[i] + (1.0 - self.beta1) * grads[i].opacity;
                    self.v_opacity[i] = self.beta2 * self.v_opacity[i] + (1.0 - self.beta2) * grads[i].opacity.powi(2);

                    let m_hat_o = self.m_opacity[i] / bias_correction1;
                    let v_hat_o = self.v_opacity[i] / bias_correction2;

                    gaussians[i].color.a -= lr_opacity * m_hat_o / (v_hat_o.sqrt() + self.epsilon);
                    gaussians[i].color.a = gaussians[i].color.a.clamp(0.01, 1.0);
                }
            }

            if loss < 1e-4 { break; }
        }

        // Restore best parameters (imperative - don't return degraded state)
        if best_iteration > 0 && best_iteration < self.max_iterations {
            gaussians.clone_from_slice(&best_gaussians);
            println!("  Restored best from iteration {}/{} (loss: {:.6})",
                     best_iteration, self.max_iterations, best_loss);
        }

        // Final flush
        if let Some(ref mut log) = logger {
            let _ = log.flush();
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
                    // Defensive: Ensure scales are valid to prevent NaN in gradient computation
                    let scale_x = gaussian.shape.scale_x.max(0.001);
                    let scale_y = gaussian.shape.scale_y.max(0.001);

                    let dx = px - gaussian.position.x;
                    let dy = py - gaussian.position.y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq > 0.1 { continue; }

                    // Defensive: Clamp denominator to prevent division by zero
                    let scale_product = (scale_x * scale_y).max(1e-6);
                    let weight = (-0.5 * dist_sq / scale_product).exp();

                    // Skip if weight is too small (no contribution) or invalid
                    if weight < 1e-6 || weight.is_nan() || weight.is_infinite() {
                        continue;
                    }

                    gradients[i].color.r += error_r * weight;
                    gradients[i].color.g += error_g * weight;
                    gradients[i].color.b += error_b * weight;

                    let error_weighted = error_r * gaussian.color.r + error_g * gaussian.color.g + error_b * gaussian.color.b;
                    gradients[i].position.x += error_weighted * weight * dx;
                    gradients[i].position.y += error_weighted * weight * dy;

                    // Defensive: Prevent division by tiny scales
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

    fn compute_loss(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
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
    opacity: f32,
}

impl GaussianGradient {
    fn zero() -> Self {
        Self {
            position: Vector2::zero(),
            opacity: 0.0,
            color: Color4::new(0.0, 0.0, 0.0, 0.0),
            scale_x: 0.0,
            scale_y: 0.0,
        }
    }
}
