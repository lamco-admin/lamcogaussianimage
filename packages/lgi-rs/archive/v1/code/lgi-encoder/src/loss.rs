//! Loss functions for Gaussian optimization

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;

/// Loss function type
pub trait LossFunction {
    /// Compute loss between rendered and target images
    fn compute(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32;

    /// Compute per-pixel gradient (for backpropagation)
    fn gradient(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> ImageBuffer<f32>;
}

/// L2 (Mean Squared Error) loss
pub struct L2Loss;

impl LossFunction for L2Loss {
    fn compute(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
        assert_eq!(rendered.width, target.width);
        assert_eq!(rendered.height, target.height);

        let mut sum = 0.0;
        let count = rendered.data.len() as f32;

        for (r, t) in rendered.data.iter().zip(target.data.iter()) {
            let diff_r = r.r - t.r;
            let diff_g = r.g - t.g;
            let diff_b = r.b - t.b;

            sum += diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
        }

        sum / count
    }

    fn gradient(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> ImageBuffer<f32> {
        let mut grad = ImageBuffer::new(rendered.width, rendered.height);

        for (idx, (r, t)) in rendered.data.iter().zip(target.data.iter()).enumerate() {
            grad.data[idx] = Color4::new(
                2.0 * (r.r - t.r),
                2.0 * (r.g - t.g),
                2.0 * (r.b - t.b),
                0.0,
            );
        }

        grad
    }
}

/// SSIM (Structural Similarity) loss
///
/// Simplified implementation for now (full fused-ssim would be better)
pub struct SSIMLoss {
    window_size: usize,
}

impl SSIMLoss {
    /// Create new SSIM loss with window size
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }
}

impl Default for SSIMLoss {
    fn default() -> Self {
        Self::new(11)
    }
}

impl LossFunction for SSIMLoss {
    fn compute(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
        // Simplified SSIM computation
        let c1 = 0.01 * 0.01;
        let c2 = 0.03 * 0.03;

        let mut ssim_sum = 0.0;
        let mut count = 0;

        let half_window = (self.window_size / 2) as i32;

        for y in half_window..(rendered.height as i32 - half_window) {
            for x in half_window..(rendered.width as i32 - half_window) {
                // Compute local means and variances
                let mut mean_r = 0.0;
                let mut mean_t = 0.0;
                let mut window_count = 0;

                for dy in -half_window..=half_window {
                    for dx in -half_window..=half_window {
                        let px = (x + dx) as u32;
                        let py = (y + dy) as u32;

                        if let (Some(r_pixel), Some(t_pixel)) = (
                            rendered.get_pixel(px, py),
                            target.get_pixel(px, py),
                        ) {
                            let r_luma = (r_pixel.r + r_pixel.g + r_pixel.b) / 3.0;
                            let t_luma = (t_pixel.r + t_pixel.g + t_pixel.b) / 3.0;

                            mean_r += r_luma;
                            mean_t += t_luma;
                            window_count += 1;
                        }
                    }
                }

                mean_r /= window_count as f32;
                mean_t /= window_count as f32;

                // Compute variances and covariance
                let mut var_r = 0.0;
                let mut var_t = 0.0;
                let mut covar = 0.0;

                for dy in -half_window..=half_window {
                    for dx in -half_window..=half_window {
                        let px = (x + dx) as u32;
                        let py = (y + dy) as u32;

                        if let (Some(r_pixel), Some(t_pixel)) = (
                            rendered.get_pixel(px, py),
                            target.get_pixel(px, py),
                        ) {
                            let r_luma = (r_pixel.r + r_pixel.g + r_pixel.b) / 3.0;
                            let t_luma = (t_pixel.r + t_pixel.g + t_pixel.b) / 3.0;

                            let diff_r = r_luma - mean_r;
                            let diff_t = t_luma - mean_t;

                            var_r += diff_r * diff_r;
                            var_t += diff_t * diff_t;
                            covar += diff_r * diff_t;
                        }
                    }
                }

                var_r /= window_count as f32;
                var_t /= window_count as f32;
                covar /= window_count as f32;

                // SSIM formula
                let numerator = (2.0 * mean_r * mean_t + c1) * (2.0 * covar + c2);
                let denominator = (mean_r * mean_r + mean_t * mean_t + c1) * (var_r + var_t + c2);

                let ssim = numerator / denominator;
                ssim_sum += ssim;
                count += 1;
            }
        }

        // Return 1 - SSIM as loss (we want to minimize)
        1.0 - (ssim_sum / count as f32)
    }

    fn gradient(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> ImageBuffer<f32> {
        // For now, use finite differences (proper SSIM gradient is complex)
        // In production, use fused-ssim library

        // Simplified: approximate with L2 gradient
        let l2 = L2Loss;
        l2.gradient(rendered, target)
    }
}

/// Combined loss functions
pub struct LossFunctions {
    l2_weight: f32,
    ssim_weight: f32,
    l2: L2Loss,
    ssim: SSIMLoss,
}

impl LossFunctions {
    /// Create new combined loss
    pub fn new(l2_weight: f32, ssim_weight: f32) -> Self {
        Self {
            l2_weight,
            ssim_weight,
            l2: L2Loss,
            ssim: SSIMLoss::default(),
        }
    }

    /// Compute combined loss
    pub fn compute(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
        let l2_loss = self.l2.compute(rendered, target);
        let ssim_loss = self.ssim.compute(rendered, target);

        self.l2_weight * l2_loss + self.ssim_weight * ssim_loss
    }

    /// Compute combined gradient
    pub fn gradient(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> ImageBuffer<f32> {
        let mut grad_l2 = self.l2.gradient(rendered, target);
        let grad_ssim = self.ssim.gradient(rendered, target);

        // Weighted combination
        for (idx, pixel) in grad_l2.data.iter_mut().enumerate() {
            let ssim_grad = grad_ssim.data[idx];
            pixel.r = self.l2_weight * pixel.r + self.ssim_weight * ssim_grad.r;
            pixel.g = self.l2_weight * pixel.g + self.ssim_weight * ssim_grad.g;
            pixel.b = self.l2_weight * pixel.b + self.ssim_weight * ssim_grad.b;
        }

        grad_l2
    }
}
