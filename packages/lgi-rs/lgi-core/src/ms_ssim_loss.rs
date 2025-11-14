//! MS-SSIM as differentiable loss function
//! Currently ms_ssim.rs only computes metric
//! This module makes it usable as optimization objective

use crate::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, color::Color4};

/// Differentiable MS-SSIM loss for optimization
pub struct MsssimLoss {
    /// Weight for MS-SSIM term vs L2
    pub alpha: f32,  // 0.0 = pure L2, 1.0 = pure MS-SSIM

}

impl Default for MsssimLoss {
    fn default() -> Self {
        Self {
            alpha: 0.84,  // From research: 0.84 MS-SSIM + 0.16 L2
        }
    }
}

impl MsssimLoss {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Compute combined loss: (1-α)·L2 + α·(1-MS-SSIM)
    pub fn compute_loss(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
        // L2 component
        let mut mse = 0.0;
        for (r, t) in rendered.data.iter().zip(target.data.iter()) {
            mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
        }
        mse /= (rendered.width * rendered.height * 3) as f32;

        // MS-SSIM component
        let ms_ssim = crate::ms_ssim::MSSSIM::default();
        let ms_ssim_score = ms_ssim.compute(target, rendered);
        let ms_ssim_loss = 1.0 - ms_ssim_score;  // Convert to loss (0 = perfect)

        // Combined
        (1.0 - self.alpha) * mse + self.alpha * ms_ssim_loss
    }

    /// Compute gradients (approximate via finite differences for now)
    /// TODO: Implement analytical MS-SSIM gradients
    pub fn compute_gradients_approx(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
        renderer: impl Fn(&[Gaussian2D<f32, Euler<f32>>], u32, u32) -> ImageBuffer<f32>,
    ) -> Vec<GaussianGradient> {
        let base_loss = self.compute_loss(rendered, target);
        let mut gradients = vec![GaussianGradient::zero(); gaussians.len()];

        let epsilon = 0.001;

        for (i, gaussian) in gaussians.iter().enumerate() {
            // Color gradients
            for channel in 0..3 {
                let mut g_plus = gaussians.to_vec();
                match channel {
                    0 => g_plus[i].color.r += epsilon,
                    1 => g_plus[i].color.g += epsilon,
                    2 => g_plus[i].color.b += epsilon,
                    _ => {}
                }
                let r_plus = renderer(&g_plus, rendered.width, rendered.height);
                let loss_plus = self.compute_loss(&r_plus, target);
                let grad = (loss_plus - base_loss) / epsilon;

                match channel {
                    0 => gradients[i].color.r = grad,
                    1 => gradients[i].color.g = grad,
                    2 => gradients[i].color.b = grad,
                    _ => {}
                }
            }
        }

        gradients
    }
}

#[derive(Clone)]
struct GaussianGradient {
    color: Color4<f32>,
}

impl GaussianGradient {
    fn zero() -> Self {
        Self { color: Color4::new(0.0, 0.0, 0.0, 0.0) }
    }
}
