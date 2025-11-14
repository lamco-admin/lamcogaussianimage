//! Renderer v2 - For testing encoder quality
//!
//! Simple CPU renderer to validate encoded Gaussians
//! Uses log-Cholesky covariances correctly

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, color::Color4};

/// Simple renderer for validation
pub struct RendererV2;

impl RendererV2 {
    /// Render Gaussians to image
    pub fn render(
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        width: u32,
        height: u32,
    ) -> ImageBuffer<f32> {
        let mut output = ImageBuffer::new(width, height);

        // For each pixel
        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / width as f32;
                let py = y as f32 / height as f32;

                // WEIGHTED AVERAGE RENDERER - Debug Plan Section 0.2
                // Accumulate: W = Σ w_i, C = Σ w_i × c_i
                // Output: C / max(W, ε)

                let mut weight_sum = 0.0;
                let mut color_sum = Color4::new(0.0, 0.0, 0.0, 0.0);

                // Accumulate contributions from all Gaussians
                for gaussian in gaussians {
                    let dx = px - gaussian.position.x;
                    let dy = py - gaussian.position.y;

                    // Compute using covariance
                    let sx = gaussian.shape.scale_x;
                    let sy = gaussian.shape.scale_y;
                    let theta = gaussian.shape.rotation;

                    // Rotate to Gaussian's local frame
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();
                    let dx_rot = dx * cos_t + dy * sin_t;
                    let dy_rot = -dx * sin_t + dy * cos_t;

                    // Mahalanobis distance (using scales as variances)
                    let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);

                    // Cutoff for efficiency
                    if dist_sq > 12.25 {  // 3.5 sigma
                        continue;
                    }

                    // Gaussian weight: w_i = α_i × exp(-0.5 × d²)
                    let gaussian_val = (-0.5 * dist_sq).exp();
                    let weight = gaussian.opacity * gaussian_val;

                    // Accumulate
                    weight_sum += weight;
                    color_sum.r += weight * gaussian.color.r;
                    color_sum.g += weight * gaussian.color.g;
                    color_sum.b += weight * gaussian.color.b;
                }

                // NORMALIZE by weight sum (Debug Plan Section 0.2)
                let final_color = if weight_sum > 1e-10 {
                    Color4::new(
                        color_sum.r / weight_sum,
                        color_sum.g / weight_sum,
                        color_sum.b / weight_sum,
                        1.0,
                    )
                } else {
                    // No Gaussian coverage - black
                    Color4::new(0.0, 0.0, 0.0, 1.0)
                };

                output.set_pixel(x, y, final_color);
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::vec::Vector2;

    #[test]
    fn test_render_single_gaussian() {
        let gaussians = vec![
            Gaussian2D::new(
                Vector2::new(0.5, 0.5),  // Center
                Euler::isotropic(0.1),
                Color4::new(1.0, 0.0, 0.0, 1.0),  // Red
                1.0,
            )
        ];

        let rendered = RendererV2::render(&gaussians, 64, 64);

        // Center pixel should be red
        let center = rendered.get_pixel(32, 32).unwrap();
        assert!(center.r > 0.5, "Center should be red");
        assert!(center.g < 0.1, "Center should not be green");
    }
}
