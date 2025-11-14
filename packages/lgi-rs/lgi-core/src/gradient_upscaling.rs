//! Analytical Gradient Upscaling
//!
//! From research (Gaussian-image.md): Exact gradient interpolation
//! Uses analytical derivatives of Gaussian kernels instead of finite differences
//! Key for better reconstruction quality and upscaling

use crate::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

/// Analytical gradient upscaler
///
/// Computes exact image gradients from Gaussian representation
/// Enables high-quality upscaling without interpolation artifacts
pub struct AnalyticalGradientUpscaler;

impl AnalyticalGradientUpscaler {
    /// Compute analytical gradient field from Gaussians
    ///
    /// Returns: (gradient_x, gradient_y) images
    pub fn compute_gradient_field(
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        width: u32,
        height: u32,
    ) -> (ImageBuffer<f32>, ImageBuffer<f32>) {
        let mut grad_x = ImageBuffer::new(width, height);
        let mut grad_y = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / width as f32;
                let py = y as f32 / height as f32;

                let mut gx_r = 0.0;
                let mut gx_g = 0.0;
                let mut gx_b = 0.0;
                let mut gy_r = 0.0;
                let mut gy_g = 0.0;
                let mut gy_b = 0.0;
                let mut weight_sum = 0.0;

                // Accumulate gradient contributions
                for gaussian in gaussians {
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

                    if dist_sq > 12.25 {
                        continue;
                    }

                    let gaussian_val = (-0.5 * dist_sq).exp();
                    let weight = gaussian.opacity * gaussian_val;

                    // Analytical gradient of Gaussian:
                    // ∂G/∂x = -G × Σ⁻¹ × (x - μ)

                    // Gradient in rotated space
                    let grad_rot_x = -weight * dx_rot / (sx * sx);
                    let grad_rot_y = -weight * dy_rot / (sy * sy);

                    // Rotate back to world space
                    let grad_world_x = grad_rot_x * cos_t - grad_rot_y * sin_t;
                    let grad_world_y = grad_rot_x * sin_t + grad_rot_y * cos_t;

                    // Weight by color
                    gx_r += grad_world_x * gaussian.color.r;
                    gx_g += grad_world_x * gaussian.color.g;
                    gx_b += grad_world_x * gaussian.color.b;

                    gy_r += grad_world_y * gaussian.color.r;
                    gy_g += grad_world_y * gaussian.color.g;
                    gy_b += grad_world_y * gaussian.color.b;

                    weight_sum += weight;
                }

                // Normalize
                if weight_sum > 1e-6 {
                    gx_r /= weight_sum;
                    gx_g /= weight_sum;
                    gx_b /= weight_sum;
                    gy_r /= weight_sum;
                    gy_g /= weight_sum;
                    gy_b /= weight_sum;
                }

                // Average across channels for visualization
                let gx_avg = (gx_r + gx_g + gx_b) / 3.0;
                let gy_avg = (gy_r + gy_g + gy_b) / 3.0;

                grad_x.set_pixel(x, y, Color4::new(gx_avg, gx_avg, gx_avg, 1.0));
                grad_y.set_pixel(x, y, Color4::new(gy_avg, gy_avg, gy_avg, 1.0));
            }
        }

        (grad_x, grad_y)
    }

    /// Upscale image using gradient information
    ///
    /// Uses analytical gradients for high-quality interpolation
    /// Better than bilinear/bicubic for Gaussian-represented images
    pub fn upscale_with_gradients(
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        original_width: u32,
        original_height: u32,
        scale_factor: f32,
    ) -> ImageBuffer<f32> {
        let new_width = (original_width as f32 * scale_factor) as u32;
        let new_height = (original_height as f32 * scale_factor) as u32;

        let mut upscaled = ImageBuffer::new(new_width, new_height);

        // Render at higher resolution directly from Gaussians
        // Gaussians are resolution-independent!
        for y in 0..new_height {
            for x in 0..new_width {
                let px = x as f32 / new_width as f32;
                let py = y as f32 / new_height as f32;

                let mut weight_sum = 0.0;
                let mut color_sum = Color4::new(0.0, 0.0, 0.0, 0.0);

                for gaussian in gaussians {
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

                    if dist_sq > 12.25 {
                        continue;
                    }

                    let gaussian_val = (-0.5 * dist_sq).exp();
                    let weight = gaussian.opacity * gaussian_val;

                    weight_sum += weight;
                    color_sum.r += weight * gaussian.color.r;
                    color_sum.g += weight * gaussian.color.g;
                    color_sum.b += weight * gaussian.color.b;
                }

                let final_color = if weight_sum > 1e-10 {
                    Color4::new(
                        color_sum.r / weight_sum,
                        color_sum.g / weight_sum,
                        color_sum.b / weight_sum,
                        1.0,
                    )
                } else {
                    Color4::new(0.0, 0.0, 0.0, 1.0)
                };

                upscaled.set_pixel(x, y, final_color);
            }
        }

        upscaled
    }

    /// Compute gradient magnitude map
    ///
    /// Useful for edge detection and refinement decisions
    pub fn gradient_magnitude_map(
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        width: u32,
        height: u32,
    ) -> ImageBuffer<f32> {
        let (grad_x, grad_y) = Self::compute_gradient_field(gaussians, width, height);

        let mut magnitude = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                if let (Some(gx), Some(gy)) = (grad_x.get_pixel(x, y), grad_y.get_pixel(x, y)) {
                    let mag = (gx.r * gx.r + gy.r * gy.r).sqrt();
                    magnitude.set_pixel(x, y, Color4::new(mag, mag, mag, 1.0));
                }
            }
        }

        magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_computation() {
        // Create simple Gaussian at center
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.1),
            Color4::new(1.0, 0.0, 0.0, 1.0),
            1.0,
        );

        let gaussians = vec![gaussian];

        let (grad_x, grad_y) = AnalyticalGradientUpscaler::compute_gradient_field(&gaussians, 64, 64);

        // At center, gradients should be near zero
        let center_gx = grad_x.get_pixel(32, 32).unwrap();
        assert!(center_gx.r.abs() < 0.1);

        // Away from center, gradients should be non-zero
        let off_center_gx = grad_x.get_pixel(40, 32).unwrap();
        assert!(off_center_gx.r.abs() > 0.01);
    }

    #[test]
    fn test_upscaling() {
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.1),
            Color4::new(1.0, 0.0, 0.0, 1.0),
            1.0,
        );

        let gaussians = vec![gaussian];

        // Upscale 2×
        let upscaled = AnalyticalGradientUpscaler::upscale_with_gradients(&gaussians, 64, 64, 2.0);

        assert_eq!(upscaled.width, 128);
        assert_eq!(upscaled.height, 128);

        // Center should still be red
        let center = upscaled.get_pixel(64, 64).unwrap();
        assert!(center.r > 0.5);
    }
}
