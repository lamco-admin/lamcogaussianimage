//! EWA (Elliptical Weighted Average) Splatting
//! Alias-free Gaussian rendering for zoom stability
//! From Zwicker et al. 2001

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use crate::ImageBuffer;

/// EWA splatting renderer
pub struct EWASplatter {
    /// Filter radius multiplier
    pub radius_multiplier: f32,
}

impl Default for EWASplatter {
    fn default() -> Self {
        Self { radius_multiplier: 3.5 }
    }
}

impl EWASplatter {
    /// Render with EWA splatting
    pub fn render(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        width: u32,
        height: u32,
    ) -> ImageBuffer<f32> {
        let mut output = ImageBuffer::new(width, height);

        for gaussian in gaussians {
            self.splat_gaussian(gaussian, &mut output, width, height);
        }

        output
    }

    fn splat_gaussian(
        &self,
        gaussian: &Gaussian2D<f32, Euler<f32>>,
        output: &mut ImageBuffer<f32>,
        width: u32,
        height: u32,
    ) {
        // Screen-space Gaussian parameters
        let cx = gaussian.position.x * width as f32;
        let cy = gaussian.position.y * height as f32;
        let sx = gaussian.shape.scale_x * width as f32;
        let sy = gaussian.shape.scale_y * height as f32;
        let theta = gaussian.shape.rotation;

        // Bounding box (conservative)
        let radius = self.radius_multiplier * sx.max(sy);
        let x_min = (cx - radius).floor().max(0.0) as u32;
        let x_max = (cx + radius).ceil().min(width as f32) as u32;
        let y_min = (cy - radius).floor().max(0.0) as u32;
        let y_max = (cy + radius).ceil().min(height as f32) as u32;

        // Rotation matrix
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Splat within bounding box
        for y in y_min..y_max {
            for x in x_min..x_max {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;

                // Rotate to Gaussian frame
                let dx_rot = dx * cos_t + dy * sin_t;
                let dy_rot = -dx * sin_t + dy * cos_t;

                // EWA filter weight
                let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);

                if dist_sq < self.radius_multiplier.powi(2) {
                    let weight = (-0.5 * dist_sq).exp();

                    if let Some(pixel) = output.get_pixel_mut(x, y) {
                        // Accumulate with weight
                        pixel.r += weight * gaussian.color.r * gaussian.opacity;
                        pixel.g += weight * gaussian.color.g * gaussian.opacity;
                        pixel.b += weight * gaussian.color.b * gaussian.opacity;
                        pixel.a += weight * gaussian.opacity;
                    }
                }
            }
        }
    }

    /// Normalize output after splatting
    pub fn normalize(output: &mut ImageBuffer<f32>) {
        for pixel in &mut output.data {
            if pixel.a > 1e-10 {
                pixel.r /= pixel.a;
                pixel.g /= pixel.a;
                pixel.b /= pixel.a;
                pixel.a = 1.0;
            }
        }
    }
}
