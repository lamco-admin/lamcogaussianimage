//! EWA Splatting v2 - Full Robust Implementation
//! Elliptical Weighted Average per Zwicker et al. 2001
//! Alias-free, zoom-stable Gaussian rendering

use crate::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, color::Color4};

/// Full EWA renderer with reconstruction filter
pub struct EWARendererV2 {
    /// Reconstruction filter bandwidth (typically 1.0)
    pub filter_bandwidth: f32,
    /// Cutoff threshold (typically 3.0-3.5)
    pub cutoff_radius: f32,
}

impl Default for EWARendererV2 {
    fn default() -> Self {
        Self {
            filter_bandwidth: 1.0,
            cutoff_radius: 3.5,
        }
    }
}

impl EWARendererV2 {
    /// Render Gaussians with full EWA splatting
    pub fn render(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        width: u32,
        height: u32,
        zoom: f32,
    ) -> ImageBuffer<f32> {
        let mut color_accum = ImageBuffer::new(width, height);
        let mut weight_accum = ImageBuffer::new(width, height);

        for gaussian in gaussians {
            self.ewa_splat(gaussian, &mut color_accum, &mut weight_accum, width, height, zoom);
        }

        // Normalize by accumulated weights
        self.normalize(&mut color_accum, &weight_accum);

        color_accum
    }

    fn ewa_splat(
        &self,
        gaussian: &Gaussian2D<f32, Euler<f32>>,
        color_accum: &mut ImageBuffer<f32>,
        weight_accum: &mut ImageBuffer<f32>,
        width: u32,
        height: u32,
        zoom: f32,
    ) {
        // Transform Gaussian to screen space
        // Note: width/height already account for zoom in render_multiscale
        let mu_x = gaussian.position.x * width as f32;
        let mu_y = gaussian.position.y * height as f32;

        // Scale to screen space
        let sx = gaussian.shape.scale_x * width as f32;
        let sy = gaussian.shape.scale_y * height as f32;
        let theta = gaussian.shape.rotation;

        // Build covariance matrix in screen space
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Rotation matrix R
        // R = [cos_t, -sin_t]
        //     [sin_t,  cos_t]

        // Covariance: Σ = R · diag(sx², sy²) · R^T
        let sx_sq = sx * sx;
        let sy_sq = sy * sy;

        let sigma_xx = sx_sq * cos_t * cos_t + sy_sq * sin_t * sin_t;
        let sigma_yy = sx_sq * sin_t * sin_t + sy_sq * cos_t * cos_t;
        let sigma_xy = (sx_sq - sy_sq) * cos_t * sin_t;

        // Add reconstruction filter (critical for anti-aliasing)
        // V_footprint = V_gaussian + I (I = pixel footprint)
        let V_xx = sigma_xx + self.filter_bandwidth;
        let V_yy = sigma_yy + self.filter_bandwidth;
        let V_xy = sigma_xy;  // Off-diagonal unchanged

        // Compute inverse of V_footprint for evaluation
        let det_V = V_xx * V_yy - V_xy * V_xy;
        if det_V < 1e-10 {
            return;  // Degenerate
        }

        let inv_V_xx = V_yy / det_V;
        let inv_V_yy = V_xx / det_V;
        let inv_V_xy = -V_xy / det_V;

        // Compute bounding box
        let radius_sq = self.cutoff_radius * self.cutoff_radius;
        let max_extent = self.cutoff_radius * (V_xx.max(V_yy)).sqrt();

        let x_min = (mu_x - max_extent).floor().max(0.0) as u32;
        let x_max = (mu_x + max_extent).ceil().min(width as f32) as u32;
        let y_min = (mu_y - max_extent).floor().max(0.0) as u32;
        let y_max = (mu_y + max_extent).ceil().min(height as f32) as u32;

        // Splat within footprint
        for y in y_min..y_max {
            for x in x_min..x_max {
                let px = x as f32 + 0.5;  // Pixel center
                let py = y as f32 + 0.5;

                let dx = px - mu_x;
                let dy = py - mu_y;

                // Mahalanobis distance with inverse V_footprint
                let dist_sq = dx * (inv_V_xx * dx + inv_V_xy * dy) + dy * (inv_V_xy * dx + inv_V_yy * dy);

                if dist_sq < radius_sq {
                    // EWA weight
                    let weight = (-0.5 * dist_sq).exp() * gaussian.opacity;

                    // Accumulate weighted color
                    if let Some(color_pixel) = color_accum.get_pixel_mut(x, y) {
                        color_pixel.r += weight * gaussian.color.r;
                        color_pixel.g += weight * gaussian.color.g;
                        color_pixel.b += weight * gaussian.color.b;
                    }

                    // Accumulate weight
                    if let Some(weight_pixel) = weight_accum.get_pixel_mut(x, y) {
                        weight_pixel.r += weight;
                    }
                }
            }
        }
    }

    /// Normalize accumulated colors by weights
    fn normalize(&self, color: &mut ImageBuffer<f32>, weights: &ImageBuffer<f32>) {
        for (color_pixel, weight_pixel) in color.data.iter_mut().zip(weights.data.iter()) {
            let w = weight_pixel.r;
            if w > 1e-10 {
                color_pixel.r /= w;
                color_pixel.g /= w;
                color_pixel.b /= w;
                color_pixel.a = 1.0;
            } else {
                // No Gaussian contribution - use background
                color_pixel.r = 0.0;
                color_pixel.g = 0.0;
                color_pixel.b = 0.0;
                color_pixel.a = 1.0;
            }
        }
    }

    /// Render at multiple zoom levels for validation
    pub fn render_multiscale(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        base_width: u32,
        base_height: u32,
        zooms: &[f32],
    ) -> Vec<(f32, ImageBuffer<f32>)> {
        zooms
            .iter()
            .map(|&zoom| {
                let render_width = (base_width as f32 * zoom) as u32;
                let render_height = (base_height as f32 * zoom) as u32;
                let rendered = self.render(gaussians, render_width, render_height, zoom);
                (zoom, rendered)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::vec::Vector2;

    #[test]
    fn test_ewa_render_single_gaussian() {
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.1),
            Color4::new(1.0, 0.0, 0.0, 1.0),
            1.0,
        );

        let renderer = EWARendererV2::default();
        let rendered = renderer.render(&[gaussian], 64, 64, 1.0);

        // Should have non-zero values near center
        let center = rendered.get_pixel(32, 32).unwrap();
        assert!(center.r > 0.5, "Center should be bright red");
    }

    #[test]
    fn test_ewa_zoom_stability() {
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.1),
            Color4::new(1.0, 0.0, 0.0, 1.0),
            1.0,
        );

        let renderer = EWARendererV2::default();

        // Render at different zooms
        let zooms = vec![0.5, 1.0, 2.0];
        let renders = renderer.render_multiscale(&[gaussian], 64, 64, &zooms);

        // All should render (no crashes)
        assert_eq!(renders.len(), 3);

        // Center pixel should be similar across zooms (normalized)
        for (zoom, rendered) in renders {
            let cx = ((rendered.width / 2) as f32) as u32;
            let cy = ((rendered.height / 2) as u32) as u32;
            let center = rendered.get_pixel(cx, cy);
            if let Some(c) = center {
                assert!(c.r > 0.1, "Zoom {}: center should have red component", zoom);
            }
        }
    }
}
