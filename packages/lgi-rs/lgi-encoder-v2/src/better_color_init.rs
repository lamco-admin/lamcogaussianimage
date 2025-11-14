//! Better color initialization
//! Instead of sampling single pixel, average over Gaussian's footprint

use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, vec::Vector2};

/// Compute weighted average color over Gaussian's footprint
pub fn gaussian_weighted_color(
    image: &ImageBuffer<f32>,
    center: Vector2<f32>,  // Normalized [0,1]
    sigma_x: f32,          // Normalized
    sigma_y: f32,
    rotation: f32,
) -> Color4<f32> {
    let width = image.width as f32;
    let height = image.height as f32;

    // Convert to pixel coordinates
    let cx_px = center.x * width;
    let cy_px = center.y * height;
    let sx_px = sigma_x * width;
    let sy_px = sigma_y * height;

    // Sample region: ±3σ
    let sample_radius_x = (3.0 * sx_px) as u32;
    let sample_radius_y = (3.0 * sy_px) as u32;

    let x_start = (cx_px as i32 - sample_radius_x as i32).max(0) as u32;
    let x_end = (cx_px as u32 + sample_radius_x).min(image.width);
    let y_start = (cy_px as i32 - sample_radius_y as i32).max(0) as u32;
    let y_end = (cy_px as u32 + sample_radius_y).min(image.height);

    let mut weighted_r = 0.0;
    let mut weighted_g = 0.0;
    let mut weighted_b = 0.0;
    let mut weight_sum = 0.0;

    let cos_t = rotation.cos();
    let sin_t = rotation.sin();

    for y in y_start..y_end {
        for x in x_start..x_end {
            let dx = x as f32 - cx_px;
            let dy = y as f32 - cy_px;

            // Rotate to Gaussian frame
            let dx_rot = dx * cos_t + dy * sin_t;
            let dy_rot = -dx * sin_t + dy * cos_t;

            // Gaussian weight
            let dist_sq = (dx_rot / sx_px).powi(2) + (dy_rot / sy_px).powi(2);

            if dist_sq < 9.0 {  // Within 3σ
                let weight = (-0.5 * dist_sq).exp();

                if let Some(pixel) = image.get_pixel(x, y) {
                    weighted_r += weight * pixel.r;
                    weighted_g += weight * pixel.g;
                    weighted_b += weight * pixel.b;
                    weight_sum += weight;
                }
            }
        }
    }

    if weight_sum > 1e-6 {
        Color4::new(
            weighted_r / weight_sum,
            weighted_g / weight_sum,
            weighted_b / weight_sum,
            1.0,
        )
    } else {
        // Fallback: sample center pixel
        image.get_pixel(cx_px as u32, cy_px as u32)
            .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_color_uniform() {
        let mut img = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                img.set_pixel(x, y, Color4::new(1.0, 0.0, 0.0, 1.0));
            }
        }

        let color = gaussian_weighted_color(
            &img,
            Vector2::new(0.5, 0.5),
            0.1,
            0.1,
            0.0,
        );

        assert!((color.r - 1.0).abs() < 0.01);
        assert!(color.g < 0.01);
    }
}
