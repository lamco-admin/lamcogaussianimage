//! Per-Primitive Texture Mapping
//!
//! From research (Gaussian-image.md): Decouple geometry from material appearance
//! Allows high-frequency spatial variability without increasing primitive count
//! Key for representing fine textures in photos (fabric, skin, hair)

use lgi_math::{vec::Vector2, color::Color4};

/// Small texture map attached to a Gaussian primitive
/// Represents fine-scale color variation within the Gaussian's footprint
#[derive(Clone, Debug)]
pub struct TextureMap {
    /// Texture resolution (typically 8×8 or 16×16)
    pub width: usize,
    pub height: usize,

    /// Texture data (stored in local Gaussian coordinates)
    pub data: Vec<Color4<f32>>,
}

impl TextureMap {
    /// Create empty texture map
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![Color4::new(0.5, 0.5, 0.5, 1.0); width * height],
        }
    }

    /// Sample texture at local coordinates (u, v) ∈ [-1, 1]²
    /// Uses bilinear interpolation
    pub fn sample(&self, u: f32, v: f32) -> Color4<f32> {
        // Convert from [-1, 1] to [0, width-1] × [0, height-1]
        let x_cont = ((u + 1.0) * 0.5 * (self.width - 1) as f32).clamp(0.0, (self.width - 1) as f32);
        let y_cont = ((v + 1.0) * 0.5 * (self.height - 1) as f32).clamp(0.0, (self.height - 1) as f32);

        let x0 = x_cont.floor() as usize;
        let y0 = y_cont.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let fx = x_cont - x0 as f32;
        let fy = y_cont - y0 as f32;

        // Bilinear interpolation
        let c00 = &self.data[y0 * self.width + x0];
        let c10 = &self.data[y0 * self.width + x1];
        let c01 = &self.data[y1 * self.width + x0];
        let c11 = &self.data[y1 * self.width + x1];

        Color4::new(
            (1.0 - fx) * (1.0 - fy) * c00.r + fx * (1.0 - fy) * c10.r +
            (1.0 - fx) * fy * c01.r + fx * fy * c11.r,

            (1.0 - fx) * (1.0 - fy) * c00.g + fx * (1.0 - fy) * c10.g +
            (1.0 - fx) * fy * c01.g + fx * fy * c11.g,

            (1.0 - fx) * (1.0 - fy) * c00.b + fx * (1.0 - fy) * c10.b +
            (1.0 - fx) * fy * c01.b + fx * fy * c11.b,

            1.0,
        )
    }

    /// Set texture value at grid position
    pub fn set(&mut self, x: usize, y: usize, color: Color4<f32>) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = color;
        }
    }

    /// Get texture value at grid position
    pub fn get(&self, x: usize, y: usize) -> Color4<f32> {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            Color4::new(0.5, 0.5, 0.5, 1.0)
        }
    }

    /// Extract texture from image region covered by Gaussian
    pub fn extract_from_image(
        image: &crate::ImageBuffer<f32>,
        center: Vector2<f32>,    // Gaussian center (normalized)
        scale_x: f32,            // Gaussian scale x (normalized)
        scale_y: f32,
        rotation: f32,
        texture_size: usize,     // e.g., 8 or 16
    ) -> Self {
        let mut texture = TextureMap::new(texture_size, texture_size);

        let width_f = image.width as f32;
        let height_f = image.height as f32;

        let cx = center.x * width_f;
        let cy = center.y * height_f;
        let sx = scale_x * width_f;
        let sy = scale_y * height_f;
        let rot = rotation;

        let cos_t = rot.cos();
        let sin_t = rot.sin();

        // Sample texture grid
        for ty in 0..texture_size {
            for tx in 0..texture_size {
                // Local coords in [-1, 1]
                let u = (tx as f32 / (texture_size - 1) as f32) * 2.0 - 1.0;
                let v = (ty as f32 / (texture_size - 1) as f32) * 2.0 - 1.0;

                // Convert to Gaussian's elliptical coordinates
                let local_x = u * sx * 3.0;  // ±3σ coverage
                let local_y = v * sy * 3.0;

                // Rotate to image space
                let img_x = cx + local_x * cos_t - local_y * sin_t;
                let img_y = cy + local_x * sin_t + local_y * cos_t;

                // Sample image (with bilinear interpolation)
                let color = sample_image_bilinear(image, img_x, img_y);

                texture.set(tx, ty, Color4::new(
                    color.r,
                    color.g,
                    color.b,
                    1.0,
                ));
            }
        }

        texture
    }
}

/// Sample image with bilinear interpolation
fn sample_image_bilinear(
    image: &crate::ImageBuffer<f32>,
    x: f32,
    y: f32,
) -> Color4<f32> {
    let x_clamped = x.clamp(0.0, (image.width - 1) as f32);
    let y_clamped = y.clamp(0.0, (image.height - 1) as f32);

    let x0 = x_clamped.floor() as u32;
    let y0 = y_clamped.floor() as u32;
    let x1 = (x0 + 1).min(image.width - 1);
    let y1 = (y0 + 1).min(image.height - 1);

    let fx = x_clamped - x0 as f32;
    let fy = y_clamped - y0 as f32;

    if let (Some(c00), Some(c10), Some(c01), Some(c11)) = (
        image.get_pixel(x0, y0),
        image.get_pixel(x1, y0),
        image.get_pixel(x0, y1),
        image.get_pixel(x1, y1),
    ) {
        Color4::new(
            (1.0-fx)*(1.0-fy)*c00.r + fx*(1.0-fy)*c10.r +
            (1.0-fx)*fy*c01.r + fx*fy*c11.r,

            (1.0-fx)*(1.0-fy)*c00.g + fx*(1.0-fy)*c10.g +
            (1.0-fx)*fy*c01.g + fx*fy*c11.g,

            (1.0-fx)*(1.0-fy)*c00.b + fx*(1.0-fy)*c10.b +
            (1.0-fx)*fy*c01.b + fx*fy*c11.b,

            1.0,
        )
    } else {
        Color4::new(0.5, 0.5, 0.5, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texture_map_creation() {
        let tex = TextureMap::new(8, 8);
        assert_eq!(tex.width, 8);
        assert_eq!(tex.height, 8);
        assert_eq!(tex.data.len(), 64);
    }

    #[test]
    fn test_texture_sampling() {
        let mut tex = TextureMap::new(4, 4);
        tex.set(0, 0, Color4::new(1.0, 0.0, 0.0, 1.0));  // Red corner
        tex.set(3, 3, Color4::new(0.0, 0.0, 1.0, 1.0));  // Blue corner

        // Sample at corners
        let c1 = tex.sample(-1.0, -1.0);  // Should be red
        let c2 = tex.sample(1.0, 1.0);     // Should be blue

        assert!(c1.r > 0.5);
        assert!(c2.b > 0.5);
    }
}
