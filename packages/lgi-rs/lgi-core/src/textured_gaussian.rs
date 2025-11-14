//! Textured Gaussian - Gaussian with optional texture map
//!
//! Commercial-grade implementation for production use
//! Per-primitive textures enable fine detail without increasing Gaussian count

use crate::texture_map::TextureMap;
use crate::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

/// Gaussian primitive with optional texture map
///
/// For photos: base Gaussian represents smooth variation,
/// texture map represents fine detail (skin pores, fabric weave, etc.)
#[derive(Clone)]
pub struct TexturedGaussian2D {
    /// Base Gaussian (position, scale, rotation, base color, opacity)
    pub gaussian: Gaussian2D<f32, Euler<f32>>,

    /// Optional texture map (8×8 or 16×16 typical)
    /// None = use base color only (for smooth regions)
    /// Some = modulate base color by texture (for detailed regions)
    pub texture: Option<TextureMap>,
}

impl TexturedGaussian2D {
    /// Create from base Gaussian without texture
    pub fn from_gaussian(gaussian: Gaussian2D<f32, Euler<f32>>) -> Self {
        Self {
            gaussian,
            texture: None,
        }
    }

    /// Create with texture map
    pub fn with_texture(gaussian: Gaussian2D<f32, Euler<f32>>, texture: TextureMap) -> Self {
        Self {
            gaussian,
            texture: Some(texture),
        }
    }

    /// Evaluate color at world position, incorporating texture if present
    pub fn evaluate_color(&self, world_pos: Vector2<f32>) -> Color4<f32> {
        // Base color from Gaussian
        let mut color = self.gaussian.color;

        // If texture present, modulate by texture
        if let Some(ref texture) = self.texture {
            // Convert world position to local Gaussian coordinates
            let local = self.world_to_local(world_pos);

            // Sample texture at local coords
            let tex_color = texture.sample(local.x, local.y);

            // ADDITIVE detail modulation (preserves base brightness, adds detail)
            // Texture is centered around 0.5, detail = texture - 0.5
            let detail_strength = 0.5;  // Scale factor for detail
            color.r += (tex_color.r - 0.5) * detail_strength;
            color.g += (tex_color.g - 0.5) * detail_strength;
            color.b += (tex_color.b - 0.5) * detail_strength;

            // Clamp to valid range
            color.r = color.r.clamp(0.0, 1.0);
            color.g = color.g.clamp(0.0, 1.0);
            color.b = color.b.clamp(0.0, 1.0);
        }

        color
    }

    /// Convert world position to Gaussian's local coordinates [-1, 1]²
    pub fn world_to_local(&self, world_pos: Vector2<f32>) -> Vector2<f32> {
        let dx = world_pos.x - self.gaussian.position.x;
        let dy = world_pos.y - self.gaussian.position.y;

        // Get Euler parameters
        let sx = self.gaussian.shape.scale_x;
        let sy = self.gaussian.shape.scale_y;
        let theta = self.gaussian.shape.rotation;

        // Rotate to Gaussian's local frame
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let dx_rot = dx * cos_t + dy * sin_t;
        let dy_rot = -dx * sin_t + dy * cos_t;

        // Normalize to [-1, 1] range (±3σ coverage)
        let u = dx_rot / (sx * 3.0);
        let v = dy_rot / (sy * 3.0);

        Vector2::new(u.clamp(-1.0, 1.0), v.clamp(-1.0, 1.0))
    }

    /// Extract texture from image region covered by this Gaussian
    ///
    /// Samples the image within the Gaussian's footprint and creates
    /// a texture map representing local detail
    pub fn extract_texture_from_image(
        &mut self,
        image: &ImageBuffer<f32>,
        texture_size: usize,
    ) {
        let texture = TextureMap::extract_from_image(
            image,
            self.gaussian.position,
            self.gaussian.shape.scale_x,
            self.gaussian.shape.scale_y,
            self.gaussian.shape.rotation,
            texture_size,
        );

        self.texture = Some(texture);
    }

    /// Check if texture should be added based on image complexity
    ///
    /// Uses local variance to decide if detail worth capturing
    pub fn should_add_texture(
        &self,
        image: &ImageBuffer<f32>,
        variance_threshold: f32,
    ) -> bool {
        // Sample region variance
        let variance = self.compute_local_variance(image);
        variance > variance_threshold
    }

    /// Compute variance in image region covered by Gaussian
    fn compute_local_variance(&self, image: &ImageBuffer<f32>) -> f32 {
        let cx = (self.gaussian.position.x * image.width as f32) as u32;
        let cy = (self.gaussian.position.y * image.height as f32) as u32;
        let radius = (self.gaussian.shape.scale_x * image.width as f32 * 3.0) as u32;

        let x_start = cx.saturating_sub(radius);
        let x_end = (cx + radius).min(image.width);
        let y_start = cy.saturating_sub(radius);
        let y_end = (cy + radius).min(image.height);

        let mut mean = 0.0;
        let mut count = 0.0;

        // Compute mean
        for y in y_start..y_end {
            for x in x_start..x_end {
                if let Some(pixel) = image.get_pixel(x, y) {
                    mean += (pixel.r + pixel.g + pixel.b) / 3.0;
                    count += 1.0;
                }
            }
        }

        if count > 0.0 {
            mean /= count;
        }

        // Compute variance
        let mut variance = 0.0;
        for y in y_start..y_end {
            for x in x_start..x_end {
                if let Some(pixel) = image.get_pixel(x, y) {
                    let val = (pixel.r + pixel.g + pixel.b) / 3.0;
                    let diff = val - mean;
                    variance += diff * diff;
                }
            }
        }

        if count > 0.0 {
            variance / count
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_textured_gaussian_creation() {
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.1),
            Color4::new(1.0, 0.0, 0.0, 1.0),
            1.0,
        );

        let textured = TexturedGaussian2D::from_gaussian(gaussian);
        assert!(textured.texture.is_none());
    }

    #[test]
    fn test_local_coordinates() {
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.1),
            Color4::new(1.0, 1.0, 1.0, 1.0),
            1.0,
        );

        let textured = TexturedGaussian2D::from_gaussian(gaussian);

        // Center should map to (0, 0)
        let local = textured.world_to_local(Vector2::new(0.5, 0.5));
        assert!((local.x).abs() < 0.01);
        assert!((local.y).abs() < 0.01);

        // Edge of 3σ should map to ~1.0
        let edge = textured.world_to_local(Vector2::new(0.5 + 0.3, 0.5));
        assert!(edge.x.abs() > 0.9);
    }
}
