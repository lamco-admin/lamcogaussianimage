//! Gaussian splitting for error-driven refinement

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use crate::ImageBuffer;

pub struct GaussianSplitter {
    pub split_threshold: f32,  // Error threshold for splitting
}

impl Default for GaussianSplitter {
    fn default() -> Self {
        Self { split_threshold: 0.01 }
    }
}

impl GaussianSplitter {
    pub fn split_gaussian(
        &self,
        gaussian: &Gaussian2D<f32, Euler<f32>>,
        error_map: &[f32],
        image_width: u32,
        image_height: u32,
    ) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        let mut children = Vec::new();

        // Split along major axis
        let offset = gaussian.shape.scale_x * 0.5;
        let cos_t = gaussian.shape.rotation.cos();
        let sin_t = gaussian.shape.rotation.sin();

        // Child 1: offset in +major direction
        let pos1 = Vector2::new(
            gaussian.position.x + offset * cos_t,
            gaussian.position.y + offset * sin_t,
        );

        // Child 2: offset in -major direction
        let pos2 = Vector2::new(
            gaussian.position.x - offset * cos_t,
            gaussian.position.y - offset * sin_t,
        );

        // Each child is smaller
        let child_scale_x = gaussian.shape.scale_x * 0.7;
        let child_scale_y = gaussian.shape.scale_y * 0.7;

        children.push(Gaussian2D::new(
            pos1,
            Euler::new(child_scale_x, child_scale_y, gaussian.shape.rotation),
            gaussian.color,
            gaussian.opacity,
        ));

        children.push(Gaussian2D::new(
            pos2,
            Euler::new(child_scale_x, child_scale_y, gaussian.shape.rotation),
            gaussian.color,
            gaussian.opacity,
        ));

        children
    }
}
