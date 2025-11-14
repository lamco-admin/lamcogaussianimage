//! Gaussian merging for flat regions

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};

pub struct GaussianMerger {
    pub merge_distance_threshold: f32,
    pub color_similarity_threshold: f32,
}

impl Default for GaussianMerger {
    fn default() -> Self {
        Self {
            merge_distance_threshold: 0.02,
            color_similarity_threshold: 0.1,
        }
    }
}

impl GaussianMerger {
    pub fn merge_pair(
        &self,
        g1: &Gaussian2D<f32, Euler<f32>>,
        g2: &Gaussian2D<f32, Euler<f32>>,
    ) -> Gaussian2D<f32, Euler<f32>> {
        // Weighted average by opacity
        let w1 = g1.opacity;
        let w2 = g2.opacity;
        let total_w = w1 + w2;

        let pos = Vector2::new(
            (g1.position.x * w1 + g2.position.x * w2) / total_w,
            (g1.position.y * w1 + g2.position.y * w2) / total_w,
        );

        let scale_x = (g1.shape.scale_x * w1 + g2.shape.scale_x * w2) / total_w;
        let scale_y = (g1.shape.scale_y * w1 + g2.shape.scale_y * w2) / total_w;

        let color_r = (g1.color.r * w1 + g2.color.r * w2) / total_w;
        let color_g = (g1.color.g * w1 + g2.color.g * w2) / total_w;
        let color_b = (g1.color.b * w1 + g2.color.b * w2) / total_w;

        Gaussian2D::new(
            pos,
            Euler::new(scale_x, scale_y, 0.0),
            lgi_math::color::Color4::new(color_r, color_g, color_b, 1.0),
            total_w.min(1.0),
        )
    }

    pub fn can_merge(&self, g1: &Gaussian2D<f32, Euler<f32>>, g2: &Gaussian2D<f32, Euler<f32>>) -> bool {
        let dist = ((g1.position.x - g2.position.x).powi(2) + (g1.position.y - g2.position.y).powi(2)).sqrt();
        let color_diff = ((g1.color.r - g2.color.r).powi(2) + (g1.color.g - g2.color.g).powi(2) + (g1.color.b - g2.color.b).powi(2)).sqrt();

        dist < self.merge_distance_threshold && color_diff < self.color_similarity_threshold
    }
}
