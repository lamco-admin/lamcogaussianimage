//! Text and stroke detection for special handling

use crate::{ImageBuffer, structure_tensor::StructureTensorField};

pub struct TextStrokeDetector {
    pub coherence_threshold: f32,  // > 0.8 for strokes
    pub width_max: f32,  // < 2 pixels for text
}

impl Default for TextStrokeDetector {
    fn default() -> Self {
        Self {
            coherence_threshold: 0.8,
            width_max: 2.0,
        }
    }
}

impl TextStrokeDetector {
    pub fn detect_strokes(
        &self,
        image: &ImageBuffer<f32>,
        tensor_field: &StructureTensorField,
    ) -> Vec<bool> {
        let mut is_stroke = vec![false; (image.width * image.height) as usize];

        for y in 0..image.height {
            for x in 0..image.width {
                let tensor = tensor_field.get(x, y);

                // High coherence + thin = stroke
                if tensor.coherence > self.coherence_threshold {
                    let minor_eigenvalue = tensor.eigenvalue_minor;
                    let width_estimate = minor_eigenvalue.sqrt() * 2.0;  // Approximate width

                    if width_estimate < self.width_max {
                        is_stroke[(y * image.width + x) as usize] = true;
                    }
                }
            }
        }

        is_stroke
    }

    pub fn create_stroke_preset(
        &self,
        position: (f32, f32),
        tangent_angle: f32,
        color: (f32, f32, f32),
    ) -> lgi_math::gaussian::Gaussian2D<f32, lgi_math::parameterization::Euler<f32>> {
        use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

        // Text preset from Debug Plan: σ⊥=0.3px, σ∥=5px
        Gaussian2D::new(
            Vector2::new(position.0, position.1),
            Euler::new(5.0, 0.3, tangent_angle),  // Long along stroke, thin across
            Color4::new(color.0, color.1, color.2, 1.0),
            1.0,
        )
    }
}
