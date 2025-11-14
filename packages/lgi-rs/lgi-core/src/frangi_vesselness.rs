//! Frangi vesselness filter for tubular structures (medical, hair)

use crate::{ImageBuffer, structure_tensor::StructureTensorField};

pub struct FrangiFilter {
    pub alpha: f32,  // Plate-like structures (0.5)
    pub beta: f32,   // Background (15.0)
    pub c: f32,      // Second order structureness (500.0)
}

impl Default for FrangiFilter {
    fn default() -> Self {
        Self { alpha: 0.5, beta: 15.0, c: 500.0 }
    }
}

impl FrangiFilter {
    pub fn compute_vesselness(
        &self,
        tensor_field: &StructureTensorField,
    ) -> Vec<f32> {
        let size = (tensor_field.width * tensor_field.height) as usize;
        let mut vesselness = vec![0.0; size];

        for y in 0..tensor_field.height {
            for x in 0..tensor_field.width {
                let tensor = tensor_field.get(x, y);
                let l1 = tensor.eigenvalue_major;
                let l2 = tensor.eigenvalue_minor;

                if l2 < 0.0 {
                    vesselness[(y * tensor_field.width + x) as usize] = 0.0;
                    continue;
                }

                let rb = l1 / l2;  // Blobness
                let s = (l1.powi(2) + l2.powi(2)).sqrt();  // Second order structureness

                let v = (-rb.powi(2) / (2.0 * self.alpha.powi(2))).exp() *
                        (1.0 - (-s.powi(2) / (2.0 * self.c.powi(2))).exp());

                vesselness[(y * tensor_field.width + x) as usize] = v;
            }
        }

        vesselness
    }
}
