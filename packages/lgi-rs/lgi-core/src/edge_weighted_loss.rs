//! Edge-Weighted Loss Function
//! 3× weight at edges for sharper boundaries

use crate::{ImageBuffer, structure_tensor::StructureTensorField};

/// Edge-weighted loss computer
pub struct EdgeWeightedLoss {
    /// Weight multiplier at edges (3.0 typical)
    pub edge_weight: f32,

    /// Coherence threshold (>= this is edge)
    pub edge_threshold: f32,
}

impl Default for EdgeWeightedLoss {
    fn default() -> Self {
        Self {
            edge_weight: 3.0,
            edge_threshold: 0.5,
        }
    }
}

impl EdgeWeightedLoss {
    pub fn new(edge_weight: f32, edge_threshold: f32) -> Self {
        Self { edge_weight, edge_threshold }
    }

    /// Compute edge-weighted L2 loss
    pub fn compute_loss(
        &self,
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
        tensor_field: &StructureTensorField,
    ) -> f32 {
        let mut weighted_error = 0.0;
        let mut total_weight = 0.0;

        for y in 0..rendered.height {
            for x in 0..rendered.width {
                if let (Some(r), Some(t)) = (rendered.get_pixel(x, y), target.get_pixel(x, y)) {
                    // Get structure tensor at this pixel
                    let tensor = tensor_field.get(x, y);

                    // Weight based on coherence (edges get higher weight)
                    let weight = if tensor.coherence >= self.edge_threshold {
                        self.edge_weight  // Edge pixel
                    } else {
                        1.0  // Flat region
                    };

                    // Per-channel error
                    let error = (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);

                    weighted_error += weight * error;
                    total_weight += weight;
                }
            }
        }

        weighted_error / total_weight
    }

    /// Compute per-pixel weight map (for gradient computation)
    pub fn compute_weight_map(
        &self,
        width: u32,
        height: u32,
        tensor_field: &StructureTensorField,
    ) -> Vec<f32> {
        let mut weights = Vec::with_capacity((width * height) as usize);

        for y in 0..height {
            for x in 0..width {
                let tensor = tensor_field.get(x, y);
                let weight = if tensor.coherence >= self.edge_threshold {
                    self.edge_weight
                } else {
                    1.0
                };
                weights.push(weight);
            }
        }

        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_weighted_loss_creation() {
        let loss = EdgeWeightedLoss::default();
        assert_eq!(loss.edge_weight, 3.0);
        assert_eq!(loss.edge_threshold, 0.5);
    }
}
