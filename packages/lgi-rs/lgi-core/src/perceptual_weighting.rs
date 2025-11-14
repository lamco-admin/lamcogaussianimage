//! Perceptual weighting for loss functions

use crate::ImageBuffer;

pub struct PerceptualWeighter {
    pub use_ms_ssim: bool,
    pub use_edge_weight: bool,
    pub use_saliency: bool,
}

impl Default for PerceptualWeighter {
    fn default() -> Self {
        Self {
            use_ms_ssim: true,
            use_edge_weight: true,
            use_saliency: false,
        }
    }
}

impl PerceptualWeighter {
    pub fn compute_weight_map(
        &self,
        image: &ImageBuffer<f32>,
        edge_map: Option<&[f32]>,
        saliency_map: Option<&[f32]>,
    ) -> Vec<f32> {
        let size = (image.width * image.height) as usize;
        let mut weights = vec![1.0; size];

        if self.use_edge_weight {
            if let Some(edges) = edge_map {
                for (w, &e) in weights.iter_mut().zip(edges.iter()) {
                    if e > 0.5 {
                        *w *= 3.0;  // 3× weight at edges
                    }
                }
            }
        }

        if self.use_saliency {
            if let Some(saliency) = saliency_map {
                for (w, &s) in weights.iter_mut().zip(saliency.iter()) {
                    *w *= 1.0 + s;  // Boost salient regions
                }
            }
        }

        weights
    }
}
