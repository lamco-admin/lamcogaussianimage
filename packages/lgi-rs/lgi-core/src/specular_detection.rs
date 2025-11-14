//! Specular highlight detection via log-luminance

use crate::ImageBuffer;

pub struct SpecularDetector {
    pub percentile: f32,  // e.g., 95th percentile
}

impl Default for SpecularDetector {
    fn default() -> Self {
        Self { percentile: 0.95 }
    }
}

impl SpecularDetector {
    pub fn detect_speculars(&self, image: &ImageBuffer<f32>) -> Vec<bool> {
        // Compute log-luminance
        let mut luminances: Vec<f32> = image.data.iter()
            .map(|c| {
                let y = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
                (y + 1e-6).ln()
            })
            .collect();

        // Find threshold
        let mut sorted = luminances.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_idx = (sorted.len() as f32 * self.percentile) as usize;
        let threshold = sorted[threshold_idx.min(sorted.len() - 1)];

        // Mark speculars
        luminances.iter().map(|&l| l > threshold).collect()
    }
}
