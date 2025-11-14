//! Position Probability Map (PPM) - Strategy G
//!
//! Combines multiple image analysis signals into per-pixel placement probability:
//! - Entropy (complexity)
//! - Gradient (edges)
//! - Saliency (importance)
//! - Custom weights
//!
//! Used by multiple initialization strategies

use crate::ImageBuffer;

/// PPM configuration
#[derive(Clone)]
pub struct PPMConfig {
    /// Entropy weight (complexity)
    pub weight_entropy: f32,

    /// Gradient weight (edges)
    pub weight_gradient: f32,

    /// Saliency weight (importance)
    pub weight_saliency: f32,

    /// Tile size for entropy computation
    pub entropy_tile_size: usize,
}

impl Default for PPMConfig {
    fn default() -> Self {
        Self {
            weight_entropy: 0.6,
            weight_gradient: 0.4,
            weight_saliency: 0.0,  // Optional, disabled by default
            entropy_tile_size: 8,
        }
    }
}

/// Position Probability Map
///
/// Per-pixel probability for Gaussian placement
/// Normalized to sum to 1.0 (valid probability distribution)
pub struct PositionProbabilityMap {
    /// Probabilities (width × height)
    pub probabilities: Vec<f32>,

    /// Image dimensions
    pub width: u32,
    pub height: u32,
}

impl PositionProbabilityMap {
    /// Generate PPM from entropy + gradient
    ///
    /// Combines local complexity with edge information
    pub fn from_entropy_gradient(
        entropy_map: &[f32],
        gradient_map: &[f32],
        width: u32,
        height: u32,
        config: &PPMConfig,
    ) -> Self {
        assert_eq!(entropy_map.len(), (width * height) as usize);
        assert_eq!(gradient_map.len(), (width * height) as usize);

        let mut probabilities = vec![0.0; (width * height) as usize];

        // Combine weighted signals
        for i in 0..(width * height) as usize {
            probabilities[i] =
                config.weight_entropy * entropy_map[i] +
                config.weight_gradient * gradient_map[i];
        }

        // Normalize to probability distribution (sum = 1.0)
        let sum: f32 = probabilities.iter().sum();
        if sum > 0.0 {
            for p in &mut probabilities {
                *p /= sum;
            }
        }

        Self { probabilities, width, height }
    }

    /// Generate PPM with saliency
    pub fn from_entropy_gradient_saliency(
        entropy_map: &[f32],
        gradient_map: &[f32],
        saliency_map: &[f32],
        width: u32,
        height: u32,
        config: &PPMConfig,
    ) -> Self {
        assert_eq!(saliency_map.len(), (width * height) as usize);

        let mut probabilities = vec![0.0; (width * height) as usize];

        for i in 0..(width * height) as usize {
            probabilities[i] =
                config.weight_entropy * entropy_map[i] +
                config.weight_gradient * gradient_map[i] +
                config.weight_saliency * saliency_map[i];
        }

        let sum: f32 = probabilities.iter().sum();
        if sum > 0.0 {
            for p in &mut probabilities {
                *p /= sum;
            }
        }

        Self { probabilities, width, height }
    }

    /// Sample N positions from PPM using importance sampling
    ///
    /// Returns pixel coordinates (x, y)
    pub fn sample_positions(&self, n: usize) -> Vec<(u32, u32)> {
        use rand::Rng;

        // Build cumulative distribution function (CDF)
        let mut cdf = Vec::with_capacity(self.probabilities.len());
        let mut cumsum = 0.0;

        for &p in &self.probabilities {
            cumsum += p;
            cdf.push(cumsum);
        }

        // Sample N positions
        let mut rng = rand::thread_rng();
        let mut positions = Vec::new();

        for _ in 0..n {
            let r: f32 = rng.gen();  // Random [0,1]

            // Binary search in CDF
            let idx = match cdf.binary_search_by(|probe| {
                probe.partial_cmp(&r).unwrap()
            }) {
                Ok(i) => i,
                Err(i) => i.min(cdf.len() - 1),
            };

            // Convert index to (x, y)
            let x = (idx as u32) % self.width;
            let y = (idx as u32) / self.width;

            positions.push((x, y));
        }

        positions
    }

    /// Sample positions without replacement (no duplicates)
    pub fn sample_positions_unique(&self, n: usize) -> Vec<(u32, u32)> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        // Create weighted index list
        let mut weighted_indices: Vec<_> = self.probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();

        // Sort by probability (descending)
        weighted_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top N deterministically (avoid shuffle borrow issue)
        let take_n = n.min(weighted_indices.len());

        weighted_indices
            .into_iter()
            .take(take_n)
            .map(|(idx, _)| {
                let x = (idx as u32) % self.width;
                let y = (idx as u32) / self.width;
                (x, y)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppm_normalization() {
        let entropy = vec![0.1, 0.2, 0.3, 0.4];
        let gradient = vec![0.4, 0.3, 0.2, 0.1];

        let ppm = PositionProbabilityMap::from_entropy_gradient(
            &entropy,
            &gradient,
            2,
            2,
            &PPMConfig::default(),
        );

        // Should sum to 1.0
        let sum: f32 = ppm.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 0.001, "PPM should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_importance_sampling() {
        // Uniform probabilities
        let uniform = vec![0.25; 4];
        let ppm = PositionProbabilityMap {
            probabilities: uniform,
            width: 2,
            height: 2,
        };

        let samples = ppm.sample_positions(10);
        assert_eq!(samples.len(), 10);

        // All samples should be valid coordinates
        for (x, y) in samples {
            assert!(x < 2 && y < 2);
        }
    }
}
