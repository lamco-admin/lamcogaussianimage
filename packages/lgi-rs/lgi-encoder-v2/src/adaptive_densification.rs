//! Adaptive Densification (from 3D Gaussian Splatting)
//!
//! Instead of just ADDING Gaussians at hotspots:
//! - SPLIT large Gaussians in high-error regions
//! - CLONE small Gaussians in high-error regions
//! - PRUNE low-opacity Gaussians
//!
//! This is Strategy U from research (Kerbl et al. 2023)

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use crate::renderer_v2::RendererV2;

/// Adaptive densification controller
pub struct AdaptiveDensifier {
    /// Gradient magnitude threshold for densification
    pub densify_grad_threshold: f32,

    /// Scale threshold for split vs clone decision
    pub split_scale_threshold: f32,

    /// Opacity threshold for pruning
    pub prune_opacity_threshold: f32,

    /// How often to densify (every N iterations)
    pub densify_interval: usize,

    /// How often to prune (every N iterations)
    pub prune_interval: usize,
}

impl Default for AdaptiveDensifier {
    fn default() -> Self {
        Self {
            densify_grad_threshold: 0.0002,  // Tune empirically
            split_scale_threshold: 0.05,      // If σ > 0.05, split; else clone
            prune_opacity_threshold: 0.005,   // Remove if α < 0.005
            densify_interval: 100,            // Every 100 iters
            prune_interval: 300,              // Every 300 iters
        }
    }
}

impl AdaptiveDensifier {
    /// Adaptive densification: split large, clone small
    ///
    /// Returns: (new_gaussians, indices_to_remove)
    pub fn densify(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        gradient_magnitudes: &[f32],  // Per-Gaussian gradient magnitude
    ) -> (Vec<Gaussian2D<f32, Euler<f32>>>, Vec<usize>) {
        let mut to_add = Vec::new();
        let mut to_remove = Vec::new();

        for (i, gaussian) in gaussians.iter().enumerate() {
            let grad_mag = gradient_magnitudes[i];

            // Only densify if gradient is high
            if grad_mag < self.densify_grad_threshold {
                continue;
            }

            // Decision: Split or Clone?
            let scale_avg = (gaussian.shape.scale_x + gaussian.shape.scale_y) / 2.0;

            if scale_avg > self.split_scale_threshold {
                // LARGE Gaussian with high gradient → SPLIT into 2
                let (g1, g2) = self.split_gaussian(gaussian);
                to_add.push(g1);
                to_add.push(g2);
                to_remove.push(i);  // Remove parent
            } else {
                // SMALL Gaussian with high gradient → CLONE
                let clone = self.clone_gaussian(gaussian);
                to_add.push(clone);
                // Keep original
            }
        }

        (to_add, to_remove)
    }

    /// Split Gaussian along major axis
    fn split_gaussian(&self, gaussian: &Gaussian2D<f32, Euler<f32>>) -> (Gaussian2D<f32, Euler<f32>>, Gaussian2D<f32, Euler<f32>>) {
        // Split distance: move along major axis by σ/2
        let major_axis = Vector2::new(
            gaussian.shape.rotation.cos(),
            gaussian.shape.rotation.sin(),
        );

        let offset = major_axis * (gaussian.shape.scale_x / 2.0);

        // Child 1: Move in +major direction, reduce scale
        let g1 = Gaussian2D::new(
            Vector2::new(
                (gaussian.position.x + offset.x).clamp(0.0, 1.0),
                (gaussian.position.y + offset.y).clamp(0.0, 1.0),
            ),
            Euler::new(
                gaussian.shape.scale_x * 0.8,  // 80% of parent scale
                gaussian.shape.scale_y * 0.8,
                gaussian.shape.rotation,
            ),
            gaussian.color,
            gaussian.opacity * 0.5,  // Half opacity each
        );

        // Child 2: Move in -major direction, reduce scale
        let g2 = Gaussian2D::new(
            Vector2::new(
                (gaussian.position.x - offset.x).clamp(0.0, 1.0),
                (gaussian.position.y - offset.y).clamp(0.0, 1.0),
            ),
            Euler::new(
                gaussian.shape.scale_x * 0.8,
                gaussian.shape.scale_y * 0.8,
                gaussian.shape.rotation,
            ),
            gaussian.color,
            gaussian.opacity * 0.5,
        );

        (g1, g2)
    }

    /// Clone Gaussian with small perturbation
    fn clone_gaussian(&self, gaussian: &Gaussian2D<f32, Euler<f32>>) -> Gaussian2D<f32, Euler<f32>> {
        // Perturb position slightly (random direction, small distance)
        let angle = (gaussian.position.x * 1000.0) % (2.0 * std::f32::consts::PI);  // Pseudo-random
        let distance = gaussian.shape.scale_x * 0.1;  // 10% of scale

        let offset = Vector2::new(
            angle.cos() * distance,
            angle.sin() * distance,
        );

        Gaussian2D::new(
            Vector2::new(
                (gaussian.position.x + offset.x).clamp(0.0, 1.0),
                (gaussian.position.y + offset.y).clamp(0.0, 1.0),
            ),
            gaussian.shape.clone(),  // Same shape
            gaussian.color,
            gaussian.opacity,  // Same opacity
        )
    }

    /// Prune low-opacity Gaussians
    pub fn prune(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Vec<usize> {
        gaussians
            .iter()
            .enumerate()
            .filter_map(|(i, g)| {
                if g.opacity < self.prune_opacity_threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute per-Gaussian gradient magnitude (approximation)
    pub fn compute_gaussian_gradients(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> Vec<f32> {
        let mut gradient_mags = vec![0.0; gaussians.len()];

        // Approximate: Measure how much each Gaussian's position gradient would be
        // For now, use contribution to total error as proxy
        let rendered = RendererV2::render(gaussians, target.width, target.height);

        for (i, gaussian) in gaussians.iter().enumerate() {
            // Approximate gradient by measuring error in Gaussian's footprint
            let mut error_sum = 0.0;
            let mut count = 0;

            let px_x = (gaussian.position.x * target.width as f32) as u32;
            let px_y = (gaussian.position.y * target.height as f32) as u32;

            // Check 5×5 region around Gaussian center
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let x = (px_x as i32 + dx).clamp(0, target.width as i32 - 1) as u32;
                    let y = (px_y as i32 + dy).clamp(0, target.height as i32 - 1) as u32;

                    if let (Some(t), Some(r)) = (target.get_pixel(x, y), rendered.get_pixel(x, y)) {
                        let err = (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
                        error_sum += err;
                        count += 1;
                    }
                }
            }

            gradient_mags[i] = if count > 0 {
                (error_sum / count as f32).sqrt()
            } else {
                0.0
            };
        }

        gradient_mags
    }
}

/// Apply densification operations (split/clone/prune)
pub fn apply_densification(
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    to_add: Vec<Gaussian2D<f32, Euler<f32>>>,
    to_remove: Vec<usize>,
) {
    // Remove in reverse order to preserve indices
    let mut sorted_removes = to_remove;
    sorted_removes.sort_unstable_by(|a, b| b.cmp(a));  // Descending
    for &idx in &sorted_removes {
        if idx < gaussians.len() {
            gaussians.swap_remove(idx);
        }
    }

    // Add new Gaussians
    gaussians.extend(to_add);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_gaussian() {
        let densifier = AdaptiveDensifier::default();

        let original = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::new(0.1, 0.05, 0.0),  // Large, anisotropic
            Color4::new(1.0, 0.0, 0.0, 1.0),
            1.0,
        );

        let (g1, g2) = densifier.split_gaussian(&original);

        // Children should be offset along major axis
        assert_ne!(g1.position.x, g2.position.x);
        // Children should be smaller
        assert!(g1.shape.scale_x < original.shape.scale_x);
        // Opacity should sum to ~original
        assert!((g1.opacity + g2.opacity - original.opacity).abs() < 0.01);
    }

    #[test]
    fn test_prune() {
        let densifier = AdaptiveDensifier::default();

        let gaussians = vec![
            Gaussian2D::new(Vector2::new(0.5, 0.5), Euler::isotropic(0.05), Color4::white(), 0.1),  // Keep
            Gaussian2D::new(Vector2::new(0.3, 0.3), Euler::isotropic(0.05), Color4::white(), 0.001), // Prune
            Gaussian2D::new(Vector2::new(0.7, 0.7), Euler::isotropic(0.05), Color4::white(), 0.5),   // Keep
        ];

        let to_prune = densifier.prune(&gaussians);

        assert_eq!(to_prune.len(), 1);
        assert_eq!(to_prune[0], 1);  // Middle Gaussian (low opacity)
    }
}
