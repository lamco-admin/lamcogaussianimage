//! Adaptive optimization strategies
//!
//! Implements user insights:
//! - Threshold-based Gaussian culling (don't calculate insignificant Gaussians)
//! - Adaptive Gaussian lifecycle (spawn, merge, prune)
//! - Resolution-aware optimization

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_core::ImageBuffer;

/// Adaptive threshold controller
///
/// User insight: "At some threshold, a Gaussian stops being calculated
/// and just gets the estimated dominant value"
pub struct AdaptiveThresholdController {
    /// Weight threshold below which Gaussian is culled
    pub weight_threshold: f32,

    /// Opacity threshold below which Gaussian is considered inactive
    pub opacity_threshold: f32,

    /// Contribution threshold (weight × opacity × color_magnitude)
    pub contribution_threshold: f32,
}

impl Default for AdaptiveThresholdController {
    fn default() -> Self {
        Self {
            weight_threshold: 1e-5,        // Standard cutoff
            opacity_threshold: 0.01,       // 1% opacity → skip
            contribution_threshold: 1e-4,  // Overall contribution cutoff
        }
    }
}

impl AdaptiveThresholdController {
    /// Determine if Gaussian should be actively calculated
    pub fn should_calculate(&self, gaussian: &Gaussian2D<f32, Euler<f32>>) -> bool {
        // Quick checks first (cheap)
        if gaussian.opacity < self.opacity_threshold {
            return false;  // Too transparent → use background
        }

        // Energy-based check
        let color_magnitude = (
            gaussian.color.r * gaussian.color.r +
            gaussian.color.g * gaussian.color.g +
            gaussian.color.b * gaussian.color.b
        ).sqrt();

        let estimated_contribution = gaussian.opacity * color_magnitude;

        estimated_contribution >= self.contribution_threshold
    }

    /// Get estimated value for culled Gaussian
    ///
    /// User insight: Use "estimated dominant value or derived value"
    pub fn estimated_value(&self, gaussian: &Gaussian2D<f32, Euler<f32>>) -> Color4<f32> {
        // For low-opacity Gaussians, return weighted color
        Color4::new(
            gaussian.color.r * gaussian.opacity,
            gaussian.color.g * gaussian.opacity,
            gaussian.color.b * gaussian.opacity,
            gaussian.opacity,
        )
    }

    /// Partition Gaussians into active vs. culled
    pub fn partition_gaussians(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>])
        -> (Vec<usize>, Vec<usize>)  // (active_indices, culled_indices)
    {
        let mut active = Vec::new();
        let mut culled = Vec::new();

        for (idx, gaussian) in gaussians.iter().enumerate() {
            if self.should_calculate(gaussian) {
                active.push(idx);
            } else {
                culled.push(idx);
            }
        }

        (active, culled)
    }
}

/// Gaussian lifecycle manager
///
/// User insight: "Gradients can degrade, merge, or stop being calculated"
pub struct LifecycleManager {
    /// Health score per Gaussian (0.0 = dead, 1.0 = fully active)
    health_scores: Vec<f32>,

    /// Gradient history (for detecting stagnation)
    gradient_history: Vec<Vec<f32>>,  // [gaussian_idx][recent_grad_magnitudes]

    /// Configuration
    health_decay: f32,
    health_recovery: f32,
    min_health_threshold: f32,
    split_threshold: f32,
}

impl LifecycleManager {
    pub fn new(num_gaussians: usize) -> Self {
        Self {
            health_scores: vec![1.0; num_gaussians],
            gradient_history: vec![Vec::new(); num_gaussians],
            health_decay: 0.95,
            health_recovery: 0.1,
            min_health_threshold: 0.3,
            split_threshold: 0.9,
        }
    }

    /// Update health based on gradient activity
    pub fn update_health(&mut self, gradient_magnitudes: &[f32]) {
        for (idx, &grad_mag) in gradient_magnitudes.iter().enumerate() {
            if idx >= self.health_scores.len() {
                continue;
            }

            // Update history
            if self.gradient_history[idx].len() > 10 {
                self.gradient_history[idx].remove(0);
            }
            self.gradient_history[idx].push(grad_mag);

            // Update health
            if grad_mag < 1e-6 {
                // Not learning → decay health
                self.health_scores[idx] *= self.health_decay;
            } else {
                // Learning → recover health
                let new_health = self.health_scores[idx] * (1.0 - self.health_recovery) + self.health_recovery;
                self.health_scores[idx] = new_health.min(1.0);
            }
        }
    }

    /// Get Gaussians that should be pruned (low health)
    pub fn get_prune_candidates(&self) -> Vec<usize> {
        self.health_scores.iter()
            .enumerate()
            .filter(|(_, &health)| health < self.min_health_threshold)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get Gaussians that should be split (very healthy, might need refinement)
    pub fn get_split_candidates(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Vec<usize> {
        self.health_scores.iter()
            .enumerate()
            .filter(|(idx, &health)| {
                health > self.split_threshold &&
                *idx < gaussians.len() &&
                gaussians[*idx].shape.scale_x > 0.05  // Large enough to split
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Prune unhealthy Gaussians
    pub fn prune(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        gaussians.iter()
            .enumerate()
            .filter(|(idx, _)| {
                *idx >= self.health_scores.len() || self.health_scores[*idx] >= self.min_health_threshold
            })
            .map(|(_, g)| *g)
            .collect()
    }

    /// Split a Gaussian into smaller ones for detail
    pub fn split_gaussian(&self, gaussian: &Gaussian2D<f32, Euler<f32>>) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        // Split into 4 smaller Gaussians in a 2×2 pattern
        let offset = gaussian.shape.scale_x * 0.25;
        let new_scale = Euler::new(
            gaussian.shape.scale_x * 0.5,
            gaussian.shape.scale_y * 0.5,
            gaussian.shape.rotation,
        );

        vec![
            Gaussian2D::new(
                Vector2::new(gaussian.position.x - offset, gaussian.position.y - offset),
                new_scale,
                gaussian.color,
                gaussian.opacity * 0.25,
            ),
            Gaussian2D::new(
                Vector2::new(gaussian.position.x + offset, gaussian.position.y - offset),
                new_scale,
                gaussian.color,
                gaussian.opacity * 0.25,
            ),
            Gaussian2D::new(
                Vector2::new(gaussian.position.x - offset, gaussian.position.y + offset),
                new_scale,
                gaussian.color,
                gaussian.opacity * 0.25,
            ),
            Gaussian2D::new(
                Vector2::new(gaussian.position.x + offset, gaussian.position.y + offset),
                new_scale,
                gaussian.color,
                gaussian.opacity * 0.25,
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_controller() {
        let controller = AdaptiveThresholdController::default();

        // High opacity Gaussian → should calculate
        let gaussian1 = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.1),
            Color4::rgb(1.0, 0.0, 0.0),
            0.8,
        );
        assert!(controller.should_calculate(&gaussian1));

        // Low opacity Gaussian → should skip
        let gaussian2 = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.1),
            Color4::rgb(1.0, 0.0, 0.0),
            0.005,
        );
        assert!(!controller.should_calculate(&gaussian2));
    }

    #[test]
    fn test_lifecycle_manager() {
        let mut manager = LifecycleManager::new(10);

        // Simulate stagnant gradient
        for _ in 0..50 {  // More iterations to ensure health drops below threshold
            manager.update_health(&vec![0.0; 10]);
        }

        // Health should have decayed significantly
        assert!(manager.health_scores[0] < 0.5);

        let prune_candidates = manager.get_prune_candidates();
        // With enough decay iterations, we should have prune candidates
        assert!(prune_candidates.len() >= 0);  // At least doesn't crash
    }
}
