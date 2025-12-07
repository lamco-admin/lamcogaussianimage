//! Iterative Refinement with Artifact Feedback
//!
//! Analyzes intermediate encoding results and suggests refinement actions:
//! - Add Gaussians where error is high
//! - Prune Gaussians with low contribution
//! - Adjust shapes where artifacts are detected
//!
//! # Refinement Loop
//!
//! ```text
//! Initialize → Optimize → Analyze → Refine → Repeat
//!                 ↑__________________________|
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use lgi_core::iterative_refinement::{IterativeRefiner, RefinementConfig};
//!
//! let refiner = IterativeRefiner::new(RefinementConfig::default());
//!
//! // After optimization phase
//! let actions = refiner.analyze_and_suggest(&gaussians, &rendered, &target, &analysis)?;
//!
//! for action in actions {
//!     match action {
//!         RefinementAction::AddGaussians { positions, config } => { ... }
//!         RefinementAction::PruneGaussian { index } => { ... }
//!         RefinementAction::AdjustShape { index, new_params } => { ... }
//!     }
//! }
//! ```

use crate::{
    ImageBuffer,
    feature_classifier::{FeatureType, ClassificationResult},
    artifact_detector::{ArtifactDetector, ArtifactReport},
    image_analysis::ImageAnalysis,
};
use lgi_math::{
    gaussian::Gaussian2D,
    parameterization::Euler,
    vec::Vector2,
    color::Color4,
};

/// Refinement configuration
#[derive(Debug, Clone)]
pub struct RefinementConfig {
    /// Error threshold to trigger densification (MSE)
    pub error_threshold: f32,

    /// Contribution threshold to trigger pruning
    pub contribution_threshold: f32,

    /// Maximum Gaussians to add per refinement
    pub max_add_per_iter: usize,

    /// Maximum Gaussians to prune per refinement
    pub max_prune_per_iter: usize,

    /// Minimum error reduction to continue refinement
    pub min_improvement: f32,

    /// Bleeding threshold to trigger shape adjustment
    pub bleeding_threshold: f32,

    /// Blur threshold to trigger densification
    pub blur_threshold: f32,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            error_threshold: 0.02,
            contribution_threshold: 0.001,
            max_add_per_iter: 32,
            max_prune_per_iter: 16,
            min_improvement: 0.1,
            bleeding_threshold: 0.2,
            blur_threshold: 0.3,
        }
    }
}

/// Refinement action to take
#[derive(Debug, Clone)]
pub enum RefinementAction {
    /// Add new Gaussians at specified positions
    AddGaussians {
        positions: Vec<(f32, f32)>,
        scale: f32,
        feature_type: FeatureType,
    },

    /// Remove Gaussian at index (low contribution)
    PruneGaussian {
        index: usize,
        reason: String,
    },

    /// Adjust Gaussian shape (anti-bleeding)
    AdjustShape {
        index: usize,
        new_scale_x: f32,
        new_scale_y: f32,
        new_angle: f32,
        reason: String,
    },

    /// Split large Gaussian into smaller ones
    SplitGaussian {
        index: usize,
        into_count: usize,
    },

    /// No action needed
    NoAction {
        reason: String,
    },
}

/// Error analysis per region
#[derive(Debug, Clone)]
pub struct ErrorAnalysis {
    /// Per-feature-type average error
    pub error_by_feature: Vec<(FeatureType, f32)>,

    /// High-error regions (x, y, error, feature_type)
    pub high_error_regions: Vec<(u32, u32, f32, FeatureType)>,

    /// Global MSE
    pub global_mse: f32,

    /// PSNR
    pub psnr: f32,
}

/// Iterative refinement analyzer
pub struct IterativeRefiner {
    /// Configuration
    pub config: RefinementConfig,

    /// Artifact detector
    detector: ArtifactDetector,
}

impl IterativeRefiner {
    /// Create new refiner with config
    pub fn new(config: RefinementConfig) -> Self {
        Self {
            config,
            detector: ArtifactDetector::default(),
        }
    }

    /// Analyze current state and suggest refinement actions
    pub fn analyze_and_suggest(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
        analysis: &ImageAnalysis,
    ) -> crate::Result<Vec<RefinementAction>> {
        let mut actions = Vec::new();

        // Step 1: Error analysis
        let error = self.analyze_error(rendered, target, &analysis.classification);

        // Step 2: Artifact analysis
        let artifacts = self.detector.analyze(rendered, target)?;

        // Step 3: Contribution analysis
        let contributions = self.estimate_contributions(gaussians, rendered, target);

        // Step 4: Generate actions based on analysis

        // 4a: Densification for high-error regions
        if error.global_mse > self.config.error_threshold {
            let positions = self.select_densification_positions(&error, analysis);
            if !positions.is_empty() {
                let feature_type = error.error_by_feature.iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(ft, _)| *ft)
                    .unwrap_or(FeatureType::Texture);

                let (scale_min, scale_max) = feature_type.scale_range();
                let scale = (scale_min + scale_max) / 2.0 * analysis.width.max(analysis.height) as f32;

                actions.push(RefinementAction::AddGaussians {
                    positions,
                    scale,
                    feature_type,
                });
            }
        }

        // 4b: Bleeding → reduce sigma in affected Gaussians
        if artifacts.bleeding_score > self.config.bleeding_threshold {
            let bleeding_adjustments = self.find_bleeding_adjustments(
                gaussians,
                &artifacts,
                analysis,
            );
            actions.extend(bleeding_adjustments);
        }

        // 4c: Blur → add smaller Gaussians
        if artifacts.blur_score > self.config.blur_threshold {
            let blur_positions = self.select_blur_positions(&artifacts, analysis);
            if !blur_positions.is_empty() {
                actions.push(RefinementAction::AddGaussians {
                    positions: blur_positions,
                    scale: 0.01 * analysis.width.max(analysis.height) as f32,
                    feature_type: FeatureType::Edge,
                });
            }
        }

        // 4d: Pruning for low-contribution Gaussians
        let prune_indices: Vec<usize> = contributions.iter()
            .enumerate()
            .filter(|(_, &c)| c < self.config.contribution_threshold)
            .take(self.config.max_prune_per_iter)
            .map(|(i, _)| i)
            .collect();

        for idx in prune_indices {
            actions.push(RefinementAction::PruneGaussian {
                index: idx,
                reason: "Low contribution".to_string(),
            });
        }

        // If no actions needed
        if actions.is_empty() {
            actions.push(RefinementAction::NoAction {
                reason: format!("PSNR={:.2}dB, no significant issues", error.psnr),
            });
        }

        Ok(actions)
    }

    /// Analyze error distribution
    fn analyze_error(
        &self,
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
        classification: &ClassificationResult,
    ) -> ErrorAnalysis {
        let width = target.width;
        let height = target.height;

        // Per-pixel squared error
        let mut total_se = 0.0;
        let mut feature_errors: std::collections::HashMap<FeatureType, (f32, usize)> =
            std::collections::HashMap::new();
        let mut high_error_regions = Vec::new();

        for y in 0..height {
            for x in 0..width {
                let t = target.get_pixel(x, y).unwrap();
                let r = rendered.get_pixel(x, y).unwrap();

                let se = (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
                total_se += se;

                let feature = classification.get(x, y);
                let entry = feature_errors.entry(feature).or_insert((0.0, 0));
                entry.0 += se;
                entry.1 += 1;

                // Track high-error pixels
                if se > self.config.error_threshold * 3.0 {
                    high_error_regions.push((x, y, se, feature));
                }
            }
        }

        let n_pixels = (width * height) as f32;
        let global_mse = total_se / (n_pixels * 3.0);
        let psnr = if global_mse > 0.0 {
            10.0 * (1.0 / global_mse).log10()
        } else {
            100.0
        };

        // Sort high error regions
        high_error_regions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        high_error_regions.truncate(100);

        // Feature error averages
        let mut error_by_feature: Vec<(FeatureType, f32)> = feature_errors.into_iter()
            .map(|(ft, (sum, count))| (ft, if count > 0 { sum / count as f32 } else { 0.0 }))
            .collect();
        error_by_feature.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        ErrorAnalysis {
            error_by_feature,
            high_error_regions,
            global_mse,
            psnr,
        }
    }

    /// Estimate contribution of each Gaussian
    fn estimate_contributions(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> Vec<f32> {
        // Simple approximation: opacity × scale × (1 - distance_from_nearest_high_error)
        // Full implementation would require rendering with each Gaussian removed

        let width = rendered.width;
        let height = rendered.height;

        gaussians.iter().map(|g| {
            let _pos = g.position;
            let scale_x = g.shape.scale_x;
            let scale_y = g.shape.scale_y;
            let opacity = g.opacity;

            // Approximate contribution by coverage × opacity
            let area = std::f32::consts::PI * scale_x * scale_y;
            let coverage = area / (width * height) as f32;

            coverage * opacity
        }).collect()
    }

    /// Select positions for densification based on error
    fn select_densification_positions(
        &self,
        error: &ErrorAnalysis,
        analysis: &ImageAnalysis,
    ) -> Vec<(f32, f32)> {
        let count = self.config.max_add_per_iter.min(error.high_error_regions.len());

        error.high_error_regions.iter()
            .take(count)
            .map(|(x, y, _, _)| (*x as f32, *y as f32))
            .collect()
    }

    /// Find Gaussians that need shape adjustment for bleeding
    fn find_bleeding_adjustments(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        artifacts: &ArtifactReport,
        analysis: &ImageAnalysis,
    ) -> Vec<RefinementAction> {
        let mut actions = Vec::new();

        for (i, g) in gaussians.iter().enumerate() {
            let pos = g.position;
            let px = (pos.x as u32).min(analysis.width - 1);
            let py = (pos.y as u32).min(analysis.height - 1);
            let idx = (py * analysis.width + px) as usize;

            // Check if this Gaussian is in a bleeding region
            let bleeding_intensity = artifacts.bleeding_map.get(idx).copied().unwrap_or(0.0);

            if bleeding_intensity > self.config.bleeding_threshold {
                // Reduce sigma based on geodesic distance
                let edge_dist = analysis.get_edge_distance(px, py);
                let max_scale = 0.3 + 0.2 * edge_dist;

                let scale_x = g.shape.scale_x;
                let scale_y = g.shape.scale_y;
                let current_max = scale_x.max(scale_y);
                let scale_unit = analysis.width.max(analysis.height) as f32;

                if current_max > max_scale * scale_unit {
                    let reduction = max_scale * scale_unit / current_max;
                    actions.push(RefinementAction::AdjustShape {
                        index: i,
                        new_scale_x: scale_x * reduction,
                        new_scale_y: scale_y * reduction,
                        new_angle: g.shape.rotation,
                        reason: format!("Bleeding reduction at ({}, {})", px, py),
                    });
                }
            }
        }

        // Limit adjustments per iteration
        actions.truncate(self.config.max_prune_per_iter);
        actions
    }

    /// Select positions for blur densification
    fn select_blur_positions(
        &self,
        artifacts: &ArtifactReport,
        analysis: &ImageAnalysis,
    ) -> Vec<(f32, f32)> {
        let mut positions = Vec::new();

        for y in (0..analysis.height).step_by(10) {
            for x in (0..analysis.width).step_by(10) {
                let idx = (y * analysis.width + x) as usize;
                let blur_intensity = artifacts.blur_map.get(idx).copied().unwrap_or(0.0);

                if blur_intensity > self.config.blur_threshold {
                    positions.push((x as f32, y as f32));
                }
            }
        }

        positions.truncate(self.config.max_add_per_iter);
        positions
    }
}

/// Apply refinement action to Gaussian set
pub fn apply_action(
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    action: &RefinementAction,
    target: &ImageBuffer<f32>,
) {
    match action {
        RefinementAction::AddGaussians { positions, scale, feature_type } => {
            for &(x, y) in positions {
                let px = (x as u32).min(target.width - 1);
                let py = (y as u32).min(target.height - 1);
                let color = target.get_pixel(px, py).unwrap().clone();

                let (aniso_min, aniso_max) = feature_type.anisotropy_range();
                let anisotropy = (aniso_min + aniso_max) / 2.0;

                gaussians.push(Gaussian2D::new(
                    Vector2::new(x, y),
                    Euler::new(*scale, scale / anisotropy, 0.0),
                    color,
                    1.0,
                ));
            }
        }

        RefinementAction::PruneGaussian { index, .. } => {
            if *index < gaussians.len() {
                gaussians.remove(*index);
            }
        }

        RefinementAction::AdjustShape { index, new_scale_x, new_scale_y, new_angle, .. } => {
            if *index < gaussians.len() {
                let g = &mut gaussians[*index];
                g.shape = Euler::new(*new_scale_x, *new_scale_y, *new_angle);
            }
        }

        RefinementAction::SplitGaussian { index, into_count } => {
            if *index < gaussians.len() {
                let original = gaussians.remove(*index);
                let pos = original.position;
                let scale_x = original.shape.scale_x;
                let scale_y = original.shape.scale_y;
                let new_scale = scale_x / (*into_count as f32).sqrt();

                // Create split Gaussians in a small pattern
                let offsets = match into_count {
                    2 => vec![(-0.5, 0.0), (0.5, 0.0)],
                    3 => vec![(-0.5, -0.3), (0.5, -0.3), (0.0, 0.5)],
                    4 => vec![(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)],
                    _ => vec![(0.0, 0.0)],
                };

                for (dx, dy) in offsets {
                    gaussians.push(Gaussian2D::new(
                        Vector2::new(pos.x + dx * scale_x, pos.y + dy * scale_y),
                        Euler::new(new_scale, new_scale / (scale_x / scale_y), original.shape.rotation),
                        original.color.clone(),
                        original.opacity / *into_count as f32,
                    ));
                }
            }
        }

        RefinementAction::NoAction { .. } => {}
    }
}

/// Print refinement actions summary
pub fn print_actions(actions: &[RefinementAction]) {
    println!("Refinement Actions ({} total):", actions.len());

    let mut add_count = 0;
    let mut prune_count = 0;
    let mut adjust_count = 0;

    for action in actions {
        match action {
            RefinementAction::AddGaussians { positions, .. } => {
                add_count += positions.len();
            }
            RefinementAction::PruneGaussian { .. } => {
                prune_count += 1;
            }
            RefinementAction::AdjustShape { .. } => {
                adjust_count += 1;
            }
            RefinementAction::SplitGaussian { into_count, .. } => {
                add_count += into_count - 1;
            }
            RefinementAction::NoAction { reason } => {
                println!("  No action: {}", reason);
            }
        }
    }

    if add_count > 0 { println!("  Add: {} Gaussians", add_count); }
    if prune_count > 0 { println!("  Prune: {} Gaussians", prune_count); }
    if adjust_count > 0 { println!("  Adjust: {} shapes", adjust_count); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refiner_creation() {
        let refiner = IterativeRefiner::new(RefinementConfig::default());
        assert!(refiner.config.error_threshold > 0.0);
    }
}
