//! Progressive loading support for LGI format
//! Implements importance-based ordering for streaming decode

use lgi_math::{Float, gaussian::Gaussian2D, parameterization::Parameterization};

/// Importance metric for a Gaussian
#[derive(Debug, Clone, Copy)]
pub struct GaussianImportance {
    pub gaussian_index: usize,
    pub importance: f32,
}

/// Compute importance score for progressive ordering
/// Higher importance = should be transmitted/decoded first
pub fn compute_importance(
    gaussian: &Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>,
    canvas_width: u32,
    canvas_height: u32,
) -> f32 {
    // Importance factors:
    // 1. Coverage area (larger = more important)
    // 2. Opacity (higher = more important)
    // 3. Central position (closer to center = more important for typical viewing)

    let scale_x = gaussian.shape.scale_x;
    let scale_y = gaussian.shape.scale_y;
    let opacity = gaussian.opacity;

    // Coverage area (in normalized canvas space)
    let area = scale_x * scale_y * std::f32::consts::PI;

    // Opacity contribution
    let opacity_weight = opacity;

    // Position weight (center bias)
    let px = gaussian.position.x;
    let py = gaussian.position.y;
    let center_x = 0.5f32;
    let center_y = 0.5f32;
    let dist_from_center = ((px - center_x).powi(2) + (py - center_y).powi(2)).sqrt();
    let position_weight = 1.0f32 - dist_from_center.min(1.0f32);

    // Combined importance score
    // Large, opaque, centrally-located Gaussians are most important
    let importance = area * opacity_weight * (0.5f32 + 0.5f32 * position_weight);

    importance
}

/// Order Gaussians by importance for progressive transmission
pub fn order_by_importance(
    gaussians: &[Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>],
    canvas_width: u32,
    canvas_height: u32,
) -> Vec<usize> {
    let mut importance_scores: Vec<GaussianImportance> = gaussians
        .iter()
        .enumerate()
        .map(|(idx, g)| GaussianImportance {
            gaussian_index: idx,
            importance: compute_importance(g, canvas_width, canvas_height),
        })
        .collect();

    // Sort by importance (descending)
    importance_scores.sort_by(|a, b| {
        b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Return ordered indices
    importance_scores
        .iter()
        .map(|is| is.gaussian_index)
        .collect()
}

/// Reorder Gaussians according to importance indices
pub fn reorder_gaussians(
    gaussians: &[Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>],
    order: &[usize],
) -> Vec<Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>> {
    order.iter().map(|&idx| gaussians[idx]).collect()
}

/// LOD (Level of Detail) band classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LODBand {
    Coarse,   // Large Gaussians (background, large structures)
    Medium,   // Medium Gaussians (details)
    Fine,     // Small Gaussians (fine details, edges)
}

impl LODBand {
    pub fn to_u8(&self) -> u8 {
        match self {
            LODBand::Coarse => 0,
            LODBand::Medium => 1,
            LODBand::Fine => 2,
        }
    }

    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(LODBand::Coarse),
            1 => Some(LODBand::Medium),
            2 => Some(LODBand::Fine),
            _ => None,
        }
    }
}

/// Classify Gaussian into LOD band based on scale
pub fn classify_lod(
    gaussian: &Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>,
) -> LODBand {
    let scale_x = gaussian.shape.scale_x;
    let scale_y = gaussian.shape.scale_y;

    // Use geometric mean of scales
    let avg_scale = (scale_x * scale_y).sqrt();

    // Thresholds (normalized to canvas, typical values)
    if avg_scale > 0.1f32 {
        LODBand::Coarse  // Large background elements
    } else if avg_scale > 0.03f32 {
        LODBand::Medium  // Medium details
    } else {
        LODBand::Fine    // Fine details and edges
    }
}

/// Progressive loading strategy
#[derive(Debug, Clone, Copy)]
pub enum ProgressiveStrategy {
    /// Load by importance score (default)
    Importance,
    /// Load coarse-to-fine (LOD-based)
    CoarseToFine,
    /// Load by spatial tiles (for viewport-based streaming)
    Spatial,
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

    #[test]
    fn test_importance_ordering() {
        let gaussians = vec![
            // Small, low opacity (least important)
            Gaussian2D::new(
                Vector2::new(0.5, 0.5),
                Euler::isotropic(0.01),
                Color4::new(1.0, 0.0, 0.0, 1.0),
                0.3,
            ),
            // Large, high opacity (most important)
            Gaussian2D::new(
                Vector2::new(0.5, 0.5),
                Euler::isotropic(0.2),
                Color4::new(0.0, 1.0, 0.0, 1.0),
                0.9,
            ),
            // Medium
            Gaussian2D::new(
                Vector2::new(0.5, 0.5),
                Euler::isotropic(0.05),
                Color4::new(0.0, 0.0, 1.0, 1.0),
                0.7,
            ),
        ];

        let order = order_by_importance(&gaussians, 512, 512);

        // Should be ordered: 1 (large+opaque), 2 (medium), 0 (small+transparent)
        assert_eq!(order[0], 1);
        assert_eq!(order[2], 0);
    }

    #[test]
    fn test_lod_classification() {
        let coarse = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.15),
            Color4::new(1.0, 0.0, 0.0, 1.0),
            0.9,
        );

        let medium = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.05),
            Color4::new(0.0, 1.0, 0.0, 1.0),
            0.8,
        );

        let fine = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.01),
            Color4::new(0.0, 0.0, 1.0, 1.0),
            0.7,
        );

        assert_eq!(classify_lod(&coarse), LODBand::Coarse);
        assert_eq!(classify_lod(&medium), LODBand::Medium);
        assert_eq!(classify_lod(&fine), LODBand::Fine);
    }
}
