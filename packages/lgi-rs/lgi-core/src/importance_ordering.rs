//! Importance-Based Gaussian Ordering
//! For progressive encoding and view-dependent selection

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

/// Importance metric for Gaussian
pub fn compute_importance(gaussian: &Gaussian2D<f32, Euler<f32>>) -> f32 {
    // Energy-based importance: |α · σx · σy · max(RGB)|
    let scale_product = gaussian.shape.scale_x * gaussian.shape.scale_y;
    let color_max = gaussian.color.r.max(gaussian.color.g).max(gaussian.color.b);
    gaussian.opacity * scale_product * color_max
}

/// Sort Gaussians by importance (descending)
pub fn sort_by_importance(gaussians: &mut [Gaussian2D<f32, Euler<f32>>]) {
    gaussians.sort_by(|a, b| {
        let imp_a = compute_importance(a);
        let imp_b = compute_importance(b);
        imp_b.partial_cmp(&imp_a).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Get top K most important Gaussians
pub fn select_top_k(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    k: usize,
) -> Vec<usize> {
    let mut indexed: Vec<_> = gaussians.iter().enumerate()
        .map(|(i, g)| (i, compute_importance(g)))
        .collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    indexed.iter().take(k).map(|(i, _)| *i).collect()
}
