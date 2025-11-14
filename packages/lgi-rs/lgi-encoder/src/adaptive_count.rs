//! Adaptive Gaussian count determination based on image complexity
//!
//! Key principle: Number of Gaussians should match image FEATURES, not pixels
//! - 1 solid color = 1 Gaussian
//! - Simple gradient = 5-10 Gaussians
//! - Complex photo = 100-5000 Gaussians based on edge density and scale

use lgi_core::ImageBuffer;

/// Analyze image and determine appropriate Gaussian count
///
/// This uses a multi-scale edge analysis approach:
/// 1. Detect edges at multiple scales
/// 2. Count distinct regions/features
/// 3. Adjust for target rendering resolution
pub fn estimate_gaussian_count(
    image: &ImageBuffer<f32>,
    target_quality: QualityTarget,
) -> usize {
    let width = image.width;
    let height = image.height;

    // Step 1: Check if image is essentially uniform (1 Gaussian case!)
    if is_uniform_color(image) {
        return 1;
    }

    // Step 2: Multi-scale edge detection
    let edge_pixels = count_edge_pixels(image, 0.05); // 5% threshold
    let edge_ratio = edge_pixels as f32 / (width * height) as f32;

    // Step 3: Estimate number of distinct regions/features
    let feature_count = estimate_feature_count(image);

    // Step 4: Calculate base Gaussian count from features
    let base_from_features = match target_quality {
        QualityTarget::Fast => feature_count.max(5),
        QualityTarget::Balanced => (feature_count as f32 * 1.5) as usize,
        QualityTarget::High => (feature_count as f32 * 3.0) as usize,
        QualityTarget::Ultra => (feature_count as f32 * 5.0) as usize,
    };

    // Step 5: Adjust for edge density (more edges = need more Gaussians)
    let edge_multiplier = if edge_ratio < 0.05 {
        1.0 // Very few edges (simple image)
    } else if edge_ratio < 0.2 {
        1.5 // Moderate edges
    } else {
        2.5 // High edge density (complex image)
    };

    let adjusted = (base_from_features as f32 * edge_multiplier) as usize;

    // Step 6: Clamp to reasonable limits based on image size
    let pixel_count = (width * height) as usize;
    let max_reasonable = pixel_count / 20; // At most 1 Gaussian per 20 pixels
    let min_reasonable = 1; // At least 1 Gaussian!

    adjusted.max(min_reasonable).min(max_reasonable)
}

#[derive(Debug, Clone, Copy)]
pub enum QualityTarget {
    Fast,      // Minimal Gaussians, fast encode
    Balanced,  // Good quality/speed tradeoff
    High,      // High quality
    Ultra,     // Maximum quality
}

/// Check if image is essentially one uniform color
/// Returns true if variance is extremely low (solid color = 1 Gaussian!)
fn is_uniform_color(image: &ImageBuffer<f32>) -> bool {
    let variance = compute_color_variance(image);
    variance < 0.01 // Less than 1% variance = uniform
}

/// Count pixels that are part of edges
fn count_edge_pixels(image: &ImageBuffer<f32>, threshold: f32) -> usize {
    let width = image.width;
    let height = image.height;
    let mut edge_pixels = 0;

    // Use Sobel-like edge detection
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            if let Some(center) = image.get_pixel(x, y) {
                // Check 4-connected neighbors
                let mut max_diff = 0.0f32;

                for (dx, dy) in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let nx = (x as i32 + dx) as u32;
                    let ny = (y as i32 + dy) as u32;

                    if let Some(neighbor) = image.get_pixel(nx, ny) {
                        let diff = ((center.r - neighbor.r).powi(2) +
                                   (center.g - neighbor.g).powi(2) +
                                   (center.b - neighbor.b).powi(2)).sqrt();
                        max_diff = max_diff.max(diff);
                    }
                }

                if max_diff > threshold {
                    edge_pixels += 1;
                }
            }
        }
    }

    edge_pixels
}

/// Estimate number of distinct features/regions using connected components
fn estimate_feature_count(image: &ImageBuffer<f32>) -> usize {
    // Simple region counting using color similarity
    let width = image.width;
    let height = image.height;

    // Sample at reduced resolution for speed
    let sample_step = (width.max(height) / 64).max(1);
    let mut regions = std::collections::HashSet::new();

    for y in (0..height).step_by(sample_step as usize) {
        for x in (0..width).step_by(sample_step as usize) {
            if let Some(pixel) = image.get_pixel(x, y) {
                // Quantize color to identify regions
                let r_bucket = (pixel.r * 8.0) as u8;
                let g_bucket = (pixel.g * 8.0) as u8;
                let b_bucket = (pixel.b * 8.0) as u8;
                regions.insert((r_bucket, g_bucket, b_bucket));
            }
        }
    }

    // Number of distinct color regions is a good proxy for features
    regions.len().max(1)
}

/// Compute color variance (0.0 = uniform, 1.0 = very varied)
fn compute_color_variance(image: &ImageBuffer<f32>) -> f32 {
    let pixel_count = (image.width * image.height) as f32;

    // Compute mean color
    let mut mean_r = 0.0;
    let mut mean_g = 0.0;
    let mut mean_b = 0.0;

    for pixel in &image.data {
        mean_r += pixel.r;
        mean_g += pixel.g;
        mean_b += pixel.b;
    }

    mean_r /= pixel_count;
    mean_g /= pixel_count;
    mean_b /= pixel_count;

    // Compute variance
    let mut variance = 0.0;
    for pixel in &image.data {
        variance += (pixel.r - mean_r).powi(2);
        variance += (pixel.g - mean_g).powi(2);
        variance += (pixel.b - mean_b).powi(2);
    }

    variance /= pixel_count * 3.0;

    // Normalize (approximate)
    (variance.sqrt() * 2.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solid_color_needs_few_gaussians() {
        let mut image = ImageBuffer::new(256, 256);
        // Solid red
        for pixel in &mut image.data {
            pixel.r = 1.0;
        }

        let count = estimate_gaussian_count(&image, QualityTarget::Balanced);
        assert!(count < 100); // Solid color should need very few
    }
}
