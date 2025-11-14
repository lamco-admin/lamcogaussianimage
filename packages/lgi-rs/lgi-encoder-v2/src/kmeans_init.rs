//! K-Means Clustering Initialization (Strategy D)
//!
//! Cluster pixel features (color, position, gradient) using k-means
//! Place Gaussian at each cluster centroid
//!
//! Classic approach, well-proven in computer vision

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use rand::Rng;

/// K-means configuration
#[derive(Clone)]
pub struct KMeansConfig {
    /// Number of clusters (= number of Gaussians)
    pub k: usize,

    /// Maximum k-means iterations
    pub max_iterations: usize,

    /// Feature weights
    pub weight_color: f32,
    pub weight_position: f32,
    pub weight_gradient: f32,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 100,
            max_iterations: 50,
            weight_color: 1.0,
            weight_position: 0.5,
            weight_gradient: 0.3,
        }
    }
}

/// Feature vector for clustering
#[derive(Clone, Copy)]
struct PixelFeature {
    r: f32,
    g: f32,
    b: f32,
    x: f32,  // Normalized position
    y: f32,
    gradient: f32,
}

/// Initialize Gaussians using k-means clustering
///
/// Clusters pixels by (color, position, gradient) features
/// Places Gaussian at each cluster centroid
pub fn initialize_kmeans(
    image: &ImageBuffer<f32>,
    structure_tensor: &StructureTensorField,
    config: &KMeansConfig,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    log::info!("🔍 K-Means Clustering Initialization (k={})", config.k);

    // Extract pixel features
    log::info!("  Extracting pixel features...");
    let features = extract_features(image, structure_tensor, config);

    // Run k-means
    log::info!("  Running k-means (max {} iterations)...", config.max_iterations);
    let (centroids, assignments) = kmeans(&features, config.k, config.max_iterations);

    log::info!("  Creating Gaussians from {} clusters...", centroids.len());

    // Create Gaussian for each cluster
    let mut gaussians = Vec::new();

    for (cluster_id, centroid) in centroids.iter().enumerate() {
        // Get pixels in this cluster
        let cluster_pixels: Vec<_> = assignments
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == cluster_id)
            .map(|(i, _)| i)
            .collect();

        if cluster_pixels.is_empty() {
            continue;
        }

        // Position from centroid
        let position = Vector2::new(centroid.x, centroid.y);

        // Color from centroid
        let color = Color4::new(centroid.r, centroid.g, centroid.b, 1.0);

        // Compute cluster covariance for Gaussian shape
        let (scale_x, scale_y, rotation) = compute_cluster_covariance(
            &cluster_pixels,
            &features,
            image.width,
            image.height,
        );

        gaussians.push(Gaussian2D::new(
            position,
            Euler::new(scale_x, scale_y, rotation),
            color,
            1.0,
        ));
    }

    log::info!("✅ K-means initialization complete: {} Gaussians", gaussians.len());

    gaussians
}

/// Extract features from all pixels
fn extract_features(
    image: &ImageBuffer<f32>,
    structure_tensor: &StructureTensorField,
    config: &KMeansConfig,
) -> Vec<PixelFeature> {
    let mut features = Vec::new();

    for y in 0..image.height {
        for x in 0..image.width {
            let pixel = image.get_pixel(x, y).unwrap();
            let tensor = structure_tensor.get(x, y);

            let gradient = tensor.eigenvalue_major.sqrt();

            features.push(PixelFeature {
                r: pixel.r * config.weight_color,
                g: pixel.g * config.weight_color,
                b: pixel.b * config.weight_color,
                x: (x as f32 / image.width as f32) * config.weight_position,
                y: (y as f32 / image.height as f32) * config.weight_position,
                gradient: gradient * config.weight_gradient,
            });
        }
    }

    features
}

/// Simple k-means clustering
///
/// Returns: (centroids, assignments)
fn kmeans(features: &[PixelFeature], k: usize, max_iters: usize) -> (Vec<PixelFeature>, Vec<usize>) {
    let mut rng = rand::thread_rng();

    // Initialize centroids (random selection from features)
    let mut centroids: Vec<PixelFeature> = (0..k)
        .map(|_| features[rng.gen_range(0..features.len())])
        .collect();

    let mut assignments = vec![0; features.len()];

    for iter in 0..max_iters {
        let mut changed = 0;

        // Assignment step
        for (i, feature) in features.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_distance = f32::INFINITY;

            for (j, centroid) in centroids.iter().enumerate() {
                let dist = distance(feature, centroid);
                if dist < best_distance {
                    best_distance = dist;
                    best_cluster = j;
                }
            }

            if assignments[i] != best_cluster {
                changed += 1;
            }
            assignments[i] = best_cluster;
        }

        // Convergence check
        if changed == 0 {
            log::debug!("K-means converged at iteration {}", iter);
            break;
        }

        // Update step: recompute centroids
        let mut counts = vec![0; k];
        let mut sums = vec![PixelFeature { r: 0.0, g: 0.0, b: 0.0, x: 0.0, y: 0.0, gradient: 0.0 }; k];

        for (i, feature) in features.iter().enumerate() {
            let cluster = assignments[i];
            counts[cluster] += 1;
            sums[cluster].r += feature.r;
            sums[cluster].g += feature.g;
            sums[cluster].b += feature.b;
            sums[cluster].x += feature.x;
            sums[cluster].y += feature.y;
            sums[cluster].gradient += feature.gradient;
        }

        for j in 0..k {
            if counts[j] > 0 {
                let count = counts[j] as f32;
                centroids[j] = PixelFeature {
                    r: sums[j].r / count,
                    g: sums[j].g / count,
                    b: sums[j].b / count,
                    x: sums[j].x / count,
                    y: sums[j].y / count,
                    gradient: sums[j].gradient / count,
                };
            }
        }
    }

    (centroids, assignments)
}

/// Euclidean distance in feature space
fn distance(a: &PixelFeature, b: &PixelFeature) -> f32 {
    let dr = a.r - b.r;
    let dg = a.g - b.g;
    let db = a.b - b.b;
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dgrad = a.gradient - b.gradient;

    (dr*dr + dg*dg + db*db + dx*dx + dy*dy + dgrad*dgrad).sqrt()
}

/// Compute Gaussian covariance from cluster
fn compute_cluster_covariance(
    cluster_pixels: &[usize],
    features: &[PixelFeature],
    image_width: u32,
    image_height: u32,
) -> (f32, f32, f32) {
    if cluster_pixels.is_empty() {
        return (0.02, 0.02, 0.0);  // Default isotropic
    }

    // Compute position variance
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;

    for &idx in cluster_pixels {
        mean_x += features[idx].x;
        mean_y += features[idx].y;
    }

    mean_x /= cluster_pixels.len() as f32;
    mean_y /= cluster_pixels.len() as f32;

    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut cov_xy = 0.0;

    for &idx in cluster_pixels {
        let dx = features[idx].x - mean_x;
        let dy = features[idx].y - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
        cov_xy += dx * dy;
    }

    var_x /= cluster_pixels.len() as f32;
    var_y /= cluster_pixels.len() as f32;
    cov_xy /= cluster_pixels.len() as f32;

    // Eigenvalues of 2×2 covariance matrix
    let trace = var_x + var_y;
    let det = var_x * var_y - cov_xy * cov_xy;
    let discriminant = (trace * trace - 4.0 * det).max(0.0).sqrt();

    let lambda1 = (trace + discriminant) / 2.0;
    let lambda2 = (trace - discriminant) / 2.0;

    let scale_x = lambda1.sqrt().max(0.01).min(0.2);
    let scale_y = lambda2.sqrt().max(0.01).min(0.2);

    // Rotation from eigenvector
    let rotation = if cov_xy.abs() > 1e-6 {
        (lambda1 - var_x).atan2(cov_xy)
    } else {
        0.0
    };

    (scale_x, scale_y, rotation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        // Create simple test image
        let mut image = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let val = if x < 32 { 0.2 } else { 0.8 };
                image.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let tensor = StructureTensorField::compute(&image, 1.0, 1.0).unwrap();
        let config = KMeansConfig { k: 10, ..Default::default() };

        let gaussians = initialize_kmeans(&image, &tensor, &config);

        // Should create approximately k Gaussians
        assert!(gaussians.len() >= 5 && gaussians.len() <= 10);
    }
}
