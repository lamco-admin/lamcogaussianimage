//! Gaussian initialization strategies
//!
//! Provides various methods for placing initial Gaussians:
//! - Random placement
//! - Gradient-based (edge-aware)
//! - Grid-based (uniform coverage)
//! - Importance sampling

use lgi_math::{Float, gaussian::Gaussian2D, parameterization::Euler, color::Color4, vec::Vector2};
use crate::{ImageBuffer, Result, LgiError};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Initialization strategy
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum InitStrategy {
    /// Random uniform placement
    Random,
    /// Grid-based uniform placement
    Grid,
    /// Gradient-based (edge-aware) placement
    Gradient,
    /// Importance sampling based on pixel variance
    Importance,
    /// SLIC superpixel-based initialization (content-adaptive)
    SLIC,
}

/// Gaussian initializer
pub struct Initializer {
    strategy: InitStrategy,
    /// Initial scale for Gaussians (in normalized coordinates)
    pub initial_scale: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Initializer {
    /// Create new initializer
    pub fn new(strategy: InitStrategy) -> Self {
        Self {
            strategy,
            initial_scale: 0.01, // 1% of image size
            seed: None,
        }
    }

    /// Set initial Gaussian scale
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.initial_scale = scale;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Initialize Gaussians for an image
    pub fn initialize(
        &self,
        target: &ImageBuffer<f32>,
        num_gaussians: usize,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        match self.strategy {
            InitStrategy::Random => self.init_random(target, num_gaussians),
            InitStrategy::Grid => self.init_grid(target, num_gaussians),
            InitStrategy::Gradient => self.init_gradient(target, num_gaussians),
            InitStrategy::Importance => self.init_importance(target, num_gaussians),
            InitStrategy::SLIC => self.init_slic(target, num_gaussians),
        }
    }

    /// Random initialization
    fn init_random(
        &self,
        target: &ImageBuffer<f32>,
        num_gaussians: usize,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        let mut rng = if let Some(seed) = self.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let mut gaussians = Vec::with_capacity(num_gaussians);

        for _ in 0..num_gaussians {
            let x = rng.gen::<f32>();
            let y = rng.gen::<f32>();

            // Sample color from nearby pixels
            let px = (x * target.width as f32) as u32;
            let py = (y * target.height as f32) as u32;
            let color = target.get_pixel(px.min(target.width - 1), py.min(target.height - 1))
                .unwrap_or(Color4::white());

            let gaussian = Gaussian2D::new(
                Vector2::new(x, y),
                Euler::isotropic(self.initial_scale),
                color,
                0.5, // Initial opacity
            );

            gaussians.push(gaussian);
        }

        Ok(gaussians)
    }

    /// Grid-based initialization
    fn init_grid(
        &self,
        target: &ImageBuffer<f32>,
        num_gaussians: usize,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        let grid_size = (num_gaussians as f32).sqrt() as usize;
        let mut gaussians = Vec::with_capacity(grid_size * grid_size);

        let step = 1.0 / grid_size as f32;

        for gy in 0..grid_size {
            for gx in 0..grid_size {
                let x = (gx as f32 + 0.5) * step;
                let y = (gy as f32 + 0.5) * step;

                // Sample color from grid position
                let px = (x * target.width as f32) as u32;
                let py = (y * target.height as f32) as u32;
                let color = target.get_pixel(px.min(target.width - 1), py.min(target.height - 1))
                    .unwrap_or(Color4::white());

                let gaussian = Gaussian2D::new(
                    Vector2::new(x, y),
                    Euler::isotropic(self.initial_scale),
                    color,
                    0.5,
                );

                gaussians.push(gaussian);
            }
        }

        Ok(gaussians)
    }

    /// Gradient-based initialization (edge-aware)
    fn init_gradient(
        &self,
        target: &ImageBuffer<f32>,
        num_gaussians: usize,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        // Compute gradient magnitude using Sobel filter
        let gradient_map = self.compute_gradient_map(target);

        // Sample positions based on gradient magnitude (edges get more Gaussians)
        let positions = self.sample_by_importance(&gradient_map, num_gaussians);

        let mut gaussians = Vec::with_capacity(positions.len());

        for pos in positions {
            let px = (pos.x * target.width as f32) as u32;
            let py = (pos.y * target.height as f32) as u32;
            let color = target.get_pixel(px.min(target.width - 1), py.min(target.height - 1))
                .unwrap_or(Color4::white());

            let gaussian = Gaussian2D::new(
                pos,
                Euler::isotropic(self.initial_scale),
                color,
                0.5,
            );

            gaussians.push(gaussian);
        }

        Ok(gaussians)
    }

    /// Importance sampling initialization
    fn init_importance(
        &self,
        target: &ImageBuffer<f32>,
        num_gaussians: usize,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        // Compute pixel variance/complexity map
        let importance_map = self.compute_importance_map(target);

        // Sample based on importance
        let positions = self.sample_by_importance(&importance_map, num_gaussians);

        let mut gaussians = Vec::with_capacity(positions.len());

        for pos in positions {
            let px = (pos.x * target.width as f32) as u32;
            let py = (pos.y * target.height as f32) as u32;
            let color = target.get_pixel(px.min(target.width - 1), py.min(target.height - 1))
                .unwrap_or(Color4::white());

            let gaussian = Gaussian2D::new(
                pos,
                Euler::isotropic(self.initial_scale),
                color,
                0.5,
            );

            gaussians.push(gaussian);
        }

        Ok(gaussians)
    }

    /// Compute gradient magnitude map (Sobel filter)
    fn compute_gradient_map(&self, target: &ImageBuffer<f32>) -> Vec<f32> {
        let mut gradient = vec![0.0f32; (target.width * target.height) as usize];

        // Sobel kernels
        let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
        let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

        for y in 1..(target.height - 1) {
            for x in 1..(target.width - 1) {
                let mut gx = 0.0;
                let mut gy = 0.0;

                // Apply Sobel filter
                for dy in 0..3 {
                    for dx in 0..3 {
                        let px = x + dx - 1;
                        let py = y + dy - 1;

                        if let Some(pixel) = target.get_pixel(px, py) {
                            let intensity = (pixel.r + pixel.g + pixel.b) / 3.0;
                            gx += sobel_x[dy as usize][dx as usize] * intensity;
                            gy += sobel_y[dy as usize][dx as usize] * intensity;
                        }
                    }
                }

                let magnitude = (gx * gx + gy * gy).sqrt();
                gradient[(y * target.width + x) as usize] = magnitude;
            }
        }

        gradient
    }

    /// Compute importance map (variance-based)
    fn compute_importance_map(&self, target: &ImageBuffer<f32>) -> Vec<f32> {
        let mut importance = vec![0.0f32; (target.width * target.height) as usize];
        let window_size = 5;

        for y in window_size..(target.height - window_size) {
            for x in window_size..(target.width - window_size) {
                // Compute local variance
                let mut mean = Color4::new(0.0, 0.0, 0.0, 0.0);
                let mut count = 0;

                // Compute mean
                for dy in -(window_size as i32)..(window_size as i32) {
                    for dx in -(window_size as i32)..(window_size as i32) {
                        let px = (x as i32 + dx) as u32;
                        let py = (y as i32 + dy) as u32;

                        if let Some(pixel) = target.get_pixel(px, py) {
                            mean.r += pixel.r;
                            mean.g += pixel.g;
                            mean.b += pixel.b;
                            count += 1;
                        }
                    }
                }

                mean.r /= count as f32;
                mean.g /= count as f32;
                mean.b /= count as f32;

                // Compute variance
                let mut variance = 0.0;
                for dy in -(window_size as i32)..(window_size as i32) {
                    for dx in -(window_size as i32)..(window_size as i32) {
                        let px = (x as i32 + dx) as u32;
                        let py = (y as i32 + dy) as u32;

                        if let Some(pixel) = target.get_pixel(px, py) {
                            let diff_r = pixel.r - mean.r;
                            let diff_g = pixel.g - mean.g;
                            let diff_b = pixel.b - mean.b;
                            variance += diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
                        }
                    }
                }

                importance[(y * target.width + x) as usize] = variance;
            }
        }

        importance
    }

    /// Sample positions based on importance map
    fn sample_by_importance(&self, importance_map: &[f32], num_samples: usize) -> Vec<Vector2<f32>> {
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        // Normalize importance map to probabilities
        let total: f32 = importance_map.iter().sum();
        let probabilities: Vec<f32> = importance_map.iter()
            .map(|&v| v / total)
            .collect();

        // Cumulative distribution
        let mut cumulative = vec![0.0; probabilities.len()];
        cumulative[0] = probabilities[0];
        for i in 1..probabilities.len() {
            cumulative[i] = cumulative[i - 1] + probabilities[i];
        }

        // Sample positions
        let mut positions = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let r: f32 = rng.gen();

            // Binary search in cumulative distribution
            let idx = cumulative.partition_point(|&v| v < r);
            let idx = idx.min(cumulative.len() - 1);

            // Convert index to (x, y)
            let width = ((importance_map.len() as f32).sqrt()) as usize;
            let x = (idx % width) as f32 / width as f32;
            let y = (idx / width) as f32 / width as f32;

            positions.push(Vector2::new(x, y));
        }

        positions
    }

    /// SLIC superpixel-based initialization
    /// Uses Simple Linear Iterative Clustering to segment image into superpixels,
    /// then initializes one Gaussian per superpixel
    fn init_slic(
        &self,
        target: &ImageBuffer<f32>,
        num_gaussians: usize,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        let width = target.width as usize;
        let height = target.height as usize;

        // SLIC parameters
        let m = 10.0; // Compactness parameter (spatial vs color weight)
        let iterations = 10; // Number of k-means iterations

        // Calculate grid spacing
        let s = ((width * height) as f32 / num_gaussians as f32).sqrt();
        let grid_w = (width as f32 / s).ceil() as usize;
        let grid_h = (height as f32 / s).ceil() as usize;
        let actual_num = grid_w * grid_h;

        // Initialize cluster centers on a grid
        let mut centers: Vec<SlicCenter> = Vec::with_capacity(actual_num);
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let x = ((gx as f32 + 0.5) * s).min((width - 1) as f32) as usize;
                let y = ((gy as f32 + 0.5) * s).min((height - 1) as f32) as usize;

                if let Some(pixel) = target.get_pixel(x as u32, y as u32) {
                    centers.push(SlicCenter {
                        l: pixel.r * 0.299 + pixel.g * 0.587 + pixel.b * 0.114, // Luminance
                        a: pixel.r,
                        b: pixel.g,
                        x: x as f32,
                        y: y as f32,
                        count: 0,
                    });
                }
            }
        }

        // Move centers to lowest gradient position in 3x3 neighborhood
        centers = self.move_to_low_gradient(target, centers);

        // Assignment and update iterations
        let mut labels = vec![0usize; width * height];
        let mut distances = vec![f32::INFINITY; width * height];

        for _iter in 0..iterations {
            // Reset distances
            distances.fill(f32::INFINITY);

            // Assign pixels to nearest center
            for (i, center) in centers.iter().enumerate() {
                let cx = center.x as i32;
                let cy = center.y as i32;
                let search_region = (2.0 * s) as i32;

                for dy in -search_region..=search_region {
                    for dx in -search_region..=search_region {
                        let px = (cx + dx).max(0).min((width - 1) as i32) as usize;
                        let py = (cy + dy).max(0).min((height - 1) as i32) as usize;

                        if let Some(pixel) = target.get_pixel(px as u32, py as u32) {
                            let l = pixel.r * 0.299 + pixel.g * 0.587 + pixel.b * 0.114;

                            // Compute 5D distance (Lab + XY)
                            let dc = ((l - center.l).powi(2) +
                                     (pixel.r - center.a).powi(2) +
                                     (pixel.g - center.b).powi(2)).sqrt();
                            let ds = ((px as f32 - center.x).powi(2) +
                                     (py as f32 - center.y).powi(2)).sqrt();

                            let distance = dc + (m / s) * ds;

                            let idx = py * width + px;
                            if distance < distances[idx] {
                                distances[idx] = distance;
                                labels[idx] = i;
                            }
                        }
                    }
                }
            }

            // Update centers
            for center in centers.iter_mut() {
                center.l = 0.0;
                center.a = 0.0;
                center.b = 0.0;
                center.x = 0.0;
                center.y = 0.0;
                center.count = 0;
            }

            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    let label = labels[idx];

                    if let Some(pixel) = target.get_pixel(x as u32, y as u32) {
                        let l = pixel.r * 0.299 + pixel.g * 0.587 + pixel.b * 0.114;
                        centers[label].l += l;
                        centers[label].a += pixel.r;
                        centers[label].b += pixel.g;
                        centers[label].x += x as f32;
                        centers[label].y += y as f32;
                        centers[label].count += 1;
                    }
                }
            }

            // Normalize centers
            for center in centers.iter_mut() {
                if center.count > 0 {
                    let count = center.count as f32;
                    center.l /= count;
                    center.a /= count;
                    center.b /= count;
                    center.x /= count;
                    center.y /= count;
                }
            }
        }

        // Convert superpixels to Gaussians
        let mut gaussians = Vec::with_capacity(centers.len());

        for (i, center) in centers.iter().enumerate() {
            if center.count == 0 {
                continue; // Skip empty clusters
            }

            // Calculate superpixel statistics
            let mut sum_x = 0.0f32;
            let mut sum_y = 0.0f32;
            let mut sum_xx = 0.0f32;
            let mut sum_yy = 0.0f32;
            let mut sum_xy = 0.0f32;
            let mut pixel_count = 0;

            for y in 0..height {
                for x in 0..width {
                    if labels[y * width + x] == i {
                        let fx = x as f32;
                        let fy = y as f32;
                        sum_x += fx;
                        sum_y += fy;
                        sum_xx += fx * fx;
                        sum_yy += fy * fy;
                        sum_xy += fx * fy;
                        pixel_count += 1;
                    }
                }
            }

            if pixel_count == 0 {
                continue;
            }

            // Compute centroid (normalized)
            let mean_x = sum_x / pixel_count as f32 / width as f32;
            let mean_y = sum_y / pixel_count as f32 / height as f32;

            // Compute covariance matrix for scale and rotation
            let count_f = pixel_count as f32;
            let var_x = (sum_xx / count_f - (sum_x / count_f).powi(2)) / (width * width) as f32;
            let var_y = (sum_yy / count_f - (sum_y / count_f).powi(2)) / (height * height) as f32;
            let cov_xy = (sum_xy / count_f - (sum_x / count_f) * (sum_y / count_f)) / (width * height) as f32;

            // Compute eigenvalues for scale (principal axes)
            let trace = var_x + var_y;
            let det = var_x * var_y - cov_xy * cov_xy;
            let discriminant = (trace * trace / 4.0 - det).max(0.0).sqrt();
            let lambda1 = (trace / 2.0 + discriminant).max(0.001);
            let lambda2 = (trace / 2.0 - discriminant).max(0.001);

            let scale_x = lambda1.sqrt().max(0.005).min(0.2);
            let scale_y = lambda2.sqrt().max(0.005).min(0.2);

            // Compute rotation from eigenvector
            let rotation = if cov_xy.abs() > 1e-6 {
                (cov_xy).atan2(lambda1 - var_y)
            } else {
                0.0
            };

            // Color from center
            let color = Color4::new(center.a, center.b, center.l, 1.0);

            gaussians.push(Gaussian2D::new(
                Vector2::new(mean_x, mean_y),
                Euler::new(scale_x, scale_y, rotation),
                color,
                0.8, // Initial opacity
            ));
        }

        Ok(gaussians)
    }

    /// Move SLIC centers to lowest gradient position in 3x3 neighborhood
    fn move_to_low_gradient(
        &self,
        target: &ImageBuffer<f32>,
        centers: Vec<SlicCenter>,
    ) -> Vec<SlicCenter> {
        let width = target.width as i32;
        let height = target.height as i32;

        centers.into_iter().map(|mut center| {
            let cx = center.x as i32;
            let cy = center.y as i32;

            let mut min_gradient = f32::INFINITY;
            let mut best_x = center.x;
            let mut best_y = center.y;

            for dy in -1..=1 {
                for dx in -1..=1 {
                    let x = (cx + dx).max(0).min(width - 1);
                    let y = (cy + dy).max(0).min(height - 1);

                    // Compute gradient magnitude
                    let gradient = self.compute_gradient_magnitude(target, x as u32, y as u32);

                    if gradient < min_gradient {
                        min_gradient = gradient;
                        best_x = x as f32;
                        best_y = y as f32;
                    }
                }
            }

            center.x = best_x;
            center.y = best_y;

            // Update color at new position
            if let Some(pixel) = target.get_pixel(best_x as u32, best_y as u32) {
                center.l = pixel.r * 0.299 + pixel.g * 0.587 + pixel.b * 0.114;
                center.a = pixel.r;
                center.b = pixel.g;
            }

            center
        }).collect()
    }

    /// Compute gradient magnitude at a pixel
    fn compute_gradient_magnitude(&self, target: &ImageBuffer<f32>, x: u32, y: u32) -> f32 {
        let width = target.width as i32;
        let height = target.height as i32;

        let x = x as i32;
        let y = y as i32;

        // Sobel operator
        let dx_left = target.get_pixel((x - 1).max(0) as u32, y as u32);
        let dx_right = target.get_pixel((x + 1).min(width - 1) as u32, y as u32);
        let dy_top = target.get_pixel(x as u32, (y - 1).max(0) as u32);
        let dy_bottom = target.get_pixel(x as u32, (y + 1).min(height - 1) as u32);

        if let (Some(left), Some(right), Some(top), Some(bottom)) = (dx_left, dx_right, dy_top, dy_bottom) {
            let gx = (right.r - left.r).abs() + (right.g - left.g).abs() + (right.b - left.b).abs();
            let gy = (bottom.r - top.r).abs() + (bottom.g - top.g).abs() + (bottom.b - top.b).abs();
            (gx * gx + gy * gy).sqrt()
        } else {
            0.0
        }
    }
}

/// SLIC cluster center
#[derive(Debug, Clone, Copy)]
struct SlicCenter {
    l: f32,  // Luminance
    a: f32,  // R channel (approximating Lab a*)
    b: f32,  // G channel (approximating Lab b*)
    x: f32,  // X position
    y: f32,  // Y position
    count: usize, // Number of pixels in cluster
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_init() {
        let target = ImageBuffer::<f32>::with_background(100, 100, Color4::white());
        let initializer = Initializer::new(InitStrategy::Random).with_seed(42);

        let gaussians = initializer.initialize(&target, 100).unwrap();

        assert_eq!(gaussians.len(), 100);
        // All positions should be in [0, 1]
        for g in &gaussians {
            assert!(g.position.x >= 0.0 && g.position.x <= 1.0);
            assert!(g.position.y >= 0.0 && g.position.y <= 1.0);
        }
    }

    #[test]
    fn test_grid_init() {
        let target = ImageBuffer::<f32>::with_background(100, 100, Color4::white());
        let initializer = Initializer::new(InitStrategy::Grid);

        let gaussians = initializer.initialize(&target, 100).unwrap();

        // Grid init creates ~sqrt(n)^2 Gaussians
        assert!(gaussians.len() >= 90 && gaussians.len() <= 110);
    }
}
