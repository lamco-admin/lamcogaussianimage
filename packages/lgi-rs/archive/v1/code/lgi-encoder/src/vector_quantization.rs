//! Vector Quantization for Gaussian Compression
//!
//! Implements k-means based vector quantization (from GaussianImage ECCV 2024)
//! Achieves 5-10× compression with <1 dB quality loss

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use rand::Rng;
use std::f32;
use serde::{Deserialize, Serialize};

/// Flattened Gaussian representation for VQ
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GaussianVector {
    pub data: [f32; 9],  // [μx, μy, σx, σy, θ, r, g, b, α]
}

impl GaussianVector {
    /// Convert Gaussian2D to vector
    pub fn from_gaussian(gaussian: &Gaussian2D<f32, Euler<f32>>) -> Self {
        Self {
            data: [
                gaussian.position.x,
                gaussian.position.y,
                gaussian.shape.scale_x,
                gaussian.shape.scale_y,
                gaussian.shape.rotation,
                gaussian.color.r,
                gaussian.color.g,
                gaussian.color.b,
                gaussian.opacity,
            ],
        }
    }

    /// Convert vector back to Gaussian2D
    pub fn to_gaussian(&self) -> Gaussian2D<f32, Euler<f32>> {
        Gaussian2D::new(
            Vector2::new(self.data[0], self.data[1]),
            Euler::new(self.data[2], self.data[3], self.data[4]),
            Color4::new(self.data[5], self.data[6], self.data[7], self.data[8]),
            self.data[8],
        )
    }

    /// Squared Euclidean distance to another vector
    pub fn distance_squared(&self, other: &Self) -> f32 {
        self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum()
    }
}

/// Vector Quantizer using k-means clustering
pub struct VectorQuantizer {
    /// Codebook entries
    pub codebook: Vec<GaussianVector>,

    /// Codebook size (typically 256 for 8-bit indices)
    pub codebook_size: usize,

    /// Whether codebook is trained
    pub trained: bool,
}

impl VectorQuantizer {
    /// Create new untrained quantizer
    pub fn new(codebook_size: usize) -> Self {
        Self {
            codebook: Vec::with_capacity(codebook_size),
            codebook_size,
            trained: false,
        }
    }

    /// Train codebook using k-means++ initialization and Lloyd's algorithm
    pub fn train(&mut self, gaussians: &[Gaussian2D<f32, Euler<f32>>], max_iterations: usize) {
        // Convert to vectors
        let vectors: Vec<GaussianVector> = gaussians.iter()
            .map(|g| GaussianVector::from_gaussian(g))
            .collect();

        println!("Training VQ codebook: {} entries from {} Gaussians", self.codebook_size, vectors.len());

        // K-means++ initialization (better than random)
        self.codebook = self.kmeans_plus_plus_init(&vectors);

        // Lloyd's algorithm
        for iteration in 0..max_iterations {
            // Assign each vector to nearest codebook entry
            let assignments = self.assign_to_nearest(&vectors);

            // Recompute centroids
            let old_codebook = self.codebook.clone();
            self.update_centroids(&vectors, &assignments);

            // Check convergence
            let movement = self.codebook.iter()
                .zip(old_codebook.iter())
                .map(|(new, old)| new.distance_squared(old))
                .sum::<f32>();

            if movement < 1e-6 {
                println!("VQ converged at iteration {}", iteration);
                break;
            }

            if iteration % 10 == 0 {
                println!("VQ iteration {}: movement = {:.6}", iteration, movement);
            }
        }

        self.trained = true;
        println!("VQ training complete!");
    }

    /// K-means++ initialization (better centroid placement)
    fn kmeans_plus_plus_init(&self, vectors: &[GaussianVector]) -> Vec<GaussianVector> {
        let mut centroids = Vec::with_capacity(self.codebook_size);

        // First centroid: random
        let first_idx = rand::random::<usize>() % vectors.len();
        centroids.push(vectors[first_idx]);

        // Remaining centroids: weighted by distance to nearest existing centroid
        for _ in 1..self.codebook_size {
            let distances: Vec<f32> = vectors.iter()
                .map(|v| {
                    // Distance to nearest centroid
                    centroids.iter()
                        .map(|c| v.distance_squared(c))
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                })
                .collect();

            // Sample proportional to squared distance
            let total: f32 = distances.iter().sum();
            let mut threshold = rand::random::<f32>() * total;

            let mut next_idx = 0;
            for (idx, &dist) in distances.iter().enumerate() {
                threshold -= dist;
                if threshold <= 0.0 {
                    next_idx = idx;
                    break;
                }
            }

            centroids.push(vectors[next_idx]);
        }

        centroids
    }

    /// Assign vectors to nearest codebook entries
    fn assign_to_nearest(&self, vectors: &[GaussianVector]) -> Vec<usize> {
        vectors.iter()
            .map(|v| {
                self.codebook.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let dist_a = v.distance_squared(a);
                        let dist_b = v.distance_squared(b);
                        dist_a.partial_cmp(&dist_b).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Update centroids based on assignments
    fn update_centroids(&mut self, vectors: &[GaussianVector], assignments: &[usize]) {
        // Reset centroids
        for centroid in &mut self.codebook {
            centroid.data = [0.0; 9];
        }

        let mut counts = vec![0; self.codebook_size];

        // Accumulate
        for (vector, &assignment) in vectors.iter().zip(assignments.iter()) {
            for i in 0..9 {
                self.codebook[assignment].data[i] += vector.data[i];
            }
            counts[assignment] += 1;
        }

        // Average
        for (centroid, count) in self.codebook.iter_mut().zip(counts.iter()) {
            if *count > 0 {
                for i in 0..9 {
                    centroid.data[i] /= *count as f32;
                }
            }
        }
    }

    /// Quantize a Gaussian to codebook index
    pub fn quantize(&self, gaussian: &Gaussian2D<f32, Euler<f32>>) -> u8 {
        let vector = GaussianVector::from_gaussian(gaussian);

        self.codebook.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let dist_a = vector.distance_squared(a);
                let dist_b = vector.distance_squared(b);
                dist_a.partial_cmp(&dist_b).unwrap()
            })
            .map(|(idx, _)| idx as u8)
            .unwrap_or(0)
    }

    /// Dequantize from codebook index
    pub fn dequantize(&self, index: u8) -> Gaussian2D<f32, Euler<f32>> {
        self.codebook[index as usize].to_gaussian()
    }

    /// Quantize all Gaussians
    pub fn quantize_all(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Vec<u8> {
        gaussians.iter().map(|g| self.quantize(g)).collect()
    }

    /// Dequantize all indices
    pub fn dequantize_all(&self, indices: &[u8]) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        indices.iter().map(|&idx| self.dequantize(idx)).collect()
    }

    /// Measure quantization distortion
    pub fn measure_distortion(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> f32 {
        let mut total_distortion = 0.0;

        for gaussian in gaussians {
            let vector = GaussianVector::from_gaussian(gaussian);
            let idx = self.quantize(gaussian);
            let reconstructed = &self.codebook[idx as usize];

            total_distortion += vector.distance_squared(reconstructed);
        }

        total_distortion / gaussians.len() as f32
    }

    /// Get codebook size in bytes
    pub fn codebook_size_bytes(&self) -> usize {
        self.codebook.len() * 9 * 4  // 9 floats × 4 bytes
    }

    /// Get compressed size for N Gaussians
    pub fn compressed_size_bytes(&self, num_gaussians: usize) -> usize {
        self.codebook_size_bytes() + num_gaussians  // Codebook + indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_vector_conversion() {
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::new(0.1, 0.1, 0.0),
            Color4::rgb(1.0, 0.0, 0.0),
            0.8,
        );

        let vector = GaussianVector::from_gaussian(&gaussian);
        let reconstructed = vector.to_gaussian();

        assert!((reconstructed.position.x - 0.5).abs() < 1e-5);
        assert!((reconstructed.color.r - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vq_basic() {
        // Create test Gaussians
        let gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = (0..100)
            .map(|i| {
                Gaussian2D::new(
                    Vector2::new(i as f32 / 100.0, 0.5),
                    Euler::isotropic(0.01),
                    Color4::rgb(i as f32 / 100.0, 0.5, 0.5),
                    0.8,
                )
            })
            .collect();

        // Train VQ
        let mut vq = VectorQuantizer::new(16);  // Small codebook for test
        vq.train(&gaussians, 50);

        assert!(vq.trained);
        assert_eq!(vq.codebook.len(), 16);

        // Test quantization
        let idx = vq.quantize(&gaussians[0]);
        assert!(idx < 16);

        let reconstructed = vq.dequantize(idx);
        // Should be close to original
        let dist = (reconstructed.position.x - gaussians[0].position.x).abs();
        assert!(dist < 0.1);  // Some distortion expected from VQ
    }
}
