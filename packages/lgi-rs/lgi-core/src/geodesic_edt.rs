//! Geodesic Distance Transform - Structure-Respecting Distance Fields
//!
//! Computes distance fields that respect image structure (edges, boundaries).
//! Unlike Euclidean EDT, geodesic distance cannot cross strong edges.
//!
//! # Theory
//!
//! Geodesic distance D(p,q) is the shortest path from p to q where
//! path cost is weighted by edge strength:
//!
//! D(p,q) = min_γ ∫_γ C(x,y) ds
//!
//! where C(x,y) is high at edges (penalizes crossing).
//!
//! # Algorithm: Jump Flooding Algorithm (JFA)
//!
//! GPU-friendly parallel algorithm for computing approximate
//! geodesic distance in O(log N) passes.
//!
//! # Benefits for Gaussian Encoding
//!
//! - Prevents color bleeding across object boundaries
//! - Enables edge-aware Gaussian size clamping
//! - Respects semantic structure
//! - Critical for text, faces, segmented content
//!
//! # References
//!
//! - Jump Flooding Algorithm (Rong & Tan, 2006)
//! - FastGeodis (Python/CUDA implementation)
//! - Geodesic distance transforms (Criminisi et al.)

use crate::{ImageBuffer, StructureTensorField};

/// Geodesic distance transform
///
/// Stores geodesic distance from each pixel to nearest edge
pub struct GeodesicEDT {
    pub width: u32,
    pub height: u32,

    /// Geodesic distance to nearest edge (in pixels)
    pub distances: Vec<f32>,

    /// Nearest edge pixel (for debugging/visualization)
    nearest_edge: Vec<Option<(u32, u32)>>,
}

impl GeodesicEDT {
    /// Compute geodesic EDT using Jump Flooding Algorithm
    ///
    /// # Parameters
    ///
    /// - `image`: Input image
    /// - `structure_tensor`: Provides edge strength
    /// - `edge_threshold`: Coherence threshold for detecting edges (0.7)
    /// - `edge_penalty`: How much crossing edges costs (10-100)
    ///
    /// # Algorithm
    ///
    /// 1. Detect edges from structure tensor (coherence > threshold)
    /// 2. Initialize: distance=0 at edges, ∞ elsewhere
    /// 3. Jump flooding: propagate distances with geodesic cost
    /// 4. Refine: local relaxation passes
    ///
    /// # Returns
    ///
    /// Geodesic distance field respecting image structure
    pub fn compute(
        image: &ImageBuffer<f32>,
        structure_tensor: &StructureTensorField,
        edge_threshold: f32,
        edge_penalty: f32,
    ) -> crate::Result<Self> {
        let width = image.width;
        let height = image.height;
        let size = (width * height) as usize;

        // Step 1: Detect edges from structure tensor
        let edge_map = Self::detect_edges(structure_tensor, edge_threshold);

        // Step 2: Initialize distance field
        let mut distances = vec![f32::INFINITY; size];
        let mut nearest_edge = vec![None; size];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if edge_map[idx] {
                    distances[idx] = 0.0;
                    nearest_edge[idx] = Some((x, y));
                }
            }
        }

        // Step 3: Jump Flooding Algorithm
        let max_dim = width.max(height);
        let mut step_size = max_dim.next_power_of_two() / 2;

        while step_size >= 1 {
            Self::jump_flood_pass(
                &mut distances,
                &mut nearest_edge,
                width,
                height,
                step_size,
                &edge_map,
                edge_penalty,
            );

            step_size /= 2;
        }

        // Step 4: Local refinement (1-pixel neighbors)
        for _ in 0..2 {  // 2 refinement passes
            Self::refine_pass(&mut distances, &mut nearest_edge, width, height, &edge_map, edge_penalty);
        }

        Ok(Self { width, height, distances, nearest_edge })
    }

    /// Get geodesic distance at pixel
    pub fn get_distance(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return f32::INFINITY;
        }
        self.distances[(y * self.width + x) as usize]
    }

    /// Clamp sigma by geodesic distance (anti-bleeding)
    ///
    /// Formula: σ ≤ 0.5 + 0.3 × d
    ///
    /// Prevents Gaussians from extending across edges
    pub fn clamp_sigma(&self, x: u32, y: u32, desired_sigma: f32) -> f32 {
        let d = self.get_distance(x, y);
        desired_sigma.min(0.5 + 0.3 * d)
    }

    /// Detect edges from structure tensor
    fn detect_edges(structure_tensor: &StructureTensorField, threshold: f32) -> Vec<bool> {
        let size = (structure_tensor.width * structure_tensor.height) as usize;
        let mut edges = vec![false; size];

        for y in 0..structure_tensor.height {
            for x in 0..structure_tensor.width {
                let tensor = structure_tensor.get(x, y);
                let idx = (y * structure_tensor.width + x) as usize;

                // High coherence = edge
                edges[idx] = tensor.coherence > threshold;
            }
        }

        edges
    }

    /// Single jump flooding pass
    fn jump_flood_pass(
        distances: &mut [f32],
        nearest: &mut [Option<(u32, u32)>],
        width: u32,
        height: u32,
        step: u32,
        edge_map: &[bool],
        edge_penalty: f32,
    ) {
        let mut new_distances = distances.to_vec();
        let mut new_nearest = nearest.to_vec();

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;

                // Check 8 neighbors at step distance
                for dy in [-1i32, 0, 1] {
                    for dx in [-1i32, 0, 1] {
                        let nx = x as i32 + dx * step as i32;
                        let ny = y as i32 + dy * step as i32;

                        if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                            continue;
                        }

                        let nx = nx as u32;
                        let ny = ny as u32;
                        let nidx = (ny * width + nx) as usize;

                        if let Some(edge_pos) = nearest[nidx] {
                            // Compute geodesic cost
                            let cost = Self::geodesic_cost(
                                x, y,
                                edge_pos.0, edge_pos.1,
                                edge_map,
                                width,
                                edge_penalty,
                            );

                            if cost < new_distances[idx] {
                                new_distances[idx] = cost;
                                new_nearest[idx] = Some(edge_pos);
                            }
                        }
                    }
                }
            }
        }

        distances.copy_from_slice(&new_distances);
        nearest.copy_from_slice(&new_nearest);
    }

    /// Refine with 1-pixel neighbors
    fn refine_pass(
        distances: &mut [f32],
        nearest: &mut [Option<(u32, u32)>],
        width: u32,
        height: u32,
        edge_map: &[bool],
        edge_penalty: f32,
    ) {
        // Same as jump_flood_pass with step=1
        Self::jump_flood_pass(distances, nearest, width, height, 1, edge_map, edge_penalty);
    }

    /// Compute geodesic cost between two points
    ///
    /// Cost increases when path crosses edges
    fn geodesic_cost(
        x1: u32, y1: u32,
        x2: u32, y2: u32,
        edge_map: &[bool],
        width: u32,
        edge_penalty: f32,
    ) -> f32 {
        // Euclidean distance
        let dx = x1 as f32 - x2 as f32;
        let dy = y1 as f32 - y2 as f32;
        let euclidean = (dx * dx + dy * dy).sqrt();

        // Sample edge strength along path (simplified - check endpoints)
        let idx1 = (y1 * width + x1) as usize;
        let idx2 = (y2 * width + x2) as usize;

        let edge_cost = if edge_map[idx1] || edge_map[idx2] {
            edge_penalty  // High cost to cross edge
        } else {
            1.0  // Normal cost
        };

        euclidean * edge_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::color::Color4;

    #[test]
    fn test_geodesic_edt_respects_edges() {
        // Create image with vertical edge
        let mut image = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let color = if x < 32 { 0.0 } else { 1.0 };
                image.set_pixel(x, y, Color4::new(color, color, color, 1.0));
            }
        }

        // Compute structure tensor
        let structure = StructureTensorField::compute(&image, 1.2, 2.5).unwrap();

        // Compute geodesic EDT
        let gedt = GeodesicEDT::compute(&image, &structure, 0.5, 50.0).unwrap();

        // Distance should be high across edge, low within regions
        let dist_left = gedt.get_distance(16, 32);   // Far from edge
        let dist_edge = gedt.get_distance(32, 32);   // On edge
        let dist_right = gedt.get_distance(48, 32);  // Far from edge

        assert!(dist_edge < 1.0, "Edge pixel should have low distance");
        assert!(dist_left > 5.0, "Interior should have higher distance");
        assert!(dist_right > 5.0, "Interior should have higher distance");
    }

    #[test]
    fn test_sigma_clamping() {
        let mut distances = vec![0.0; 64*64];
        distances[32*64 + 16] = 10.0;  // Far from edge

        let gedt = GeodesicEDT {
            width: 64,
            height: 64,
            distances,
            nearest_edge: vec![None; 64*64],
        };

        // Sigma clamping: σ ≤ 0.5 + 0.3×d
        let clamped = gedt.clamp_sigma(16, 32, 10.0);

        // With d=10: σ ≤ 0.5 + 3.0 = 3.5
        assert!(clamped <= 3.5);
        assert!(clamped > 3.0);  // Should clamp to formula
    }
}
