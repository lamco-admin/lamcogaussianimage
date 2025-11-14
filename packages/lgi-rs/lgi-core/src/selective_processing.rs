//! Selective Processing Framework
//!
//! Apply advanced techniques only where needed (not entire image)
//! From research: Spatial masking + multi-resolution + region-growing
//! Commercial-grade adaptive processing

use crate::{ImageBuffer, analytical_triggers::AnalyticalTriggers};
use lgi_math::color::Color4;

/// Complexity map for selective processing
#[derive(Clone)]
pub struct ComplexityMap {
    pub width: u32,
    pub height: u32,

    /// Gradient magnitude (edge strength)
    pub gradient_map: Vec<f32>,

    /// Local variance (texture complexity)
    pub variance_map: Vec<f32>,

    /// Combined complexity score [0, 1]
    pub complexity_score: Vec<f32>,
}

impl ComplexityMap {
    /// Compute comprehensive complexity analysis
    pub fn compute(image: &ImageBuffer<f32>) -> Self {
        let width = image.width;
        let height = image.height;

        let gradient_map = compute_gradient_magnitude(image);
        let variance_map = compute_local_variance(image, 5); // 5px window

        // Combine gradient and variance into complexity score
        let mut complexity_score = vec![0.0; (width * height) as usize];

        // Normalize gradients
        let max_grad = gradient_map.iter().cloned().fold(0.0f32, f32::max);
        // Normalize variance
        let max_var = variance_map.iter().cloned().fold(0.0f32, f32::max);

        for i in 0..(width * height) as usize {
            let norm_grad = if max_grad > 1e-6 { gradient_map[i] / max_grad } else { 0.0 };
            let norm_var = if max_var > 1e-6 { variance_map[i] / max_var } else { 0.0 };

            // Weighted combination
            complexity_score[i] = 0.6 * norm_grad + 0.4 * norm_var;
        }

        Self {
            width,
            height,
            gradient_map,
            variance_map,
            complexity_score,
        }
    }

    /// Generate binary mask for high-complexity regions
    pub fn threshold_mask(&self, threshold: f32) -> Vec<bool> {
        self.complexity_score.iter().map(|&s| s > threshold).collect()
    }

    /// Generate soft mask (0-1) for weighted processing
    pub fn soft_mask(&self, threshold: f32, falloff: f32) -> Vec<f32> {
        self.complexity_score.iter().map(|&s| {
            if s > threshold {
                ((s - threshold) / falloff).min(1.0)
            } else {
                0.0
            }
        }).collect()
    }

    /// Grow regions using connected components
    ///
    /// Prevents pixel-level noise, creates coherent processing regions
    pub fn region_growing(&self, initial_mask: &[bool], min_region_size: usize) -> Vec<bool> {
        let mut mask = initial_mask.to_vec();
        let mut visited = vec![false; mask.len()];
        let mut final_mask = vec![false; mask.len()];

        for start_idx in 0..mask.len() {
            if mask[start_idx] && !visited[start_idx] {
                // Flood fill to find connected region
                let region = self.flood_fill(start_idx, &mask, &mut visited);

                // Keep region if large enough
                if region.len() >= min_region_size {
                    for idx in region {
                        final_mask[idx] = true;
                    }
                }
            }
        }

        final_mask
    }

    /// Flood fill for connected component detection
    fn flood_fill(&self, start: usize, mask: &[bool], visited: &mut [bool]) -> Vec<usize> {
        let mut region = Vec::new();
        let mut stack = vec![start];

        while let Some(idx) = stack.pop() {
            if visited[idx] || !mask[idx] {
                continue;
            }

            visited[idx] = true;
            region.push(idx);

            // Add 4-connected neighbors
            let x = (idx as u32) % self.width;
            let y = (idx as u32) / self.width;

            for (dx, dy) in [(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                    let neighbor_idx = (ny as u32 * self.width + nx as u32) as usize;
                    stack.push(neighbor_idx);
                }
            }
        }

        region
    }
}

/// Selective processor - applies techniques only where needed
pub struct SelectiveProcessor {
    pub complexity_map: ComplexityMap,
    pub triggers: AnalyticalTriggers,
}

impl SelectiveProcessor {
    /// Create from image analysis
    pub fn new(
        original: &ImageBuffer<f32>,
        rendered: &ImageBuffer<f32>,
        gaussians: &[lgi_math::gaussian::Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>],
        structure_tensor: &crate::StructureTensorField,
    ) -> Self {
        let complexity_map = ComplexityMap::compute(original);
        let triggers = AnalyticalTriggers::analyze(original, rendered, gaussians, structure_tensor);

        Self {
            complexity_map,
            triggers,
        }
    }

    /// Get texture recommendation mask
    pub fn texture_mask(&self, threshold: f32) -> Vec<bool> {
        // High complexity OR SED trigger
        self.complexity_map.threshold_mask(threshold)
    }

    /// Get residual recommendation mask
    pub fn residual_mask(&self, threshold: f32) -> Vec<bool> {
        // Very high complexity OR ERR trigger
        self.complexity_map.threshold_mask(threshold * 1.5)
    }

    /// Get refinement recommendation mask
    pub fn refinement_mask(&self) -> Vec<bool> {
        // LCC or AGD triggers
        self.triggers.refinement_mask.iter().map(|&v| v > 0.5).collect()
    }
}

// Helper functions

fn compute_gradient_magnitude(image: &ImageBuffer<f32>) -> Vec<f32> {
    let mut result = vec![0.0; (image.width * image.height) as usize];

    for y in 1..image.height-1 {
        for x in 1..image.width-1 {
            if let (Some(c), Some(cx), Some(cy)) = (
                image.get_pixel(x, y),
                image.get_pixel(x+1, y),
                image.get_pixel(x, y+1),
            ) {
                let gx = ((cx.r - c.r) + (cx.g - c.g) + (cx.b - c.b)) / 3.0;
                let gy = ((cy.r - c.r) + (cy.g - c.g) + (cy.b - c.b)) / 3.0;

                let magnitude = (gx * gx + gy * gy).sqrt();
                result[(y * image.width + x) as usize] = magnitude;
            }
        }
    }

    result
}

fn compute_local_variance(image: &ImageBuffer<f32>, window: u32) -> Vec<f32> {
    let mut result = vec![0.0; (image.width * image.height) as usize];
    let w = window as i32;

    for y in 0..image.height {
        for x in 0..image.width {
            // Compute local mean
            let mut sum = 0.0;
            let mut count = 0.0;

            for dy in -w..=w {
                for dx in -w..=w {
                    let nx = (x as i32 + dx).clamp(0, image.width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, image.height as i32 - 1) as u32;

                    if let Some(pixel) = image.get_pixel(nx, ny) {
                        let intensity = (pixel.r + pixel.g + pixel.b) / 3.0;
                        sum += intensity;
                        count += 1.0;
                    }
                }
            }

            let mean = sum / count;

            // Compute variance
            let mut variance = 0.0;

            for dy in -w..=w {
                for dx in -w..=w {
                    let nx = (x as i32 + dx).clamp(0, image.width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, image.height as i32 - 1) as u32;

                    if let Some(pixel) = image.get_pixel(nx, ny) {
                        let intensity = (pixel.r + pixel.g + pixel.b) / 3.0;
                        let diff = intensity - mean;
                        variance += diff * diff;
                    }
                }
            }

            result[(y * image.width + x) as usize] = variance / count;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_map() {
        let mut img = ImageBuffer::new(64, 64);

        // Create edge
        for y in 0..64 {
            for x in 0..64 {
                let val = if x < 32 { 0.0 } else { 1.0 };
                img.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let complexity = ComplexityMap::compute(&img);

        // Complexity should be high at edge
        let edge_complexity = complexity.complexity_score[(32 * 64 + 32) as usize];
        let flat_complexity = complexity.complexity_score[(16 * 64 + 16) as usize];

        assert!(edge_complexity > flat_complexity);
    }

    #[test]
    fn test_masking() {
        let mut img = ImageBuffer::new(64, 64);

        for y in 0..64 {
            for x in 0..64 {
                let val = if x < 32 { 0.0 } else { 1.0 };
                img.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let complexity = ComplexityMap::compute(&img);
        let mask = complexity.threshold_mask(0.5);

        // Should have some masked pixels (high complexity regions)
        let masked_count = mask.iter().filter(|&&m| m).count();
        assert!(masked_count > 0);
        assert!(masked_count < mask.len());  // Not all pixels
    }
}
