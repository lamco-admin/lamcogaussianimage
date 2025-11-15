//! Content Type Detection and Adaptive Strategies
//!
//! Automatically detect image content type and select optimal parameters
//! Commercial-grade adaptive processing

use crate::{ImageBuffer, StructureTensorField};

/// Image content type classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContentType {
    /// Smooth gradients, uniform regions (γ=0.6-0.8)
    Smooth,

    /// Sharp edges, boundaries (γ=0.5-0.6)
    Sharp,

    /// Mixed smooth and sharp (γ=0.6-0.7)
    Mixed,

    /// High-frequency patterns, text (γ=0.4-0.5)
    HighFrequency,

    /// Photos with natural variation (γ=0.7-0.9)
    Photo,
}

/// Content analyzer
pub struct ContentAnalyzer;

impl ContentAnalyzer {
    /// Detect content type from image
    ///
    /// Uses gradient statistics, coherence, and entropy
    pub fn detect_content_type(
        image: &ImageBuffer<f32>,
        structure_tensor: &StructureTensorField,
    ) -> ContentType {
        // Compute statistics
        let avg_gradient = Self::average_gradient_magnitude(image);
        let avg_coherence = Self::average_coherence(structure_tensor);
        let entropy = Self::compute_entropy(image);
        let edge_density = Self::edge_density(structure_tensor);

        // Decision tree based on statistics
        if entropy > 6.0 && edge_density > 0.3 {
            // High entropy + many edges = high-frequency or text
            ContentType::HighFrequency
        } else if entropy < 2.0 && avg_coherence < 0.5 {
            // Low entropy (few distinct values) + low coherence = sharp edges
            ContentType::Sharp
        } else if avg_coherence > 0.8 || avg_gradient < 0.01 {
            // High coherence (aligned gradients everywhere) = smooth gradients
            // OR very low gradient magnitude = smooth/flat
            ContentType::Smooth
        } else if entropy > 5.0 {
            // High entropy, moderate everything = photo
            ContentType::Photo
        } else {
            // Mixed characteristics
            ContentType::Mixed
        }
    }

    /// Get optimal gamma for content type and Gaussian count
    pub fn optimal_gamma(content: ContentType, num_gaussians: usize) -> f32 {
        // Base gamma from N
        let base_gamma = match num_gaussians {
            n if n < 100 => 1.2,
            n if n < 200 => 0.8,
            n if n < 500 => 0.7,
            n if n < 1000 => 0.6,
            n if n < 2000 => 0.5,
            _ => 0.4,
        };

        // Adjust by content type (from EXP-020 data)
        match content {
            ContentType::Smooth => base_gamma * 1.1,      // Larger for smooth
            ContentType::Sharp => base_gamma * 0.9,       // Smaller for sharp
            ContentType::Mixed => base_gamma,              // Use base
            ContentType::HighFrequency => base_gamma * 0.8, // Smaller for detail
            ContentType::Photo => base_gamma * 1.05,      // Slightly larger for photos
        }
    }

    /// Get optimal learning rates for content
    pub fn optimal_learning_rates(content: ContentType) -> (f32, f32, f32) {
        // (LR_color, LR_position, LR_scale)
        match content {
            ContentType::Smooth => (0.3, 0.05, 0.05),      // Standard
            ContentType::Sharp => (0.2, 0.03, 0.03),       // More conservative
            ContentType::Mixed => (0.25, 0.04, 0.04),      // Between
            ContentType::HighFrequency => (0.15, 0.02, 0.02), // Very conservative
            ContentType::Photo => (0.25, 0.04, 0.05),      // Balanced
        }
    }

    // Helper methods

    fn average_gradient_magnitude(image: &ImageBuffer<f32>) -> f32 {
        let mut sum = 0.0;
        let mut count = 0.0;

        for y in 1..image.height-1 {
            for x in 1..image.width-1 {
                if let (Some(c), Some(cx), Some(cy)) = (
                    image.get_pixel(x, y),
                    image.get_pixel(x+1, y),
                    image.get_pixel(x, y+1),
                ) {
                    let gx = ((cx.r - c.r) + (cx.g - c.g) + (cx.b - c.b)) / 3.0;
                    let gy = ((cy.r - c.r) + (cy.g - c.g) + (cy.b - c.b)) / 3.0;

                    sum += (gx * gx + gy * gy).sqrt();
                    count += 1.0;
                }
            }
        }

        if count > 0.0 { sum / count } else { 0.0 }
    }

    fn average_coherence(structure_tensor: &StructureTensorField) -> f32 {
        let mut sum = 0.0;
        let mut count = 0.0;

        for y in 0..structure_tensor.height {
            for x in 0..structure_tensor.width {
                sum += structure_tensor.get(x, y).coherence;
                count += 1.0;
            }
        }

        if count > 0.0 { sum / count } else { 0.0 }
    }

    fn compute_entropy(image: &ImageBuffer<f32>) -> f32 {
        let mut histogram = vec![0u32; 256];

        for pixel in &image.data {
            let intensity = ((pixel.r + pixel.g + pixel.b) / 3.0 * 255.0) as usize;
            let bin = intensity.min(255);
            histogram[bin] += 1;
        }

        let total = image.data.len() as f32;
        let mut entropy = 0.0;

        for &count in &histogram {
            if count > 0 {
                let p = count as f32 / total;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    fn edge_density(structure_tensor: &StructureTensorField) -> f32 {
        let mut edge_count = 0.0;
        let total = (structure_tensor.width * structure_tensor.height) as f32;

        for y in 0..structure_tensor.height {
            for x in 0..structure_tensor.width {
                if structure_tensor.get(x, y).coherence > 0.5 {
                    edge_count += 1.0;
                }
            }
        }

        edge_count / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::color::Color4;

    #[test]
    fn test_smooth_detection() {
        let mut img = ImageBuffer::new(64, 64);

        // Smooth gradient
        for y in 0..64 {
            for x in 0..64 {
                let val = x as f32 / 64.0;
                img.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let tensor = StructureTensorField::compute(&img, 1.2, 1.0).unwrap();
        let content_type = ContentAnalyzer::detect_content_type(&img, &tensor);

        // Should detect as smooth (low gradients, low coherence)
        assert_eq!(content_type, ContentType::Smooth);
    }

    #[test]
    fn test_sharp_detection() {
        let mut img = ImageBuffer::new(64, 64);

        // Sharp edge
        for y in 0..64 {
            for x in 0..64 {
                let val = if x < 32 { 0.0 } else { 1.0 };
                img.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let tensor = StructureTensorField::compute(&img, 1.2, 1.0).unwrap();
        let content_type = ContentAnalyzer::detect_content_type(&img, &tensor);

        // Should detect as sharp (high coherence at edges)
        assert_eq!(content_type, ContentType::Sharp);
    }
}
