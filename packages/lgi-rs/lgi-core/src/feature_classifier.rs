//! Per-Pixel Feature Classification for Adaptive Gaussian Encoding
//!
//! This module combines structure tensor, Laplacian, and luminance analysis
//! to classify each pixel into actionable feature types that drive
//! Gaussian initialization and optimization strategies.
//!
//! # Feature Types
//!
//! | Type | Characteristics | Gaussian Strategy |
//! |------|-----------------|-------------------|
//! | Smooth | Low energy, low texture | Large overlapping, few needed |
//! | SmoothGradient | Low energy, color varying | Medium, color-interpolating |
//! | Edge | High coherence, high energy | Anisotropic, aligned to edge |
//! | Corner | Low coherence, high energy | Small isotropic, precise |
//! | TextStroke | Very high coherence, thin | Very small, elongated along stroke |
//! | Texture | High local variance | Medium, potentially anisotropic |
//! | Specular | Very high luminance | Small, high opacity |
//!
//! # Usage
//!
//! ```ignore
//! use lgi_core::{ImageBuffer, feature_classifier::{FeatureClassifier, FeatureType}};
//!
//! let image = ImageBuffer::load("photo.png")?;
//! let classifier = FeatureClassifier::default();
//! let feature_map = classifier.classify(&image)?;
//!
//! // Count feature types
//! let edge_count = feature_map.iter().filter(|&&f| f == FeatureType::Edge).count();
//! ```

use crate::{ImageBuffer, structure_tensor::StructureTensorField, laplacian::compute_laplacian};

/// Feature type for each pixel - drives Gaussian strategy selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureType {
    /// Smooth region - large Gaussians, low density needed
    Smooth,

    /// Smooth but color-varying (gradient) - medium Gaussians
    SmoothGradient,

    /// Edge pixel - anisotropic Gaussians aligned to edge tangent
    Edge,

    /// Corner or junction - small isotropic Gaussians
    Corner,

    /// Text stroke - very small elongated Gaussians
    TextStroke,

    /// Textured region - medium Gaussians, potentially directional
    Texture,

    /// Specular highlight - small bright Gaussians
    Specular,
}

impl FeatureType {
    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            FeatureType::Smooth => "Smooth",
            FeatureType::SmoothGradient => "SmoothGradient",
            FeatureType::Edge => "Edge",
            FeatureType::Corner => "Corner",
            FeatureType::TextStroke => "TextStroke",
            FeatureType::Texture => "Texture",
            FeatureType::Specular => "Specular",
        }
    }

    /// Suggested scale range (σ_min, σ_max) relative to image dimension
    pub fn scale_range(&self) -> (f32, f32) {
        match self {
            FeatureType::Smooth => (0.05, 0.15),       // Large
            FeatureType::SmoothGradient => (0.03, 0.08),
            FeatureType::Edge => (0.01, 0.05),         // Medium-thin
            FeatureType::Corner => (0.005, 0.02),      // Small
            FeatureType::TextStroke => (0.002, 0.01),  // Very small
            FeatureType::Texture => (0.02, 0.06),      // Medium
            FeatureType::Specular => (0.005, 0.02),    // Small
        }
    }

    /// Suggested anisotropy range (1.0 = isotropic)
    pub fn anisotropy_range(&self) -> (f32, f32) {
        match self {
            FeatureType::Smooth => (1.0, 1.5),         // Nearly isotropic
            FeatureType::SmoothGradient => (1.0, 2.0),
            FeatureType::Edge => (2.0, 8.0),           // Highly anisotropic
            FeatureType::Corner => (1.0, 1.2),         // Isotropic
            FeatureType::TextStroke => (3.0, 15.0),    // Very elongated
            FeatureType::Texture => (1.0, 3.0),        // Depends on directionality
            FeatureType::Specular => (1.0, 1.5),       // Nearly isotropic
        }
    }

    /// Suggested density multiplier (relative to average)
    pub fn density_multiplier(&self) -> f32 {
        match self {
            FeatureType::Smooth => 0.3,       // Few needed
            FeatureType::SmoothGradient => 0.5,
            FeatureType::Edge => 2.0,         // Many needed for sharpness
            FeatureType::Corner => 3.0,       // Critical points
            FeatureType::TextStroke => 4.0,   // High density for readability
            FeatureType::Texture => 1.5,      // Above average
            FeatureType::Specular => 2.0,     // Need precise capture
        }
    }

    /// Should use structure tensor orientation?
    pub fn use_orientation(&self) -> bool {
        matches!(self, FeatureType::Edge | FeatureType::TextStroke | FeatureType::Texture)
    }

    /// Should use geodesic sigma clamping?
    pub fn use_geodesic_clamping(&self) -> bool {
        matches!(self, FeatureType::Edge | FeatureType::TextStroke | FeatureType::Corner)
    }
}

/// Classification thresholds - tunable parameters
#[derive(Debug, Clone)]
pub struct ClassifierThresholds {
    /// Structure tensor coherence threshold for edge detection (default: 0.6)
    pub edge_coherence: f32,

    /// Structure tensor coherence threshold for corner detection (default: 0.2)
    pub corner_coherence: f32,

    /// Structure tensor energy threshold for corner detection (default: 0.05)
    pub corner_energy: f32,

    /// Structure tensor energy threshold for flat/smooth regions (default: 0.01)
    pub smooth_energy: f32,

    /// Structure tensor coherence for text stroke (default: 0.85)
    pub text_coherence: f32,

    /// Maximum width estimate for text stroke (default: 3.0 pixels)
    pub text_width_max: f32,

    /// Luminance percentile for specular highlights (default: 0.97)
    pub specular_percentile: f32,

    /// Local variance threshold for texture (default: 0.015)
    pub texture_variance: f32,

    /// Color gradient threshold for smooth gradient regions (default: 0.02)
    pub gradient_threshold: f32,

    /// Structure tensor smoothing sigma (default: 1.2)
    pub sigma_gradient: f32,

    /// Structure tensor integration sigma (default: 2.5)
    pub sigma_smooth: f32,

    /// Local variance window size (default: 5)
    pub variance_window: usize,
}

impl Default for ClassifierThresholds {
    fn default() -> Self {
        Self {
            edge_coherence: 0.6,
            corner_coherence: 0.2,
            corner_energy: 0.05,
            smooth_energy: 0.01,
            text_coherence: 0.85,
            text_width_max: 3.0,
            specular_percentile: 0.97,
            texture_variance: 0.015,
            gradient_threshold: 0.02,
            sigma_gradient: 1.2,
            sigma_smooth: 2.5,
            variance_window: 5,
        }
    }
}

/// Per-pixel feature classifier
pub struct FeatureClassifier {
    /// Classification thresholds
    pub thresholds: ClassifierThresholds,
}

impl Default for FeatureClassifier {
    fn default() -> Self {
        Self {
            thresholds: ClassifierThresholds::default(),
        }
    }
}

impl FeatureClassifier {
    /// Create classifier with custom thresholds
    pub fn with_thresholds(thresholds: ClassifierThresholds) -> Self {
        Self { thresholds }
    }

    /// Classify every pixel in the image
    ///
    /// Returns a Vec<FeatureType> with one entry per pixel (row-major order).
    ///
    /// # Algorithm
    ///
    /// 1. Compute structure tensor field (coherence, energy, orientation)
    /// 2. Compute local variance map (texture indicator)
    /// 3. Compute color gradient magnitude (gradient regions)
    /// 4. Detect specular highlights (luminance threshold)
    /// 5. Apply decision tree per pixel:
    ///    - Specular: luminance > 97th percentile
    ///    - TextStroke: coherence > 0.85 AND width < 3px
    ///    - Corner: coherence < 0.2 AND energy > 0.05
    ///    - Edge: coherence > 0.6
    ///    - Texture: local variance > 0.015
    ///    - SmoothGradient: energy < 0.01 AND color gradient > 0.02
    ///    - Smooth: everything else
    pub fn classify(&self, image: &ImageBuffer<f32>) -> crate::Result<ClassificationResult> {
        let width = image.width;
        let height = image.height;
        let n_pixels = (width * height) as usize;

        // Step 1: Compute structure tensor field
        let tensor_field = StructureTensorField::compute(
            image,
            self.thresholds.sigma_gradient,
            self.thresholds.sigma_smooth,
        )?;

        // Step 2: Compute local variance (texture indicator)
        let local_variance = self.compute_local_variance(image);

        // Step 3: Compute color gradient magnitude
        let color_gradient = self.compute_color_gradient(image);

        // Step 4: Compute luminance and find specular threshold
        let luminance = self.extract_luminance(image);
        let specular_threshold = self.compute_percentile(&luminance, self.thresholds.specular_percentile);

        // Step 5: Classify each pixel
        let mut feature_map = vec![FeatureType::Smooth; n_pixels];
        let mut counts = FeatureCounts::default();

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let tensor = tensor_field.get(x, y);
                let energy = tensor.eigenvalue_major + tensor.eigenvalue_minor;

                // Decision tree (order matters - more specific first)
                let feature = if luminance[idx] > specular_threshold && specular_threshold > 0.8 {
                    // Specular highlight: very bright
                    FeatureType::Specular
                } else if tensor.coherence > self.thresholds.text_coherence {
                    // Potential text stroke: check width
                    let width_estimate = tensor.eigenvalue_minor.sqrt() * 2.0;
                    if width_estimate < self.thresholds.text_width_max {
                        FeatureType::TextStroke
                    } else {
                        FeatureType::Edge
                    }
                } else if tensor.coherence < self.thresholds.corner_coherence
                          && energy > self.thresholds.corner_energy {
                    // Corner: low coherence but high energy
                    FeatureType::Corner
                } else if tensor.coherence > self.thresholds.edge_coherence {
                    // Edge: high coherence
                    FeatureType::Edge
                } else if local_variance[idx] > self.thresholds.texture_variance {
                    // Texture: high local variance
                    FeatureType::Texture
                } else if energy < self.thresholds.smooth_energy {
                    // Smooth region: low energy
                    if color_gradient[idx] > self.thresholds.gradient_threshold {
                        FeatureType::SmoothGradient
                    } else {
                        FeatureType::Smooth
                    }
                } else {
                    // Default: texture if variance is moderate, else smooth
                    if local_variance[idx] > self.thresholds.texture_variance * 0.5 {
                        FeatureType::Texture
                    } else {
                        FeatureType::SmoothGradient
                    }
                };

                feature_map[idx] = feature;
                counts.increment(feature);
            }
        }

        Ok(ClassificationResult {
            width,
            height,
            feature_map,
            counts,
            tensor_field,
            local_variance,
        })
    }

    /// Compute local variance in a window around each pixel
    fn compute_local_variance(&self, image: &ImageBuffer<f32>) -> Vec<f32> {
        let width = image.width as usize;
        let height = image.height as usize;
        let mut variance = vec![0.0; width * height];
        let half = self.thresholds.variance_window / 2;

        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                let mut count = 0;

                let y_start = y.saturating_sub(half);
                let y_end = (y + half + 1).min(height);
                let x_start = x.saturating_sub(half);
                let x_end = (x + half + 1).min(width);

                for wy in y_start..y_end {
                    for wx in x_start..x_end {
                        let pixel = image.get_pixel(wx as u32, wy as u32).unwrap();
                        let luma = 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;
                        sum += luma;
                        sum_sq += luma * luma;
                        count += 1;
                    }
                }

                let mean = sum / count as f32;
                let var = (sum_sq / count as f32) - mean * mean;
                variance[y * width + x] = var.max(0.0);
            }
        }

        variance
    }

    /// Compute color gradient magnitude (change in color across neighbors)
    fn compute_color_gradient(&self, image: &ImageBuffer<f32>) -> Vec<f32> {
        let width = image.width;
        let height = image.height;
        let mut gradient = vec![0.0; (width * height) as usize];

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let idx = (y * width + x) as usize;
                let center = image.get_pixel(x, y).unwrap();

                // Sobel-like gradient in color space
                let left = image.get_pixel(x - 1, y).unwrap();
                let right = image.get_pixel(x + 1, y).unwrap();
                let up = image.get_pixel(x, y - 1).unwrap();
                let down = image.get_pixel(x, y + 1).unwrap();

                let dx_r = (right.r - left.r) / 2.0;
                let dx_g = (right.g - left.g) / 2.0;
                let dx_b = (right.b - left.b) / 2.0;

                let dy_r = (down.r - up.r) / 2.0;
                let dy_g = (down.g - up.g) / 2.0;
                let dy_b = (down.b - up.b) / 2.0;

                // Combined magnitude
                let mag = ((dx_r*dx_r + dx_g*dx_g + dx_b*dx_b +
                           dy_r*dy_r + dy_g*dy_g + dy_b*dy_b) / 6.0).sqrt();
                gradient[idx] = mag;
            }
        }

        gradient
    }

    /// Extract luminance channel
    fn extract_luminance(&self, image: &ImageBuffer<f32>) -> Vec<f32> {
        image.data.iter()
            .map(|p| 0.2126 * p.r + 0.7152 * p.g + 0.0722 * p.b)
            .collect()
    }

    /// Compute percentile value
    fn compute_percentile(&self, data: &[f32], percentile: f32) -> f32 {
        let mut sorted: Vec<f32> = data.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((sorted.len() - 1) as f32 * percentile) as usize;
        sorted[idx]
    }
}

/// Counts of each feature type
#[derive(Debug, Clone, Default)]
pub struct FeatureCounts {
    pub smooth: usize,
    pub smooth_gradient: usize,
    pub edge: usize,
    pub corner: usize,
    pub text_stroke: usize,
    pub texture: usize,
    pub specular: usize,
}

impl FeatureCounts {
    fn increment(&mut self, feature: FeatureType) {
        match feature {
            FeatureType::Smooth => self.smooth += 1,
            FeatureType::SmoothGradient => self.smooth_gradient += 1,
            FeatureType::Edge => self.edge += 1,
            FeatureType::Corner => self.corner += 1,
            FeatureType::TextStroke => self.text_stroke += 1,
            FeatureType::Texture => self.texture += 1,
            FeatureType::Specular => self.specular += 1,
        }
    }

    /// Total pixel count
    pub fn total(&self) -> usize {
        self.smooth + self.smooth_gradient + self.edge + self.corner +
        self.text_stroke + self.texture + self.specular
    }

    /// Percentage of each type
    pub fn percentages(&self) -> [(FeatureType, f32); 7] {
        let total = self.total() as f32;
        [
            (FeatureType::Smooth, self.smooth as f32 / total * 100.0),
            (FeatureType::SmoothGradient, self.smooth_gradient as f32 / total * 100.0),
            (FeatureType::Edge, self.edge as f32 / total * 100.0),
            (FeatureType::Corner, self.corner as f32 / total * 100.0),
            (FeatureType::TextStroke, self.text_stroke as f32 / total * 100.0),
            (FeatureType::Texture, self.texture as f32 / total * 100.0),
            (FeatureType::Specular, self.specular as f32 / total * 100.0),
        ]
    }
}

/// Complete classification result with supporting data
pub struct ClassificationResult {
    /// Image width
    pub width: u32,

    /// Image height
    pub height: u32,

    /// Per-pixel feature classification (row-major)
    pub feature_map: Vec<FeatureType>,

    /// Counts of each feature type
    pub counts: FeatureCounts,

    /// Structure tensor field (for orientation data)
    pub tensor_field: StructureTensorField,

    /// Local variance map (for texture analysis)
    pub local_variance: Vec<f32>,
}

impl ClassificationResult {
    /// Get feature type at pixel (x, y)
    pub fn get(&self, x: u32, y: u32) -> FeatureType {
        self.feature_map[(y * self.width + x) as usize]
    }

    /// Create a visualization of the feature map as an RGB image
    ///
    /// Color coding:
    /// - Smooth: Dark blue (0, 0, 100)
    /// - SmoothGradient: Light blue (100, 150, 255)
    /// - Edge: Green (0, 200, 0)
    /// - Corner: Red (255, 0, 0)
    /// - TextStroke: Yellow (255, 255, 0)
    /// - Texture: Orange (255, 150, 0)
    /// - Specular: White (255, 255, 255)
    pub fn visualize(&self) -> ImageBuffer<f32> {
        use lgi_math::color::Color4;

        let mut vis = ImageBuffer::new(self.width, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let feature = self.get(x, y);
                let color = match feature {
                    FeatureType::Smooth => Color4::new(0.0, 0.0, 0.4, 1.0),
                    FeatureType::SmoothGradient => Color4::new(0.4, 0.6, 1.0, 1.0),
                    FeatureType::Edge => Color4::new(0.0, 0.8, 0.0, 1.0),
                    FeatureType::Corner => Color4::new(1.0, 0.0, 0.0, 1.0),
                    FeatureType::TextStroke => Color4::new(1.0, 1.0, 0.0, 1.0),
                    FeatureType::Texture => Color4::new(1.0, 0.6, 0.0, 1.0),
                    FeatureType::Specular => Color4::new(1.0, 1.0, 1.0, 1.0),
                };
                vis.set_pixel(x, y, color);
            }
        }

        vis
    }

    /// Generate Gaussian budget allocation based on feature distribution
    ///
    /// Returns a map from FeatureType to suggested count
    pub fn allocate_budget(&self, total_gaussians: usize) -> std::collections::HashMap<FeatureType, usize> {
        use std::collections::HashMap;

        let mut allocation = HashMap::new();

        // Weight each feature type by count × density_multiplier
        let weights: Vec<(FeatureType, f32)> = self.counts.percentages().iter()
            .map(|(ft, pct)| (*ft, pct * ft.density_multiplier()))
            .collect();

        let total_weight: f32 = weights.iter().map(|(_, w)| w).sum();

        // Allocate proportionally
        let mut allocated = 0;
        for (ft, weight) in &weights {
            let count = ((weight / total_weight) * total_gaussians as f32).round() as usize;
            allocation.insert(*ft, count);
            allocated += count;
        }

        // Adjust for rounding errors
        if allocated < total_gaussians {
            // Add remainder to edges (usually most important)
            *allocation.entry(FeatureType::Edge).or_insert(0) += total_gaussians - allocated;
        }

        allocation
    }

    /// Get pixel positions for a specific feature type
    pub fn positions_for_feature(&self, feature: FeatureType) -> Vec<(u32, u32)> {
        let mut positions = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) == feature {
                    positions.push((x, y));
                }
            }
        }
        positions
    }

    /// Sample positions for a feature type using importance weighting
    ///
    /// Uses structure tensor energy as importance for edges,
    /// local variance for texture, uniform for smooth regions.
    pub fn sample_positions(
        &self,
        feature: FeatureType,
        count: usize,
        seed: u64,
    ) -> Vec<(f32, f32)> {
        use std::collections::HashSet;

        let positions = self.positions_for_feature(feature);
        if positions.is_empty() {
            return Vec::new();
        }

        // Build importance weights
        let weights: Vec<f32> = positions.iter()
            .map(|&(x, y)| {
                let idx = (y * self.width + x) as usize;
                match feature {
                    FeatureType::Edge | FeatureType::Corner | FeatureType::TextStroke => {
                        let t = self.tensor_field.get(x, y);
                        t.eigenvalue_major + t.eigenvalue_minor + 0.001
                    }
                    FeatureType::Texture => {
                        self.local_variance[idx] + 0.001
                    }
                    _ => 1.0,
                }
            })
            .collect();

        // Weighted sampling (reservoir sampling with weights)
        let total_weight: f32 = weights.iter().sum();
        let mut selected = HashSet::new();
        let mut result = Vec::with_capacity(count.min(positions.len()));

        // Simple weighted sampling with LCG for reproducibility
        let mut rng_state = seed;
        let lcg_next = |state: &mut u64| -> f32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state >> 33) as f32 / (1u64 << 31) as f32
        };

        while result.len() < count && result.len() < positions.len() {
            let target = lcg_next(&mut rng_state) * total_weight;
            let mut cumulative = 0.0;

            for (i, &w) in weights.iter().enumerate() {
                cumulative += w;
                if cumulative >= target && !selected.contains(&i) {
                    selected.insert(i);
                    let (x, y) = positions[i];
                    result.push((x as f32, y as f32));
                    break;
                }
            }

            // Fallback: random selection if weighted sampling fails
            if result.len() == selected.len() - 1 {
                let idx = (lcg_next(&mut rng_state) * positions.len() as f32) as usize % positions.len();
                if !selected.contains(&idx) {
                    selected.insert(idx);
                    let (x, y) = positions[idx];
                    result.push((x as f32, y as f32));
                }
            }
        }

        result
    }
}

/// Print summary of classification results
pub fn print_classification_summary(result: &ClassificationResult) {
    println!("Feature Classification Summary:");
    println!("  Image: {}x{} ({} pixels)", result.width, result.height, result.counts.total());
    println!();
    println!("  Feature Distribution:");
    for (ft, pct) in result.counts.percentages() {
        let count = match ft {
            FeatureType::Smooth => result.counts.smooth,
            FeatureType::SmoothGradient => result.counts.smooth_gradient,
            FeatureType::Edge => result.counts.edge,
            FeatureType::Corner => result.counts.corner,
            FeatureType::TextStroke => result.counts.text_stroke,
            FeatureType::Texture => result.counts.texture,
            FeatureType::Specular => result.counts.specular,
        };
        println!("    {:14}: {:6} ({:5.1}%)", ft.name(), count, pct);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::color::Color4;

    #[test]
    fn test_classify_flat_image() {
        let flat = ImageBuffer::with_background(64, 64, Color4::new(0.5, 0.5, 0.5, 1.0));
        let classifier = FeatureClassifier::default();
        let result = classifier.classify(&flat).unwrap();

        // Should be mostly smooth
        assert!(result.counts.smooth > result.counts.total() / 2,
            "Flat image should be mostly Smooth");
    }

    #[test]
    fn test_classify_edge() {
        let mut image = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let val = if x < 32 { 0.0 } else { 1.0 };
                image.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let classifier = FeatureClassifier::default();
        let result = classifier.classify(&image).unwrap();

        // Should have some edges detected
        assert!(result.counts.edge > 0, "Vertical edge should be detected");
    }

    #[test]
    fn test_visualize() {
        let flat = ImageBuffer::with_background(32, 32, Color4::new(0.5, 0.5, 0.5, 1.0));
        let classifier = FeatureClassifier::default();
        let result = classifier.classify(&flat).unwrap();
        let vis = result.visualize();

        assert_eq!(vis.width, 32);
        assert_eq!(vis.height, 32);
    }

    #[test]
    fn test_budget_allocation() {
        let flat = ImageBuffer::with_background(64, 64, Color4::new(0.5, 0.5, 0.5, 1.0));
        let classifier = FeatureClassifier::default();
        let result = classifier.classify(&flat).unwrap();
        let budget = result.allocate_budget(256);

        let total: usize = budget.values().sum();
        assert_eq!(total, 256, "Budget should sum to requested total");
    }
}
