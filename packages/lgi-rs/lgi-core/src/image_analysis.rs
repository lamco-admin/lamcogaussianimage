//! Unified Image Analysis Pipeline
//!
//! Combines feature classification, structure tensor, geodesic distance,
//! and provides a comprehensive analysis for adaptive Gaussian encoding.
//!
//! # Architecture
//!
//! ```text
//! Image → ImageAnalysis
//!         ├── Feature Map (per-pixel: Smooth/Edge/Corner/Texture/...)
//!         ├── Structure Tensor Field (orientation, coherence)
//!         ├── Geodesic Distance Map (distance to edges)
//!         ├── Importance Map (placement priority)
//!         └── Content Statistics (global summary)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use lgi_core::image_analysis::{ImageAnalysis, AnalysisConfig};
//!
//! let config = AnalysisConfig::default();
//! let analysis = ImageAnalysis::compute(&image, config)?;
//!
//! // Use for Gaussian initialization
//! let gaussians = analysis.generate_gaussians(256);
//!
//! // Or for adaptive refinement
//! let positions = analysis.sample_positions_by_importance(64);
//! ```

use crate::{
    ImageBuffer,
    feature_classifier::{FeatureClassifier, ClassificationResult, FeatureType, ClassifierThresholds},
    structure_tensor::StructureTensorField,
    geodesic_edt::GeodesicEDT,
    position_probability_map::PositionProbabilityMap,
};
use lgi_math::{
    gaussian::Gaussian2D,
    parameterization::Euler,
    vec::Vector2,
    color::Color4,
};

/// Configuration for image analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Feature classification thresholds
    pub classifier_thresholds: ClassifierThresholds,

    /// Structure tensor gradient sigma
    pub sigma_gradient: f32,

    /// Structure tensor smoothing sigma
    pub sigma_smooth: f32,

    /// Edge threshold for geodesic EDT
    pub edge_threshold: f32,

    /// Gradient weight for geodesic EDT (vs Euclidean)
    pub geodesic_weight: f32,

    /// Edge weight for importance map
    pub importance_edge_weight: f32,

    /// Gradient weight for importance map
    pub importance_gradient_weight: f32,

    /// Laplacian weight for importance map
    pub importance_laplacian_weight: f32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            classifier_thresholds: ClassifierThresholds::default(),
            sigma_gradient: 1.2,
            sigma_smooth: 2.5,
            edge_threshold: 0.1,
            geodesic_weight: 0.5,
            importance_edge_weight: 0.4,
            importance_gradient_weight: 0.3,
            importance_laplacian_weight: 0.3,
        }
    }
}

/// Global content statistics
#[derive(Debug, Clone)]
pub struct ContentStats {
    /// Dominant feature type
    pub dominant_type: FeatureType,

    /// Average image entropy (0-8 bits)
    pub entropy: f32,

    /// Edge density (fraction of edge pixels)
    pub edge_density: f32,

    /// Average texture directionality (0 = random, 1 = strongly aligned)
    pub texture_directionality: f32,

    /// Color complexity (number of distinct color clusters)
    pub color_complexity: f32,

    /// Smooth region fraction
    pub smooth_fraction: f32,

    /// Texture region fraction
    pub texture_fraction: f32,
}

/// Comprehensive image analysis result
pub struct ImageAnalysis {
    /// Image dimensions
    pub width: u32,
    pub height: u32,

    /// Per-pixel feature classification
    pub classification: ClassificationResult,

    /// Geodesic distance to edges
    pub edge_distance: Vec<f32>,

    /// Importance map for placement priority
    pub importance: Vec<f32>,

    /// Global content statistics
    pub content_stats: ContentStats,

    /// Configuration used
    pub config: AnalysisConfig,
}

impl ImageAnalysis {
    /// Compute comprehensive image analysis
    pub fn compute(image: &ImageBuffer<f32>, config: AnalysisConfig) -> crate::Result<Self> {
        let width = image.width;
        let height = image.height;

        // Step 1: Feature classification (includes structure tensor)
        let classifier = FeatureClassifier::with_thresholds(config.classifier_thresholds.clone());
        let classification = classifier.classify(image)?;

        // Step 2: Geodesic distance to edges
        // Uses structure tensor for edge detection
        let gedt = GeodesicEDT::compute(
            image,
            &classification.tensor_field,
            config.edge_threshold,
            config.geodesic_weight * 50.0,  // Scale to edge_penalty range
        )?;
        let edge_distance = gedt.distances;

        // Step 3: Importance map
        let importance = Self::compute_importance(
            image,
            &classification,
            &config,
        );

        // Step 4: Content statistics
        let content_stats = Self::compute_content_stats(&classification);

        Ok(Self {
            width,
            height,
            classification,
            edge_distance,
            importance,
            content_stats,
            config,
        })
    }

    /// Create binary edge mask from classification
    fn create_edge_mask(classification: &ClassificationResult) -> Vec<bool> {
        classification.feature_map.iter()
            .map(|&ft| matches!(ft, FeatureType::Edge | FeatureType::Corner | FeatureType::TextStroke))
            .collect()
    }

    /// Compute importance map for placement priority
    fn compute_importance(
        image: &ImageBuffer<f32>,
        classification: &ClassificationResult,
        config: &AnalysisConfig,
    ) -> Vec<f32> {
        let width = image.width;
        let height = image.height;
        let n = (width * height) as usize;

        // Compute gradient magnitude as importance signal
        let mut gradient_map = vec![0.0f32; n];
        let mut edge_map = vec![0.0f32; n];

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let idx = (y * width + x) as usize;

                // Gradient from structure tensor energy
                let tensor = classification.tensor_field.get(x, y);
                gradient_map[idx] = (tensor.eigenvalue_major + tensor.eigenvalue_minor).sqrt();

                // Edge importance from feature type
                let feature = classification.get(x, y);
                edge_map[idx] = match feature {
                    FeatureType::Edge | FeatureType::Corner | FeatureType::TextStroke => 1.0,
                    FeatureType::Texture | FeatureType::Specular => 0.7,
                    FeatureType::SmoothGradient => 0.4,
                    FeatureType::Smooth => 0.2,
                };
            }
        }

        // Combine using weights
        let ppm_config = crate::position_probability_map::PPMConfig {
            weight_entropy: config.importance_laplacian_weight,
            weight_gradient: config.importance_gradient_weight,
            weight_saliency: config.importance_edge_weight,
            entropy_tile_size: 8,
        };

        let ppm = PositionProbabilityMap::from_entropy_gradient_saliency(
            &gradient_map,
            &gradient_map,
            &edge_map,
            width,
            height,
            &ppm_config,
        );

        ppm.probabilities
    }

    /// Compute global content statistics
    fn compute_content_stats(classification: &ClassificationResult) -> ContentStats {
        let total = classification.counts.total() as f32;

        // Find dominant type
        let percentages = classification.counts.percentages();
        let dominant_type = percentages.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(ft, _)| *ft)
            .unwrap_or(FeatureType::Smooth);

        // Edge density
        let edge_density = (classification.counts.edge + classification.counts.corner +
                          classification.counts.text_stroke) as f32 / total;

        // Texture fraction
        let texture_fraction = classification.counts.texture as f32 / total;

        // Smooth fraction
        let smooth_fraction = (classification.counts.smooth + classification.counts.smooth_gradient) as f32 / total;

        // Estimate texture directionality from structure tensor coherence
        let mut total_coherence = 0.0;
        let mut coherence_count = 0;
        for y in 0..classification.height {
            for x in 0..classification.width {
                if classification.get(x, y) == FeatureType::Texture {
                    let tensor = classification.tensor_field.get(x, y);
                    total_coherence += tensor.coherence;
                    coherence_count += 1;
                }
            }
        }
        let texture_directionality = if coherence_count > 0 {
            total_coherence / coherence_count as f32
        } else {
            0.0
        };

        ContentStats {
            dominant_type,
            entropy: 0.0, // TODO: compute from histogram
            edge_density,
            texture_directionality,
            color_complexity: 0.0, // TODO: compute from color clustering
            smooth_fraction,
            texture_fraction,
        }
    }

    /// Get feature type at pixel
    pub fn get_feature(&self, x: u32, y: u32) -> FeatureType {
        self.classification.get(x, y)
    }

    /// Get structure tensor at pixel
    pub fn get_tensor(&self, x: u32, y: u32) -> &crate::structure_tensor::StructureTensor {
        self.classification.tensor_field.get(x, y)
    }

    /// Get geodesic distance to nearest edge
    pub fn get_edge_distance(&self, x: u32, y: u32) -> f32 {
        self.edge_distance[(y * self.width + x) as usize]
    }

    /// Get importance value (higher = more important for Gaussian placement)
    pub fn get_importance(&self, x: u32, y: u32) -> f32 {
        self.importance[(y * self.width + x) as usize]
    }

    /// Sample positions weighted by importance
    pub fn sample_positions_by_importance(&self, count: usize, seed: u64) -> Vec<(f32, f32)> {
        // Weighted sampling by importance
        let total_importance: f32 = self.importance.iter().sum();

        let mut positions = Vec::with_capacity(count);
        let mut rng_state = seed;

        let lcg_next = |state: &mut u64| -> f32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state >> 33) as f32 / (1u64 << 31) as f32
        };

        for _ in 0..count {
            let target = lcg_next(&mut rng_state) * total_importance;
            let mut cumulative = 0.0;

            for y in 0..self.height {
                for x in 0..self.width {
                    cumulative += self.importance[(y * self.width + x) as usize];
                    if cumulative >= target {
                        positions.push((x as f32 + lcg_next(&mut rng_state) * 0.5 - 0.25,
                                       y as f32 + lcg_next(&mut rng_state) * 0.5 - 0.25));
                        break;
                    }
                }
                if positions.len() > positions.capacity() - count + count { break; }
            }
        }

        positions
    }

    /// Generate Gaussians based on analysis
    ///
    /// Uses feature-aware budget allocation and orientation from structure tensor.
    pub fn generate_gaussians(
        &self,
        total_count: usize,
        image: &ImageBuffer<f32>,
        seed: u64,
    ) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        let mut gaussians = Vec::with_capacity(total_count);
        let mut rng_state = seed;

        let lcg_next = |state: &mut u64| -> f32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*state >> 33) as f32 / (1u64 << 31) as f32
        };

        // Allocate budget per feature type
        let budget = self.classification.allocate_budget(total_count);

        for (feature_type, count) in budget {
            if count == 0 {
                continue;
            }

            let (scale_min, scale_max) = feature_type.scale_range();
            let (aniso_min, aniso_max) = feature_type.anisotropy_range();
            let use_orientation = feature_type.use_orientation();
            let use_clamping = feature_type.use_geodesic_clamping();

            // Sample positions for this feature type
            let positions = self.classification.sample_positions(feature_type, count, seed);

            for (x, y) in positions {
                let px = (x as u32).min(self.width - 1);
                let py = (y as u32).min(self.height - 1);

                // Sample color from image
                let color = image.get_pixel(px, py).unwrap().clone();

                // Base scale from range
                let scale_factor = scale_min + lcg_next(&mut rng_state) * (scale_max - scale_min);
                let mut base_scale = scale_factor * (self.width.max(self.height) as f32);

                // Apply geodesic clamping if enabled
                if use_clamping {
                    let edge_dist = self.get_edge_distance(px, py);
                    let max_scale = 0.5 + 0.3 * edge_dist;  // Linear clamping
                    base_scale = base_scale.min(max_scale * self.width.max(self.height) as f32 / 100.0);
                }

                // Get orientation and anisotropy
                let (sigma_major, sigma_minor, angle) = if use_orientation {
                    let tensor = self.get_tensor(px, py);

                    // Anisotropy from coherence
                    let coherence_aniso = 1.0 + tensor.coherence * (aniso_max - 1.0);
                    let anisotropy = coherence_aniso.clamp(aniso_min, aniso_max);

                    // Orientation from edge tangent (minor eigenvector)
                    let edge_tangent_angle = tensor.eigenvector_minor.y.atan2(tensor.eigenvector_minor.x);

                    (base_scale, base_scale / anisotropy, edge_tangent_angle)
                } else {
                    let anisotropy = aniso_min + lcg_next(&mut rng_state) * (aniso_max - aniso_min);
                    let angle = lcg_next(&mut rng_state) * std::f32::consts::PI;
                    (base_scale, base_scale / anisotropy, angle)
                };

                gaussians.push(Gaussian2D::new(
                    Vector2::new(x, y),
                    Euler::new(sigma_major, sigma_minor, angle),
                    color,
                    1.0,
                ));
            }
        }

        gaussians
    }

    /// Print analysis summary
    pub fn print_summary(&self) {
        println!("Image Analysis Summary:");
        println!("  Dimensions: {}x{}", self.width, self.height);
        println!();
        println!("  Content Statistics:");
        println!("    Dominant type: {:?}", self.content_stats.dominant_type);
        println!("    Edge density: {:.1}%", self.content_stats.edge_density * 100.0);
        println!("    Smooth fraction: {:.1}%", self.content_stats.smooth_fraction * 100.0);
        println!("    Texture fraction: {:.1}%", self.content_stats.texture_fraction * 100.0);
        println!("    Texture directionality: {:.2}", self.content_stats.texture_directionality);
        println!();
        crate::feature_classifier::print_classification_summary(&self.classification);
    }

    /// Suggest encoding strategy based on analysis
    pub fn suggest_strategy(&self) -> EncodingStrategy {
        let stats = &self.content_stats;

        // Base Gaussian count on complexity
        let base_count = if stats.smooth_fraction > 0.6 {
            128  // Simple image
        } else if stats.edge_density > 0.2 {
            512  // Complex edges
        } else {
            256  // Average
        };

        // Adjust for texture
        let gaussian_count = if stats.texture_fraction > 0.3 {
            (base_count as f32 * 1.5) as usize
        } else {
            base_count
        };

        // Use geodesic clamping if significant edges
        let use_geodesic_clamping = stats.edge_density > 0.1;

        // Use orientation if texture is directional
        let use_orientation = stats.texture_directionality > 0.5 || stats.edge_density > 0.1;

        // Use coarse-to-fine if both smooth and detailed regions
        let use_coarse_to_fine = stats.smooth_fraction > 0.2 && stats.edge_density > 0.1;

        EncodingStrategy {
            gaussian_count,
            use_geodesic_clamping,
            use_orientation,
            use_coarse_to_fine,
            dominant_type: stats.dominant_type,
        }
    }
}

/// Suggested encoding strategy based on image analysis
#[derive(Debug, Clone)]
pub struct EncodingStrategy {
    /// Recommended Gaussian count
    pub gaussian_count: usize,

    /// Whether to use geodesic sigma clamping
    pub use_geodesic_clamping: bool,

    /// Whether to use structure tensor orientation
    pub use_orientation: bool,

    /// Whether to use coarse-to-fine hierarchy
    pub use_coarse_to_fine: bool,

    /// Dominant feature type
    pub dominant_type: FeatureType,
}

impl EncodingStrategy {
    /// Print strategy summary
    pub fn print(&self) {
        println!("Suggested Encoding Strategy:");
        println!("  Gaussian count: {}", self.gaussian_count);
        println!("  Geodesic clamping: {}", self.use_geodesic_clamping);
        println!("  Orientation: {}", self.use_orientation);
        println!("  Coarse-to-fine: {}", self.use_coarse_to_fine);
        println!("  Dominant type: {:?}", self.dominant_type);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_flat_image() {
        let flat = ImageBuffer::with_background(64, 64, Color4::new(0.5, 0.5, 0.5, 1.0));
        let analysis = ImageAnalysis::compute(&flat, AnalysisConfig::default()).unwrap();

        assert!(analysis.content_stats.smooth_fraction > 0.5,
            "Flat image should be mostly smooth");
    }

    #[test]
    fn test_suggest_strategy() {
        let flat = ImageBuffer::with_background(64, 64, Color4::new(0.5, 0.5, 0.5, 1.0));
        let analysis = ImageAnalysis::compute(&flat, AnalysisConfig::default()).unwrap();
        let strategy = analysis.suggest_strategy();

        // Simple image should suggest fewer Gaussians
        assert!(strategy.gaussian_count <= 256,
            "Simple image should suggest fewer Gaussians");
    }
}
