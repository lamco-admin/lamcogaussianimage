//! Level-of-Detail (LOD) System for View-Dependent Rendering
//!
//! Classifies Gaussians into coarse/medium/fine bands based on det(Σ)
//! Enables progressive loading and view-dependent selection
//!
//! Specification Requirement: LGI Format Spec Section 7 (Progressive and Streaming)

use crate::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

/// LOD band classification based on Gaussian scale
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LODBand {
    /// Coarse band: det(Σ) > 0.04 (large Gaussians, preview quality)
    Coarse = 0,
    /// Medium band: 0.01 < det(Σ) ≤ 0.04 (medium Gaussians, usable quality)
    Medium = 1,
    /// Fine band: det(Σ) ≤ 0.01 (small Gaussians, full detail)
    Fine = 2,
}

impl LODBand {
    /// Classify Gaussian by scale determinant
    /// Thresholds adjusted for normalized coordinates (σ ~ 0.01 typical)
    pub fn classify(gaussian: &Gaussian2D<f32, Euler<f32>>) -> Self {
        let det = gaussian.shape.scale_x * gaussian.shape.scale_y;

        // Adjusted thresholds based on actual scale distribution
        // Typical σ in normalized coords: 0.05-0.3
        // det(Σ) = σ_x × σ_y:
        //   Coarse: > 0.04 (σ > 0.2)
        //   Medium: 0.01-0.04 (σ ~ 0.1-0.2)
        //   Fine: < 0.01 (σ < 0.1)
        if det > 0.04 {
            LODBand::Coarse  // Large Gaussians (σ > 0.2)
        } else if det > 0.01 {
            LODBand::Medium  // Medium Gaussians (σ ~ 0.1-0.2)
        } else {
            LODBand::Fine    // Small details (σ < 0.1)
        }
    }

    /// Get band name
    pub fn name(&self) -> &'static str {
        match self {
            LODBand::Coarse => "Coarse",
            LODBand::Medium => "Medium",
            LODBand::Fine => "Fine",
        }
    }

    /// Get expected quality contribution
    pub fn quality_factor(&self) -> f32 {
        match self {
            LODBand::Coarse => 0.6,   // ~60% of final quality
            LODBand::Medium => 0.85,  // ~85% with coarse+medium
            LODBand::Fine => 1.0,     // 100% with all bands
        }
    }
}

/// LOD system managing multiscale Gaussian organization
pub struct LODSystem {
    /// Gaussians in each band
    pub bands: [Vec<Gaussian2D<f32, Euler<f32>>>; 3],

    /// Original indices (for reconstruction)
    pub indices: [Vec<usize>; 3],

    /// Statistics
    pub coarse_count: usize,
    pub medium_count: usize,
    pub fine_count: usize,
}

impl LODSystem {
    /// Classify Gaussians into LOD bands
    pub fn classify(gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Self {
        let mut bands: [Vec<Gaussian2D<f32, Euler<f32>>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut indices: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];

        for (idx, gaussian) in gaussians.iter().enumerate() {
            let band = LODBand::classify(gaussian);
            let band_idx = band as usize;

            bands[band_idx].push(gaussian.clone());
            indices[band_idx].push(idx);
        }

        let coarse_count = bands[0].len();
        let medium_count = bands[1].len();
        let fine_count = bands[2].len();

        Self {
            bands,
            indices,
            coarse_count,
            medium_count,
            fine_count,
        }
    }

    /// Select Gaussians for given zoom level
    pub fn select_for_zoom(&self, zoom: f32) -> Vec<&Gaussian2D<f32, Euler<f32>>> {
        let mut selected = Vec::new();

        if zoom <= 0.5 {
            // Zoomed out → coarse only (preview)
            selected.extend(self.bands[0].iter());
        } else if zoom <= 1.0 {
            // Normal → coarse + medium
            selected.extend(self.bands[0].iter());
            selected.extend(self.bands[1].iter());
        } else if zoom <= 2.0 {
            // Slightly zoomed in → medium + fine
            selected.extend(self.bands[1].iter());
            selected.extend(self.bands[2].iter());
        } else {
            // Zoomed in → fine only (detail)
            selected.extend(self.bands[2].iter());
        }

        selected
    }

    /// Select for screen-space rendering with culling
    pub fn select_for_viewport(
        &self,
        zoom: f32,
        viewport_width: u32,
        viewport_height: u32,
        dpr: f32,  // Device pixel ratio (e.g., 2.0 for Retina)
    ) -> Vec<&Gaussian2D<f32, Euler<f32>>> {
        let mut selected = Vec::new();

        // Screen-space minimum σ threshold (sub-pixel culling)
        let min_screen_sigma = 0.25;  // pixels

        for band in &self.bands {
            for gaussian in band {
                // Project to screen space
                let screen_sigma_x = gaussian.shape.scale_x * viewport_width as f32 * zoom * dpr;
                let screen_sigma_y = gaussian.shape.scale_y * viewport_height as f32 * zoom * dpr;
                let screen_sigma_min = screen_sigma_x.min(screen_sigma_y);

                // Cull if sub-pixel
                if screen_sigma_min >= min_screen_sigma {
                    selected.push(gaussian);
                }
            }
        }

        selected
    }

    /// Progressive rendering (3 phases)
    /// Returns (preview, usable, full) quality levels
    /// TODO: Replace with GPU-accelerated renderer
    pub fn render_progressive_cpu(
        &self,
        width: u32,
        height: u32,
    ) -> (ImageBuffer<f32>, ImageBuffer<f32>, ImageBuffer<f32>) {
        // Simple CPU rendering for now
        // TODO: Use proper Renderer once we understand the API better
        // For now, just return the bands info

        let preview = ImageBuffer::new(width, height);  // Placeholder
        let usable = preview.clone();
        let full = preview.clone();

        (preview, usable, full)
    }

    /// Get band by name
    pub fn get_band(&self, band: LODBand) -> &[Gaussian2D<f32, Euler<f32>>] {
        &self.bands[band as usize]
    }

    /// Get statistics
    pub fn stats(&self) -> LODStats {
        let total = self.coarse_count + self.medium_count + self.fine_count;

        LODStats {
            total_gaussians: total,
            coarse_count: self.coarse_count,
            coarse_percent: 100.0 * self.coarse_count as f32 / total as f32,
            medium_count: self.medium_count,
            medium_percent: 100.0 * self.medium_count as f32 / total as f32,
            fine_count: self.fine_count,
            fine_percent: 100.0 * self.fine_count as f32 / total as f32,
        }
    }

    /// Print statistics
    pub fn print_stats(&self) {
        let stats = self.stats();
        println!("LOD System Statistics:");
        println!("  Total Gaussians: {}", stats.total_gaussians);
        println!("  Coarse band:  {} ({:.1}%)", stats.coarse_count, stats.coarse_percent);
        println!("  Medium band:  {} ({:.1}%)", stats.medium_count, stats.medium_percent);
        println!("  Fine band:    {} ({:.1}%)", stats.fine_count, stats.fine_percent);
    }
}

/// LOD statistics
pub struct LODStats {
    pub total_gaussians: usize,
    pub coarse_count: usize,
    pub coarse_percent: f32,
    pub medium_count: usize,
    pub medium_percent: f32,
    pub fine_count: usize,
    pub fine_percent: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::{parameterization::Euler, vec::Vector2, color::Color4};

    #[test]
    fn test_lod_classification() {
        // Create Gaussians with different scales
        let mut gaussians = Vec::new();

        // Coarse (det > 0.04, e.g., σx=0.21, σy=0.21 → det=0.0441)
        gaussians.push(Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::new(0.21, 0.21, 0.0),
            Color4::new(1.0, 0.0, 0.0, 1.0),
            1.0,
        ));

        // Medium (0.01 < det ≤ 0.04, e.g., σx=0.15, σy=0.15 → det=0.0225)
        gaussians.push(Gaussian2D::new(
            Vector2::new(0.3, 0.3),
            Euler::new(0.15, 0.15, 0.0),
            Color4::new(0.0, 1.0, 0.0, 1.0),
            1.0,
        ));

        // Fine (det ≤ 0.01, e.g., σx=0.08, σy=0.08 → det=0.0064)
        gaussians.push(Gaussian2D::new(
            Vector2::new(0.7, 0.7),
            Euler::new(0.08, 0.08, 0.0),
            Color4::new(0.0, 0.0, 1.0, 1.0),
            1.0,
        ));

        let lod = LODSystem::classify(&gaussians);

        assert_eq!(lod.coarse_count, 1);
        assert_eq!(lod.medium_count, 1);
        assert_eq!(lod.fine_count, 1);
    }

    #[test]
    fn test_zoom_selection() {
        // Create test Gaussians
        let gaussians = vec![
            // Coarse
            Gaussian2D::new(Vector2::new(0.5, 0.5), Euler::new(0.3, 0.3, 0.0), Color4::new(1.0, 0.0, 0.0, 1.0), 1.0),
            // Medium
            Gaussian2D::new(Vector2::new(0.3, 0.3), Euler::new(0.15, 0.15, 0.0), Color4::new(0.0, 1.0, 0.0, 1.0), 1.0),
            // Fine
            Gaussian2D::new(Vector2::new(0.7, 0.7), Euler::new(0.05, 0.05, 0.0), Color4::new(0.0, 0.0, 1.0, 1.0), 1.0),
        ];

        let lod = LODSystem::classify(&gaussians);

        // At 0.5× zoom: Only coarse
        let selected = lod.select_for_zoom(0.5);
        assert_eq!(selected.len(), 1);

        // At 1.0× zoom: Coarse + medium
        let selected = lod.select_for_zoom(1.0);
        assert_eq!(selected.len(), 2);

        // At 2.0× zoom: Medium + fine
        let selected = lod.select_for_zoom(2.0);
        assert_eq!(selected.len(), 2);

        // At 4.0× zoom: Fine only
        let selected = lod.select_for_zoom(4.0);
        assert_eq!(selected.len(), 1);
    }
}
