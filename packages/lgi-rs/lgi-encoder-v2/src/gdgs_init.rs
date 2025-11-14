//! GDGS (Gradient Domain Gaussian Splatting) Initialization
//!
//! Key innovation: Place Gaussians ONLY at Laplacian peaks (edges, corners)
//! Claims: 10-100× fewer Gaussians for same quality!
//!
//! Algorithm:
//! 1. Compute Laplacian (∇²I)
//! 2. Find local maxima (peaks)
//! 3. Place Gaussian at each peak, oriented along gradient
//! 4. (Optional) Poisson reconstruction for smoothness
//!
//! Paper: arXiv:2405.05446v1 "Gradient Domain Gaussian Splatting"

use lgi_core::{ImageBuffer, StructureTensorField, laplacian};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

/// GDGS initialization configuration
#[derive(Clone)]
pub struct GDGSConfig {
    /// Laplacian threshold for peak detection
    pub lap_threshold: f32,

    /// Window size for local maxima detection
    pub peak_window_size: usize,

    /// Maximum number of Gaussians (take top N peaks)
    pub max_gaussians: usize,

    /// Base Gaussian scale (will be modulated by Laplacian magnitude)
    pub base_sigma: f32,
}

impl Default for GDGSConfig {
    fn default() -> Self {
        Self {
            lap_threshold: 0.01,   // Tune: Higher = fewer Gaussians
            peak_window_size: 5,   // 5×5 local window
            max_gaussians: 500,    // Budget limit
            base_sigma: 0.02,      // Base scale
        }
    }
}

/// Initialize Gaussians using GDGS (Gradient Domain) strategy
///
/// Places Gaussians ONLY at high-curvature points (edges, corners)
/// Much sparser than grid initialization
pub fn initialize_gdgs(
    image: &ImageBuffer<f32>,
    structure_tensor: &StructureTensorField,
    config: &GDGSConfig,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    log::info!("🔬 GDGS Initialization (Gradient Domain)");

    // Step 1: Compute Laplacian
    log::info!("  Computing Laplacian (∇²I)...");
    let laplacian_map = lgi_core::laplacian::compute_laplacian(image);

    // Step 2: Find peaks (local maxima)
    log::info!("  Finding Laplacian peaks (threshold={:.4})...", config.lap_threshold);
    let peaks = lgi_core::laplacian::find_local_maxima(
        &laplacian_map,
        image.width,
        image.height,
        config.lap_threshold,
        config.peak_window_size,
    );

    log::info!("  Found {} peaks", peaks.len());

    // Step 3: Place Gaussian at each peak (up to max)
    let mut gaussians = Vec::new();
    let n = peaks.len().min(config.max_gaussians);

    log::info!("  Creating {} Gaussians at peak locations", n);

    for (x, y, lap_magnitude) in peaks.iter().take(n) {
        // Position (normalized)
        let position = Vector2::new(
            *x as f32 / image.width as f32,
            *y as f32 / image.height as f32,
        );

        // Color from image
        let color = image.get_pixel(*x, *y)
            .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        // Get local structure (gradient direction)
        let tensor = structure_tensor.get(*x, *y);

        // Scale modulated by Laplacian magnitude
        // Higher Laplacian → sharper feature → smaller Gaussian
        let scale_factor = (1.0 / (1.0 + lap_magnitude * 10.0)).max(0.3);
        let sigma = config.base_sigma * scale_factor;

        // Align to gradient direction if coherent edge
        let (sig_x, sig_y, rotation) = if tensor.coherence > 0.3 {
            // Edge: thin perpendicular, long parallel
            let sig_perp = sigma * 0.5;
            let sig_para = sigma * 2.0;
            let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
            (sig_para, sig_perp, angle)
        } else {
            // Corner/junction: isotropic
            (sigma, sigma, 0.0)
        };

        gaussians.push(Gaussian2D::new(
            position,
            Euler::new(sig_x, sig_y, rotation),
            color,
            1.0,
        ));
    }

    log::info!("✅ GDGS initialization complete: {} Gaussians", gaussians.len());
    log::info!("   (vs grid would need {} for same coverage)", (n as f32).sqrt().ceil().powi(2));

    gaussians
}

/// GDGS with target Gaussian count
///
/// Adjusts threshold to get approximately N Gaussians
pub fn initialize_gdgs_with_target_n(
    image: &ImageBuffer<f32>,
    structure_tensor: &StructureTensorField,
    target_n: usize,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    // Binary search for threshold that gives ~target_n peaks
    let mut low = 0.001;
    let mut high = 0.5;
    let mut best_config = GDGSConfig::default();

    for _ in 0..10 {  // Max 10 binary search iterations
        let mid = (low + high) / 2.0;
        let config = GDGSConfig {
            lap_threshold: mid,
            max_gaussians: target_n * 2,  // Allow overshoot
            ..Default::default()
        };

        let laplacian_map = lgi_core::laplacian::compute_laplacian(image);
        let peaks = lgi_core::laplacian::find_local_maxima(
            &laplacian_map,
            image.width,
            image.height,
            mid,
            5,
        );

        let n_peaks = peaks.len();

        if n_peaks < target_n {
            // Too few peaks, lower threshold
            high = mid;
        } else if n_peaks > ((target_n as f32) * 1.2) as usize {
            // Too many peaks, raise threshold
            low = mid;
        } else {
            // Close enough
            best_config = config;
            break;
        }

        best_config.lap_threshold = mid;
    }

    best_config.max_gaussians = target_n;
    initialize_gdgs(image, structure_tensor, &best_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_core::structure_tensor::StructureTensorField;

    #[test]
    fn test_gdgs_creates_gaussians() {
        // Create test image with edge
        let mut image = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let val = if x < 32 { 0.0 } else { 1.0 };
                image.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let tensor = StructureTensorField::compute(&image, 1.0, 1.0).unwrap();
        let config = GDGSConfig::default();

        let gaussians = initialize_gdgs(&image, &tensor, &config);

        // Should create Gaussians at edge
        assert!(gaussians.len() > 0, "Should create some Gaussians");
        assert!(gaussians.len() < 500, "Should be sparse (not grid)");

        // Gaussians should be near edge (x~32)
        let near_edge = gaussians.iter().filter(|g| {
            let px = (g.position.x * 64.0) as u32;
            (px as i32 - 32).abs() < 5
        }).count();

        assert!(near_edge > gaussians.len() / 2, "Most Gaussians should be near edge");
    }
}
