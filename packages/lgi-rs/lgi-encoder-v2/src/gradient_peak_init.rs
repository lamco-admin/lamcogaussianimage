//! Gradient Peak Initialization (Strategy H)
//!
//! Place Gaussians at local gradient maxima (edges, corners, detail)
//! Simpler than GDGS, focuses on high-frequency content
//!
//! Algorithm:
//! 1. Compute gradient magnitude at each pixel
//! 2. Find local maxima (peaks)
//! 3. Place Gaussian at each peak
//! 4. Add background Gaussians for smooth regions (hybrid approach)

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

/// Gradient peak initialization config
#[derive(Clone)]
pub struct GradientPeakConfig {
    /// Gradient magnitude threshold for peak detection
    pub gradient_threshold: f32,

    /// Window size for local maxima
    pub peak_window_size: usize,

    /// Target number of peak Gaussians
    pub target_peak_gaussians: usize,

    /// Add background grid? (hybrid approach)
    pub add_background_grid: bool,

    /// Background grid density (if enabled)
    pub background_grid_size: u32,
}

impl Default for GradientPeakConfig {
    fn default() -> Self {
        Self {
            gradient_threshold: 0.01,
            peak_window_size: 5,
            target_peak_gaussians: 300,
            add_background_grid: true,   // Hybrid: peaks + sparse background
            background_grid_size: 5,     // 5×5 = 25 background Gaussians
        }
    }
}

/// Initialize Gaussians at gradient peaks
///
/// Places Gaussians where gradient magnitude is locally maximal
/// Optionally adds sparse background grid for smooth regions
pub fn initialize_gradient_peaks(
    image: &ImageBuffer<f32>,
    structure_tensor: &StructureTensorField,
    config: &GradientPeakConfig,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    log::info!("🔍 Gradient Peak Initialization");

    // Step 1: Extract gradient magnitude from structure tensor
    log::info!("  Computing gradient magnitudes...");
    let mut gradient_map = vec![0.0; (image.width * image.height) as usize];

    for y in 0..image.height {
        for x in 0..image.width {
            let tensor = structure_tensor.get(x, y);
            // Gradient magnitude ≈ sqrt(larger eigenvalue)
            let grad_mag = tensor.eigenvalue_major.sqrt();
            gradient_map[(y * image.width + x) as usize] = grad_mag;
        }
    }

    // Step 2: Find local maxima
    log::info!("  Finding gradient peaks (threshold={:.4})...", config.gradient_threshold);
    let peaks = find_gradient_peaks(
        &gradient_map,
        image.width,
        image.height,
        config.gradient_threshold,
        config.peak_window_size,
    );

    log::info!("  Found {} gradient peaks", peaks.len());

    // Step 3: Place Gaussians at peaks
    let mut gaussians = Vec::new();
    let n_peaks = peaks.len().min(config.target_peak_gaussians);

    log::info!("  Creating {} Gaussians at peaks", n_peaks);

    for (x, y, grad_mag) in peaks.iter().take(n_peaks) {
        let position = Vector2::new(
            *x as f32 / image.width as f32,
            *y as f32 / image.height as f32,
        );

        let color = image.get_pixel(*x, *y)
            .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        // Get local structure
        let tensor = structure_tensor.get(*x, *y);

        // Scale inversely proportional to gradient (sharp edge = smaller Gaussian)
        let base_sigma = 0.02;
        let scale_factor = (1.0 / (1.0 + grad_mag * 5.0)).max(0.5);
        let sigma = base_sigma * scale_factor;

        // Orient along edge if coherent
        let (sig_x, sig_y, rotation) = if tensor.coherence > 0.3 {
            let sig_perp = sigma * 0.6;
            let sig_para = sigma * 2.0;
            let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
            (sig_para, sig_perp, angle)
        } else {
            (sigma, sigma, 0.0)
        };

        gaussians.push(Gaussian2D::new(
            position,
            Euler::new(sig_x, sig_y, rotation),
            color,
            1.0,
        ));
    }

    // Step 4: Add sparse background grid (hybrid approach)
    if config.add_background_grid {
        let bg_gaussians = add_background_grid(
            image,
            structure_tensor,
            config.background_grid_size,
            &gaussians,  // Avoid overlap with peaks
        );

        log::info!("  Added {} background Gaussians", bg_gaussians.len());
        gaussians.extend(bg_gaussians);
    }

    log::info!("✅ Gradient peak initialization complete: {} total Gaussians", gaussians.len());

    gaussians
}

/// Find peaks in gradient field
fn find_gradient_peaks(
    gradient_map: &[f32],
    width: u32,
    height: u32,
    threshold: f32,
    window_size: usize,
) -> Vec<(u32, u32, f32)> {
    let mut peaks = Vec::new();
    let half_window = window_size / 2;

    for y in half_window..(height as usize - half_window) {
        for x in half_window..(width as usize - half_window) {
            let idx = y * width as usize + x;
            let value = gradient_map[idx];

            if value < threshold {
                continue;
            }

            // Check if local maximum
            let mut is_max = true;
            for dy in -(half_window as i32)..=(half_window as i32) {
                for dx in -(half_window as i32)..=(half_window as i32) {
                    if dx == 0 && dy == 0 {
                        continue;
                    }

                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;

                    if nx >= width as usize || ny >= height as usize {
                        continue;
                    }

                    if gradient_map[ny * width as usize + nx] > value {
                        is_max = false;
                        break;
                    }
                }
                if !is_max { break; }
            }

            if is_max {
                peaks.push((x as u32, y as u32, value));
            }
        }
    }

    // Sort by gradient magnitude (descending)
    peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    peaks
}

/// Add sparse background grid for smooth region coverage
fn add_background_grid(
    image: &ImageBuffer<f32>,
    structure_tensor: &StructureTensorField,
    grid_size: u32,
    existing_gaussians: &[Gaussian2D<f32, Euler<f32>>],
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut background = Vec::new();
    let step_x = image.width / grid_size;
    let step_y = image.height / grid_size;

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx * step_x + step_x / 2).min(image.width - 1);
            let y = (gy * step_y + step_y / 2).min(image.height - 1);

            let position = Vector2::new(
                x as f32 / image.width as f32,
                y as f32 / image.height as f32,
            );

            // Skip if too close to existing peak Gaussian
            let too_close = existing_gaussians.iter().any(|g| {
                let dx = g.position.x - position.x;
                let dy = g.position.y - position.y;
                (dx * dx + dy * dy).sqrt() < 0.05  // 5% of image
            });

            if too_close {
                continue;
            }

            let color = image.get_pixel(x, y).unwrap();
            let tensor = structure_tensor.get(x, y);

            // Larger Gaussians for background (cover smooth regions)
            let sigma = 0.04;  // 2× larger than peak Gaussians

            let (sig_x, sig_y, rotation) = if tensor.coherence > 0.3 {
                let sig_perp = sigma * 0.6;
                let sig_para = sigma * 2.0;
                let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
                (sig_para, sig_perp, angle)
            } else {
                (sigma, sigma, 0.0)
            };

            background.push(Gaussian2D::new(
                position,
                Euler::new(sig_x, sig_y, rotation),
                color,
                1.0,
            ));
        }
    }

    background
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_peaks_finds_edges() {
        // Create edge image
        let mut image = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let val = if x < 32 { 0.0 } else { 1.0 };
                image.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let tensor = StructureTensorField::compute(&image, 1.0, 1.0).unwrap();
        let config = GradientPeakConfig::default();

        let gaussians = initialize_gradient_peaks(&image, &tensor, &config);

        // Should find Gaussians near edge (x≈32)
        let near_edge = gaussians.iter().filter(|g| {
            let px = (g.position.x * 64.0) as u32;
            (px as i32 - 32).abs() < 8
        }).count();

        assert!(near_edge > 0, "Should have some Gaussians near edge");
    }
}
