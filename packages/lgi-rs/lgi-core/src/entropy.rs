//! Entropy-based adaptive Gaussian count
//!
//! From Instant-GaussianImage (2025): Automatically determine optimal
//! Gaussian count based on image complexity/entropy

use crate::ImageBuffer;

/// Compute optimal Gaussian count for an image based on entropy
///
/// Simple images (solid colors, gradients) → few Gaussians
/// Complex images (photos, textures) → many Gaussians
/// High-frequency images (noise, patterns) → very many Gaussians
pub fn adaptive_gaussian_count(image: &ImageBuffer<f32>) -> usize {
    let entropy = compute_image_entropy(image);

    // Base density: ~0.005 Gaussians per pixel (reduced for simpler images)
    let base_density = 0.005;

    // Entropy amplification factor (tuned for solid→complex range)
    let entropy_factor = 3.0;

    // Formula: count = pixels × density × (1 + entropy_factor × normalized_entropy)
    let pixels = (image.width * image.height) as f32;
    let count = pixels * base_density * (1.0 + entropy_factor * entropy);

    // Clamp to reasonable range
    count.max(50.0).min(50000.0) as usize
}

/// Compute image entropy (Shannon entropy approximation via local variance)
pub fn compute_image_entropy(image: &ImageBuffer<f32>) -> f32 {
    let window_size = 16;
    let mut total_entropy = 0.0;
    let mut tile_count = 0;

    // Divide image into tiles
    for y in (0..image.height).step_by(window_size) {
        for x in (0..image.width).step_by(window_size) {
            let tile_width = window_size.min((image.width - x) as usize) as u32;
            let tile_height = window_size.min((image.height - y) as usize) as u32;

            // Compute tile variance
            let variance = compute_tile_variance(image, x, y, tile_width, tile_height);

            // Shannon entropy approximation via variance
            // High variance = high entropy (complex)
            // Low variance = low entropy (simple)
            if variance > 1e-8 {
                // Normalize variance and use as entropy proxy
                let entropy = variance.sqrt();  // Simpler: just use std dev as entropy
                total_entropy += entropy;
            }
            // If variance ≈ 0, entropy = 0 (don't add anything)

            tile_count += 1;
        }
    }

    // Normalize by tile count
    let mean_entropy = total_entropy / tile_count as f32;

    // Normalize to [0, 1] range (empirically, std dev is ~0-0.5 for images)
    // Solid colors: ~0.0
    // Photos: ~0.1-0.3
    // High-freq: ~0.3-0.5
    (mean_entropy * 2.0).min(1.0).max(0.0)
}

/// Compute variance within a tile
fn compute_tile_variance(
    image: &ImageBuffer<f32>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> f32 {
    let mut sum_r = 0.0;
    let mut sum_g = 0.0;
    let mut sum_b = 0.0;
    let mut count = 0;

    // Compute mean
    for dy in 0..height {
        for dx in 0..width {
            if let Some(pixel) = image.get_pixel(x + dx, y + dy) {
                sum_r += pixel.r;
                sum_g += pixel.g;
                sum_b += pixel.b;
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    let mean_r = sum_r / count as f32;
    let mean_g = sum_g / count as f32;
    let mean_b = sum_b / count as f32;

    // Compute variance
    let mut var = 0.0;
    for dy in 0..height {
        for dx in 0..width {
            if let Some(pixel) = image.get_pixel(x + dx, y + dy) {
                let diff_r = pixel.r - mean_r;
                let diff_g = pixel.g - mean_g;
                let diff_b = pixel.b - mean_b;

                var += diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
            }
        }
    }

    var / (count as f32 * 3.0)  // 3 channels
}

/// Compute per-tile entropy map (for visualization/analysis)
pub fn compute_entropy_map(image: &ImageBuffer<f32>, tile_size: usize) -> Vec<Vec<f32>> {
    let tiles_x = (image.width as usize + tile_size - 1) / tile_size;
    let tiles_y = (image.height as usize + tile_size - 1) / tile_size;

    let mut entropy_map = vec![vec![0.0; tiles_x]; tiles_y];

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x = (tx * tile_size) as u32;
            let y = (ty * tile_size) as u32;
            let w = tile_size.min((image.width - x) as usize) as u32;
            let h = tile_size.min((image.height - y) as usize) as u32;

            let variance = compute_tile_variance(image, x, y, w, h);
            let entropy = if variance > 1e-6 {
                -variance * variance.ln()
            } else {
                0.0
            };

            entropy_map[ty][tx] = entropy;
        }
    }

    entropy_map
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::color::Color4;

    #[test]
    fn test_entropy_solid_color() {
        // Solid color should have very low entropy
        let image = ImageBuffer::with_background(256, 256, Color4::rgb(0.5, 0.5, 0.5));
        let entropy = compute_image_entropy(&image);

        assert!(entropy < 0.1, "Solid color should have low entropy, got {}", entropy);

        let count = adaptive_gaussian_count(&image);
        assert!(count < 500, "Solid color should need few Gaussians, got {}", count);
    }

    #[test]
    fn test_entropy_varies() {
        // Solid image
        let solid = ImageBuffer::with_background(256, 256, Color4::white());
        let entropy_solid = compute_image_entropy(&solid);

        // Noisy image (higher entropy)
        let mut noisy = ImageBuffer::new(256, 256);
        for y in 0..256 {
            for x in 0..256 {
                // Checkerboard pattern (higher entropy than solid)
                let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                noisy.set_pixel(x, y, Color4::rgb(v, v, v));
            }
        }
        let entropy_noisy = compute_image_entropy(&noisy);

        assert!(entropy_noisy > entropy_solid,
            "Checkerboard should have higher entropy than solid");

        let count_solid = adaptive_gaussian_count(&solid);
        let count_noisy = adaptive_gaussian_count(&noisy);

        assert!(count_noisy > count_solid,
            "Complex image should need more Gaussians");
    }
}
