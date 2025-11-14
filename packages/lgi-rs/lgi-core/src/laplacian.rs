//! Laplacian Operator (∇²I) - Second Derivative
//!
//! Computes Laplacian (sum of second derivatives) for edge detection
//! Used in GDGS (Gradient Domain Gaussian Splatting)
//!
//! Laplacian kernel (discrete approximation):
//!     [ 0  1  0 ]
//!     [ 1 -4  1 ]
//!     [ 0  1  0 ]

use crate::ImageBuffer;
use lgi_math::color::Color4;

/// Compute Laplacian (∇²I) of image
///
/// Returns magnitude of Laplacian per pixel (always positive)
/// High values = edges, corners, texture boundaries
/// Low values = smooth regions, gradients
pub fn compute_laplacian(image: &ImageBuffer<f32>) -> Vec<f32> {
    let width = image.width;
    let height = image.height;
    let mut laplacian = vec![0.0; (width * height) as usize];

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = (y * width + x) as usize;

            let center = image.get_pixel(x, y).unwrap();
            let up = image.get_pixel(x, y - 1).unwrap();
            let down = image.get_pixel(x, y + 1).unwrap();
            let left = image.get_pixel(x - 1, y).unwrap();
            let right = image.get_pixel(x + 1, y).unwrap();

            // Laplacian = up + down + left + right - 4×center
            let lap_r = up.r + down.r + left.r + right.r - 4.0 * center.r;
            let lap_g = up.g + down.g + left.g + right.g - 4.0 * center.g;
            let lap_b = up.b + down.b + left.b + right.b - 4.0 * center.b;

            // Magnitude (unsigned)
            let magnitude = (lap_r.abs() + lap_g.abs() + lap_b.abs()) / 3.0;
            laplacian[idx] = magnitude;
        }
    }

    laplacian
}

/// Find local maxima (peaks) in a scalar field
///
/// Returns (x, y, value) tuples for peaks above threshold
pub fn find_local_maxima(
    field: &[f32],
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
            let value = field[idx];

            // Skip if below threshold
            if value < threshold {
                continue;
            }

            // Check if local maximum in window
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

                    let neighbor_idx = ny * width as usize + nx;
                    if field[neighbor_idx] > value {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }

            if is_max {
                peaks.push((x as u32, y as u32, value));
            }
        }
    }

    // Sort by magnitude (descending)
    peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_flat() {
        // Flat image should have zero Laplacian
        let flat = ImageBuffer::with_background(100, 100, Color4::new(0.5, 0.5, 0.5, 1.0));
        let lap = compute_laplacian(&flat);

        // Interior should be near zero (edges may have boundary effects)
        let center_idx = (50 * 100 + 50) as usize;
        assert!(lap[center_idx] < 0.01, "Flat region should have low Laplacian");
    }

    #[test]
    fn test_laplacian_edge() {
        // Step edge should have high Laplacian
        let mut image = ImageBuffer::new(100, 100);

        for y in 0..100 {
            for x in 0..100 {
                let val = if x < 50 { 0.0 } else { 1.0 };
                image.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let lap = compute_laplacian(&image);

        // Edge at x=50 should have high Laplacian
        let edge_idx = (50 * 100 + 50) as usize;
        assert!(lap[edge_idx] > 0.1, "Edge should have high Laplacian, got {}", lap[edge_idx]);
    }

    #[test]
    fn test_find_peaks() {
        // Create field with known peaks
        let mut field = vec![0.0; 100 * 100];

        // Peak at (25, 25)
        field[25 * 100 + 25] = 1.0;

        // Peak at (75, 75)
        field[75 * 100 + 75] = 0.8;

        let peaks = find_local_maxima(&field, 100, 100, 0.5, 5);

        assert_eq!(peaks.len(), 2);
        assert_eq!(peaks[0], (25, 25, 1.0));  // Sorted by magnitude
        assert_eq!(peaks[1], (75, 75, 0.8));
    }
}
