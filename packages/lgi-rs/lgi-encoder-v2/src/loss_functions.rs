//! Loss functions for Gaussian optimization
//!
//! Implements various loss functions used in the reference papers:
//! - L1 (Mean Absolute Error)
//! - L2 (Mean Squared Error)
//! - SSIM (Structural Similarity Index)
//! - D-SSIM (1 - SSIM)

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;

/// Compute L1 loss (Mean Absolute Error)
pub fn compute_l1_loss(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut loss = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        loss += (r.r - t.r).abs() + (r.g - t.g).abs() + (r.b - t.b).abs();
    }
    loss / (rendered.width * rendered.height * 3) as f32
}

/// Compute L2 loss (Mean Squared Error)
pub fn compute_l2_loss(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut loss = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        loss += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    loss / (rendered.width * rendered.height * 3) as f32
}

/// Compute SSIM for a single channel between two windows
fn compute_ssim_window(
    window1: &[f32],
    window2: &[f32],
    c1: f32,
    c2: f32,
) -> f32 {
    let n = window1.len() as f32;

    // Compute means
    let mean1: f32 = window1.iter().sum::<f32>() / n;
    let mean2: f32 = window2.iter().sum::<f32>() / n;

    // Compute variances and covariance
    let mut var1 = 0.0;
    let mut var2 = 0.0;
    let mut covar = 0.0;

    for i in 0..window1.len() {
        let diff1 = window1[i] - mean1;
        let diff2 = window2[i] - mean2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
        covar += diff1 * diff2;
    }

    var1 /= n;
    var2 /= n;
    covar /= n;

    // SSIM formula
    let numerator = (2.0 * mean1 * mean2 + c1) * (2.0 * covar + c2);
    let denominator = (mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2);

    numerator / denominator
}

/// Extract a channel from an image buffer
fn extract_channel(buffer: &ImageBuffer<f32>, channel: usize) -> Vec<f32> {
    buffer.data.iter().map(|c| match channel {
        0 => c.r,
        1 => c.g,
        2 => c.b,
        _ => panic!("Invalid channel"),
    }).collect()
}

/// Compute SSIM between two images (averaged over RGB channels)
/// Returns a value between 0 (completely different) and 1 (identical)
pub fn compute_ssim(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let width = rendered.width as usize;
    let height = rendered.height as usize;

    // SSIM constants (from original paper)
    let k1: f32 = 0.01;
    let k2: f32 = 0.03;
    let l: f32 = 1.0; // Dynamic range (we use normalized [0,1] colors)
    let c1 = (k1 * l).powi(2);
    let c2 = (k2 * l).powi(2);

    // Window size (typically 11x11, but we'll use 8x8 for efficiency)
    let window_size = 8;
    let half_window = window_size / 2;

    let mut ssim_sum = 0.0;
    let mut count = 0;

    // Process each RGB channel separately
    for channel in 0..3 {
        let rendered_channel = extract_channel(rendered, channel);
        let target_channel = extract_channel(target, channel);

        // Slide window across image
        for y in half_window..(height - half_window) {
            for x in half_window..(width - half_window) {
                let mut window1 = Vec::with_capacity(window_size * window_size);
                let mut window2 = Vec::with_capacity(window_size * window_size);

                // Extract windows
                for wy in (y - half_window)..(y + half_window) {
                    for wx in (x - half_window)..(x + half_window) {
                        let idx = wy * width + wx;
                        window1.push(rendered_channel[idx]);
                        window2.push(target_channel[idx]);
                    }
                }

                ssim_sum += compute_ssim_window(&window1, &window2, c1, c2);
                count += 1;
            }
        }
    }

    // Average SSIM across all windows and channels
    ssim_sum / count as f32
}

/// Compute D-SSIM loss (1 - SSIM)
/// Used in 3D Gaussian Splatting paper
pub fn compute_dssim_loss(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    1.0 - compute_ssim(rendered, target)
}

/// Compute combined loss as used in GaussianImage and Video Codec papers
/// Loss = λ1×L1 + λ2×L2 + λ3×(1-SSIM)
pub fn compute_combined_loss(
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
    lambda_l1: f32,
    lambda_l2: f32,
    lambda_ssim: f32,
) -> f32 {
    let l1 = compute_l1_loss(rendered, target);
    let l2 = compute_l2_loss(rendered, target);
    let ssim = compute_ssim(rendered, target);

    lambda_l1 * l1 + lambda_l2 * l2 + lambda_ssim * (1.0 - ssim)
}

/// Compute 3D-GS style loss (L1 + D-SSIM)
pub fn compute_3dgs_loss(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let l1 = compute_l1_loss(rendered, target);
    let dssim = compute_dssim_loss(rendered, target);
    l1 + dssim
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::color::Color4;

    #[test]
    fn test_ssim_identical() {
        let mut img = ImageBuffer::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                img.set_pixel(x, y, Color4::new(0.5, 0.5, 0.5, 1.0));
            }
        }

        let ssim = compute_ssim(&img, &img);
        assert!((ssim - 1.0).abs() < 0.01, "Identical images should have SSIM ≈ 1.0");
    }

    #[test]
    fn test_losses_identical() {
        let mut img = ImageBuffer::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                img.set_pixel(x, y, Color4::new(0.5, 0.5, 0.5, 1.0));
            }
        }

        let l1 = compute_l1_loss(&img, &img);
        let l2 = compute_l2_loss(&img, &img);

        assert!(l1 < 1e-6, "Identical images should have L1 ≈ 0");
        assert!(l2 < 1e-6, "Identical images should have L2 ≈ 0");
    }
}
