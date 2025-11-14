//! Quality metrics for image comparison

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;

/// Quality metrics result
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Peak Signal-to-Noise Ratio (dB)
    pub psnr: f32,
    /// Structural Similarity Index
    pub ssim: f32,
    /// Mean Squared Error
    pub mse: f32,
    /// Mean Absolute Error
    pub mae: f32,
    /// Per-channel PSNR
    pub psnr_r: f32,
    pub psnr_g: f32,
    pub psnr_b: f32,
}

/// Compute PSNR between two images
pub fn compute_psnr(img1: &ImageBuffer<f32>, img2: &ImageBuffer<f32>) -> f32 {
    assert_eq!(img1.width, img2.width);
    assert_eq!(img1.height, img2.height);

    let mse = compute_mse(img1, img2);

    if mse < 1e-10 {
        100.0 // Essentially perfect
    } else {
        20.0 * (1.0f32 / mse.sqrt()).log10()
    }
}

/// Compute MSE
pub fn compute_mse(img1: &ImageBuffer<f32>, img2: &ImageBuffer<f32>) -> f32 {
    let mut sum = 0.0;
    let count = (img1.width * img1.height * 3) as f32; // 3 channels

    for (p1, p2) in img1.data.iter().zip(img2.data.iter()) {
        let diff_r = p1.r - p2.r;
        let diff_g = p1.g - p2.g;
        let diff_b = p1.b - p2.b;

        sum += diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
    }

    sum / count
}

/// Compute MAE
pub fn compute_mae(img1: &ImageBuffer<f32>, img2: &ImageBuffer<f32>) -> f32 {
    let mut sum = 0.0;
    let count = (img1.width * img1.height * 3) as f32;

    for (p1, p2) in img1.data.iter().zip(img2.data.iter()) {
        sum += (p1.r - p2.r).abs() + (p1.g - p2.g).abs() + (p1.b - p2.b).abs();
    }

    sum / count
}

/// Compute SSIM (Structural Similarity Index)
pub fn compute_ssim(img1: &ImageBuffer<f32>, img2: &ImageBuffer<f32>) -> f32 {
    assert_eq!(img1.width, img2.width);
    assert_eq!(img1.height, img2.height);

    let c1 = 0.01 * 0.01;
    let c2 = 0.03 * 0.03;
    let window_size = 11i32;
    let half_window = window_size / 2;

    let mut ssim_sum = 0.0;
    let mut count = 0;

    for y in half_window..(img1.height as i32 - half_window) {
        for x in half_window..(img1.width as i32 - half_window) {
            // Compute local statistics in window
            let mut mean1 = 0.0;
            let mut mean2 = 0.0;
            let mut window_count = 0;

            // Mean
            for dy in -half_window..=half_window {
                for dx in -half_window..=half_window {
                    let px = (x + dx) as u32;
                    let py = (y + dy) as u32;

                    if let (Some(p1), Some(p2)) = (img1.get_pixel(px, py), img2.get_pixel(px, py)) {
                        let luma1 = (p1.r + p1.g + p1.b) / 3.0;
                        let luma2 = (p2.r + p2.g + p2.b) / 3.0;

                        mean1 += luma1;
                        mean2 += luma2;
                        window_count += 1;
                    }
                }
            }

            mean1 /= window_count as f32;
            mean2 /= window_count as f32;

            // Variance and covariance
            let mut var1 = 0.0;
            let mut var2 = 0.0;
            let mut covar = 0.0;

            for dy in -half_window..=half_window {
                for dx in -half_window..=half_window {
                    let px = (x + dx) as u32;
                    let py = (y + dy) as u32;

                    if let (Some(p1), Some(p2)) = (img1.get_pixel(px, py), img2.get_pixel(px, py)) {
                        let luma1 = (p1.r + p1.g + p1.b) / 3.0;
                        let luma2 = (p2.r + p2.g + p2.b) / 3.0;

                        let diff1 = luma1 - mean1;
                        let diff2 = luma2 - mean2;

                        var1 += diff1 * diff1;
                        var2 += diff2 * diff2;
                        covar += diff1 * diff2;
                    }
                }
            }

            var1 /= window_count as f32;
            var2 /= window_count as f32;
            covar /= window_count as f32;

            // SSIM formula
            let numerator = (2.0 * mean1 * mean2 + c1) * (2.0 * covar + c2);
            let denominator = (mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2);

            ssim_sum += numerator / denominator;
            count += 1;
        }
    }

    ssim_sum / count as f32
}

/// Compute all quality metrics
pub fn compute_all_metrics(original: &ImageBuffer<f32>, reconstructed: &ImageBuffer<f32>) -> QualityMetrics {
    let mse = compute_mse(original, reconstructed);
    let mae = compute_mae(original, reconstructed);
    let psnr = compute_psnr(original, reconstructed);
    let ssim = compute_ssim(original, reconstructed);

    // Per-channel PSNR
    let (psnr_r, psnr_g, psnr_b) = compute_per_channel_psnr(original, reconstructed);

    QualityMetrics {
        psnr,
        ssim,
        mse,
        mae,
        psnr_r,
        psnr_g,
        psnr_b,
    }
}

fn compute_per_channel_psnr(img1: &ImageBuffer<f32>, img2: &ImageBuffer<f32>) -> (f32, f32, f32) {
    let mut mse_r = 0.0;
    let mut mse_g = 0.0;
    let mut mse_b = 0.0;
    let count = (img1.width * img1.height) as f32;

    for (p1, p2) in img1.data.iter().zip(img2.data.iter()) {
        mse_r += (p1.r - p2.r) * (p1.r - p2.r);
        mse_g += (p1.g - p2.g) * (p1.g - p2.g);
        mse_b += (p1.b - p2.b) * (p1.b - p2.b);
    }

    mse_r /= count;
    mse_g /= count;
    mse_b /= count;

    let psnr_r = if mse_r < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse_r.sqrt()).log10() };
    let psnr_g = if mse_g < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse_g.sqrt()).log10() };
    let psnr_b = if mse_b < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse_b.sqrt()).log10() };

    (psnr_r, psnr_g, psnr_b)
}
