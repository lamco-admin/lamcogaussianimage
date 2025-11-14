//! MS-SSIM (Multi-Scale Structural Similarity Index)
//!
//! Perceptual image quality metric that correlates better with human perception than PSNR
//! Based on Wang et al. (2003)

use crate::ImageBuffer;
use lgi_math::color::Color4;

/// MS-SSIM loss computer
pub struct MSSSIM {
    /// Number of scales (typically 5)
    pub num_scales: usize,
    /// Window size for SSIM computation
    pub window_size: usize,
    /// Scale weights (from Wang et al.)
    pub weights: Vec<f32>,
}

impl Default for MSSSIM {
    fn default() -> Self {
        Self {
            num_scales: 5,
            window_size: 11,
            // Weights from MS-SSIM paper
            weights: vec![0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        }
    }
}

impl MSSSIM {
    /// Compute MS-SSIM between two images
    /// Returns value in [0, 1] where 1 = identical
    pub fn compute(
        &self,
        img1: &ImageBuffer<f32>,
        img2: &ImageBuffer<f32>,
    ) -> f32 {
        assert_eq!(img1.width, img2.width);
        assert_eq!(img1.height, img2.height);

        let mut ssim_values = Vec::new();
        let mut contrast_values = Vec::new();
        let mut structure_values = Vec::new();

        let mut current1 = img1.clone();
        let mut current2 = img2.clone();

        for scale in 0..self.num_scales {
            // Compute SSIM at this scale
            let (luminance, contrast, structure) = self.compute_ssim_components(&current1, &current2);

            if scale < self.num_scales - 1 {
                // Store contrast and structure for all but last scale
                contrast_values.push(contrast);
                structure_values.push(structure);

                // Downsample for next scale
                current1 = downsample_2x(&current1);
                current2 = downsample_2x(&current2);
            } else {
                // Last scale: use full SSIM
                ssim_values.push(luminance * contrast * structure);
            }
        }

        // Combine scales with weights
        let mut ms_ssim = ssim_values[0];  // Last scale full SSIM
        for (i, &cs) in contrast_values.iter().enumerate() {
            let s = structure_values[i];
            ms_ssim *= (cs * s).powf(self.weights[i]);
        }

        ms_ssim
    }

    /// Compute MS-SSIM as loss (1 - MS-SSIM)
    pub fn compute_loss(
        &self,
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> f32 {
        let ms_ssim = self.compute(rendered, target);
        1.0 - ms_ssim
    }

    /// Compute SSIM components (luminance, contrast, structure)
    fn compute_ssim_components(
        &self,
        img1: &ImageBuffer<f32>,
        img2: &ImageBuffer<f32>,
    ) -> (f32, f32, f32) {
        let c1 = (0.01_f32).powi(2);
        let c2 = (0.03_f32).powi(2);

        // Compute means
        let (mu1, mu2) = self.compute_means(img1, img2);

        // Compute variances and covariance
        let (var1, var2, cov) = self.compute_variances_covariance(img1, img2, mu1, mu2);

        // SSIM components
        let luminance = (2.0 * mu1 * mu2 + c1) / (mu1 * mu1 + mu2 * mu2 + c1);
        let contrast = (2.0 * var1.sqrt() * var2.sqrt() + c2) / (var1 + var2 + c2);
        let structure = (cov + c2 / 2.0) / (var1.sqrt() * var2.sqrt() + c2 / 2.0);

        (luminance, contrast, structure)
    }

    fn compute_means(&self, img1: &ImageBuffer<f32>, img2: &ImageBuffer<f32>) -> (f32, f32) {
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;
        let count = (img1.width * img1.height * 3) as f32;

        for (p1, p2) in img1.data.iter().zip(img2.data.iter()) {
            sum1 += p1.r + p1.g + p1.b;
            sum2 += p2.r + p2.g + p2.b;
        }

        (sum1 / count, sum2 / count)
    }

    fn compute_variances_covariance(
        &self,
        img1: &ImageBuffer<f32>,
        img2: &ImageBuffer<f32>,
        mu1: f32,
        mu2: f32,
    ) -> (f32, f32, f32) {
        let mut var1 = 0.0;
        let mut var2 = 0.0;
        let mut cov = 0.0;
        let count = (img1.width * img1.height * 3) as f32;

        for (p1, p2) in img1.data.iter().zip(img2.data.iter()) {
            // Average across RGB channels
            let v1 = (p1.r + p1.g + p1.b) / 3.0;
            let v2 = (p2.r + p2.g + p2.b) / 3.0;

            let diff1 = v1 - mu1;
            let diff2 = v2 - mu2;

            var1 += diff1 * diff1;
            var2 += diff2 * diff2;
            cov += diff1 * diff2;
        }

        (var1 / count, var2 / count, cov / count)
    }
}

/// Downsample image by factor of 2 (average pooling)
fn downsample_2x(img: &ImageBuffer<f32>) -> ImageBuffer<f32> {
    let new_width = (img.width / 2).max(1);
    let new_height = (img.height / 2).max(1);

    let mut downsampled = ImageBuffer::new(new_width, new_height);

    for y in 0..new_height {
        for x in 0..new_width {
            let x2 = x * 2;
            let y2 = y * 2;

            // Average 2×2 block
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            let mut count = 0.0;

            for dy in 0..2 {
                for dx in 0..2 {
                    if let Some(pixel) = img.get_pixel(x2 + dx, y2 + dy) {
                        r += pixel.r;
                        g += pixel.g;
                        b += pixel.b;
                        count += 1.0;
                    }
                }
            }

            if count > 0.0 {
                downsampled.set_pixel(x, y, Color4::new(r / count, g / count, b / count, 1.0));
            }
        }
    }

    downsampled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ms_ssim_identical_images() {
        let img1 = ImageBuffer::new(64, 64);
        let img2 = img1.clone();

        let ms_ssim = MSSSIM::default();
        let similarity = ms_ssim.compute(&img1, &img2);

        assert!((similarity - 1.0).abs() < 0.01, "Identical images should have MS-SSIM ≈ 1.0");
    }

    #[test]
    fn test_downsample() {
        let mut img = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                img.set_pixel(x, y, Color4::new(1.0, 0.0, 0.0, 1.0));
            }
        }

        let down = downsample_2x(&img);
        assert_eq!(down.width, 32);
        assert_eq!(down.height, 32);

        let pixel = down.get_pixel(16, 16).unwrap();
        assert!((pixel.r - 1.0).abs() < 0.01);
    }
}
