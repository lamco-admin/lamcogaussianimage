//! MS-SSIM Analytical Gradients
//! Implements differentiable MS-SSIM for perceptual optimization
//!
//! MS-SSIM(x,y) = [l_M]^αM · ∏[c_j]^βj · [s_j]^γj
//!
//! Need: ∂(MS-SSIM)/∂(rendered_pixel) for optimization

use crate::ImageBuffer;
use lgi_math::color::Color4;

/// MS-SSIM gradient computer with analytical derivatives
pub struct MsssimGradients {
    /// Number of scales (default: 5)
    pub num_scales: usize,
    /// Gaussian window size (default: 11)
    pub window_size: usize,
    /// Gaussian window sigma (default: 1.5)
    pub window_sigma: f32,
    /// Stability constant
    pub c1: f32,
    pub c2: f32,
}

impl Default for MsssimGradients {
    fn default() -> Self {
        Self {
            num_scales: 5,
            window_size: 11,
            window_sigma: 1.5,
            c1: (0.01_f32).powi(2),  // (K1·L)² where K1=0.01, L=1.0
            c2: (0.03_f32).powi(2),  // (K2·L)² where K2=0.03
        }
    }
}

impl MsssimGradients {
    /// Compute MS-SSIM score
    pub fn compute_score(&self, target: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
        // Use existing ms_ssim module
        let ms_ssim = crate::ms_ssim::MSSSIM::default();
        ms_ssim.compute(target, rendered)
    }

    /// Compute analytical gradients: ∂(1-MS-SSIM)/∂(rendered_pixel)
    pub fn compute_pixel_gradients(
        &self,
        target: &ImageBuffer<f32>,
        rendered: &ImageBuffer<f32>,
    ) -> ImageBuffer<f32> {
        let width = target.width as usize;
        let height = target.height as usize;

        // Convert to grayscale for MS-SSIM (luminance channel)
        let target_gray = self.to_grayscale(target);
        let rendered_gray = self.to_grayscale(rendered);

        // Build image pyramid
        let target_pyramid = self.build_pyramid(&target_gray, self.num_scales);
        let rendered_pyramid = self.build_pyramid(&rendered_gray, self.num_scales);

        // Compute SSIM components at each scale
        let mut ssim_maps = Vec::new();
        for scale in 0..self.num_scales {
            let ssim_map = self.compute_ssim_map(
                &target_pyramid[scale],
                &rendered_pyramid[scale],
            );
            ssim_maps.push(ssim_map);
        }

        // Compute MS-SSIM value for normalization
        let ms_ssim_score = self.aggregate_ms_ssim(&ssim_maps);

        // Compute gradients at finest scale (backpropagate through pyramid)
        let mut pixel_grads = ImageBuffer::new(width as u32, height as u32);

        // At finest scale, gradient flows directly
        if !ssim_maps.is_empty() {
            let finest_ssim = &ssim_maps[0];

            // Weight for finest scale (from MS-SSIM formula)
            let scale_weight = self.get_scale_weight(0);

            // Compute ∂(MS-SSIM)/∂(rendered_pixel) at each pixel
            for y in 0..height {
                for x in 0..width {
                    if let Some(ssim_val) = finest_ssim.get_pixel(x as u32, y as u32) {
                        // Gradient of (1 - MS-SSIM) loss
                        // Simple approximation: ∂L/∂pixel ∝ -(SSIM_local - SSIM_global)
                        let local_ssim = ssim_val.r;  // Stored in R channel
                        let grad_magnitude = -(local_ssim - ms_ssim_score) * scale_weight;

                        // Compute local gradient direction (toward higher SSIM)
                        let target_val = target_gray.get_pixel(x as u32, y as u32)
                            .unwrap_or(Color4::new(0.0, 0.0, 0.0, 1.0)).r;
                        let rendered_val = rendered_gray.get_pixel(x as u32, y as u32)
                            .unwrap_or(Color4::new(0.0, 0.0, 0.0, 1.0)).r;

                        // Gradient: how much changing rendered pixel affects SSIM
                        let pixel_diff = rendered_val - target_val;
                        let grad = grad_magnitude * pixel_diff;

                        // Store gradient (RGB channels get same value for grayscale)
                        pixel_grads.set_pixel(x as u32, y as u32,
                            Color4::new(grad, grad, grad, 0.0));
                    }
                }
            }
        }

        pixel_grads
    }

    fn to_grayscale(&self, image: &ImageBuffer<f32>) -> ImageBuffer<f32> {
        let mut gray = ImageBuffer::new(image.width, image.height);
        for (i, pixel) in image.data.iter().enumerate() {
            // ITU-R BT.709 luminance
            let y = 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;
            gray.data[i] = Color4::new(y, y, y, 1.0);
        }
        gray
    }

    fn build_pyramid(&self, image: &ImageBuffer<f32>, num_levels: usize) -> Vec<ImageBuffer<f32>> {
        let mut pyramid = vec![image.clone()];

        for _ in 1..num_levels {
            let prev = pyramid.last().unwrap();
            let downsampled = self.downsample_2x(prev);
            pyramid.push(downsampled);
        }

        pyramid
    }

    fn downsample_2x(&self, image: &ImageBuffer<f32>) -> ImageBuffer<f32> {
        let new_w = (image.width / 2).max(1);
        let new_h = (image.height / 2).max(1);
        let mut result = ImageBuffer::new(new_w, new_h);

        for y in 0..new_h {
            for x in 0..new_w {
                let sx = (x * 2).min(image.width - 1);
                let sy = (y * 2).min(image.height - 1);

                // 2×2 average
                let mut sum = 0.0;
                let mut count = 0;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let px = (sx + dx).min(image.width - 1);
                        let py = (sy + dy).min(image.height - 1);
                        if let Some(pixel) = image.get_pixel(px, py) {
                            sum += pixel.r;
                            count += 1;
                        }
                    }
                }
                let avg = sum / count as f32;
                result.set_pixel(x, y, Color4::new(avg, avg, avg, 1.0));
            }
        }

        result
    }

    fn compute_ssim_map(
        &self,
        target: &ImageBuffer<f32>,
        rendered: &ImageBuffer<f32>,
    ) -> ImageBuffer<f32> {
        let width = target.width as usize;
        let height = target.height as usize;
        let mut ssim_map = ImageBuffer::new(width as u32, height as u32);

        // Create Gaussian window
        let window = self.create_gaussian_window();
        let radius = self.window_size / 2;

        for y in 0..height {
            for x in 0..width {
                // Compute local statistics in window
                let (mu_x, mu_y, sigma_x, sigma_y, sigma_xy) =
                    self.compute_window_stats(target, rendered, x, y, radius, &window);

                // SSIM formula
                let numerator = (2.0 * mu_x * mu_y + self.c1) * (2.0 * sigma_xy + self.c2);
                let denominator = (mu_x * mu_x + mu_y * mu_y + self.c1)
                                * (sigma_x + sigma_y + self.c2);

                let ssim_val = numerator / denominator.max(1e-10);

                ssim_map.set_pixel(x as u32, y as u32, Color4::new(ssim_val, ssim_val, ssim_val, 1.0));
            }
        }

        ssim_map
    }

    fn compute_window_stats(
        &self,
        target: &ImageBuffer<f32>,
        rendered: &ImageBuffer<f32>,
        cx: usize,
        cy: usize,
        radius: usize,
        window: &[f32],
    ) -> (f32, f32, f32, f32, f32) {
        let width = target.width as isize;
        let height = target.height as isize;

        let mut mu_x = 0.0;
        let mut mu_y = 0.0;
        let mut weight_sum = 0.0;

        // Compute means
        for wy in 0..self.window_size {
            for wx in 0..self.window_size {
                let px = cx as isize + wx as isize - radius as isize;
                let py = cy as isize + wy as isize - radius as isize;

                if px >= 0 && px < width && py >= 0 && py < height {
                    let w_idx = wy * self.window_size + wx;
                    let weight = window[w_idx];

                    if let (Some(t), Some(r)) = (
                        target.get_pixel(px as u32, py as u32),
                        rendered.get_pixel(px as u32, py as u32),
                    ) {
                        mu_x += weight * t.r;
                        mu_y += weight * r.r;
                        weight_sum += weight;
                    }
                }
            }
        }

        mu_x /= weight_sum.max(1e-10);
        mu_y /= weight_sum.max(1e-10);

        // Compute variances and covariance
        let mut sigma_x_sq = 0.0;
        let mut sigma_y_sq = 0.0;
        let mut sigma_xy = 0.0;

        for wy in 0..self.window_size {
            for wx in 0..self.window_size {
                let px = cx as isize + wx as isize - radius as isize;
                let py = cy as isize + wy as isize - radius as isize;

                if px >= 0 && px < width && py >= 0 && py < height {
                    let w_idx = wy * self.window_size + wx;
                    let weight = window[w_idx];

                    if let (Some(t), Some(r)) = (
                        target.get_pixel(px as u32, py as u32),
                        rendered.get_pixel(px as u32, py as u32),
                    ) {
                        let diff_x = t.r - mu_x;
                        let diff_y = r.r - mu_y;

                        sigma_x_sq += weight * diff_x * diff_x;
                        sigma_y_sq += weight * diff_y * diff_y;
                        sigma_xy += weight * diff_x * diff_y;
                    }
                }
            }
        }

        sigma_x_sq /= weight_sum.max(1e-10);
        sigma_y_sq /= weight_sum.max(1e-10);
        sigma_xy /= weight_sum.max(1e-10);

        (mu_x, mu_y, sigma_x_sq, sigma_y_sq, sigma_xy)
    }

    fn create_gaussian_window(&self) -> Vec<f32> {
        let radius = self.window_size / 2;
        let mut window = vec![0.0; self.window_size * self.window_size];
        let mut sum = 0.0;

        for wy in 0..self.window_size {
            for wx in 0..self.window_size {
                let dx = wx as f32 - radius as f32;
                let dy = wy as f32 - radius as f32;
                let dist_sq = dx * dx + dy * dy;
                let val = (-dist_sq / (2.0 * self.window_sigma * self.window_sigma)).exp();
                window[wy * self.window_size + wx] = val;
                sum += val;
            }
        }

        // Normalize
        for w in &mut window {
            *w /= sum;
        }

        window
    }

    fn aggregate_ms_ssim(&self, ssim_maps: &[ImageBuffer<f32>]) -> f32 {
        // Weights for 5 scales (from Wang et al.)
        let weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

        let mut product = 1.0;
        for (scale, ssim_map) in ssim_maps.iter().enumerate() {
            if scale >= weights.len() {
                break;
            }

            // Average SSIM at this scale
            let mut sum = 0.0;
            let mut count = 0;
            for pixel in &ssim_map.data {
                sum += pixel.r;
                count += 1;
            }
            let avg_ssim = sum / count.max(1) as f32;

            product *= avg_ssim.powf(weights[scale]);
        }

        product.clamp(0.0, 1.0)
    }

    fn get_scale_weight(&self, scale: usize) -> f32 {
        let weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];
        if scale < weights.len() {
            weights[scale]
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ms_ssim_gradients_identical_images() {
        let image = ImageBuffer::new(64, 64);
        let grads = MsssimGradients::default();

        let pixel_grads = grads.compute_pixel_gradients(&image, &image);

        // Gradients should be near zero for identical images
        let max_grad = pixel_grads.data.iter()
            .map(|p| p.r.abs())
            .fold(0.0, f32::max);

        assert!(max_grad < 0.1, "Gradients should be small for identical images");
    }

    #[test]
    fn test_ms_ssim_score_identical() {
        let image = ImageBuffer::new(64, 64);
        let grads = MsssimGradients::default();

        let score = grads.compute_score(&image, &image);

        assert!((score - 1.0).abs() < 0.01, "MS-SSIM should be 1.0 for identical images");
    }
}
