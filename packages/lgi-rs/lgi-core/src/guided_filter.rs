//! Guided Filter - Edge-Preserving Image Smoothing
//!
//! Based on He et al. "Guided Image Filtering" (2010)
//! O(N) algorithm using box filters
//! CRITICAL for real photo color initialization

use crate::ImageBuffer;
use lgi_math::color::Color4;

/// Guided filter for edge-preserving smoothing
pub struct GuidedFilter {
    pub radius: usize,   // Window radius (default: 4-8)
    pub epsilon: f32,    // Regularization (default: 0.01² = 0.0001)
}

impl Default for GuidedFilter {
    fn default() -> Self {
        Self {
            radius: 6,
            epsilon: 0.0004,  // 0.02²
        }
    }
}

impl GuidedFilter {
    /// Apply guided filter (image guides itself)
    pub fn filter(&self, image: &ImageBuffer<f32>) -> ImageBuffer<f32> {
        self.filter_with_guide(image, image)
    }

    /// Apply guided filter with separate guide image
    pub fn filter_with_guide(
        &self,
        input: &ImageBuffer<f32>,
        guide: &ImageBuffer<f32>,
    ) -> ImageBuffer<f32> {
        assert_eq!(input.width, guide.width);
        assert_eq!(input.height, guide.height);

        let width = input.width;
        let height = input.height;

        // Convert to grayscale guide (luminance)
        let mut guide_gray = vec![0.0f32; (width * height) as usize];
        for i in 0..guide.data.len() {
            guide_gray[i] = guide.data[i].r * 0.299 + guide.data[i].g * 0.587 + guide.data[i].b * 0.114;
        }

        // Compute box filter statistics
        let mean_I = box_filter_gray(&guide_gray, width, height, self.radius);
        let mean_p_r = box_filter_channel(&input.data, width, height, self.radius, |c| c.r);
        let mean_p_g = box_filter_channel(&input.data, width, height, self.radius, |c| c.g);
        let mean_p_b = box_filter_channel(&input.data, width, height, self.radius, |c| c.b);

        // Compute correlation and variance
        let corr_Ip_r = box_filter_correlation(&guide_gray, &input.data, width, height, self.radius, |c| c.r);
        let corr_Ip_g = box_filter_correlation(&guide_gray, &input.data, width, height, self.radius, |c| c.g);
        let corr_Ip_b = box_filter_correlation(&guide_gray, &input.data, width, height, self.radius, |c| c.b);

        let var_I = box_filter_variance(&guide_gray, &mean_I, width, height, self.radius);

        // Compute linear coefficients a, b
        let mut a_r = vec![0.0; (width * height) as usize];
        let mut a_g = vec![0.0; (width * height) as usize];
        let mut a_b = vec![0.0; (width * height) as usize];
        let mut b_r = vec![0.0; (width * height) as usize];
        let mut b_g = vec![0.0; (width * height) as usize];
        let mut b_b = vec![0.0; (width * height) as usize];

        for i in 0..(width * height) as usize {
            let var_eps = var_I[i] + self.epsilon;

            a_r[i] = (corr_Ip_r[i] - mean_I[i] * mean_p_r[i]) / var_eps;
            a_g[i] = (corr_Ip_g[i] - mean_I[i] * mean_p_g[i]) / var_eps;
            a_b[i] = (corr_Ip_b[i] - mean_I[i] * mean_p_b[i]) / var_eps;

            b_r[i] = mean_p_r[i] - a_r[i] * mean_I[i];
            b_g[i] = mean_p_g[i] - a_g[i] * mean_I[i];
            b_b[i] = mean_p_b[i] - a_b[i] * mean_I[i];
        }

        // Average coefficients
        let mean_a_r = box_filter_gray(&a_r, width, height, self.radius);
        let mean_a_g = box_filter_gray(&a_g, width, height, self.radius);
        let mean_a_b = box_filter_gray(&a_b, width, height, self.radius);
        let mean_b_r = box_filter_gray(&b_r, width, height, self.radius);
        let mean_b_g = box_filter_gray(&b_g, width, height, self.radius);
        let mean_b_b = box_filter_gray(&b_b, width, height, self.radius);

        // Apply linear transform
        let mut output = ImageBuffer::new(width, height);
        for i in 0..(width * height) as usize {
            let I = guide_gray[i];
            let r = mean_a_r[i] * I + mean_b_r[i];
            let g = mean_a_g[i] * I + mean_b_g[i];
            let b = mean_a_b[i] * I + mean_b_b[i];

            output.data[i] = Color4::new(
                r.clamp(0.0, 1.0),
                g.clamp(0.0, 1.0),
                b.clamp(0.0, 1.0),
                1.0,
            );
        }

        output
    }
}

/// Fast box filter on grayscale
fn box_filter_gray(data: &[f32], width: u32, height: u32, radius: usize) -> Vec<f32> {
    let r = radius as i32;
    let mut output = vec![0.0; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut count = 0.0;

            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;

                    sum += data[(ny * width + nx) as usize];
                    count += 1.0;
                }
            }

            output[(y * width + x) as usize] = sum / count;
        }
    }

    output
}

/// Box filter on image channel
fn box_filter_channel<F>(data: &[Color4<f32>], width: u32, height: u32, radius: usize, extract: F) -> Vec<f32>
where
    F: Fn(&Color4<f32>) -> f32,
{
    let r = radius as i32;
    let mut output = vec![0.0; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut count = 0.0;

            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;

                    sum += extract(&data[(ny * width + nx) as usize]);
                    count += 1.0;
                }
            }

            output[(y * width + x) as usize] = sum / count;
        }
    }

    output
}

/// Box filter correlation between guide and input
fn box_filter_correlation<F>(
    guide_gray: &[f32],
    input: &[Color4<f32>],
    width: u32,
    height: u32,
    radius: usize,
    extract: F,
) -> Vec<f32>
where
    F: Fn(&Color4<f32>) -> f32,
{
    let r = radius as i32;
    let mut output = vec![0.0; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut count = 0.0;

            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
                    let idx = (ny * width + nx) as usize;

                    sum += guide_gray[idx] * extract(&input[idx]);
                    count += 1.0;
                }
            }

            output[(y * width + x) as usize] = sum / count;
        }
    }

    output
}

/// Box filter variance
fn box_filter_variance(
    data: &[f32],
    mean: &[f32],
    width: u32,
    height: u32,
    radius: usize,
) -> Vec<f32> {
    let r = radius as i32;
    let mut output = vec![0.0; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut count = 0.0;

            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;
                    let idx = (ny * width + nx) as usize;

                    let diff = data[idx] - mean[idx];
                    sum += diff * diff;
                    count += 1.0;
                }
            }

            output[(y * width + x) as usize] = sum / count;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guided_filter_preserves_edges() {
        // Create image with edge
        let mut img = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let val = if x < 32 { 0.0 } else { 1.0 };
                img.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let filter = GuidedFilter::default();
        let filtered = filter.filter(&img);

        // Check edge is preserved (not blurred like Gaussian)
        let left = filtered.get_pixel(16, 32).unwrap();
        let right = filtered.get_pixel(48, 32).unwrap();

        assert!(left.r < 0.3, "Left side should stay dark");
        assert!(right.r > 0.7, "Right side should stay bright");
    }
}
