//! Gaussian filtering and derivative-of-Gaussian for structure tensor computation
//!
//! Implements separable Gaussian convolution and derivative filters
//! for efficient, high-quality gradient computation.
//!
//! # Theory
//!
//! Gaussian kernel: G(x, σ) = (1/√(2πσ²)) × exp(-x²/(2σ²))
//!
//! Derivative: ∂G/∂x = -x/(σ²) × G(x, σ)
//!
//! For images, use separable convolution:
//! - First: convolve rows
//! - Then: convolve columns
//! - Complexity: O(W×H×K) instead of O(W×H×K²)

/// Generate 1D Gaussian kernel
///
/// # Parameters
/// - `sigma`: Standard deviation in pixels
/// - `truncate`: Number of standard deviations to include (default: 3.0)
///
/// # Returns
/// Kernel coefficients (normalized to sum to 1.0)
pub fn gaussian_kernel_1d(sigma: f32, truncate: f32) -> Vec<f32> {
    let radius = (sigma * truncate).ceil() as usize;
    let size = 2 * radius + 1;

    let mut kernel = Vec::with_capacity(size);
    let sigma_sq = sigma * sigma;
    let norm_factor = 1.0 / (2.0 * std::f32::consts::PI * sigma_sq).sqrt();

    let mut sum = 0.0;

    for i in 0..size {
        let x = (i as f32) - (radius as f32);
        let value = norm_factor * (-x * x / (2.0 * sigma_sq)).exp();
        kernel.push(value);
        sum += value;
    }

    // Normalize to sum to 1.0
    for value in &mut kernel {
        *value /= sum;
    }

    kernel
}

/// Generate 1D derivative-of-Gaussian kernel
///
/// Computes ∂G/∂x = -x/σ² × G(x, σ)
///
/// # Parameters
/// - `sigma`: Standard deviation
/// - `truncate`: Kernel extent (default: 3.0)
///
/// # Returns
/// Derivative kernel (sums to approximately 0)
pub fn derivative_gaussian_kernel_1d(sigma: f32, truncate: f32) -> Vec<f32> {
    let radius = (sigma * truncate).ceil() as usize;
    let size = 2 * radius + 1;

    let mut kernel = Vec::with_capacity(size);
    let sigma_sq = sigma * sigma;
    let norm_factor = 1.0 / (2.0 * std::f32::consts::PI * sigma_sq).sqrt();

    for i in 0..size {
        let x = (i as f32) - (radius as f32);
        let gaussian = norm_factor * (-x * x / (2.0 * sigma_sq)).exp();
        let derivative = (-x / sigma_sq) * gaussian;
        kernel.push(derivative);
    }

    kernel
}

/// Convolve image with 1D kernel (horizontal direction)
///
/// # Parameters
/// - `input`: Source data (row-major)
/// - `width`, `height`: Image dimensions
/// - `kernel`: 1D convolution kernel
///
/// # Returns
/// Convolved image (same size as input)
pub fn convolve_horizontal(
    input: &[f32],
    width: u32,
    height: u32,
    kernel: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; (width * height) as usize];
    let radius = kernel.len() / 2;

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;

            for k in 0..kernel.len() {
                let offset = k as i32 - radius as i32;
                let sample_x = (x as i32 + offset).clamp(0, width as i32 - 1) as u32;

                let input_idx = (y * width + sample_x) as usize;
                sum += input[input_idx] * kernel[k];
            }

            let output_idx = (y * width + x) as usize;
            output[output_idx] = sum;
        }
    }

    output
}

/// Convolve image with 1D kernel (vertical direction)
pub fn convolve_vertical(
    input: &[f32],
    width: u32,
    height: u32,
    kernel: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; (width * height) as usize];
    let radius = kernel.len() / 2;

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;

            for k in 0..kernel.len() {
                let offset = k as i32 - radius as i32;
                let sample_y = (y as i32 + offset).clamp(0, height as i32 - 1) as u32;

                let input_idx = (sample_y * width + x) as usize;
                sum += input[input_idx] * kernel[k];
            }

            let output_idx = (y * width + x) as usize;
            output[output_idx] = sum;
        }
    }

    output
}

/// Convolve image with 2D Gaussian (separable)
///
/// Applies Gaussian blur using two 1D convolutions
pub fn gaussian_blur(
    input: &[f32],
    width: u32,
    height: u32,
    sigma: f32,
) -> Vec<f32> {
    let kernel = gaussian_kernel_1d(sigma, 3.0);

    // Separable: convolve horizontal then vertical
    let temp = convolve_horizontal(input, width, height, &kernel);
    convolve_vertical(&temp, width, height, &kernel)
}

/// Compute image gradient using derivative-of-Gaussian
///
/// Returns (grad_x, grad_y) where each is same size as input
pub fn derivative_of_gaussian(
    input: &[f32],
    width: u32,
    height: u32,
    sigma: f32,
) -> (Vec<f32>, Vec<f32>) {
    let gaussian_kernel = gaussian_kernel_1d(sigma, 3.0);
    let derivative_kernel = derivative_gaussian_kernel_1d(sigma, 3.0);

    // Gradient in X: convolve with [derivative, gaussian]
    let temp_x = convolve_horizontal(input, width, height, &derivative_kernel);
    let grad_x = convolve_vertical(&temp_x, width, height, &gaussian_kernel);

    // Gradient in Y: convolve with [gaussian, derivative]
    let temp_y = convolve_horizontal(input, width, height, &gaussian_kernel);
    let grad_y = convolve_vertical(&temp_y, width, height, &derivative_kernel);

    (grad_x, grad_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel_normalized() {
        let kernel = gaussian_kernel_1d(1.0, 3.0);

        // Kernel should sum to 1.0
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_derivative_kernel_antisymmetric() {
        let kernel = derivative_gaussian_kernel_1d(1.0, 3.0);

        // Derivative kernel should be antisymmetric
        let n = kernel.len();
        for i in 0..n/2 {
            assert!((kernel[i] + kernel[n - 1 - i]).abs() < 1e-6);
        }

        // Center should be ~0
        assert!(kernel[n/2].abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_blur_preserves_mean() {
        // Create constant image
        let size = 64;
        let input = vec![0.5f32; (size * size) as usize];

        let blurred = gaussian_blur(&input, size, size, 2.0);

        // Mean should be preserved
        let mean: f32 = blurred.iter().sum::<f32>() / blurred.len() as f32;
        assert!((mean - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_derivative_of_vertical_edge() {
        // Create vertical edge (left=0, right=1)
        let width = 64u32;
        let height = 64u32;
        let mut input = vec![0.0f32; (width * height) as usize];

        for y in 0..height {
            for x in width/2..width {
                input[(y * width + x) as usize] = 1.0;
            }
        }

        let (grad_x, grad_y) = derivative_of_gaussian(&input, width, height, 1.0);

        // Gradient X should be high at edge (x≈32)
        let edge_grad_x = grad_x[(32 * width + 32) as usize];
        assert!(edge_grad_x.abs() > 0.1, "Should have strong X gradient at vertical edge");

        // Gradient Y should be ~0 for vertical edge
        let edge_grad_y = grad_y[(32 * width + 32) as usize];
        assert!(edge_grad_y.abs() < 0.1, "Should have weak Y gradient at vertical edge");
    }
}
