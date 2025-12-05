//! Structure Tensor for Edge-Aware Image Analysis
//!
//! The structure tensor describes local image structure (edges, corners, flat regions)
//! and provides orientation information critical for anisotropic Gaussian fitting.
//!
//! # Theory
//!
//! Given image I, the structure tensor is:
//!
//! J = [[Ix²,   Ix·Iy],  ⊗ G_σs
//!      [Ix·Iy, Iy²  ]]
//!
//! where Ix, Iy are image gradients and G_σs is smoothing Gaussian.
//!
//! Eigendecomposition gives:
//! - λ1, λ2: Variance along principal axes
//! - e1, e2: Edge tangent and normal
//! - coherence = (λ1-λ2)/(λ1+λ2): Edge strength
//!
//! # Usage
//!
//! ```ignore
//! use lgi_core::ImageBuffer;
//! use lgi_math::structure_tensor::StructureTensorField;
//!
//! let image = ImageBuffer::load("photo.png")?;
//! let tensor_field = StructureTensorField::compute(&image, 1.2, 2.5)?;
//!
//! // Get tensor at specific pixel
//! let tensor = tensor_field.get(100, 100);
//! println!("Edge tangent: {:?}", tensor.eigenvector_major);
//! println!("Coherence: {:.3}", tensor.coherence);
//! ```

use lgi_math::{vec::Vector2, eigen2x2::Eigen2x2};

/// Structure tensor at a single pixel
///
/// Describes local image structure through eigendecomposition
/// of the gradient outer-product matrix.
///
/// # Note on Eigenvector Interpretation
///
/// - **Major eigenvector (λ1)**: Direction of MAXIMUM gradient variance = **edge normal**
/// - **Minor eigenvector (λ2)**: Direction of MINIMUM gradient variance = **edge tangent**
///
/// For Gaussian fitting:
/// - Make Gaussian **thin along edge normal** (e1)
/// - Make Gaussian **long along edge tangent** (e2)
#[derive(Debug, Clone, Copy)]
pub struct StructureTensor {
    /// Major eigenvector (edge NORMAL, direction of maximal gradient variance)
    pub eigenvector_major: Vector2<f32>,

    /// Minor eigenvector (edge TANGENT, direction of minimal gradient variance)
    pub eigenvector_minor: Vector2<f32>,

    /// Major eigenvalue (gradient variance along edge normal)
    pub eigenvalue_major: f32,

    /// Minor eigenvalue (gradient variance along edge tangent)
    pub eigenvalue_minor: f32,

    /// Coherence: (λ1 - λ2) / (λ1 + λ2) ∈ [0, 1]
    ///
    /// - 0: Isotropic (corner or flat region)
    /// - 1: Strong linear structure (edge or ridge)
    pub coherence: f32,
}

impl StructureTensor {
    /// Create from eigendecomposition result
    pub fn from_eigen(eigen: Eigen2x2<f32>) -> Self {
        Self {
            eigenvector_major: eigen.eigenvector1,
            eigenvector_minor: eigen.eigenvector2,
            eigenvalue_major: eigen.lambda1,
            eigenvalue_minor: eigen.lambda2,
            coherence: eigen.coherence,
        }
    }

    /// Orientation angle of major axis (edge tangent) in radians
    pub fn orientation_angle(&self) -> f32 {
        self.eigenvector_major.y.atan2(self.eigenvector_major.x)
    }

    /// Anisotropy ratio: λ1 / λ2
    pub fn anisotropy(&self) -> f32 {
        if self.eigenvalue_minor.abs() < 1e-10 {
            f32::INFINITY
        } else {
            self.eigenvalue_major / self.eigenvalue_minor
        }
    }

    /// Is this an edge (high coherence)?
    pub fn is_edge(&self, threshold: f32) -> bool {
        self.coherence > threshold
    }

    /// Is this a corner (low coherence, high energy)?
    pub fn is_corner(&self, coherence_threshold: f32, energy_threshold: f32) -> bool {
        let energy = self.eigenvalue_major + self.eigenvalue_minor;
        self.coherence < coherence_threshold && energy > energy_threshold
    }

    /// Is this a flat region (low energy)?
    pub fn is_flat(&self, energy_threshold: f32) -> bool {
        let energy = self.eigenvalue_major + self.eigenvalue_minor;
        energy < energy_threshold
    }
}

/// Structure tensor field for entire image
///
/// Stores structure tensor for each pixel, enabling efficient
/// edge-aware processing.
#[derive(Clone)]
pub struct StructureTensorField {
    /// Image width
    pub width: u32,

    /// Image height
    pub height: u32,

    /// Structure tensors (row-major order)
    tensors: Vec<StructureTensor>,
}

impl StructureTensorField {
    /// Compute structure tensor field for image
    ///
    /// # Parameters
    ///
    /// - `image`: Input image
    /// - `sigma_gradient`: Scale for gradient computation (default: 1.2 px)
    /// - `sigma_smooth`: Scale for tensor integration/smoothing (default: 2.5 px)
    ///
    /// # Algorithm
    ///
    /// 1. Compute gradients: Ix = I ⊗ ∂G/∂x, Iy = I ⊗ ∂G/∂y
    /// 2. Form tensor components: J11 = Ix², J12 = Ix·Iy, J22 = Iy²
    /// 3. Smooth components: J ← J ⊗ G_σs
    /// 4. Eigendecompose per-pixel: J = Q Λ Q^T
    ///
    /// # Performance
    ///
    /// - Time: O(W×H) with separable convolutions
    /// - Memory: ~60 bytes per pixel (temporary)
    /// - Can be parallelized (rayon)
    ///
    /// # References
    ///
    /// - Bigun & Granlund (1987): Optical flow constraint equation
    /// - Förstner & Gülch (1987): Structure tensor for corner detection
    /// - Brox et al. (2006): Optical flow using structure tensor
    pub fn compute(
        image: &crate::ImageBuffer<f32>,
        sigma_gradient: f32,
        sigma_smooth: f32,
    ) -> crate::Result<Self> {
        let width = image.width;
        let height = image.height;

        // Convert image to luma for gradient computation
        let luma = Self::extract_luma(image);

        // Step 1: Compute gradients using derivative-of-Gaussian
        let (grad_x, grad_y) = crate::gaussian_filter::derivative_of_gaussian(
            &luma,
            width,
            height,
            sigma_gradient,
        );

        // Step 2: Form tensor components (outer product)
        let mut j11 = vec![0.0f32; (width * height) as usize];
        let mut j12 = vec![0.0f32; (width * height) as usize];
        let mut j22 = vec![0.0f32; (width * height) as usize];

        for i in 0..(width * height) as usize {
            j11[i] = grad_x[i] * grad_x[i];
            j12[i] = grad_x[i] * grad_y[i];
            j22[i] = grad_y[i] * grad_y[i];
        }

        // Step 3: Smooth tensor components with Gaussian
        let j11_smooth = crate::gaussian_filter::gaussian_blur(&j11, width, height, sigma_smooth);
        let j12_smooth = crate::gaussian_filter::gaussian_blur(&j12, width, height, sigma_smooth);
        let j22_smooth = crate::gaussian_filter::gaussian_blur(&j22, width, height, sigma_smooth);

        // Step 4: Eigendecompose per-pixel
        let mut tensors = Vec::with_capacity((width * height) as usize);

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;

                let eigen = Eigen2x2::decompose(j11_smooth[idx], j12_smooth[idx], j22_smooth[idx]);
                tensors.push(StructureTensor::from_eigen(eigen));
            }
        }

        Ok(Self { width, height, tensors })
    }

    /// Get structure tensor at pixel (x, y)
    pub fn get(&self, x: u32, y: u32) -> &StructureTensor {
        assert!(x < self.width && y < self.height, "Pixel out of bounds");
        &self.tensors[(y * self.width + x) as usize]
    }

    /// Get structure tensor with bounds checking
    pub fn get_checked(&self, x: u32, y: u32) -> Option<&StructureTensor> {
        if x < self.width && y < self.height {
            Some(&self.tensors[(y * self.width + x) as usize])
        } else {
            None
        }
    }

    /// Extract luma channel from RGB image
    fn extract_luma(image: &crate::ImageBuffer<f32>) -> Vec<f32> {
        let size = (image.width * image.height) as usize;
        let mut luma = Vec::with_capacity(size);

        for pixel in &image.data {
            // Rec. 709 luma coefficients
            let y = 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;
            luma.push(y);
        }

        luma
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageBuffer;
    use lgi_math::color::Color4;

    #[test]
    fn test_structure_tensor_flat_region() {
        // Create solid color image (no structure)
        let mut image = ImageBuffer::new(64, 64);
        for pixel in &mut image.data {
            *pixel = Color4::new(0.5, 0.5, 0.5, 1.0);
        }

        let tensor_field = StructureTensorField::compute(&image, 1.2, 2.5).unwrap();

        // Check center pixel
        let tensor = tensor_field.get(32, 32);

        // Flat region should have low coherence (isotropic)
        assert!(tensor.coherence < 0.1, "Flat region should have low coherence");
    }

    #[test]
    fn test_structure_tensor_vertical_edge() {
        // Create vertical edge (left=black, right=white)
        let mut image = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let color = if x < 32 { 0.0 } else { 1.0 };
                image.set_pixel(x, y, Color4::new(color, color, color, 1.0));
            }
        }

        let tensor_field = StructureTensorField::compute(&image, 1.2, 2.5).unwrap();

        // Check pixel on edge
        let tensor = tensor_field.get(32, 32);

        // Vertical edge should have high coherence
        assert!(tensor.coherence > 0.3, "Edge should have high coherence, got {}", tensor.coherence);

        // For vertical edge: gradient points horizontally (perpendicular to edge)
        // Major eigenvector (λ1) = edge NORMAL = horizontal
        let angle = tensor.orientation_angle();
        let horizontal_angle = 0.0;

        // Allow tolerance due to discretization and smoothing
        let angle_diff = (angle - horizontal_angle).abs();
        assert!(angle_diff < 0.5, "Edge normal should be horizontal for vertical edge, got angle={}", angle);

        // Minor eigenvector should be perpendicular (vertical = edge tangent)
        let tangent_angle = tensor.eigenvector_minor.y.atan2(tensor.eigenvector_minor.x);
        let vertical_angle = std::f32::consts::PI / 2.0;
        let tangent_diff = (tangent_angle - vertical_angle).abs().min((tangent_angle + vertical_angle).abs());
        assert!(tangent_diff < 0.5, "Edge tangent should be vertical, got {}", tangent_angle);
    }
}

