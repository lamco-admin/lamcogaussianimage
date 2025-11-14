//! Automatic differentiation through Gaussian rendering
//!
//! This module implements full backpropagation through the rendering pipeline,
//! computing gradients for ALL Gaussian parameters (not just position/color).

use lgi_math::{
    gaussian::Gaussian2D,
    parameterization::Euler,
    vec::Vector2,
    color::Color4,
};
use lgi_core::ImageBuffer;

/// Complete gradient for a single Gaussian
#[derive(Debug, Clone)]
pub struct FullGaussianGradient {
    /// Gradient w.r.t. position (μx, μy)
    pub position: Vector2<f32>,

    /// Gradient w.r.t. scale (σx, σy)
    pub scale: Vector2<f32>,

    /// Gradient w.r.t. rotation (θ)
    pub rotation: f32,

    /// Gradient w.r.t. color (r, g, b, a)
    pub color: Color4<f32>,

    /// Gradient w.r.t. opacity (α)
    pub opacity: f32,
}

impl FullGaussianGradient {
    pub fn zero() -> Self {
        Self {
            position: Vector2::zero(),
            scale: Vector2::zero(),
            rotation: 0.0,
            color: Color4::new(0.0, 0.0, 0.0, 0.0),
            opacity: 0.0,
        }
    }

    /// L2 norm of gradient (for monitoring convergence)
    pub fn magnitude(&self) -> f32 {
        (self.position.length_squared() +
         self.scale.length_squared() +
         self.rotation * self.rotation +
         self.color.r * self.color.r + self.color.g * self.color.g + self.color.b * self.color.b +
         self.opacity * self.opacity).sqrt()
    }
}

/// Compute full gradients for all Gaussians
///
/// This implements the chain rule through the entire rendering pipeline:
/// ∂L/∂param = ∂L/∂pixel × ∂pixel/∂color × ∂color/∂weight × ∂weight/∂Σ⁻¹ × ∂Σ⁻¹/∂param
pub fn compute_full_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
    width: u32,
    height: u32,
) -> Vec<FullGaussianGradient> {
    let mut gradients = vec![FullGaussianGradient::zero(); gaussians.len()];

    // Image gradient (∂L/∂pixel)
    let image_grad = compute_image_gradient(rendered, target);

    // For each Gaussian, accumulate gradients from all affected pixels
    for (g_idx, gaussian) in gaussians.iter().enumerate() {
        // Determine bounding box of affected pixels
        let (min, max) = gaussian.bounding_box(3.5);

        let x_min = (min.x * width as f32).max(0.0) as u32;
        let y_min = (min.y * height as f32).max(0.0) as u32;
        let x_max = (max.x * width as f32).min((width - 1) as f32) as u32;
        let y_max = (max.y * height as f32).min((height - 1) as f32) as u32;

        // Precompute inverse covariance and its derivatives
        let inv_cov = gaussian.inverse_covariance();
        let (dinv_cov_dscale_x, dinv_cov_dscale_y, dinv_cov_drotation) =
            compute_inv_cov_derivatives(&gaussian.shape);

        // For each pixel in bounding box
        for py in y_min..=y_max {
            for px in x_min..=x_max {
                // Normalized pixel position
                let point = Vector2::new(
                    px as f32 / width as f32,
                    py as f32 / height as f32,
                );

                // Compute Gaussian weight and its derivatives
                let (weight, dweight) = evaluate_with_derivatives(
                    gaussian,
                    point,
                    &inv_cov,
                    &dinv_cov_dscale_x,
                    &dinv_cov_dscale_y,
                    &dinv_cov_drotation,
                );

                // Get pixel gradient
                if let Some(pixel_grad) = image_grad.get_pixel(px, py) {
                    // Chain rule: ∂L/∂param = ∂L/∂pixel × ∂pixel/∂param

                    // Color channel contributions
                    let color_contrib = pixel_grad.r + pixel_grad.g + pixel_grad.b;

                    // ∂L/∂position
                    gradients[g_idx].position.x += color_contrib * dweight.position.x * gaussian.opacity;
                    gradients[g_idx].position.y += color_contrib * dweight.position.y * gaussian.opacity;

                    // ∂L/∂scale
                    gradients[g_idx].scale.x += color_contrib * dweight.scale_x * gaussian.opacity;
                    gradients[g_idx].scale.y += color_contrib * dweight.scale_y * gaussian.opacity;

                    // ∂L/∂rotation
                    gradients[g_idx].rotation += color_contrib * dweight.rotation * gaussian.opacity;

                    // ∂L/∂color (simpler - direct contribution)
                    gradients[g_idx].color.r += pixel_grad.r * weight * gaussian.opacity;
                    gradients[g_idx].color.g += pixel_grad.g * weight * gaussian.opacity;
                    gradients[g_idx].color.b += pixel_grad.b * weight * gaussian.opacity;

                    // ∂L/∂opacity
                    gradients[g_idx].opacity += color_contrib * weight;
                }
            }
        }

        // Normalize by number of pixels (prevent gradient explosion)
        let pixel_count = ((x_max - x_min + 1) * (y_max - y_min + 1)) as f32;
        if pixel_count > 0.0 {
            gradients[g_idx].position = gradients[g_idx].position / pixel_count;
            gradients[g_idx].scale = gradients[g_idx].scale / pixel_count;
            gradients[g_idx].rotation /= pixel_count;
            gradients[g_idx].color.r /= pixel_count;
            gradients[g_idx].color.g /= pixel_count;
            gradients[g_idx].color.b /= pixel_count;
            gradients[g_idx].opacity /= pixel_count;
        }
    }

    gradients
}

/// Derivatives of Gaussian weight w.r.t. all parameters
#[derive(Debug, Clone)]
struct WeightDerivatives {
    position: Vector2<f32>,
    scale_x: f32,
    scale_y: f32,
    rotation: f32,
}

/// Evaluate Gaussian and compute all derivatives simultaneously
fn evaluate_with_derivatives(
    gaussian: &Gaussian2D<f32, Euler<f32>>,
    point: Vector2<f32>,
    inv_cov: &[[f32; 2]; 2],
    dinv_cov_dscale_x: &[[f32; 2]; 2],
    dinv_cov_dscale_y: &[[f32; 2]; 2],
    dinv_cov_drotation: &[[f32; 2]; 2],
) -> (f32, WeightDerivatives) {
    // Offset from Gaussian center
    let dx = point.x - gaussian.position.x;
    let dy = point.y - gaussian.position.y;

    // Mahalanobis distance squared: dᵀ Σ⁻¹ d
    let mahal_sq =
        inv_cov[0][0] * dx * dx +
        (inv_cov[0][1] + inv_cov[1][0]) * dx * dy +
        inv_cov[1][1] * dy * dy;

    // Gaussian weight: exp(-0.5 × mahal_sq)
    let weight = (-0.5 * mahal_sq).exp();

    // Now compute derivatives (chain rule)
    // ∂weight/∂param = weight × (-0.5) × ∂mahal_sq/∂param

    // ∂mahal_sq/∂position
    let dmahal_dpos_x = -2.0 * (inv_cov[0][0] * dx + inv_cov[0][1] * dy);
    let dmahal_dpos_y = -2.0 * (inv_cov[1][0] * dx + inv_cov[1][1] * dy);

    // ∂weight/∂position
    let dweight_dpos = Vector2::new(
        weight * (-0.5) * dmahal_dpos_x,
        weight * (-0.5) * dmahal_dpos_y,
    );

    // ∂mahal_sq/∂scale_x (via chain through Σ⁻¹)
    let dmahal_dscale_x =
        dinv_cov_dscale_x[0][0] * dx * dx +
        (dinv_cov_dscale_x[0][1] + dinv_cov_dscale_x[1][0]) * dx * dy +
        dinv_cov_dscale_x[1][1] * dy * dy;

    let dweight_dscale_x = weight * (-0.5) * dmahal_dscale_x;

    // ∂mahal_sq/∂scale_y
    let dmahal_dscale_y =
        dinv_cov_dscale_y[0][0] * dx * dx +
        (dinv_cov_dscale_y[0][1] + dinv_cov_dscale_y[1][0]) * dx * dy +
        dinv_cov_dscale_y[1][1] * dy * dy;

    let dweight_dscale_y = weight * (-0.5) * dmahal_dscale_y;

    // ∂mahal_sq/∂rotation
    let dmahal_drotation =
        dinv_cov_drotation[0][0] * dx * dx +
        (dinv_cov_drotation[0][1] + dinv_cov_drotation[1][0]) * dx * dy +
        dinv_cov_drotation[1][1] * dy * dy;

    let dweight_drotation = weight * (-0.5) * dmahal_drotation;

    (weight, WeightDerivatives {
        position: dweight_dpos,
        scale_x: dweight_dscale_x,
        scale_y: dweight_dscale_y,
        rotation: dweight_drotation,
    })
}

/// Compute derivatives of Σ⁻¹ w.r.t. scale and rotation
///
/// For Euler parameterization: Σ = R(θ) diag(σx², σy²) R(θ)ᵀ
/// We need: ∂Σ⁻¹/∂σx, ∂Σ⁻¹/∂σy, ∂Σ⁻¹/∂θ
fn compute_inv_cov_derivatives(shape: &Euler<f32>) -> ([[f32; 2]; 2], [[f32; 2]; 2], [[f32; 2]; 2]) {
    let sx = shape.scale_x;
    let sy = shape.scale_y;
    let theta = shape.rotation;

    let cos_t = theta.cos();
    let sin_t = theta.sin();

    // For inverse covariance:
    // Σ⁻¹ = R(θ) diag(1/σx², 1/σy²) R(θ)ᵀ

    let inv_sx2 = 1.0 / (sx * sx);
    let inv_sy2 = 1.0 / (sy * sy);

    // ∂Σ⁻¹/∂σx = -2/(σx³) × R(θ) diag(1, 0) R(θ)ᵀ
    let dinv_dsx = -2.0 / (sx * sx * sx);
    let dinv_cov_dscale_x = [
        [dinv_dsx * cos_t * cos_t, dinv_dsx * cos_t * sin_t],
        [dinv_dsx * sin_t * cos_t, dinv_dsx * sin_t * sin_t],
    ];

    // ∂Σ⁻¹/∂σy = -2/(σy³) × R(θ) diag(0, 1) R(θ)ᵀ
    let dinv_dsy = -2.0 / (sy * sy * sy);
    let dinv_cov_dscale_y = [
        [dinv_dsy * sin_t * sin_t, -dinv_dsy * sin_t * cos_t],
        [-dinv_dsy * cos_t * sin_t, dinv_dsy * cos_t * cos_t],
    ];

    // ∂Σ⁻¹/∂θ (more complex - rotation derivative)
    let diff = inv_sx2 - inv_sy2;
    let dinv_cov_drotation = [
        [-diff * 2.0 * cos_t * sin_t, diff * (cos_t * cos_t - sin_t * sin_t)],
        [diff * (cos_t * cos_t - sin_t * sin_t), diff * 2.0 * cos_t * sin_t],
    ];

    (dinv_cov_dscale_x, dinv_cov_dscale_y, dinv_cov_drotation)
}

/// Compute image gradient (∂L/∂pixel)
fn compute_image_gradient(
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
) -> ImageBuffer<f32> {
    let mut grad = ImageBuffer::new(rendered.width, rendered.height);

    for (idx, (r, t)) in rendered.data.iter().zip(target.data.iter()).enumerate() {
        // L2 loss gradient: 2 × (rendered - target)
        grad.data[idx] = Color4::new(
            2.0 * (r.r - t.r),
            2.0 * (r.g - t.g),
            2.0 * (r.b - t.b),
            0.0,
        );
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_computation() {
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::new(0.1, 0.1, 0.0),
            Color4::rgb(1.0, 0.0, 0.0),
            0.8,
        );

        let rendered = ImageBuffer::new(100, 100);
        let target = ImageBuffer::with_background(100, 100, Color4::white());

        let grads = compute_full_gradients(&[gaussian], &rendered, &target, 100, 100);

        assert_eq!(grads.len(), 1);
        // Gradient should be non-zero
        assert!(grads[0].magnitude() > 0.0);
    }

    #[test]
    fn test_inv_cov_derivatives() {
        let shape = Euler::new(0.1, 0.05, 0.3);
        let (dsx, dsy, dtheta) = compute_inv_cov_derivatives(&shape);

        // Derivatives should be finite
        assert!(dsx[0][0].is_finite());
        assert!(dsy[1][1].is_finite());
        assert!(dtheta[0][1].is_finite());
    }
}
