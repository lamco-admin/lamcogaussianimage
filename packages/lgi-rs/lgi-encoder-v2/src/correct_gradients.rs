//! Corrected Gradient Computation
//!
//! Properly implements backpropagation through the Gaussian renderer
//! with full chain rule through rotation and scaling.

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

#[derive(Clone, Debug)]
pub struct GaussianGradient {
    pub position: Vector2<f32>,
    pub color: Color4<f32>,
    pub scale_x: f32,
    pub scale_y: f32,
    pub rotation: f32,
    pub opacity: f32,
}

impl GaussianGradient {
    pub fn zero() -> Self {
        Self {
            position: Vector2::zero(),
            color: Color4::new(0.0, 0.0, 0.0, 0.0),
            scale_x: 0.0,
            scale_y: 0.0,
            rotation: 0.0,
            opacity: 0.0,
        }
    }
}

/// Compute gradients with proper chain rule through rotation
///
/// Forward pass:
///   dx = px - μx, dy = py - μy
///   dx_rot = dx*cos(θ) + dy*sin(θ)
///   dy_rot = -dx*sin(θ) + dy*cos(θ)
///   dist_sq = (dx_rot/σx)² + (dy_rot/σy)²
///   g = exp(-0.5 * dist_sq)
///   w = α * g
///   rendered = Σ w * color
///
/// Backward pass (for L2 loss):
///   ∂L/∂rendered = 2 * (rendered - target)
///   ∂rendered/∂w = color
///   ∂w/∂g = α
///   ∂g/∂dist_sq = g * (-0.5)
///   Then chain through dist_sq to each parameter
pub fn compute_gradients_correct(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
) -> Vec<GaussianGradient> {
    let width = target.width;
    let height = target.height;
    let mut gradients = vec![GaussianGradient::zero(); gaussians.len()];

    // For each pixel
    for y in 0..height {
        for x in 0..width {
            let px = x as f32 / width as f32;
            let py = y as f32 / height as f32;

            let rendered_color = rendered.get_pixel(x, y).unwrap();
            let target_color = target.get_pixel(x, y).unwrap();

            // ∂L/∂rendered for each channel (L2 loss)
            let dl_dr = 2.0 * (rendered_color.r - target_color.r);
            let dl_dg = 2.0 * (rendered_color.g - target_color.g);
            let dl_db = 2.0 * (rendered_color.b - target_color.b);

            // Compute total weight sum at this pixel (for normalization)
            let mut weight_sum = 0.0;
            let mut weights = vec![0.0; gaussians.len()];

            for (i, gaussian) in gaussians.iter().enumerate() {
                let dx = px - gaussian.position.x;
                let dy = py - gaussian.position.y;

                let cos_t = gaussian.shape.rotation.cos();
                let sin_t = gaussian.shape.rotation.sin();
                let dx_rot = dx * cos_t + dy * sin_t;
                let dy_rot = -dx * sin_t + dy * cos_t;

                let dist_sq = (dx_rot / gaussian.shape.scale_x).powi(2) +
                              (dy_rot / gaussian.shape.scale_y).powi(2);

                if dist_sq > 12.25 {
                    continue;
                }

                let gaussian_val = (-0.5 * dist_sq).exp();
                let weight = gaussian.opacity * gaussian_val;

                weights[i] = weight;
                weight_sum += weight;
            }

            // Skip if no Gaussian contributes to this pixel
            if weight_sum < 1e-10 {
                continue;
            }

            // Now backpropagate through each Gaussian
            for (i, gaussian) in gaussians.iter().enumerate() {
                let weight = weights[i];

                if weight < 1e-10 {
                    continue;
                }

                let dx = px - gaussian.position.x;
                let dy = py - gaussian.position.y;

                let cos_t = gaussian.shape.rotation.cos();
                let sin_t = gaussian.shape.rotation.sin();
                let dx_rot = dx * cos_t + dy * sin_t;
                let dy_rot = -dx * sin_t + dy * cos_t;

                let dist_sq = (dx_rot / gaussian.shape.scale_x).powi(2) +
                              (dy_rot / gaussian.shape.scale_y).powi(2);

                let gaussian_val = (-0.5 * dist_sq).exp();

                // === COLOR GRADIENTS ===
                // ∂rendered/∂color = weight / weight_sum
                // ∂L/∂color = ∂L/∂rendered * ∂rendered/∂color
                let contribution_factor = weight / weight_sum;

                gradients[i].color.r += dl_dr * contribution_factor;
                gradients[i].color.g += dl_dg * contribution_factor;
                gradients[i].color.b += dl_db * contribution_factor;

                // === Compute ∂L/∂weight ===
                // rendered = Σ(weight_i * color_i) / Σ(weight_i)
                // This is trickier because of the normalization
                // ∂rendered/∂weight_i = (color_i * weight_sum - rendered * 1) / weight_sum²
                //                      = (color_i - rendered) / weight_sum

                let drendered_dweight_r = (gaussian.color.r - rendered_color.r) / weight_sum;
                let drendered_dweight_g = (gaussian.color.g - rendered_color.g) / weight_sum;
                let drendered_dweight_b = (gaussian.color.b - rendered_color.b) / weight_sum;

                let dl_dweight = dl_dr * drendered_dweight_r +
                                dl_dg * drendered_dweight_g +
                                dl_db * drendered_dweight_b;

                // === Compute ∂weight/∂gaussian_val ===
                // weight = opacity * gaussian_val
                // ∂weight/∂gaussian_val = opacity
                let dweight_dg = gaussian.opacity;

                // === Compute ∂gaussian_val/∂dist_sq ===
                // gaussian_val = exp(-0.5 * dist_sq)
                // ∂gaussian_val/∂dist_sq = gaussian_val * (-0.5)
                let dg_ddist = gaussian_val * (-0.5);

                // === Chain to ∂L/∂dist_sq ===
                let dl_ddist = dl_dweight * dweight_dg * dg_ddist;

                // === Now compute gradients for each parameter ===

                // dist_sq = (dx_rot/σx)² + (dy_rot/σy)²
                let term_x = dx_rot / gaussian.shape.scale_x;
                let term_y = dy_rot / gaussian.shape.scale_y;

                // POSITION GRADIENTS
                // ∂dist_sq/∂μx = ∂dist_sq/∂dx_rot * ∂dx_rot/∂dx * ∂dx/∂μx +
                //                ∂dist_sq/∂dy_rot * ∂dy_rot/∂dx * ∂dx/∂μx
                // where dx = px - μx, so ∂dx/∂μx = -1

                let ddist_ddx_rot = 2.0 * term_x / gaussian.shape.scale_x;
                let ddist_ddy_rot = 2.0 * term_y / gaussian.shape.scale_y;

                // ∂dx_rot/∂dx = cos(θ), ∂dy_rot/∂dx = -sin(θ)
                let ddx_rot_ddx = cos_t;
                let ddy_rot_ddx = -sin_t;

                // ∂dx_rot/∂dy = sin(θ), ∂dy_rot/∂dy = cos(θ)
                let ddx_rot_ddy = sin_t;
                let ddy_rot_ddy = cos_t;

                let ddist_ddx = ddist_ddx_rot * ddx_rot_ddx + ddist_ddy_rot * ddy_rot_ddx;
                let ddist_ddy = ddist_ddx_rot * ddx_rot_ddy + ddist_ddy_rot * ddy_rot_ddy;

                // ∂dx/∂μx = -1, ∂dy/∂μy = -1
                gradients[i].position.x += dl_ddist * ddist_ddx * (-1.0);
                gradients[i].position.y += dl_ddist * ddist_ddy * (-1.0);

                // SCALE GRADIENTS
                // dist_sq = (dx_rot)²/σx² + (dy_rot)²/σy²
                // ∂dist_sq/∂σx = ∂((dx_rot)²/σx²)/∂σx = -2*(dx_rot)²/σx³
                // ∂dist_sq/∂σy = ∂((dy_rot)²/σy²)/∂σy = -2*(dy_rot)²/σy³

                let ddist_dscale_x = -2.0 * dx_rot.powi(2) / gaussian.shape.scale_x.powi(3);
                let ddist_dscale_y = -2.0 * dy_rot.powi(2) / gaussian.shape.scale_y.powi(3);

                gradients[i].scale_x += dl_ddist * ddist_dscale_x;
                gradients[i].scale_y += dl_ddist * ddist_dscale_y;

                // ROTATION GRADIENT
                // dx_rot = dx*cos(θ) + dy*sin(θ)
                // dy_rot = -dx*sin(θ) + dy*cos(θ)
                // ∂dx_rot/∂θ = -dx*sin(θ) + dy*cos(θ) = dy_rot
                // ∂dy_rot/∂θ = -dx*cos(θ) - dy*sin(θ) = -(dx*cos(θ) + dy*sin(θ)) = -dx_rot

                let ddx_rot_dtheta = dy_rot;
                let ddy_rot_dtheta = -dx_rot;

                let ddist_dtheta = ddist_ddx_rot * ddx_rot_dtheta + ddist_ddy_rot * ddy_rot_dtheta;

                gradients[i].rotation += dl_ddist * ddist_dtheta;

                // OPACITY GRADIENT
                // weight = opacity * gaussian_val
                // ∂weight/∂opacity = gaussian_val
                let dweight_dopacity = gaussian_val;

                gradients[i].opacity += dl_dweight * dweight_dopacity;
            }
        }
    }

    // Normalize to match L2 loss normalization
    // L2 loss divides by (width * height * 3) for R, G, B channels
    let normalization = (width * height * 3) as f32;
    for grad in &mut gradients {
        grad.position.x /= normalization;
        grad.position.y /= normalization;
        grad.color.r /= normalization;
        grad.color.g /= normalization;
        grad.color.b /= normalization;
        grad.scale_x /= normalization;
        grad.scale_y /= normalization;
        grad.rotation /= normalization;
        grad.opacity /= normalization;
    }

    gradients
}
