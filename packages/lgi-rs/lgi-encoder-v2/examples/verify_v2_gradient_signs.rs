//! GRADIENT SIGN VERIFICATION: Does V2 gradient point in the right direction?
//!
//! If FD gradient and V2 gradient have OPPOSITE signs, V2 will diverge.
//! Run: cargo run --release --example verify_v2_gradient_signs -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

fn main() {
    println!("=== V2 GRADIENT SIGN VERIFICATION ===\n");

    // Larger image for more accurate gradients
    let width = 128;
    let height = 128;

    // Create a more interesting target: white square in center
    let mut target = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let in_center = x >= 40 && x <= 88 && y >= 40 && y <= 88;
            let v = if in_center { 0.8 } else { 0.2 };
            target.set_pixel(x, y, Color4::new(v, v, v, 1.0));
        }
    }

    // Single Gaussian slightly off-center
    let gaussians = vec![
        Gaussian2D::new(
            Vector2::new(0.4, 0.5),  // Slightly left of center
            Euler::new(0.15, 0.15, 0.3),  // Some rotation
            Color4::new(0.6, 0.6, 0.6, 1.0),
            1.0,
        ),
    ];

    println!("Testing with 1 Gaussian (off-center, rotated), {}x{} image\n", width, height);

    let eps = 0.005;  // Reasonable epsilon

    // Test each parameter
    test_parameter(&gaussians, &target, width, height, eps,
                   |g, d| modify_param(g, "position.x", d), "position.x");
    test_parameter(&gaussians, &target, width, height, eps,
                   |g, d| modify_param(g, "position.y", d), "position.y");
    test_parameter(&gaussians, &target, width, height, eps,
                   |g, d| modify_param(g, "scale_x", d), "scale_x");
    test_parameter(&gaussians, &target, width, height, eps,
                   |g, d| modify_param(g, "scale_y", d), "scale_y");
    test_parameter(&gaussians, &target, width, height, eps,
                   |g, d| modify_param(g, "rotation", d), "rotation");
    test_parameter(&gaussians, &target, width, height, eps,
                   |g, d| modify_param(g, "color.r", d), "color.r");

    println!("\n=== INTERPRETATION ===");
    println!("If FD and V2 gradients have:");
    println!("  - SAME sign: V2 should converge (direction correct)");
    println!("  - OPPOSITE sign: V2 will DIVERGE (gradient points wrong way!)");
}

fn modify_param(gaussians: &[Gaussian2D<f32, Euler<f32>>], param: &str, delta: f32)
    -> Vec<Gaussian2D<f32, Euler<f32>>>
{
    let mut modified = gaussians.to_vec();
    match param {
        "position.x" => modified[0].position.x += delta,
        "position.y" => modified[0].position.y += delta,
        "scale_x" => modified[0].shape.scale_x += delta,
        "scale_y" => modified[0].shape.scale_y += delta,
        "rotation" => modified[0].shape.rotation += delta,
        "color.r" => modified[0].color.r += delta,
        "color.g" => modified[0].color.g += delta,
        "color.b" => modified[0].color.b += delta,
        _ => {}
    }
    modified
}

fn test_parameter<F>(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    width: u32,
    height: u32,
    eps: f32,
    modify: F,
    name: &str,
) where
    F: Fn(&[Gaussian2D<f32, Euler<f32>>], f32) -> Vec<Gaussian2D<f32, Euler<f32>>>,
{
    // Finite difference gradient
    let plus = modify(gaussians, eps);
    let minus = modify(gaussians, -eps);

    let loss_plus = compute_loss(&RendererV2::render(&plus, width, height), target);
    let loss_minus = compute_loss(&RendererV2::render(&minus, width, height), target);

    let fd_grad = (loss_plus - loss_minus) / (2.0 * eps);

    // V2-style analytical gradient
    let v2_grad = compute_v2_gradient(gaussians, target, width, height, name);

    // Check signs
    let fd_sign = if fd_grad > 0.0 { "+" } else if fd_grad < 0.0 { "-" } else { "0" };
    let v2_sign = if v2_grad > 0.0 { "+" } else if v2_grad < 0.0 { "-" } else { "0" };
    let sign_match = (fd_grad > 0.0) == (v2_grad > 0.0) || fd_grad.abs() < 1e-6 || v2_grad.abs() < 1e-6;

    println!("--- {} ---", name);
    println!("  FD:  {:+.6} ({})", fd_grad, fd_sign);
    println!("  V2:  {:+.6} ({})", v2_grad, v2_sign);
    println!("  Sign match: {}", if sign_match { "YES ✓" } else { "NO ✗ DIVERGENCE!" });
    if !sign_match {
        println!("  ⚠️  OPPOSITE SIGNS - V2 will diverge on this parameter!");
    }
    println!();
}

fn compute_v2_gradient(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    width: u32,
    height: u32,
    param_name: &str,
) -> f32 {
    // FIXED V2-style gradient (matches optimizer_v2.rs with position negation)
    let rendered = RendererV2::render(gaussians, width, height);
    let gaussian = &gaussians[0];

    let mut grad_position_x = 0.0f32;
    let mut grad_position_y = 0.0f32;
    let mut grad_scale_x = 0.0f32;
    let mut grad_scale_y = 0.0f32;
    let mut grad_rotation = 0.0f32;
    let mut grad_color_r = 0.0f32;

    let cos_t = gaussian.shape.rotation.cos();
    let sin_t = gaussian.shape.rotation.sin();
    let sx = gaussian.shape.scale_x;
    let sy = gaussian.shape.scale_y;

    for y in 0..height {
        for x in 0..width {
            let px = x as f32 / width as f32;
            let py = y as f32 / height as f32;

            let rendered_color = rendered.get_pixel(x, y).unwrap();
            let target_color = target.get_pixel(x, y).unwrap();

            let error_r = 2.0 * (rendered_color.r - target_color.r);
            let error_g = 2.0 * (rendered_color.g - target_color.g);
            let error_b = 2.0 * (rendered_color.b - target_color.b);

            let dx = px - gaussian.position.x;
            let dy = py - gaussian.position.y;

            // Rotate to Gaussian's frame (V2 style)
            let dx_rot = dx * cos_t + dy * sin_t;
            let dy_rot = -dx * sin_t + dy * cos_t;

            // Mahalanobis distance
            let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);
            if dist_sq > 12.25 { continue; }

            let gaussian_val = (-0.5 * dist_sq).exp();
            let weight = gaussian.opacity * gaussian_val;

            if weight < 1e-10 { continue; }

            // Color gradient: simple weighted contribution
            grad_color_r += error_r * weight;

            // error_weighted = error · color (like Adam)
            let error_weighted = error_r * gaussian.color.r +
                                error_g * gaussian.color.g +
                                error_b * gaussian.color.b;

            // Position gradient (only x negated based on FD comparison)
            let grad_weight_x = weight * (dx_rot * cos_t / (sx * sx) + dy_rot * (-sin_t) / (sy * sy));
            let grad_weight_y = weight * (dx_rot * sin_t / (sx * sx) + dy_rot * cos_t / (sy * sy));
            grad_position_x -= error_weighted * grad_weight_x;  // MINUS for x
            grad_position_y += error_weighted * grad_weight_y;  // PLUS for y

            // Scale gradient
            let grad_weight_sx = weight * (dx_rot / sx).powi(2) * (1.0 / sx);
            let grad_weight_sy = weight * (dy_rot / sy).powi(2) * (1.0 / sy);
            grad_scale_x += error_weighted * grad_weight_sx;
            grad_scale_y += error_weighted * grad_weight_sy;

            // Rotation gradient
            let d_dx_rot_dtheta = -dx * sin_t + dy * cos_t;
            let d_dy_rot_dtheta = -dx * cos_t - dy * sin_t;
            let d_dist_sq_dtheta = 2.0 * (
                (dx_rot / (sx * sx)) * d_dx_rot_dtheta +
                (dy_rot / (sy * sy)) * d_dy_rot_dtheta
            );
            let grad_weight_theta = -0.5 * weight * d_dist_sq_dtheta;
            grad_rotation += error_weighted * grad_weight_theta;
        }
    }

    let pixel_count = (width * height) as f32;

    match param_name {
        "position.x" => grad_position_x / pixel_count,
        "position.y" => grad_position_y / pixel_count,
        "scale_x" => grad_scale_x / pixel_count,
        "scale_y" => grad_scale_y / pixel_count,
        "rotation" => grad_rotation / pixel_count,
        "color.r" => grad_color_r / pixel_count,
        _ => 0.0,
    }
}

fn compute_loss(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut loss = 0.0;
    // Use V2's normalization (÷ pixels, not ÷ channels)
    let count = (rendered.width * rendered.height) as f32;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        loss += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    loss / count
}
