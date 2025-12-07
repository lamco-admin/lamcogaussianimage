//! GRADIENT VERIFICATION: Compare analytical vs finite-difference gradients
//!
//! Tests whether the gradient computations are correct by comparing
//! analytical gradients to numerical (finite-difference) gradients.
//!
//! If they match: gradient computation is correct
//! If they differ: gradient computation has a bug
//!
//! Run: cargo run --release --example verify_gradients_fd -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

fn main() {
    println!("=== GRADIENT VERIFICATION: Analytical vs Finite-Difference ===\n");

    // Small test setup
    let width = 64;
    let height = 64;

    // Create simple target: gradient from black to white
    let mut target = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let v = x as f32 / width as f32;
            target.set_pixel(x, y, Color4::new(v, v, v, 1.0));
        }
    }

    // Create single Gaussian in center
    let gaussians = vec![
        Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::new(0.1, 0.1, 0.0),
            Color4::new(0.5, 0.5, 0.5, 1.0),
            1.0,
        ),
    ];

    println!("Testing with 1 Gaussian, {}x{} image\n", width, height);

    // Compute base loss
    let rendered = RendererV2::render(&gaussians, width, height);
    let base_loss = compute_loss(&rendered, &target);
    println!("Base loss: {:.8}", base_loss);

    // Test gradient for each parameter
    // Note: eps needs to be large enough to cause measurable pixel changes
    // With a 64x64 image, 1 pixel = 1/64 ≈ 0.015 in normalized coords
    let eps = 0.01; // Larger epsilon for measurable changes

    // Test color.r gradient
    println!("\n--- Color R gradient ---");
    test_gradient(&gaussians, &target, width, height, eps, |g, delta| {
        let mut modified = g.to_vec();
        modified[0].color.r += delta;
        modified
    }, "color.r");

    // Test color.g gradient
    println!("\n--- Color G gradient ---");
    test_gradient(&gaussians, &target, width, height, eps, |g, delta| {
        let mut modified = g.to_vec();
        modified[0].color.g += delta;
        modified
    }, "color.g");

    // Test position.x gradient
    println!("\n--- Position X gradient ---");
    test_gradient(&gaussians, &target, width, height, eps, |g, delta| {
        let mut modified = g.to_vec();
        modified[0].position.x += delta;
        modified
    }, "position.x");

    // Test position.y gradient
    println!("\n--- Position Y gradient ---");
    test_gradient(&gaussians, &target, width, height, eps, |g, delta| {
        let mut modified = g.to_vec();
        modified[0].position.y += delta;
        modified
    }, "position.y");

    // Test scale_x gradient
    println!("\n--- Scale X gradient ---");
    test_gradient(&gaussians, &target, width, height, eps, |g, delta| {
        let mut modified = g.to_vec();
        modified[0].shape.scale_x += delta;
        modified
    }, "scale_x");

    // Test scale_y gradient
    println!("\n--- Scale Y gradient ---");
    test_gradient(&gaussians, &target, width, height, eps, |g, delta| {
        let mut modified = g.to_vec();
        modified[0].shape.scale_y += delta;
        modified
    }, "scale_y");

    println!("\n=== INTERPRETATION ===");
    println!("If FD gradient and analytical gradient have:");
    println!("  - Same SIGN: Gradient direction is correct");
    println!("  - Same MAGNITUDE (±10%): Gradient is numerically correct");
    println!("  - Different sign: Gradient has WRONG DIRECTION (bug!)");
}

fn test_gradient<F>(
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
    // Finite difference gradient: (f(x+eps) - f(x-eps)) / (2*eps)
    let plus = modify(gaussians, eps);
    let minus = modify(gaussians, -eps);

    let loss_plus = compute_loss(&RendererV2::render(&plus, width, height), target);
    let loss_minus = compute_loss(&RendererV2::render(&minus, width, height), target);

    let fd_grad = (loss_plus - loss_minus) / (2.0 * eps);

    // Compute analytical gradient using Adam's method
    let analytical_grad = compute_analytical_gradient_adam(gaussians, target, width, height, name);

    // Compare
    let sign_match = (fd_grad > 0.0) == (analytical_grad > 0.0) || fd_grad.abs() < 1e-8;
    let ratio = if analytical_grad.abs() > 1e-10 {
        fd_grad / analytical_grad
    } else {
        f32::NAN
    };

    println!("  Finite-diff gradient: {:+.8}", fd_grad);
    println!("  Analytical gradient:  {:+.8}", analytical_grad);
    println!("  Ratio (FD/Analytical): {:.4}", ratio);
    println!("  Sign match: {}", if sign_match { "YES ✓" } else { "NO ✗ BUG!" });

    if !sign_match && fd_grad.abs() > 1e-6 {
        println!("  ⚠️  GRADIENT SIGN MISMATCH - THIS IS A BUG!");
    }
}

fn compute_analytical_gradient_adam(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    width: u32,
    height: u32,
    param_name: &str,
) -> f32 {
    // Simplified gradient computation matching Adam's approach
    let rendered = RendererV2::render(gaussians, width, height);
    let gaussian = &gaussians[0];

    let mut grad_color_r = 0.0f32;
    let mut grad_color_g = 0.0f32;
    let mut grad_position_x = 0.0f32;
    let mut grad_position_y = 0.0f32;
    let mut grad_scale_x = 0.0f32;
    let mut grad_scale_y = 0.0f32;

    let scale_x = gaussian.shape.scale_x.max(0.001);
    let scale_y = gaussian.shape.scale_y.max(0.001);
    let scale_product = (scale_x * scale_y).max(1e-6);

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
            let dist_sq = dx * dx + dy * dy;

            if dist_sq > 0.1 { continue; }

            let weight = (-0.5 * dist_sq / scale_product).exp();
            if weight < 1e-6 { continue; }

            grad_color_r += error_r * weight;
            grad_color_g += error_g * weight;

            let error_weighted = error_r * gaussian.color.r
                + error_g * gaussian.color.g
                + error_b * gaussian.color.b;

            grad_position_x += error_weighted * weight * dx;
            grad_position_y += error_weighted * weight * dy;

            let scale_x_sq = scale_x.powi(2).max(1e-6);
            let scale_y_sq = scale_y.powi(2).max(1e-6);
            grad_scale_x += error_weighted * weight * dist_sq / scale_x_sq;
            grad_scale_y += error_weighted * weight * dist_sq / scale_y_sq;
        }
    }

    let pixel_count = (width * height) as f32;

    match param_name {
        "color.r" => grad_color_r / pixel_count,
        "color.g" => grad_color_g / pixel_count,
        "position.x" => grad_position_x / pixel_count,
        "position.y" => grad_position_y / pixel_count,
        "scale_x" => grad_scale_x / pixel_count,
        "scale_y" => grad_scale_y / pixel_count,
        _ => 0.0,
    }
}

fn compute_loss(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut loss = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        loss += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    loss / (rendered.width * rendered.height * 3) as f32
}
