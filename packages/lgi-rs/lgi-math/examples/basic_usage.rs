//! Basic usage example of lgi-math library

use lgi_math::prelude::*;

fn main() {
    println!("LGI Math Library - Basic Usage Example\n");

    // Create a Gaussian using Euler parameterization
    let gaussian = Gaussian2D::new(
        Vector2::new(0.5, 0.5),  // Center position
        Euler::new(0.1, 0.05, 0.3),  // σx=0.1, σy=0.05, θ=0.3rad
        Color4::rgb(1.0, 0.0, 0.0),  // Red color
        0.8,  // 80% opacity
    );

    println!("Created Gaussian:");
    println!("  Position: ({:.2}, {:.2})", gaussian.position.x, gaussian.position.y);
    println!("  Opacity: {:.2}", gaussian.opacity);
    println!("  Color: ({:.2}, {:.2}, {:.2})",
        gaussian.color.r, gaussian.color.g, gaussian.color.b);

    // Compute bounding box
    let (min, max) = gaussian.bounding_box(3.0);
    println!("\nBounding Box (3σ):");
    println!("  Min: ({:.3}, {:.3})", min.x, min.y);
    println!("  Max: ({:.3}, {:.3})", max.x, max.y);

    // Evaluate Gaussian at various points
    let evaluator = GaussianEvaluator::default();

    println!("\nGaussian Evaluation:");
    let test_points = [
        Vector2::new(0.5, 0.5),   // Center
        Vector2::new(0.6, 0.5),   // 1 sigma away
        Vector2::new(0.7, 0.5),   // 2 sigma away
        Vector2::new(0.9, 0.9),   // Far away
    ];

    for point in test_points.iter() {
        let weight = evaluator.evaluate(&gaussian, *point);
        println!("  At ({:.2}, {:.2}): weight = {:.4}", point.x, point.y, weight);
    }

    // Convert to different parameterizations
    let cholesky_gaussian: Gaussian2D<f32, Cholesky<f32>> = gaussian.convert();
    let log_rad_gaussian: Gaussian2D<f32, LogRadius<f32>> = gaussian.convert();

    println!("\nParameterization Conversions:");
    println!("  Cholesky: L11={:.4}, L21={:.4}, L22={:.4}",
        cholesky_gaussian.shape.l11, cholesky_gaussian.shape.l21, cholesky_gaussian.shape.l22);
    println!("  LogRadius: log_r={:.4}, e={:.4}, θ={:.4}",
        log_rad_gaussian.shape.log_radius, log_rad_gaussian.shape.eccentricity,
        log_rad_gaussian.shape.rotation);

    // Compositing example
    println!("\nCompositing Example:");
    let compositor = Compositor::default();
    let mut accum_color = Color4::black();
    let mut accum_alpha = 0.0;

    // Composite red Gaussian
    let weight = evaluator.evaluate(&gaussian, Vector2::new(0.5, 0.5));
    compositor.composite_over(
        &mut accum_color,
        &mut accum_alpha,
        gaussian.color,
        gaussian.opacity,
        weight,
    );

    println!("  After first Gaussian: color=({:.2}, {:.2}, {:.2}), alpha={:.2}",
        accum_color.r, accum_color.g, accum_color.b, accum_alpha);

    // Add a blue Gaussian on top
    let blue_gaussian = Gaussian2D::new(
        Vector2::new(0.5, 0.5),
        Euler::isotropic(0.08),
        Color4::rgb(0.0, 0.0, 1.0),
        0.5,
    );

    let weight2 = evaluator.evaluate(&blue_gaussian, Vector2::new(0.5, 0.5));
    compositor.composite_over(
        &mut accum_color,
        &mut accum_alpha,
        blue_gaussian.color,
        blue_gaussian.opacity,
        weight2,
    );

    println!("  After second Gaussian: color=({:.2}, {:.2}, {:.2}), alpha={:.2}",
        accum_color.r, accum_color.g, accum_color.b, accum_alpha);

    // Blend with background
    let final_color = compositor.blend_background(
        accum_color,
        accum_alpha,
        Color4::rgb(1.0, 1.0, 1.0),  // White background
    );

    println!("  Final color (with white bg): ({:.2}, {:.2}, {:.2})",
        final_color.r, final_color.g, final_color.b);

    println!("\n✓ Example completed successfully!");
}
