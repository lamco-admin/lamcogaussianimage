//! Test all LGIQ quantization profiles
//! Measures quality loss vs file size tradeoff for each profile

use lgi_core::quantization::QuantizedGaussian;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LGIQ Quantization Profiles Test ===\n");

    // Create diverse test Gaussians covering the parameter space
    let mut gaussians = Vec::new();

    // Small scales (common for detail)
    gaussians.push(Gaussian2D::new(
        Vector2::new(0.1, 0.1),
        Euler::new(0.002, 0.003, 0.5), // Very small scales
        Color4::new(0.8, 0.2, 0.1, 1.0),
        0.95,
    ));

    // Medium scales
    gaussians.push(Gaussian2D::new(
        Vector2::new(0.5, 0.5),
        Euler::new(0.05, 0.04, -0.7),
        Color4::new(0.3, 0.6, 0.9, 1.0),
        0.85,
    ));

    // Large scales (background)
    gaussians.push(Gaussian2D::new(
        Vector2::new(0.9, 0.9),
        Euler::new(0.15, 0.12, 1.2),
        Color4::new(0.1, 0.9, 0.3, 1.0),
        0.75,
    ));

    // Anisotropic with high rotation precision need
    gaussians.push(Gaussian2D::new(
        Vector2::new(0.25, 0.75),
        Euler::new(0.08, 0.02, 2.8), // High anisotropy
        Color4::new(0.95, 0.95, 0.05, 1.0), // Bright yellow
        0.9,
    ));

    // Edge case: very low opacity
    gaussians.push(Gaussian2D::new(
        Vector2::new(0.75, 0.25),
        Euler::isotropic(0.06),
        Color4::new(0.5, 0.5, 0.5, 1.0),
        0.05, // Very transparent
    ));

    println!("Created {} diverse test Gaussians\n", gaussians.len());

    // Test each quantization profile
    println!("Testing quantization profiles:\n");

    // LGIQ-B: Baseline (8-bit color, 12-bit scales)
    println!("1. LGIQ-B (Baseline): 8-bit color, 12-bit scales/rotation");
    test_profile(&gaussians, "baseline", |g| {
        QuantizedGaussian::quantize_baseline(
            (g.position.x, g.position.y),
            (g.shape.scale_x, g.shape.scale_y),
            g.shape.rotation,
            (g.color.r, g.color.g, g.color.b),
            g.opacity,
        )
    }, 11)?;

    // LGIQ-S: Standard (10-bit color, 12-bit scales)
    println!("\n2. LGIQ-S (Standard): 10-bit color, 12-bit scales/rotation");
    test_profile(&gaussians, "standard", |g| {
        QuantizedGaussian::quantize_standard(
            (g.position.x, g.position.y),
            (g.shape.scale_x, g.shape.scale_y),
            g.shape.rotation,
            (g.color.r, g.color.g, g.color.b),
            g.opacity,
        )
    }, 13)?;

    // LGIQ-H: High Fidelity (10-bit color, 14-bit scales)
    println!("\n3. LGIQ-H (High Fidelity): 10-bit color, 14-bit scales/rotation");
    test_profile(&gaussians, "high_fidelity", |g| {
        QuantizedGaussian::quantize_high_fidelity(
            (g.position.x, g.position.y),
            (g.shape.scale_x, g.shape.scale_y),
            g.shape.rotation,
            (g.color.r, g.color.g, g.color.b),
            g.opacity,
        )
    }, 18)?;

    println!("\n=== Summary ===");
    println!("LGIQ-B: 11 bytes/G - Good for low-detail or web delivery");
    println!("LGIQ-S: 13 bytes/G - Better color fidelity (+18%)");
    println!("LGIQ-H: 18 bytes/G - Higher geometric precision (+64%)");
    println!("LGIQ-X: 36 bytes/G - Lossless (already implemented)");

    Ok(())
}

fn test_profile<F>(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    profile_name: &str,
    quantize_fn: F,
    bytes_per_gaussian: usize,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(&Gaussian2D<f32, Euler<f32>>) -> QuantizedGaussian,
{
    let mut max_errors = ParamErrors::default();
    let mut avg_errors = ParamErrors::default();

    for (i, gaussian) in gaussians.iter().enumerate() {
        let quantized = quantize_fn(gaussian);
        let (px, py, sx, sy, rot, cr, cg, cb, op) = quantized.dequantize();

        // Compute errors
        let pos_err = ((px - gaussian.position.x).powi(2) + (py - gaussian.position.y).powi(2)).sqrt();
        let scale_x_err = (sx - gaussian.shape.scale_x).abs();
        let scale_y_err = (sy - gaussian.shape.scale_y).abs();
        let rot_err = (rot - gaussian.shape.rotation).abs();
        let color_r_err = (cr - gaussian.color.r).abs();
        let color_g_err = (cg - gaussian.color.g).abs();
        let color_b_err = (cb - gaussian.color.b).abs();
        let opacity_err = (op - gaussian.opacity).abs();

        // Track max errors
        max_errors.position = max_errors.position.max(pos_err);
        max_errors.scale_x = max_errors.scale_x.max(scale_x_err);
        max_errors.scale_y = max_errors.scale_y.max(scale_y_err);
        max_errors.rotation = max_errors.rotation.max(rot_err);
        max_errors.color_r = max_errors.color_r.max(color_r_err);
        max_errors.color_g = max_errors.color_g.max(color_g_err);
        max_errors.color_b = max_errors.color_b.max(color_b_err);
        max_errors.opacity = max_errors.opacity.max(opacity_err);

        // Accumulate for average
        avg_errors.position += pos_err;
        avg_errors.scale_x += scale_x_err;
        avg_errors.scale_y += scale_y_err;
        avg_errors.rotation += rot_err;
        avg_errors.color_r += color_r_err;
        avg_errors.color_g += color_g_err;
        avg_errors.color_b += color_b_err;
        avg_errors.opacity += opacity_err;

        if i == 0 {
            // Print detailed analysis of first Gaussian
            println!("   Sample Gaussian 0 (small scales):");
            println!("     Position:  ({:.6}, {:.6}) → ({:.6}, {:.6}) [err: {:.8}]",
                     gaussian.position.x, gaussian.position.y, px, py, pos_err);
            println!("     Scale:     ({:.6}, {:.6}) → ({:.6}, {:.6}) [err: {:.8}, {:.8}]",
                     gaussian.shape.scale_x, gaussian.shape.scale_y, sx, sy, scale_x_err, scale_y_err);
            println!("     Rotation:  {:.6} → {:.6} [err: {:.8}]",
                     gaussian.shape.rotation, rot, rot_err);
            println!("     Color:     ({:.3}, {:.3}, {:.3}) → ({:.3}, {:.3}, {:.3})",
                     gaussian.color.r, gaussian.color.g, gaussian.color.b, cr, cg, cb);
            println!("     Opacity:   {:.3} → {:.3} [err: {:.6}]",
                     gaussian.opacity, op, opacity_err);
        }
    }

    // Compute averages
    let n = gaussians.len() as f32;
    avg_errors.position /= n;
    avg_errors.scale_x /= n;
    avg_errors.scale_y /= n;
    avg_errors.rotation /= n;
    avg_errors.color_r /= n;
    avg_errors.color_g /= n;
    avg_errors.color_b /= n;
    avg_errors.opacity /= n;

    println!("\n   Max errors:");
    println!("     Position:  {:.8}", max_errors.position);
    println!("     Scales:    {:.8}, {:.8}", max_errors.scale_x, max_errors.scale_y);
    println!("     Rotation:  {:.8} rad ({:.2}°)", max_errors.rotation, max_errors.rotation.to_degrees());
    println!("     Color:     {:.6}, {:.6}, {:.6}", max_errors.color_r, max_errors.color_g, max_errors.color_b);
    println!("     Opacity:   {:.6}", max_errors.opacity);

    println!("\n   Avg errors:");
    println!("     Position:  {:.8}", avg_errors.position);
    println!("     Scales:    {:.8}, {:.8}", avg_errors.scale_x, avg_errors.scale_y);
    println!("     Rotation:  {:.8} rad ({:.2}°)", avg_errors.rotation, avg_errors.rotation.to_degrees());
    println!("     Color:     {:.6}, {:.6}, {:.6}", avg_errors.color_r, avg_errors.color_g, avg_errors.color_b);
    println!("     Opacity:   {:.6}", avg_errors.opacity);

    println!("\n   File size: {} bytes ({} bytes/G)", gaussians.len() * bytes_per_gaussian, bytes_per_gaussian);

    Ok(())
}

#[derive(Default)]
struct ParamErrors {
    position: f32,
    scale_x: f32,
    scale_y: f32,
    rotation: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    opacity: f32,
}
