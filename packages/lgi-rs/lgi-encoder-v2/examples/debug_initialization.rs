//! Debug: Check what Gaussian initialization produces

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::EncoderV2;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Create simple solid red image
    let mut image = ImageBuffer::new(64, 64);
    for pixel in &mut image.data {
        *pixel = Color4::new(1.0, 0.0, 0.0, 1.0);
    }

    println!("Creating encoder...");
    let encoder = EncoderV2::new(image).expect("Failed");

    println!("\nInitializing Gaussians (4×4 grid)...");
    let gaussians = encoder.initialize_gaussians(4);

    println!("\nGaussian Details:");
    println!("═══════════════════════════════════════");

    for (i, g) in gaussians.iter().take(5).enumerate() {
        println!("\nGaussian {}:", i);
        println!("  Position: ({:.3}, {:.3})", g.position.x, g.position.y);
        println!("  Scale: ({:.6}, {:.6})", g.shape.scale_x, g.shape.scale_y);
        println!("  Rotation: {:.3} rad ({:.1}°)", g.shape.rotation, g.shape.rotation.to_degrees());
        println!("  Color: ({:.3}, {:.3}, {:.3})", g.color.r, g.color.g, g.color.b);
        println!("  Opacity: {:.3}", g.opacity);

        let anisotropy = g.shape.scale_x.max(g.shape.scale_y) / g.shape.scale_x.min(g.shape.scale_y);
        println!("  Anisotropy: {:.2}×", anisotropy);

        // Check for issues
        if g.shape.scale_x < 0.001 || g.shape.scale_y < 0.001 {
            println!("  ⚠️  WARNING: Extremely small scale!");
        }
        if g.shape.scale_x.is_nan() || g.shape.scale_y.is_nan() {
            println!("  ❌ ERROR: NaN scale!");
        }
        if g.opacity < 0.01 {
            println!("  ⚠️  WARNING: Very low opacity!");
        }
    }

    println!("\n═══════════════════════════════════════");
    println!("Total Gaussians: {}", gaussians.len());

    let avg_scale_x: f32 = gaussians.iter().map(|g| g.shape.scale_x).sum::<f32>() / gaussians.len() as f32;
    let avg_scale_y: f32 = gaussians.iter().map(|g| g.shape.scale_y).sum::<f32>() / gaussians.len() as f32;

    println!("Average scale_x: {:.6}", avg_scale_x);
    println!("Average scale_y: {:.6}", avg_scale_y);
}
