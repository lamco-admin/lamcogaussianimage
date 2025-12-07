//! EXP-035: Commercial Encoder Test
//!
//! Complete system test with ALL features integrated
//! Goal: Demonstrate production-quality encoding

use lgi_core::ImageBuffer;
use lgi_encoder_v2::commercial_encoder::{CommercialEncoder, CommercialConfig};
use lgi_encoder_v2::renderer_v3_textured::RendererV3;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  EXP-035: Commercial Encoder - Complete System Test     ║");
    println!("║  ALL Features: Textures + Residuals + Triggers + More   ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Test on multiple content types
    test_synthetic();
    test_photo();
}

fn test_synthetic() {
    println!("═══════════════════════════════════════════════════════════");
    println!("Test 1: Synthetic Content (Validation)");
    println!("═══════════════════════════════════════════════════════════");

    // Smooth curve (should reach ~40 dB)
    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let edge_pos = 128.0 + 30.0 * (y as f32 / 256.0 - 0.5).sin();
            let distance = x as f32 - edge_pos;
            let val = 1.0 / (1.0 + (-distance / 10.0).exp());
            target.set_pixel(x, y, lgi_math::color::Color4::new(val, val, val, 1.0));
        }
    }

    let config = CommercialConfig {
        target_gaussian_count: 400,
        quality_target: 35.0,
        max_iterations: 200,
        use_textures: true,
        texture_size: 8,
        texture_threshold: 0.01,
        use_residuals: true,
        residual_threshold: 0.05,
        use_guided_filter: false,  // Not needed for synthetic
        use_triggers: true,
    };

    let encoder = CommercialEncoder::new(target.clone(), config).unwrap();
    let (gaussians, residual) = encoder.encode();

    // Render final
    let mut final_rendered = RendererV3::render(&gaussians, 256, 256);

    if let Some(ref res) = residual {
        res.apply_to_image(&mut final_rendered);
    }

    let psnr = compute_psnr(&target, &final_rendered);
    println!("\n  Final PSNR: {:.2} dB", psnr);

    if psnr >= 35.0 {
        println!("  ✅ Excellent - Reaches synthetic quality target");
    } else if psnr >= 30.0 {
        println!("  ✓ Good - Close to target");
    } else {
        println!("  ⚠️  Below synthetic target (may need tuning)");
    }
}

fn test_photo() {
    println!("\n═══════════════════════════════════════════════════════════");
    println!("Test 2: Real Photo (Production Target)");
    println!("═══════════════════════════════════════════════════════════");

    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/Lamco Head.jpg";

    let target = match ImageBuffer::load(path) {
        Ok(mut img) => {
            if img.width > 256 || img.height > 256 {
                resize_simple(&img, 256, 256)
            } else {
                img
            }
        }
        Err(e) => {
            println!("  Failed to load: {}", e);
            return;
        }
    };

    println!("\nConfig: Full commercial settings");
    let config = CommercialConfig {
        target_gaussian_count: 1024,
        quality_target: 28.0,
        max_iterations: 500,
        use_textures: true,
        texture_size: 16,
        texture_threshold: 0.005,  // Lower for photos
        use_residuals: true,
        residual_threshold: 0.03,
        use_guided_filter: true,   // Critical for photos
        use_triggers: true,
    };

    let encoder = CommercialEncoder::new(target.clone(), config).unwrap();
    let (gaussians, residual) = encoder.encode();

    // Render final
    let mut final_rendered = RendererV3::render(&gaussians, target.width, target.height);

    if let Some(ref res) = residual {
        res.apply_to_image(&mut final_rendered);
    }

    let psnr = compute_psnr(&target, &final_rendered);
    println!("\n  Final PSNR: {:.2} dB", psnr);

    // Save for visual inspection
    if let Err(e) = final_rendered.save("/tmp/commercial_encoder_output.png") {
        println!("  Warning: Could not save: {}", e);
    } else {
        println!("  Saved: /tmp/commercial_encoder_output.png");
    }

    if psnr >= 28.0 {
        println!("  ✅ SUCCESS: Reached production quality target!");
    } else if psnr >= 25.0 {
        println!("  ✓ GOOD: Close to target");
    } else if psnr >= 20.0 {
        println!("  ⚠️  MARGINAL: Need more tuning or features");
    } else {
        println!("  ❌ POOR: Significant improvements needed");
    }
}

fn resize_simple(img: &ImageBuffer<f32>, w: u32, h: u32) -> ImageBuffer<f32> {
    let mut out = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sx = (x as f32 / w as f32 * img.width as f32) as u32;
            let sy = (y as f32 / h as f32 * img.height as f32) as u32;
            if let Some(c) = img.get_pixel(sx.min(img.width-1), sy.min(img.height-1)) {
                out.set_pixel(x, y, c);
            }
        }
    }
    out
}

fn compute_psnr(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (original.width * original.height * 3) as f32;

    for (p1, p2) in original.data.iter().zip(rendered.data.iter()) {
        mse += (p1.r - p2.r).powi(2);
        mse += (p1.g - p2.g).powi(2);
        mse += (p1.b - p2.b).powi(2);
    }

    mse /= count;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
