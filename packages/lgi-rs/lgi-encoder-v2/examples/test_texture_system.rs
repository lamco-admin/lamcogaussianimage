//! EXP-033: Texture System Validation
//! Test per-primitive textures on synthetic and real content

use lgi_core::{ImageBuffer, textured_gaussian::TexturedGaussian2D};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, renderer_v3_textured::RendererV3};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-033: Texture System Validation         ║");
    println!("║  Expected: +3-5 dB on photos                 ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Test 1: Synthetic checkerboard (high frequency)
    test_checkerboard();

    // Test 2: Real photo with textures
    test_photo_with_textures();
}

fn test_checkerboard() {
    println!("Test 1: Checkerboard (high-frequency synthetic)");
    println!("═══════════════════════════════════════════════");

    let mut target = ImageBuffer::new(256, 256);
    let square_size = 16;

    for y in 0..256 {
        for x in 0..256 {
            let is_white = ((x / square_size) + (y / square_size)) % 2 == 0;
            let val = if is_white { 1.0 } else { 0.0 };
            target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    // Baseline: No textures
    println!("\n  Baseline (no textures, N=400):");
    let encoder = EncoderV2::new(target.clone()).unwrap();
    let gaussians_base = encoder.initialize_gaussians(20);

    let rendered_base = RendererV2::render(&gaussians_base, 256, 256);
    let psnr_base = compute_psnr(&target, &rendered_base);
    println!("    PSNR: {:.2} dB", psnr_base);

    // With textures
    println!("\n  With textures (N=400, 8×8 textures):");
    let mut gaussians_textured: Vec<TexturedGaussian2D> = gaussians_base
        .iter()
        .map(|g| TexturedGaussian2D::from_gaussian(g.clone()))
        .collect();

    // Extract textures for all primitives
    for gaussian in &mut gaussians_textured {
        gaussian.extract_texture_from_image(&target, 8);
    }

    let rendered_textured = RendererV3::render(&gaussians_textured, 256, 256);
    let psnr_textured = compute_psnr(&target, &rendered_textured);
    println!("    PSNR: {:.2} dB", psnr_textured);
    println!("    Improvement: {:+.2} dB", psnr_textured - psnr_base);

    if psnr_textured > psnr_base + 2.0 {
        println!("    ✅ Textures provide significant improvement!");
    } else if psnr_textured > psnr_base {
        println!("    ✓ Textures help marginally");
    } else {
        println!("    ❌ Textures don't help (may need optimization)");
    }
}

fn test_photo_with_textures() {
    println!("\n\nTest 2: Real Photo with Textures");
    println!("═══════════════════════════════════════════════");

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

    println!("\n  Baseline (no textures, N=400):");
    let encoder = EncoderV2::new(target.clone()).unwrap();
    let gaussians_base = encoder.initialize_gaussians_guided(20);

    let rendered_base = RendererV2::render(&gaussians_base, target.width, target.height);
    let psnr_base = compute_psnr(&target, &rendered_base);
    println!("    PSNR: {:.2} dB", psnr_base);

    // With textures - adaptive (only add where variance high)
    println!("\n  With adaptive textures (N=400, variance threshold=0.01):");
    let mut gaussians_textured: Vec<TexturedGaussian2D> = gaussians_base
        .iter()
        .map(|g| TexturedGaussian2D::from_gaussian(g.clone()))
        .collect();

    let mut texture_count = 0;
    for gaussian in &mut gaussians_textured {
        if gaussian.should_add_texture(&target, 0.01) {
            gaussian.extract_texture_from_image(&target, 8);
            texture_count += 1;
        }
    }

    println!("    Added textures to {}/{} Gaussians", texture_count, gaussians_textured.len());

    let rendered_textured = RendererV3::render(&gaussians_textured, target.width, target.height);
    let psnr_textured = compute_psnr(&target, &rendered_textured);
    println!("    PSNR: {:.2} dB", psnr_textured);
    println!("    Improvement: {:+.2} dB", psnr_textured - psnr_base);

    if psnr_textured > psnr_base + 3.0 {
        println!("    ✅ Textures provide major improvement!");
    } else if psnr_textured > psnr_base + 1.0 {
        println!("    ✓ Textures help significantly");
    } else if psnr_textured > psnr_base {
        println!("    ⚠️  Textures help marginally");
    } else {
        println!("    ❌ Textures don't help");
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
