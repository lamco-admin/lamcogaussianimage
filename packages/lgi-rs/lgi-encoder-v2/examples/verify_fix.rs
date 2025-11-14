//! Verify Fix - Quick Test
//!
//! Tests ONLY error-driven with structure-tensor initialization
//! Should see improvement, not regression

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2};
use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use std::path::PathBuf;

fn main() {
    let image = load_test_image();
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    let baseline = encoder.initialize_gaussians(10);
    let baseline_psnr = compute_psnr(&image, &baseline);

    println!("BEFORE FIX: Error-driven gave 9.38 dB (-12 dB regression)");
    println!("AFTER FIX:  Testing now...\n");

    println!("Baseline: {:.2} dB (N=100)", baseline_psnr);

    println!("Running error-driven (N=31→124) with FIXED initialization...");
    let fixed = encoder.encode_error_driven_adam(31, 124);
    let fixed_psnr = compute_psnr(&image, &fixed);

    println!("Error-driven: {:.2} dB (N={}) ({:+.2} dB)\n", fixed_psnr, fixed.len(), fixed_psnr - baseline_psnr);

    if fixed_psnr > baseline_psnr {
        println!("✅ FIX WORKS! Quality improved by {:.2} dB", fixed_psnr - baseline_psnr);
    } else if fixed_psnr > baseline_psnr - 2.0 {
        println!("⚠️  PARTIAL FIX: Only {:.2} dB regression (was -12 dB)", fixed_psnr - baseline_psnr);
    } else {
        println!("❌ STILL BROKEN: {:.2} dB regression", fixed_psnr - baseline_psnr);
    }
}

fn compute_psnr(target: &ImageBuffer<f32>, gaussians: &[lgi_math::gaussian::Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>]) -> f32 {
    let rendered = RendererV2::render(gaussians, target.width, target.height);
    let mut mse = 0.0f32;
    let pixel_count = target.width * target.height;

    for y in 0..target.height {
        for x in 0..target.width {
            let t = target.get_pixel(x, y).unwrap();
            let r = rendered.get_pixel(x, y).unwrap();
            mse += (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
        }
    }

    mse /= (pixel_count * 3) as f32;
    if mse < 1e-10 { 100.0 } else { -10.0 * mse.log10() }
}

fn load_test_image() -> ImageBuffer<f32> {
    let path = PathBuf::from("../../kodak-dataset/kodim02.png");
    let img = image::open(&path).expect("Failed to load");
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut buffer = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            buffer.set_pixel(x, y, Color4::new(
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
                1.0
            ));
        }
    }
    buffer
}
