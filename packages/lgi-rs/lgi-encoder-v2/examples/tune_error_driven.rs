//! EXP-019: Tune Error-Driven Parameters
//! Optimize: split_percentile, initial_N, max_N

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::error_driven::ErrorDrivenEncoder;

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-019: Error-Driven Parameter Tuning     ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Test on gradient (where error-driven excels)
    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let t = x as f32 / 255.0;
            target.set_pixel(x, y, Color4::new(t, 0.0, 1.0 - t, 1.0));
        }
    }

    println!("Test 1: Split Percentile Sweep (gradient)");
    println!("═══════════════════════════════════════");

    for &percentile in &[0.05, 0.10, 0.15, 0.20] {
        let mut encoder = ErrorDrivenEncoder::default();
        encoder.split_percentile = percentile;
        encoder.max_gaussians = 500;  // Limit for speed

        let gaussians = encoder.encode(&target);
        let rendered = lgi_encoder_v2::renderer_v2::RendererV2::render(&gaussians, 256, 256);
        let psnr = compute_psnr(&target, &rendered);

        println!("  Percentile {:.2}: N={}, PSNR={:.2} dB", percentile, gaussians.len(), psnr);
    }

    println!("\nTest 2: Initial N Sweep");
    println!("═══════════════════════════════════════");

    for &initial_n in &[64, 100, 144, 196] {
        let mut encoder = ErrorDrivenEncoder::default();
        encoder.initial_gaussians = initial_n;
        encoder.max_gaussians = 500;

        let gaussians = encoder.encode(&target);
        let rendered = lgi_encoder_v2::renderer_v2::RendererV2::render(&gaussians, 256, 256);
        let psnr = compute_psnr(&target, &rendered);

        println!("  Initial N={}: Final N={}, PSNR={:.2} dB",
            initial_n, gaussians.len(), psnr);
    }

    // Test on edge (where it struggles)
    println!("\nTest 3: Error-Driven on Edge (optimization)");
    println!("═══════════════════════════════════════");

    let mut edge_target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let val = if x < 128 { 0.0 } else { 1.0 };
            edge_target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    for &percentile in &[0.15, 0.20, 0.25] {
        let mut encoder = ErrorDrivenEncoder::default();
        encoder.split_percentile = percentile;
        encoder.initial_gaussians = 100;
        encoder.max_gaussians = 800;
        encoder.target_error = 0.005;  // Tighter for edges

        let gaussians = encoder.encode(&edge_target);
        let rendered = lgi_encoder_v2::renderer_v2::RendererV2::render(&gaussians, 256, 256);
        let psnr = compute_psnr(&edge_target, &rendered);
        let edge_psnr = compute_psnr_region(&edge_target, &rendered, 120, 136, 0, 256);

        println!("  Percentile {:.2}: N={}, Overall {:.2} dB, Edge {:.2} dB",
            percentile, gaussians.len(), psnr, edge_psnr);
    }
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

fn compute_psnr_region(
    original: &ImageBuffer<f32>,
    rendered: &ImageBuffer<f32>,
    x_start: u32, x_end: u32, y_start: u32, y_end: u32,
) -> f32 {
    let mut mse = 0.0;
    let mut count = 0.0;

    for y in y_start..y_end.min(original.height) {
        for x in x_start..x_end.min(original.width) {
            if let (Some(p1), Some(p2)) = (original.get_pixel(x, y), rendered.get_pixel(x, y)) {
                mse += (p1.r - p2.r).powi(2) + (p1.g - p2.g).powi(2) + (p1.b - p2.b).powi(2);
                count += 3.0;
            }
        }
    }

    if count > 0.0 { mse /= count; }
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
