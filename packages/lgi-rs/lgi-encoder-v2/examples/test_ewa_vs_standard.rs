//! Test EWA splatting vs standard rendering
//!
//! Compares the alias-free EWA splatting renderer (Zwicker et al. 2001)
//! against the standard Gaussian renderer to measure:
//! - Quality difference (PSNR, MS-SSIM)
//! - Rendering speed
//! - Visual artifacts (aliasing, edge quality)
//!
//! Run: cargo run --release --example test_ewa_vs_standard -p lgi-encoder-v2

use lgi_core::{ImageBuffer, ewa_splatting::EWASplatter};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   EWA SPLATTING VS STANDARD RENDERING COMPARISON            ║");
    println!("║   Testing alias-free rendering quality                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);
    println!("");

    // First, optimize Gaussians using standard renderer
    let mut gaussians = create_jittered_grid(&target, 256);
    let mut adam = AdamOptimizer::new();
    adam.max_iterations = 100;

    println!("Optimizing 256 Gaussians with Adam (100 iterations)...");
    let opt_start = Instant::now();
    adam.optimize(&mut gaussians, &target);
    let opt_time = opt_start.elapsed().as_secs_f32();
    println!("  Optimization complete in {:.2}s\n", opt_time);

    // Test 1: Standard rendering
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Standard Rendering");
    println!("═══════════════════════════════════════════════════════════════");

    let std_start = Instant::now();
    let std_render = RendererV2::render(&gaussians, width, height);
    let std_time = std_start.elapsed().as_secs_f32();
    let std_psnr = compute_psnr(&std_render, &target);
    let std_ssim = compute_ssim(&std_render, &target);

    println!("  Render time: {:.4}s", std_time);
    println!("  PSNR: {:.2} dB", std_psnr);
    println!("  SSIM: {:.4}", std_ssim);

    // Test 2: EWA splatting
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: EWA Splatting Rendering");
    println!("═══════════════════════════════════════════════════════════════");

    let ewa = EWASplatter::default();
    let ewa_start = Instant::now();
    let mut ewa_render = ewa.render(&gaussians, width, height);
    EWASplatter::normalize(&mut ewa_render);
    let ewa_time = ewa_start.elapsed().as_secs_f32();
    let ewa_psnr = compute_psnr(&ewa_render, &target);
    let ewa_ssim = compute_ssim(&ewa_render, &target);

    println!("  Render time: {:.4}s", ewa_time);
    println!("  PSNR: {:.2} dB", ewa_psnr);
    println!("  SSIM: {:.4}", ewa_ssim);

    // Compare
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("COMPARISON");
    println!("═══════════════════════════════════════════════════════════════");
    let psnr_diff = ewa_psnr - std_psnr;
    let ssim_diff = ewa_ssim - std_ssim;
    let speed_ratio = std_time / ewa_time;

    println!("  PSNR difference: {:+.2} dB (EWA {} standard)",
             psnr_diff,
             if psnr_diff > 0.0 { ">" } else { "<" });
    println!("  SSIM difference: {:+.4}", ssim_diff);
    println!("  Speed ratio: {:.2}× (Standard/EWA)", speed_ratio);

    // Save result
    let mut result = TestResult::new("ewa_vs_standard", image_path);
    result.optimizer = "EWA vs Standard Rendering".to_string();
    result.n_gaussians = gaussians.len();
    result.iterations = 100;
    result.initial_psnr = std_psnr;
    result.final_psnr = ewa_psnr;
    result.improvement_db = psnr_diff;
    result.elapsed_seconds = opt_time + std_time + ewa_time;
    result.image_dimensions = (width, height);
    result.set_param("std_psnr", std_psnr);
    result.set_param("std_ssim", std_ssim);
    result.set_param("std_time_ms", std_time * 1000.0);
    result.set_param("ewa_psnr", ewa_psnr);
    result.set_param("ewa_ssim", ewa_ssim);
    result.set_param("ewa_time_ms", ewa_time * 1000.0);
    result.set_param("speed_ratio", speed_ratio);
    result.complete();

    let _ = result.save_with_image(&ewa_render, RESULTS_DIR);
    println!("\nResult saved!");

    // Test at different zoom levels (simulated by rendering at different sizes)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("ZOOM LEVEL COMPARISON (Aliasing Sensitivity)");
    println!("═══════════════════════════════════════════════════════════════");

    let zoom_levels = [0.5, 1.0, 2.0];
    for zoom in &zoom_levels {
        let new_w = (width as f32 * zoom) as u32;
        let new_h = (height as f32 * zoom) as u32;

        let std_zoom = RendererV2::render(&gaussians, new_w, new_h);
        let mut ewa_zoom = ewa.render(&gaussians, new_w, new_h);
        EWASplatter::normalize(&mut ewa_zoom);

        // Compare to downsampled/upsampled target
        let target_resized = resize_bilinear(&target, new_w, new_h);
        let std_psnr_zoom = compute_psnr(&std_zoom, &target_resized);
        let ewa_psnr_zoom = compute_psnr(&ewa_zoom, &target_resized);

        println!("  Zoom {:.1}× ({}x{}):", zoom, new_w, new_h);
        println!("    Standard: {:.2} dB", std_psnr_zoom);
        println!("    EWA:      {:.2} dB ({:+.2} dB)", ewa_psnr_zoom, ewa_psnr_zoom - std_psnr_zoom);
    }

    println!("\n✓ EWA splatting comparison complete!");
}

fn create_jittered_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let mut gaussians = Vec::with_capacity(n);
    let mut rng = 42u64;

    for i in 0..grid_size {
        for j in 0..grid_size {
            if gaussians.len() >= n {
                break;
            }

            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let jitter_x = ((rng >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.3;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let jitter_y = ((rng >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.3;

            let x = ((i as f32 + 0.5 + jitter_x) / grid_size as f32).clamp(0.01, 0.99);
            let y = ((j as f32 + 0.5 + jitter_y) / grid_size as f32).clamp(0.01, 0.99);

            let px = (x * width as f32) as u32;
            let py = (y * height as f32) as u32;
            let color = target.get_pixel(px.min(width - 1), py.min(height - 1))
                .cloned()
                .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            gaussians.push(Gaussian2D {
                position: Vector2::new(x, y),
                shape: Euler {
                    scale_x: 0.02,
                    scale_y: 0.02,
                    rotation: 0.0,
                },
                color: Color4::rgb(color.r, color.g, color.b),
                opacity: 0.8,
            });
        }
    }

    gaussians
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mse: f32 = rendered
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(r, t)| {
            let dr = r.r - t.r;
            let dg = r.g - t.g;
            let db = r.b - t.b;
            (dr * dr + dg * dg + db * db) / 3.0
        })
        .sum::<f32>()
        / rendered.data.len() as f32;

    if mse < 1e-10 {
        100.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

fn compute_ssim(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    // Simplified SSIM
    let c1 = 0.01f32.powi(2);
    let c2 = 0.03f32.powi(2);

    let n = rendered.data.len() as f32;

    let mean_r: f32 = rendered.data.iter().map(|p| (p.r + p.g + p.b) / 3.0).sum::<f32>() / n;
    let mean_t: f32 = target.data.iter().map(|p| (p.r + p.g + p.b) / 3.0).sum::<f32>() / n;

    let var_r: f32 = rendered.data.iter()
        .map(|p| { let v = (p.r + p.g + p.b) / 3.0 - mean_r; v * v })
        .sum::<f32>() / n;
    let var_t: f32 = target.data.iter()
        .map(|p| { let v = (p.r + p.g + p.b) / 3.0 - mean_t; v * v })
        .sum::<f32>() / n;

    let covar: f32 = rendered.data.iter().zip(target.data.iter())
        .map(|(r, t)| {
            let vr = (r.r + r.g + r.b) / 3.0 - mean_r;
            let vt = (t.r + t.g + t.b) / 3.0 - mean_t;
            vr * vt
        })
        .sum::<f32>() / n;

    ((2.0 * mean_r * mean_t + c1) * (2.0 * covar + c2)) /
    ((mean_r.powi(2) + mean_t.powi(2) + c1) * (var_r + var_t + c2))
}

fn resize_bilinear(img: &ImageBuffer<f32>, new_w: u32, new_h: u32) -> ImageBuffer<f32> {
    let mut result = ImageBuffer::new(new_w, new_h);
    let scale_x = img.width as f32 / new_w as f32;
    let scale_y = img.height as f32 / new_h as f32;

    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = x as f32 * scale_x;
            let src_y = y as f32 * scale_y;

            let x0 = (src_x.floor() as u32).min(img.width - 1);
            let y0 = (src_y.floor() as u32).min(img.height - 1);
            let x1 = (x0 + 1).min(img.width - 1);
            let y1 = (y0 + 1).min(img.height - 1);

            let fx = src_x - src_x.floor();
            let fy = src_y - src_y.floor();

            if let (Some(p00), Some(p01), Some(p10), Some(p11)) = (
                img.get_pixel(x0, y0),
                img.get_pixel(x0, y1),
                img.get_pixel(x1, y0),
                img.get_pixel(x1, y1),
            ) {
                let r = p00.r * (1.0 - fx) * (1.0 - fy) + p10.r * fx * (1.0 - fy)
                      + p01.r * (1.0 - fx) * fy + p11.r * fx * fy;
                let g = p00.g * (1.0 - fx) * (1.0 - fy) + p10.g * fx * (1.0 - fy)
                      + p01.g * (1.0 - fx) * fy + p11.g * fx * fy;
                let b = p00.b * (1.0 - fx) * (1.0 - fy) + p10.b * fx * (1.0 - fy)
                      + p01.b * (1.0 - fx) * fy + p11.b * fx * fy;

                if let Some(dst) = result.get_pixel_mut(x, y) {
                    dst.r = r;
                    dst.g = g;
                    dst.b = b;
                    dst.a = 1.0;
                }
            }
        }
    }

    result
}
