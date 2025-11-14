//! Fast Benchmark Suite
//!
//! Optimized for speed while providing real PSNR measurements

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2};
use lgi_core::{ImageBuffer, quantization::LGIQProfile};
use lgi_math::color::Color4;
use std::time::Instant;

fn main() {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("         LGI v2 FAST BENCHMARK - Session 7 Validation         ");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Test on 2 key images (reduced for speed)
    let test_images = vec![
        ("Sharp Edge (128×128)", create_sharp_edge(128, 128)),
        ("Complex Pattern (128×128)", create_complex_pattern(128, 128)),
    ];

    for (name, image) in test_images {
        println!("\n▼ TEST IMAGE: {}", name);
        println!("─────────────────────────────────────────────────────────────");
        benchmark_image(&image, name);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    BENCHMARK COMPLETE                         ");
    println!("═══════════════════════════════════════════════════════════════\n");
}

fn benchmark_image(image: &ImageBuffer<f32>, image_name: &str) {
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    // Results storage
    let mut results = Vec::new();

    // =========================================================================
    // Test 1: BASELINE (No optimization)
    // =========================================================================
    println!("\n[1/6] BASELINE (Simple Grid, No Optimization)");
    let start = Instant::now();
    let baseline = encoder.initialize_gaussians(8);  // 8×8 = 64 Gaussians
    let baseline_time = start.elapsed();
    let baseline_psnr = compute_psnr(image, &baseline);
    println!("  N = {} | PSNR = {:.2} dB | Time = {:?}",
        baseline.len(), baseline_psnr, baseline_time);
    results.push(("Baseline", baseline.len(), baseline_psnr, baseline_time));

    // =========================================================================
    // Test 2: GUIDED INITIALIZATION (Better Color Init)
    // =========================================================================
    println!("\n[2/6] GUIDED FILTER + BETTER COLORS");
    let start = Instant::now();
    let guided = encoder.initialize_gaussians_guided(8);
    let guided_time = start.elapsed();
    let guided_psnr = compute_psnr(image, &guided);
    let guided_gain = guided_psnr - baseline_psnr;
    println!("  N = {} | PSNR = {:.2} dB ({:+.2} dB) | Time = {:?}",
        guided.len(), guided_psnr, guided_gain, guided_time);
    results.push(("Guided Init", guided.len(), guided_psnr, guided_time));

    // =========================================================================
    // Test 3: ERROR-DRIVEN (Short run: 15→50 Gaussians)
    // =========================================================================
    println!("\n[3/6] ERROR-DRIVEN ENCODING (Short Run)");
    let start = Instant::now();
    let error_driven = encoder.encode_error_driven(15, 50);  // Reduced for speed
    let error_time = start.elapsed();
    let error_psnr = compute_psnr(image, &error_driven);
    let error_gain = error_psnr - baseline_psnr;
    println!("  N = {} | PSNR = {:.2} dB ({:+.2} dB) | Time = {:?}",
        error_driven.len(), error_psnr, error_gain, error_time);
    results.push(("Error-Driven", error_driven.len(), error_psnr, error_time));

    // =========================================================================
    // Test 4: ADAM OPTIMIZER (RECOMMENDED)
    // =========================================================================
    println!("\n[4/6] ADAM OPTIMIZER (RECOMMENDED)");
    let start = Instant::now();
    let adam = encoder.encode_error_driven_adam(15, 50);  // Reduced for speed
    let adam_time = start.elapsed();
    let adam_psnr = compute_psnr(image, &adam);
    let adam_gain = adam_psnr - baseline_psnr;
    let speedup = error_time.as_secs_f32() / adam_time.as_secs_f32().max(0.001);
    println!("  N = {} | PSNR = {:.2} dB ({:+.2} dB) | Time = {:?} ({:.1}× faster)",
        adam.len(), adam_psnr, adam_gain, adam_time, speedup);
    results.push(("Adam", adam.len(), adam_psnr, adam_time));

    // =========================================================================
    // Test 5: GPU ACCELERATION
    // =========================================================================
    println!("\n[5/6] GPU ACCELERATION (FASTEST)");
    let start = Instant::now();
    let gpu = encoder.encode_error_driven_gpu(15, 50);  // Reduced for speed
    let gpu_time = start.elapsed();
    let gpu_psnr = compute_psnr(image, &gpu);
    let gpu_gain = gpu_psnr - baseline_psnr;
    let gpu_speedup = error_time.as_secs_f32() / gpu_time.as_secs_f32().max(0.001);
    println!("  N = {} | PSNR = {:.2} dB ({:+.2} dB) | Time = {:?} ({:.1}× faster)",
        gpu.len(), gpu_psnr, gpu_gain, gpu_time, gpu_speedup);
    results.push(("GPU", gpu.len(), gpu_psnr, gpu_time));

    // =========================================================================
    // Test 6: RATE-DISTORTION TARGETING
    // =========================================================================
    println!("\n[6/6] RATE-DISTORTION: Target PSNR = 30 dB");
    let start = Instant::now();
    let rd = encoder.encode_for_psnr(30.0, LGIQProfile::Baseline);
    let rd_time = start.elapsed();
    let rd_psnr = compute_psnr(image, &rd);
    let rd_error = rd_psnr - 30.0;
    println!("  N = {} | PSNR = {:.2} dB (target 30.0, error {:+.2} dB) | Time = {:?}",
        rd.len(), rd_psnr, rd_error, rd_time);
    results.push(("R-D Target", rd.len(), rd_psnr, rd_time));

    // =========================================================================
    // SUMMARY TABLE
    // =========================================================================
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║ SUMMARY: {}                                ", image_name);
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Method                │    N │  PSNR (dB) │   Δ dB │  Time   ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");

    for (method, n, psnr, time) in &results {
        let delta = psnr - baseline_psnr;
        let time_str = format!("{:?}", time);
        println!("║ {:21} │ {:>4} │ {:>10.2} │ {:>+6.2} │ {:>7} ║",
            method, n, psnr, delta, time_str.chars().take(7).collect::<String>());
    }

    println!("╚═══════════════════════════════════════════════════════════════╝");

    // Find best method
    let best = results.iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();

    println!("\n✅ BEST QUALITY: {} ({:.2} dB, {:+.2} dB gain)",
        best.0, best.2, best.2 - baseline_psnr);

    let fastest = results.iter()
        .skip(2)  // Skip baseline and guided (not optimization methods)
        .min_by(|a, b| a.3.cmp(&b.3))
        .unwrap();

    println!("⚡ FASTEST: {} ({:?})", fastest.0, fastest.3);
}

fn compute_psnr(
    target: &ImageBuffer<f32>,
    gaussians: &[lgi_math::gaussian::Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>]
) -> f32 {
    // Render the Gaussians to an image
    let rendered = RendererV2::render(gaussians, target.width, target.height);

    // Compute MSE over RGB channels
    let mut mse = 0.0f32;
    let pixel_count = target.width * target.height;

    for y in 0..target.height {
        for x in 0..target.width {
            let t_pixel = target.get_pixel(x, y).unwrap();
            let r_pixel = rendered.get_pixel(x, y).unwrap();

            let diff_r = t_pixel.r - r_pixel.r;
            let diff_g = t_pixel.g - r_pixel.g;
            let diff_b = t_pixel.b - r_pixel.b;

            mse += diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
        }
    }

    // Average over all pixels and RGB channels
    mse /= (pixel_count * 3) as f32;

    // Convert MSE to PSNR
    // PSNR = 10 * log10(MAX^2 / MSE) = 20 * log10(MAX / sqrt(MSE))
    // With MAX = 1.0, PSNR = -10 * log10(MSE)
    if mse < 1e-10 {
        100.0  // Cap at 100 dB for near-perfect
    } else {
        -10.0 * mse.log10()
    }
}

// Test image generators

fn create_sharp_edge(width: u32, height: u32) -> ImageBuffer<f32> {
    let mut image = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let left_half = x < width / 2;
            let top_half = y < height / 2;

            let (r, g, b) = match (left_half, top_half) {
                (true, true) => (1.0, 0.0, 0.0),    // Red
                (false, true) => (0.0, 1.0, 0.0),   // Green
                (true, false) => (0.0, 0.0, 1.0),   // Blue
                (false, false) => (1.0, 1.0, 0.0),  // Yellow
            };

            image.set_pixel(x, y, Color4::new(r, g, b, 1.0));
        }
    }

    image
}

fn create_complex_pattern(width: u32, height: u32) -> ImageBuffer<f32> {
    let mut image = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;

            let r = (fx * 8.0 * std::f32::consts::PI).sin() * 0.5 + 0.5;
            let g = (fy * 8.0 * std::f32::consts::PI).sin() * 0.5 + 0.5;
            let b = ((fx + fy) * 4.0 * std::f32::consts::PI).sin() * 0.5 + 0.5;

            image.set_pixel(x, y, Color4::new(r, g, b, 1.0));
        }
    }

    image
}
