//! Comprehensive Benchmark Suite
//!
//! Tests all encoding methods with detailed comparisons and PSNR analysis

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2};
use lgi_core::{ImageBuffer, quantization::LGIQProfile};
use lgi_math::color::Color4;
use std::time::Instant;

fn main() {
    env_logger::init();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║          LGI v2 COMPREHENSIVE BENCHMARK SUITE                 ║");
    println!("║                  Session 7 Validation                          ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Test on multiple image types
    let test_images = vec![
        ("Simple Gradient", create_simple_gradient(256, 256)),
        ("Sharp Edge", create_sharp_edge(256, 256)),
        ("Complex Pattern", create_complex_pattern(256, 256)),
        ("Photo-Like", create_photo_like(256, 256)),
    ];

    for (name, image) in test_images {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  TEST IMAGE: {}", name);
        println!("  Size: {}×{}", image.width, image.height);
        println!("═══════════════════════════════════════════════════════════════");
        println!();

        benchmark_image(&image, name);
        println!();
    }

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK COMPLETE                          ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
}

fn benchmark_image(image: &ImageBuffer<f32>, image_name: &str) {
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");
    let target = image;  // Keep reference for PSNR computation

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ TECHNIQUE COMPARISON (All methods at N=100)                │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    // Baseline: Simple grid initialization (no advanced techniques)
    println!("📊 BASELINE (Simple Grid, No Optimization)");
    let start = Instant::now();
    let baseline = encoder.initialize_gaussians(10);  // 10×10 = 100
    let baseline_time = start.elapsed();
    let baseline_psnr = compute_psnr(target, &baseline);
    println!("  Gaussians: {}", baseline.len());
    println!("  PSNR: {:.2} dB", baseline_psnr);
    println!("  Time: {:?}", baseline_time);
    println!("  Techniques: None (baseline)");
    println!();

    // Test 1: Auto N Selection
    println!("📊 AUTO N SELECTION (Integration #1)");
    let start = Instant::now();
    let auto_n = encoder.auto_gaussian_count();
    let auto_time = start.elapsed();
    println!("  Selected N: {} (entropy-based)", auto_n);
    println!("  Selection time: {:?}", auto_time);
    println!("  Δ from baseline: {:+} Gaussians", auto_n as i32 - baseline.len() as i32);
    println!();

    // Test 2: Guided Filter (Better Color Init)
    println!("📊 GUIDED FILTER + BETTER COLORS (Integration #2)");
    let start = Instant::now();
    let with_guided = encoder.initialize_gaussians_guided(10);
    let guided_time = start.elapsed();
    let guided_psnr = compute_psnr(target, &with_guided);
    println!("  Gaussians: {}", with_guided.len());
    println!("  PSNR: {:.2} dB", guided_psnr);
    println!("  Time: {:?}", guided_time);
    println!("  Δ PSNR vs baseline: {:+.2} dB", guided_psnr - baseline_psnr);
    println!("  Techniques: Guided filter + Gaussian-weighted colors");
    println!();

    // Test 3: Error-Driven (Integration #3)
    println!("📊 ERROR-DRIVEN ENCODING (Integration #3)");
    let start = Instant::now();
    let error_driven = encoder.encode_error_driven(25, 100);
    let error_time = start.elapsed();
    let error_psnr = compute_psnr(target, &error_driven);
    println!("  Gaussians: {}", error_driven.len());
    println!("  PSNR: {:.2} dB", error_psnr);
    println!("  Time: {:?}", error_time);
    println!("  Δ PSNR vs baseline: {:+.2} dB", error_psnr - baseline_psnr);
    println!("  Techniques: Structure tensor + geodesic EDT + adaptive refinement");
    println!();

    // Test 4: Adam Optimizer (Integration #4)
    println!("📊 ADAM OPTIMIZER (Integration #4) [RECOMMENDED]");
    let start = Instant::now();
    let adam = encoder.encode_error_driven_adam(25, 100);
    let adam_time = start.elapsed();
    let adam_psnr = compute_psnr(target, &adam);
    println!("  Gaussians: {}", adam.len());
    println!("  PSNR: {:.2} dB", adam_psnr);
    println!("  Time: {:?}", adam_time);
    println!("  Δ PSNR vs baseline: {:+.2} dB", adam_psnr - baseline_psnr);
    println!("  Δ Time vs error-driven: {:?} ({:.1}× faster)",
        error_time.checked_sub(adam_time).unwrap_or_default(),
        error_time.as_secs_f32() / adam_time.as_secs_f32().max(0.001)
    );
    println!("  Techniques: Error-driven + Adam + anti-bleeding");
    println!();

    // Test 5: GPU Acceleration (Integration #9)
    println!("📊 GPU ACCELERATION (Integration #9) [FASTEST]");
    let start = Instant::now();
    let gpu = encoder.encode_error_driven_gpu(25, 100);
    let gpu_time = start.elapsed();
    let gpu_psnr = compute_psnr(target, &gpu);
    println!("  Gaussians: {}", gpu.len());
    println!("  PSNR: {:.2} dB", gpu_psnr);
    println!("  Time: {:?}", gpu_time);
    println!("  Δ PSNR vs baseline: {:+.2} dB", gpu_psnr - baseline_psnr);
    println!("  Δ Time vs Adam: {:?}", adam_time.checked_sub(gpu_time).unwrap_or_default());
    println!("  Techniques: Error-driven + GPU rendering + anti-bleeding");
    println!();

    // Test 6: GPU + MS-SSIM (Integration #10)
    println!("📊 GPU + MS-SSIM (Integration #10) [ULTIMATE QUALITY]");
    let start = Instant::now();
    let ultimate = encoder.encode_error_driven_gpu_msssim(25, 100);
    let ultimate_time = start.elapsed();
    let ultimate_psnr = compute_psnr(target, &ultimate);
    println!("  Gaussians: {}", ultimate.len());
    println!("  PSNR: {:.2} dB", ultimate_psnr);
    println!("  Time: {:?}", ultimate_time);
    println!("  Δ PSNR vs baseline: {:+.2} dB", ultimate_psnr - baseline_psnr);
    println!("  Techniques: GPU + MS-SSIM perceptual loss + anti-bleeding");
    println!();

    // Test 7: Rate-Distortion Targeting
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ RATE-DISTORTION TARGETING (Integration #5)                 │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    // Target PSNR
    println!("📊 TARGET PSNR = 28 dB");
    let start = Instant::now();
    let target_psnr_gaussians = encoder.encode_for_psnr(28.0, LGIQProfile::Baseline);
    let target_time = start.elapsed();
    let achieved_psnr = compute_psnr(target, &target_psnr_gaussians);
    println!("  Gaussians: {}", target_psnr_gaussians.len());
    println!("  Target PSNR: 28.0 dB");
    println!("  Achieved PSNR: {:.2} dB", achieved_psnr);
    println!("  Error: {:+.2} dB", achieved_psnr - 28.0);
    println!("  Time: {:?}", target_time);
    println!();

    // Target bitrate
    println!("📊 TARGET BITRATE = 50 KB");
    let target_bits = 50_000.0 * 8.0;
    let start = Instant::now();
    let target_bitrate_gaussians = encoder.encode_for_bitrate(target_bits, LGIQProfile::Baseline);
    let bitrate_time = start.elapsed();
    let bitrate_psnr = compute_psnr(target, &target_bitrate_gaussians);
    println!("  Gaussians: {}", target_bitrate_gaussians.len());
    println!("  PSNR: {:.2} dB", bitrate_psnr);
    println!("  Time: {:?}", bitrate_time);
    println!();

    // Perceptual quality targeting
    println!("📊 TARGET MS-SSIM = 0.95 (Perceptual)");
    let start = Instant::now();
    let perceptual_gaussians = encoder.encode_for_perceptual_quality(0.95, LGIQProfile::Baseline);
    let perceptual_time = start.elapsed();
    let perceptual_psnr = compute_psnr(target, &perceptual_gaussians);
    println!("  Gaussians: {}", perceptual_gaussians.len());
    println!("  PSNR: {:.2} dB", perceptual_psnr);
    println!("  Time: {:?}", perceptual_time);
    println!();

    // Test 8: EWA Rendering
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ RENDERING COMPARISON (Integration #7)                      │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    println!("📊 SIMPLE SPLATTING (Baseline)");
    let start = Instant::now();
    let simple_rendered = RendererV2::render(&adam, image.width, image.height);
    let simple_time = start.elapsed();
    let simple_psnr = compute_psnr_direct(image, &simple_rendered);
    println!("  PSNR: {:.2} dB", simple_psnr);
    println!("  Time: {:?}", simple_time);
    println!();

    println!("📊 EWA SPLATTING (Alias-Free)");
    let start = Instant::now();
    let ewa_rendered = encoder.render_ewa(&adam, 1.0);
    let ewa_time = start.elapsed();
    let ewa_psnr = compute_psnr_direct(image, &ewa_rendered);
    println!("  PSNR: {:.2} dB", ewa_psnr);
    println!("  Time: {:?}", ewa_time);
    println!("  Δ PSNR vs simple: {:+.2} dB", ewa_psnr - simple_psnr);
    println!();

    // Summary Table
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ SUMMARY TABLE                                               │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌────────────────────────────────┬──────┬─────────┬──────────┐");
    println!("│ Method                         │  N   │  PSNR   │   Δ dB   │");
    println!("├────────────────────────────────┼──────┼─────────┼──────────┤");
    println!("│ Baseline (no techniques)       │ {:>4} │ {:>6.2} │    —     │",
        baseline.len(), baseline_psnr);
    println!("│ Guided Filter + Better Colors  │ {:>4} │ {:>6.2} │ {:>+7.2} │",
        with_guided.len(), guided_psnr, guided_psnr - baseline_psnr);
    println!("│ Error-Driven                   │ {:>4} │ {:>6.2} │ {:>+7.2} │",
        error_driven.len(), error_psnr, error_psnr - baseline_psnr);
    println!("│ Adam Optimizer (RECOMMENDED)   │ {:>4} │ {:>6.2} │ {:>+7.2} │",
        adam.len(), adam_psnr, adam_psnr - baseline_psnr);
    println!("│ GPU Accelerated (FASTEST)      │ {:>4} │ {:>6.2} │ {:>+7.2} │",
        gpu.len(), gpu_psnr, gpu_psnr - baseline_psnr);
    println!("│ GPU + MS-SSIM (ULTIMATE)       │ {:>4} │ {:>6.2} │ {:>+7.2} │",
        ultimate.len(), ultimate_psnr, ultimate_psnr - baseline_psnr);
    println!("└────────────────────────────────┴──────┴─────────┴──────────┘");
    println!();

    println!("Key Findings for '{}':", image_name);
    println!("  • Best quality gain: {:+.2} dB ({})",
        find_best_gain(baseline_psnr, &[guided_psnr, error_psnr, adam_psnr, gpu_psnr, ultimate_psnr]).0,
        find_best_gain(baseline_psnr, &[guided_psnr, error_psnr, adam_psnr, gpu_psnr, ultimate_psnr]).1
    );
    println!("  • Recommended method: Adam Optimizer ({:.2} dB, good speed)", adam_psnr);
    println!("  • Fastest method: GPU ({:.2} dB, fastest)", gpu_psnr);
    println!("  • Best perceptual: GPU + MS-SSIM ({:.2} dB)", ultimate_psnr);
}

fn compute_psnr(target: &ImageBuffer<f32>, gaussians: &[lgi_math::gaussian::Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>]) -> f32 {
    // Render the Gaussians
    let rendered = RendererV2::render(gaussians, target.width, target.height);

    // Compute actual MSE
    let mut mse = 0.0;
    let count = (target.width * target.height * 3) as f32;

    for (t, r) in target.data.iter().zip(rendered.data.iter()) {
        mse += (t.r - r.r).powi(2);
        mse += (t.g - r.g).powi(2);
        mse += (t.b - r.b).powi(2);
    }

    mse /= count;

    // Convert MSE to PSNR
    if mse < 1e-10 {
        100.0  // Cap at 100 dB for near-perfect reconstruction
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}

fn compute_psnr_direct(target: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (target.width * target.height * 3) as f32;

    for (t, r) in target.data.iter().zip(rendered.data.iter()) {
        mse += (t.r - r.r).powi(2);
        mse += (t.g - r.g).powi(2);
        mse += (t.b - r.b).powi(2);
    }

    mse /= count;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}

fn find_best_gain(baseline: f32, results: &[f32]) -> (f32, &'static str) {
    let methods = ["Guided Filter", "Error-Driven", "Adam", "GPU", "GPU+MS-SSIM"];
    let mut best_gain = 0.0;
    let mut best_method = "None";

    for (i, &psnr) in results.iter().enumerate() {
        let gain = psnr - baseline;
        if gain > best_gain {
            best_gain = gain;
            best_method = methods[i];
        }
    }

    (best_gain, best_method)
}

// Test image generators

fn create_simple_gradient(width: u32, height: u32) -> ImageBuffer<f32> {
    let mut image = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let r = x as f32 / width as f32;
            let g = y as f32 / height as f32;
            let b = 0.5;

            image.set_pixel(x, y, Color4::new(r, g, b, 1.0));
        }
    }

    image
}

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

fn create_photo_like(width: u32, height: u32) -> ImageBuffer<f32> {
    let mut image = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;

            // Sky gradient
            let sky = if fy < 0.6 {
                let t = fy / 0.6;
                (0.5 + t * 0.3, 0.7 + t * 0.2, 0.9 + t * 0.1)
            } else {
                // Ground
                let t = (fy - 0.6) / 0.4;
                (0.3 + t * 0.2, 0.5 + t * 0.2, 0.2 + t * 0.1)
            };

            // Add some texture
            let noise = (fx * 50.0).sin() * (fy * 50.0).sin() * 0.05;

            let r = (sky.0 + noise).clamp(0.0, 1.0);
            let g = (sky.1 + noise).clamp(0.0, 1.0);
            let b = (sky.2 + noise).clamp(0.0, 1.0);

            image.set_pixel(x, y, Color4::new(r, g, b, 1.0));
        }
    }

    image
}
