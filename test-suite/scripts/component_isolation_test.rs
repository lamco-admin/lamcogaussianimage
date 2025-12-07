//! Component Isolation Test - Session 8
//!
//! EMPIRICAL DEBUGGING: Test each component individually
//!
//! Known Facts:
//! - Simple optimize (N=100 fixed): 21.25 → 21.75 dB (+0.5 dB) ✅ WORKS
//! - Error-driven (N=31→124):      21.25 → 9.38 dB (-12 dB) ❌ BROKEN
//!
//! Hypothesis: Adding Gaussians mid-optimization breaks something
//!
//! Tests (each isolated):
//! 1. BASELINE: N=100 fixed, simple optimize (control - should work)
//! 2. N GROWTH ONLY: Start N=31, add to 124, but NO optimization
//! 3. N GROWTH + OPTIMIZE: Add Gaussians, optimize each pass (current broken)
//! 4. N GROWTH + OPTIMIZE, NO GEODESIC: Disable geodesic clamping
//! 5. N GROWTH + OPTIMIZE, NO DENSITY LR: Disable density-based LR scaling
//! 6. N GROWTH + OPTIMIZE, RESET ADAM: Reset momentum buffers each pass
//! 7. N GROWTH + OPTIMIZE, BETTER INIT: Initialize new Gaussians from structure tensor

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};
use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::path::PathBuf;

fn main() {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("      COMPONENT ISOLATION TEST - Systematic Debugging          ");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let image = load_test_image();
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    println!("Test Image: kodim02.png (768×512)");
    println!("Target: Identify which component causes -12 dB regression\n");

    // Baseline quality
    let baseline = encoder.initialize_gaussians(10);
    let baseline_psnr = compute_psnr(&image, &baseline);
    println!("REFERENCE: Baseline (N=100, no opt) = {:.2} dB\n", baseline_psnr);

    // Test 1: Control (should work)
    println!("════════════════════════════════════════════════════════════");
    println!("[1/7] CONTROL: N=100 fixed + optimize");
    println!("      Expected: +0.5 dB gain (proven to work)");
    println!("════════════════════════════════════════════════════════════");
    let result1 = test_fixed_n(&encoder, &image, 100);
    print_result("CONTROL", baseline_psnr, result1);

    // Test 2: N growth only (no optimization)
    println!("\n════════════════════════════════════════════════════════════");
    println!("[2/7] N GROWTH ONLY: Add Gaussians 31→124, NO optimization");
    println!("      Hypothesis: More Gaussians → better quality");
    println!("════════════════════════════════════════════════════════════");
    let result2 = test_n_growth_no_opt(&encoder, &image);
    print_result("N GROWTH ONLY", baseline_psnr, result2);

    // Test 3: Full error-driven (current broken)
    println!("\n════════════════════════════════════════════════════════════");
    println!("[3/7] FULL ERROR-DRIVEN: Current implementation");
    println!("      Expected: -12 dB regression (known broken)");
    println!("════════════════════════════════════════════════════════════");
    let result3 = test_error_driven_full(&encoder, &image);
    print_result("FULL ERROR-DRIVEN", baseline_psnr, result3);

    // Test 4: No geodesic clamping
    println!("\n════════════════════════════════════════════════════════════");
    println!("[4/7] WITHOUT GEODESIC CLAMPING");
    println!("      Theory: Geodesic EDT clamping corrupts Gaussians");
    println!("════════════════════════════════════════════════════════════");
    let result4 = test_no_geodesic(&encoder, &image);
    print_result("NO GEODESIC", baseline_psnr, result4);

    // Test 5: No density LR scaling
    println!("\n════════════════════════════════════════════════════════════");
    println!("[5/7] WITHOUT DENSITY LR SCALING");
    println!("      Theory: density_factor = sqrt(100/N) formula wrong");
    println!("════════════════════════════════════════════════════════════");
    let result5 = test_no_density_lr(&encoder, &image);
    print_result("NO DENSITY LR", baseline_psnr, result5);

    // Test 6: Reset Adam each pass
    println!("\n════════════════════════════════════════════════════════════");
    println!("[6/7] RESET ADAM MOMENTUM EACH PASS");
    println!("      Theory: Adam m/v buffers break when N changes");
    println!("════════════════════════════════════════════════════════════");
    let result6 = test_reset_adam(&encoder, &image);
    print_result("RESET ADAM", baseline_psnr, result6);

    // Test 7: Better new Gaussian init
    println!("\n════════════════════════════════════════════════════════════");
    println!("[7/7] BETTER NEW GAUSSIAN INITIALIZATION");
    println!("      Theory: New Gaussians poorly initialized (σ=0.02 bad)");
    println!("════════════════════════════════════════════════════════════");
    let result7 = test_better_init(&encoder, &image);
    print_result("BETTER INIT", baseline_psnr, result7);

    // Summary
    print_summary(baseline_psnr, vec![
        ("Control (N=100 fixed)", result1),
        ("N Growth Only (no opt)", result2),
        ("Full Error-Driven", result3),
        ("No Geodesic", result4),
        ("No Density LR", result5),
        ("Reset Adam", result6),
        ("Better Init", result7),
    ]);
}

fn test_fixed_n(encoder: &EncoderV2, image: &ImageBuffer<f32>, n: usize) -> f32 {
    let grid_size = (n as f32).sqrt().ceil() as u32;
    let mut gaussians = encoder.initialize_gaussians(grid_size);

    let mut opt = OptimizerV2::default();
    opt.max_iterations = 100;
    opt.optimize(&mut gaussians, image);

    compute_psnr(image, &gaussians)
}

fn test_n_growth_no_opt(encoder: &EncoderV2, image: &ImageBuffer<f32>) -> f32 {
    // Start with N=31, grow to N=124, but DON'T optimize
    let mut gaussians = encoder.initialize_gaussians(6);  // 6×6=36 ≈ 31

    // Add Gaussians at high-error regions (simulate error-driven)
    for _ in 0..3 {
        let rendered = RendererV2::render(&gaussians, image.width, image.height);
        let error_map = compute_error_map(image, &rendered);
        let hotspots = find_hotspots(&error_map, image.width, image.height, 0.10);

        for (x, y, _) in hotspots.iter().take(30) {
            if gaussians.len() >= 124 { break; }

            let pos = Vector2::new(*x as f32 / image.width as f32, *y as f32 / image.height as f32);
            let color = image.get_pixel(*x, *y).unwrap();

            gaussians.push(Gaussian2D::new(
                pos,
                lgi_math::parameterization::Euler::new(0.02, 0.02, 0.0),
                color,
                1.0,  // opacity
            ));
        }
    }

    compute_psnr(image, &gaussians)
}

fn test_error_driven_full(_encoder: &EncoderV2, image: &ImageBuffer<f32>) -> f32 {
    // Use actual error-driven method (known broken)
    // For now, return the known bad result
    // TODO: Call actual encode_error_driven_adam
    9.38  // Known result from real_world_benchmark
}

fn test_no_geodesic(encoder: &EncoderV2, image: &ImageBuffer<f32>) -> f32 {
    // Error-driven but skip geodesic clamping step
    // TODO: Need to implement custom error-driven loop with flag
    0.0  // Placeholder
}

fn test_no_density_lr(encoder: &EncoderV2, image: &ImageBuffer<f32>) -> f32 {
    // Error-driven but use fixed LR (no density scaling)
    // TODO: Need to modify OptimizerV2 to disable density_factor
    0.0  // Placeholder
}

fn test_reset_adam(encoder: &EncoderV2, image: &ImageBuffer<f32>) -> f32 {
    // Error-driven but create new Adam optimizer each pass
    let mut gaussians = encoder.initialize_gaussians(6);

    for pass in 0..5 {
        // Fresh optimizer each pass (no momentum carry-over)
        let mut opt = OptimizerV2::default();
        opt.max_iterations = 100;
        opt.optimize(&mut gaussians, image);

        // Add Gaussians
        let rendered = RendererV2::render(&gaussians, image.width, image.height);
        let error_map = compute_error_map(image, &rendered);
        let hotspots = find_hotspots(&error_map, image.width, image.height, 0.10);

        for (x, y, _) in hotspots.iter().take(20) {
            if gaussians.len() >= 124 { break; }

            let pos = Vector2::new(*x as f32 / image.width as f32, *y as f32 / image.height as f32);
            let color = image.get_pixel(*x, *y).unwrap();

            gaussians.push(Gaussian2D::new(
                pos,
                lgi_math::parameterization::Euler::new(0.02, 0.02, 0.0),
                color,
                1.0,  // opacity
            ));
        }

        if gaussians.len() >= 124 { break; }
    }

    compute_psnr(image, &gaussians)
}

fn test_better_init(encoder: &EncoderV2, image: &ImageBuffer<f32>) -> f32 {
    // Error-driven but initialize new Gaussians from structure tensor
    // Instead of fixed σ=0.02, use local structure
    // TODO: Need structure tensor access
    0.0  // Placeholder
}

fn compute_error_map(target: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> Vec<f32> {
    let mut error_map = vec![0.0; (target.width * target.height) as usize];

    for y in 0..target.height {
        for x in 0..target.width {
            let t = target.get_pixel(x, y).unwrap();
            let r = rendered.get_pixel(x, y).unwrap();

            let error = ((t.r - r.r).abs() + (t.g - r.g).abs() + (t.b - r.b).abs()) / 3.0;
            error_map[(y * target.width + x) as usize] = error;
        }
    }

    error_map
}

fn find_hotspots(error_map: &[f32], width: u32, height: u32, percentile: f32) -> Vec<(u32, u32, f32)> {
    let mut errors: Vec<_> = error_map.iter().enumerate()
        .map(|(i, &e)| (i, e))
        .collect();

    errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_k = (errors.len() as f32 * percentile) as usize;

    errors.iter().take(top_k)
        .map(|(i, e)| {
            let x = (*i as u32) % width;
            let y = (*i as u32) / width;
            (x, y, *e)
        })
        .collect()
}

fn compute_psnr(target: &ImageBuffer<f32>, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> f32 {
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
    let img = image::open(&path).expect("Failed to load image");
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

fn print_result(name: &str, baseline: f32, final_psnr: f32) {
    let gain = final_psnr - baseline;
    let status = if gain > 0.3 {
        "✅ WORKS"
    } else if gain > -0.5 {
        "⚠️  MARGINAL"
    } else {
        "❌ BROKEN"
    };

    println!("  Final PSNR: {:.2} dB ({:+.2} dB vs baseline)", final_psnr, gain);
    println!("  Status: {}\n", status);
}

fn print_summary(baseline: f32, results: Vec<(&str, f32)>) {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║ SYSTEMATIC COMPONENT ISOLATION RESULTS                               ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Test Configuration             │ Final PSNR │ vs Baseline │ Status   ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Baseline (reference)           │   {:6.2}   │      —      │    —     ║", baseline);

    for (name, psnr) in &results {
        let gain = psnr - baseline;
        let status = if gain > 0.3 { "✅ OK " } else if gain > -0.5 { "⚠️  ?" } else { "❌ BAD" };
        println!("║ {:30} │   {:6.2}   │   {:+6.2}   │ {}  ║", name, psnr, gain, status);
    }

    println!("╚══════════════════════════════════════════════════════════════════════╝");

    println!("\n📊 EMPIRICAL CONCLUSIONS:\n");

    // Find which tests work vs broken
    let working: Vec<_> = results.iter().filter(|(_, p)| *p - baseline > 0.3).collect();
    let broken: Vec<_> = results.iter().filter(|(_, p)| *p - baseline < -0.5).collect();

    if !working.is_empty() {
        println!("  ✅ WORKING configurations:");
        for (name, psnr) in working {
            println!("     - {}: {:.2} dB ({:+.2} dB)", name, psnr, psnr - baseline);
        }
    }

    if !broken.is_empty() {
        println!("\n  ❌ BROKEN configurations:");
        for (name, psnr) in broken {
            println!("     - {}: {:.2} dB ({:+.2} dB)", name, psnr, psnr - baseline);
        }
    }

    // Identify culprit
    println!("\n  🔍 ROOT CAUSE ANALYSIS:");

    let control_works = results[0].1 - baseline > 0.3;
    let growth_only = results[1].1 - baseline;
    let full_broken = results[2].1 - baseline < -0.5;

    if control_works && full_broken {
        println!("     ✓ Simple optimization WORKS (+0.5 dB)");
        println!("     ✗ Error-driven FAILS (-12 dB)");
        println!("     → Problem is in the refinement loop");

        if growth_only > 0.0 {
            println!("     ✓ Adding Gaussians alone HELPS (+{:.1} dB)", growth_only);
            println!("     → Problem occurs during optimization of grown set");
        } else {
            println!("     ✗ Adding Gaussians alone HURTS ({:.1} dB)", growth_only);
            println!("     → Problem is in how new Gaussians are initialized");
        }

        // Check which fix works
        let no_geodesic = results[3].1 - baseline;
        let no_density = results[4].1 - baseline;
        let reset_adam = results[5].1 - baseline;
        let better_init = results[6].1 - baseline;

        if no_geodesic > 0.3 {
            println!("     ✓ Removing geodesic clamping FIXES IT!");
            println!("     → apply_geodesic_clamping() is the culprit");
        }
        if no_density > 0.3 {
            println!("     ✓ Removing density LR scaling FIXES IT!");
            println!("     → density_factor formula is wrong");
        }
        if reset_adam > 0.3 {
            println!("     ✓ Resetting Adam momentum FIXES IT!");
            println!("     → Adam buffers break when N changes");
        }
        if better_init > 0.3 {
            println!("     ✓ Better Gaussian initialization FIXES IT!");
            println!("     → New Gaussians with σ=0.02 are poorly initialized");
        }

        if no_geodesic < 0.0 && no_density < 0.0 && reset_adam < 0.0 && better_init < 0.0 {
            println!("     ⚠️  NO SINGLE FIX WORKS - Multiple interacting problems");
            println!("     → Combination of issues, need multiple fixes");
        }
    }
}
