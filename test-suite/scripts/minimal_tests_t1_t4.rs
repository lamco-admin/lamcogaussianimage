//! Complete T1-T4 minimal tests from debug plan
//! T0 already validated, this completes the foundation test suite

use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use lgi_encoder_v2::{renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  Minimal Tests T1-T4 (Debug Plan)           ║");
    println!("╚══════════════════════════════════════════════╝\n");

    test_t1_solid_16_gaussians();
    test_t2_two_tone_64_gaussians();
    test_t3_sinusoid();
    test_t4_text_preset();
}

/// T1: Solid color with 16 isotropic Gaussians
/// Expected: Color-only ≥20 dB in ≤20 iters, with μ updates ≥28 dB
fn test_t1_solid_16_gaussians() {
    println!("═══════════════════════════════════════════════");
    println!("T1: Solid Color, 16 Isotropic Gaussians");
    println!("Expected: ≥20 dB color-only, ≥28 dB with position");
    println!("═══════════════════════════════════════════════");

    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            target.set_pixel(x, y, Color4::new(0.8, 0.2, 0.1, 1.0));
        }
    }

    // Create 16 isotropic Gaussians (4×4 grid) with random positions
    let mut rng_state = 12345u32;
    let mut gaussians = Vec::new();
    let sigma = 0.15;  // ~12px for 256×256

    for i in 0..16 {
        // Simple LCG random
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let rand_x = (rng_state % 256) as f32 / 256.0;
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let rand_y = (rng_state % 256) as f32 / 256.0;

        gaussians.push(Gaussian2D::new(
            Vector2::new(rand_x, rand_y),
            Euler::isotropic(sigma),
            Color4::new(0.5, 0.5, 0.5, 1.0),  // Start gray
            1.0,
        ));
    }

    // Test color-only
    let mut gaussians_color = gaussians.clone();
    let mut opt_color = OptimizerV2::default();
    opt_color.learning_rate_position = 0.0;  // Freeze position
    opt_color.learning_rate_scale = 0.0;
    opt_color.max_iterations = 20;

    opt_color.optimize(&mut gaussians_color, &target);
    let psnr_color = compute_psnr(&target, &RendererV2::render(&gaussians_color, 256, 256));

    // Test with position
    let mut gaussians_full = gaussians.clone();
    let mut opt_full = OptimizerV2::default();
    opt_full.max_iterations = 50;

    opt_full.optimize(&mut gaussians_full, &target);
    let psnr_full = compute_psnr(&target, &RendererV2::render(&gaussians_full, 256, 256));

    println!("  Color-only (20 iters): {:.2} dB", psnr_color);
    println!("  With position (50 iters): {:.2} dB", psnr_full);

    if psnr_color >= 20.0 {
        println!("  ✅ T1 PASSED (color-only ≥20 dB)");
    } else {
        println!("  ❌ T1 FAILED (color-only <20 dB)");
    }

    if psnr_full >= 28.0 {
        println!("  ✅ T1 PASSED (with position ≥28 dB)");
    } else {
        println!("  ⚠️  T1 PARTIAL ({:.2} dB, target 28 dB)", psnr_full);
    }
    println!();
}

/// T2: Two-tone vertical step with 64 Gaussians
/// Expected: ≥22 dB without Σ opt, ≥26 dB with Σ/μ
fn test_t2_two_tone_64_gaussians() {
    println!("═══════════════════════════════════════════════");
    println!("T2: Two-Tone Step, 64 Gaussians");
    println!("Expected: ≥22 dB basic, ≥26 dB optimized");
    println!("═══════════════════════════════════════════════");

    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            if x < 128 {
                target.set_pixel(x, y, Color4::new(0.2, 0.3, 0.8, 1.0));  // Blue
            } else {
                target.set_pixel(x, y, Color4::new(0.9, 0.7, 0.1, 1.0));  // Yellow
            }
        }
    }

    // Use standard initialization (8×8 = 64)
    let mut gaussians = init_grid_isotropic(&target, 8);

    let init_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, 256, 256));

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 100;
    optimizer.optimize(&mut gaussians, &target);

    let final_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, 256, 256));

    println!("  Init: {:.2} dB → Final: {:.2} dB", init_psnr, final_psnr);

    if final_psnr >= 26.0 {
        println!("  ✅ T2 PASSED (≥26 dB)");
    } else if final_psnr >= 22.0 {
        println!("  ✓ T2 ACCEPTABLE ({:.2} dB, target 26 dB)", final_psnr);
    } else {
        println!("  ❌ T2 FAILED (<22 dB)");
    }
    println!();
}

/// T3: 1D sinusoid - catches unit mistakes
/// Expected: Should represent well (PSNR ≥25 dB)
fn test_t3_sinusoid() {
    println!("═══════════════════════════════════════════════");
    println!("T3: 1D Sinusoid (Frequency Check)");
    println!("Expected: ≥25 dB");
    println!("═══════════════════════════════════════════════");

    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let freq = 4.0;  // 4 cycles
            let val = 0.5 + 0.5 * (freq * 2.0 * std::f32::consts::PI * x as f32 / 256.0).sin();
            target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    let mut gaussians = init_grid_isotropic(&target, 20);  // N=400

    let init_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, 256, 256));

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 100;
    optimizer.optimize(&mut gaussians, &target);

    let final_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, 256, 256));

    println!("  Init: {:.2} dB → Final: {:.2} dB", init_psnr, final_psnr);

    if final_psnr >= 25.0 {
        println!("  ✅ T3 PASSED (≥25 dB)");
    } else {
        println!("  ⚠️  T3 MARGINAL ({:.2} dB, target 25 dB)", final_psnr);
    }
    println!();
}

/// T4: Text patch with presets
/// Expected: Crisp stems with thin Gaussians
fn test_t4_text_preset() {
    println!("═══════════════════════════════════════════════");
    println!("T4: Text Patch (Simulated)");
    println!("Expected: Readable text, PSNR ≥20 dB");
    println!("═══════════════════════════════════════════════");

    // Create simple text-like pattern (vertical strokes)
    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            // Three vertical strokes at x=64, 128, 192
            let is_stroke = (x >= 60 && x <= 68) ||
                           (x >= 124 && x <= 132) ||
                           (x >= 188 && x <= 196);
            let val = if is_stroke { 1.0 } else { 0.0 };
            target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    // Use higher N for text detail
    let mut gaussians = init_grid_isotropic(&target, 24);  // N=576

    let init_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, 256, 256));

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 150;
    optimizer.optimize(&mut gaussians, &target);

    let final_psnr = compute_psnr(&target, &RendererV2::render(&gaussians, 256, 256));

    println!("  Init: {:.2} dB → Final: {:.2} dB", init_psnr, final_psnr);

    if final_psnr >= 20.0 {
        println!("  ✅ T4 PASSED (≥20 dB, text visible)");
    } else {
        println!("  ❌ T4 FAILED (<20 dB, text likely not readable)");
    }
    println!();
}

fn init_grid_isotropic(target: &ImageBuffer<f32>, grid_size: u32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut gaussians = Vec::new();
    let num_gaussians = grid_size * grid_size;
    let gamma = 0.8;
    let width = target.width as f32;
    let height = target.height as f32;

    let sigma_base_px = gamma * ((width * height) / num_gaussians as f32).sqrt();
    let sigma_norm = (sigma_base_px / width).clamp(0.01, 0.25);

    let step_x = target.width / grid_size;
    let step_y = target.height / grid_size;

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx * step_x + step_x / 2).min(target.width - 1);
            let y = (gy * step_y + step_y / 2).min(target.height - 1);

            let position = Vector2::new(
                x as f32 / target.width as f32,
                y as f32 / target.height as f32,
            );

            let color = target.get_pixel(x, y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            gaussians.push(Gaussian2D::new(
                position,
                Euler::isotropic(sigma_norm),
                color,
                1.0,
            ));
        }
    }

    gaussians
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
