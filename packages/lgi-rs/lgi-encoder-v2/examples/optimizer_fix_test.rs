//! Optimizer Fix Test - Session 8
//!
//! CRITICAL ISSUE: Optimizer makes quality WORSE (-10 dB regression!)
//!
//! Hypothesis: Learning rates too high
//! Test: Compare current LRs vs halved LRs vs quartered LRs
//!
//! Expected: Quality should IMPROVE after optimization, not degrade

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};
use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler};
use std::path::PathBuf;

fn main() {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("          OPTIMIZER FIX TEST - Session 8 Critical Bug          ");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Current Problem:");
    println!("  Baseline (no optimization):  18 dB");
    println!("  After Adam 100 iters:         8 dB  (-10 dB WORSE!)");
    println!();
    println!("Testing Learning Rate Reduction:");
    println!("  1. Current LRs (color=0.6, pos=0.1)");
    println!("  2. Halved LRs  (color=0.3, pos=0.05)");
    println!("  3. Quartered   (color=0.15, pos=0.025)");
    println!("  4. /10         (color=0.06, pos=0.01)");
    println!();

    // Load one test image
    let kodak_dir = PathBuf::from("../../kodak-dataset");
    let image_path = kodak_dir.join("kodim02.png");

    let image = load_png(&image_path).expect("Failed to load test image");

    println!("Test Image: kodim02.png (768×512)\n");

    // Baseline (no optimization)
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");
    let baseline = encoder.initialize_gaussians(10);  // 10×10 = 100
    let baseline_psnr = compute_psnr(&image, &baseline);
    println!("BASELINE (N={}, no optimization):", baseline.len());
    println!("  PSNR = {:.2} dB\n", baseline_psnr);

    // Test 1: Current LRs (BROKEN)
    println!("[1/4] CURRENT LRs (color=0.6, pos=0.1, scale=0.1, rot=0.02)");
    let mut opt1 = OptimizerV2::default();
    opt1.max_iterations = 100;
    let mut g1 = baseline.clone();
    let loss1 = opt1.optimize(&mut g1, &image);
    let psnr1 = compute_psnr(&image, &g1);
    let gain1 = psnr1 - baseline_psnr;
    println!("  Final: PSNR = {:.2} dB ({:+.2} dB) | loss = {:.6}", psnr1, gain1, loss1);
    println!("  Result: {}\n", if gain1 > 0.0 { "✅ IMPROVED" } else { "❌ DEGRADED" });

    // Test 2: Halved LRs
    println!("[2/4] HALVED LRs (color=0.3, pos=0.05, scale=0.05, rot=0.01)");
    let mut opt2 = OptimizerV2::default();
    opt2.learning_rate_color = 0.3;
    opt2.learning_rate_position = 0.05;
    opt2.learning_rate_scale = 0.05;
    opt2.learning_rate_rotation = 0.01;
    opt2.max_iterations = 100;
    let mut g2 = baseline.clone();
    let loss2 = opt2.optimize(&mut g2, &image);
    let psnr2 = compute_psnr(&image, &g2);
    let gain2 = psnr2 - baseline_psnr;
    println!("  Final: PSNR = {:.2} dB ({:+.2} dB) | loss = {:.6}", psnr2, gain2, loss2);
    println!("  Result: {}\n", if gain2 > 0.0 { "✅ IMPROVED" } else { "❌ DEGRADED" });

    // Test 3: Quartered LRs
    println!("[3/4] QUARTERED LRs (color=0.15, pos=0.025, scale=0.025, rot=0.005)");
    let mut opt3 = OptimizerV2::default();
    opt3.learning_rate_color = 0.15;
    opt3.learning_rate_position = 0.025;
    opt3.learning_rate_scale = 0.025;
    opt3.learning_rate_rotation = 0.005;
    opt3.max_iterations = 100;
    let mut g3 = baseline.clone();
    let loss3 = opt3.optimize(&mut g3, &image);
    let psnr3 = compute_psnr(&image, &g3);
    let gain3 = psnr3 - baseline_psnr;
    println!("  Final: PSNR = {:.2} dB ({:+.2} dB) | loss = {:.6}", psnr3, gain3, loss3);
    println!("  Result: {}\n", if gain3 > 0.0 { "✅ IMPROVED" } else { "❌ DEGRADED" });

    // Test 4: Divided by 10
    println!("[4/4] /10 LRs (color=0.06, pos=0.01, scale=0.01, rot=0.002)");
    let mut opt4 = OptimizerV2::default();
    opt4.learning_rate_color = 0.06;
    opt4.learning_rate_position = 0.01;
    opt4.learning_rate_scale = 0.01;
    opt4.learning_rate_rotation = 0.002;
    opt4.max_iterations = 100;
    let mut g4 = baseline.clone();
    let loss4 = opt4.optimize(&mut g4, &image);
    let psnr4 = compute_psnr(&image, &g4);
    let gain4 = psnr4 - baseline_psnr;
    println!("  Final: PSNR = {:.2} dB ({:+.2} dB) | loss = {:.6}", psnr4, gain4, loss4);
    println!("  Result: {}\n", if gain4 > 0.0 { "✅ IMPROVED" } else { "❌ DEGRADED" });

    // Summary
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║ SUMMARY: Does Reducing LR Fix the Optimizer?                 ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ LR Setting  │ Final PSNR │ vs Baseline │ Result              ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Baseline    │   {:6.2}   │      —      │ (No optimization)   ║", baseline_psnr);
    println!("║ Current     │   {:6.2}   │   {:+6.2}   │ {}           ║", psnr1, gain1, if gain1 > 0.0 { "✅ Works" } else { "❌ Broken" });
    println!("║ Halved      │   {:6.2}   │   {:+6.2}   │ {}           ║", psnr2, gain2, if gain2 > 0.0 { "✅ Works" } else { "❌ Broken" });
    println!("║ Quartered   │   {:6.2}   │   {:+6.2}   │ {}           ║", psnr3, gain3, if gain3 > 0.0 { "✅ Works" } else { "❌ Broken" });
    println!("║ /10         │   {:6.2}   │   {:+6.2}   │ {}           ║", psnr4, gain4, if gain4 > 0.0 { "✅ Works" } else { "❌ Broken" });
    println!("╚═══════════════════════════════════════════════════════════════╝");

    println!("\n📊 ANALYSIS:\n");

    // Find best
    let best_gain = gain1.max(gain2).max(gain3).max(gain4);
    let best_psnr = psnr1.max(psnr2).max(psnr3).max(psnr4);

    if best_gain < 0.0 {
        println!("  ❌ ALL TESTS FAILED - Optimizer broken even with reduced LRs!");
        println!("     Problem is NOT just learning rate - gradients may be wrong.");
        println!("     Next: Verify gradient computation mathematically.");
    } else if gain1 > 0.0 {
        println!("  ✅ Current LRs WORK! (+{:.2} dB gain)", gain1);
        println!("     → Problem may be elsewhere (initialization, N too low?)");
    } else if gain2 > 0.0 {
        println!("  ✅ HALVED LRs FIX IT! (+{:.2} dB gain)", gain2);
        println!("     → Update optimizer_v2.rs defaults:");
        println!("        learning_rate_color = 0.3");
        println!("        learning_rate_position = 0.05");
        println!("        learning_rate_scale = 0.05");
        println!("        learning_rate_rotation = 0.01");
    } else if gain3 > 0.0 {
        println!("  ✅ QUARTERED LRs FIX IT! (+{:.2} dB gain)", gain3);
        println!("     → Update to even more conservative LRs");
    } else if gain4 > 0.0 {
        println!("  ✅ /10 LRs FIX IT! (+{:.2} dB gain)", gain4);
        println!("     → Need VERY conservative LRs");
    }

    println!("\n  Best result: {:.2} dB (gain: {:+.2} dB)", best_psnr, best_gain);

    if best_gain > 0.0 && best_gain < 1.0 {
        println!("\n  ⚠️  Quality improves, but gain is small (+{:.1} dB)", best_gain);
        println!("      May still need:");
        println!("      - More Gaussians (N=100 may be too few)");
        println!("      - Better initialization (entropy-based N)");
        println!("      - Different optimizer (L-BFGS instead of Adam)");
    }
}

fn compute_psnr(
    target: &ImageBuffer<f32>,
    gaussians: &[Gaussian2D<f32, Euler<f32>>]
) -> f32 {
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

fn load_png(path: &PathBuf) -> Result<ImageBuffer<f32>, String> {
    let img = image::open(path)
        .map_err(|e| format!("Failed to load image: {}", e))?;
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
    Ok(buffer)
}
