//! EXP-026: Text Detection and Specialized Handling
//! Test stroke detection + thin Gaussian placement

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2, text_detection};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-026: Text Detection & Handling         ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Create text-like pattern (3 vertical strokes)
    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let is_stroke = (x >= 60 && x <= 68) ||
                           (x >= 124 && x <= 132) ||
                           (x >= 188 && x <= 196);
            let val = if is_stroke { 1.0 } else { 0.0 };
            target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    println!("Test 1: Baseline (uniform grid, N=576)");
    println!("───────────────────────────────────────");

    let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
    let mut gaussians_uniform = encoder.initialize_gaussians(24);  // 24×24

    let init_psnr_uniform = {
        let rendered = RendererV2::render(&gaussians_uniform, 256, 256);
        compute_psnr(&target, &rendered)
    };

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 150;
    optimizer.optimize(&mut gaussians_uniform, &target);

    let final_psnr_uniform = {
        let rendered = RendererV2::render(&gaussians_uniform, 256, 256);
        compute_psnr(&target, &rendered)
    };

    println!("  Uniform: {:.2} → {:.2} dB (Δ: {:+.2} dB)",
        init_psnr_uniform, final_psnr_uniform, final_psnr_uniform - init_psnr_uniform);

    println!("\nTest 2: Stroke Detection + Specialized Gaussians");
    println!("───────────────────────────────────────");

    // Detect strokes
    let tensor_field = lgi_core::StructureTensorField::compute(&target, 1.2, 0.5)
        .expect("Structure tensor failed");
    let strokes = text_detection::detect_strokes(&target, &tensor_field);

    println!("  Detected {} stroke regions", strokes.len());

    // Create stroke-specific Gaussians
    let gaussians_strokes = text_detection::create_stroke_gaussians(&strokes);

    println!("  Created {} thin stroke Gaussians", gaussians_strokes.len());

    // Add background Gaussians (sparse)
    let mut gaussians_hybrid = encoder.initialize_gaussians(12);  // 12×12 = 144 background
    gaussians_hybrid.extend(gaussians_strokes);

    println!("  Total Gaussians: {} (background) + {} (strokes) = {}",
        144, strokes.len(), gaussians_hybrid.len());

    let init_psnr_hybrid = {
        let rendered = RendererV2::render(&gaussians_hybrid, 256, 256);
        compute_psnr(&target, &rendered)
    };

    let mut opt_hybrid = OptimizerV2::default();
    opt_hybrid.max_iterations = 150;
    opt_hybrid.optimize(&mut gaussians_hybrid, &target);

    let final_psnr_hybrid = {
        let rendered = RendererV2::render(&gaussians_hybrid, 256, 256);
        compute_psnr(&target, &rendered)
    };

    println!("  Hybrid: {:.2} → {:.2} dB (Δ: {:+.2} dB)",
        init_psnr_hybrid, final_psnr_hybrid, final_psnr_hybrid - init_psnr_hybrid);

    println!("\n═══════════════════════════════════════════════");
    println!("Comparison:");
    println!("  Uniform (N=576):  {:.2} dB", final_psnr_uniform);
    println!("  Hybrid (N={}): {:.2} dB", gaussians_hybrid.len(), final_psnr_hybrid);
    println!("  Improvement:       {:+.2} dB", final_psnr_hybrid - final_psnr_uniform);

    if final_psnr_hybrid > final_psnr_uniform + 2.0 {
        println!("\n✅ Stroke detection provides significant improvement!");
    } else if final_psnr_hybrid > final_psnr_uniform {
        println!("\n✓ Minor improvement from stroke detection");
    } else {
        println!("\n⚠️  Stroke detection not better (may need tuning)");
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
