//! Quality Gate 1 - Validate Log-Cholesky + Geodesic EDT
//!
//! Tests whether our foundation improvements deliver expected gains:
//! - Log-Cholesky: +3-5 dB (better optimization, better quantization)
//! - Geodesic EDT: +5-10 dB on edges (prevents bleeding)
//! - Combined: +8-13 dB total
//!
//! PASS CRITERIA: ≥8 dB improvement on simple images

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    env_logger::init();

    println!("╔═══════════════════════════════════════════════════╗");
    println!("║     QUALITY GATE 1: Foundation Validation        ║");
    println!("║  Log-Cholesky + Geodesic EDT + Structure Tensor  ║");
    println!("╚═══════════════════════════════════════════════════╝");
    println!();

    let mut results = Vec::new();

    // Test 1: Solid Red
    println!("═══════════════════════════════════════════════════");
    println!("Test 1: Solid Red 256×256");
    println!("═══════════════════════════════════════════════════");
    let (psnr, improvement) = test_solid_color("red", 1.0, 0.0, 0.0, 17.0);
    results.push(("Solid Red", psnr, improvement));
    println!();

    // Test 2: Solid Green
    println!("═══════════════════════════════════════════════════");
    println!("Test 2: Solid Green 256×256");
    println!("═══════════════════════════════════════════════════");
    let (psnr, improvement) = test_solid_color("green", 0.0, 1.0, 0.0, 17.0);
    results.push(("Solid Green", psnr, improvement));
    println!();

    // Test 3: Linear Gradient
    println!("═══════════════════════════════════════════════════");
    println!("Test 3: Linear Gradient (Blue→Red) 256×256");
    println!("═══════════════════════════════════════════════════");
    let (psnr, improvement) = test_gradient_linear(16.0);
    results.push(("Linear Gradient", psnr, improvement));
    println!();

    // Test 4: Vertical Edge (Text-like)
    println!("═══════════════════════════════════════════════════");
    println!("Test 4: Vertical Edge (Text-like) 256×256");
    println!("═══════════════════════════════════════════════════");
    let (psnr, improvement) = test_vertical_edge(15.0);
    results.push(("Vertical Edge", psnr, improvement));
    println!();

    // Summary
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║              QUALITY GATE 1 RESULTS               ║");
    println!("╠═══════════════════════════════════════════════════╣");

    let mut total_improvement = 0.0;
    for (name, psnr, improvement) in &results {
        println!("║ {:20} {:6.2} dB  (baseline + {:+.2} dB)", name, psnr, improvement);
        total_improvement += improvement;
    }

    let avg_improvement = total_improvement / results.len() as f32;

    println!("╠═══════════════════════════════════════════════════╣");
    println!("║ Average Improvement: {:+.2} dB", avg_improvement);
    println!("╠═══════════════════════════════════════════════════╣");

    if avg_improvement >= 8.0 {
        println!("║ ✅ PASSED - Proceed to next phase                ║");
        println!("║    Foundation delivers expected gains!            ║");
    } else if avg_improvement >= 5.0 {
        println!("║ ⚠️  MARGINAL - Investigate and tune               ║");
        println!("║    Gains present but below target                 ║");
    } else {
        println!("║ ❌ FAILED - Debug before proceeding               ║");
        println!("║    Foundation not delivering expected gains       ║");
    }

    println!("╚═══════════════════════════════════════════════════╝");
    println!();
}

fn test_solid_color(name: &str, r: f32, g: f32, b: f32, baseline: f32) -> (f32, f32) {
    println!("Creating solid {} image...", name);

    let mut image = ImageBuffer::new(256, 256);
    for pixel in &mut image.data {
        *pixel = Color4::new(r, g, b, 1.0);
    }

    println!("Encoding with structure-tensor initialization...");
    let encoder = EncoderV2::new(image.clone()).expect("Encoder creation failed");

    // Initialize with 8×8 grid = 64 Gaussians
    let mut gaussians = encoder.initialize_gaussians(8);
    println!("  Initialized {} Gaussians", gaussians.len());

    // Optimize
    println!("Optimizing...");
    let optimizer = OptimizerV2::default();
    let final_loss = optimizer.optimize(&mut gaussians, &image);
    println!("  Final loss: {:.6}", final_loss);

    // Render
    println!("Rendering...");
    let rendered = RendererV2::render(&gaussians, 256, 256);

    // Compute PSNR
    let psnr = compute_psnr(&image, &rendered);
    let improvement = psnr - baseline;

    println!("Results:");
    println!("  Baseline (v1):  {:.2} dB", baseline);
    println!("  Current (v2):   {:.2} dB", psnr);
    println!("  Improvement:    {:+.2} dB", improvement);

    if improvement >= 3.0 {
        println!("  Status: ✅ Target met (+3 dB)");
    } else if improvement >= 1.0 {
        println!("  Status: ⚠️  Partial improvement");
    } else {
        println!("  Status: ❌ No significant improvement");
    }

    (psnr, improvement)
}

fn test_gradient_linear(baseline: f32) -> (f32, f32) {
    println!("Creating linear gradient (blue→red)...");

    let mut image = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let t = x as f32 / 255.0;
            image.set_pixel(x, y, Color4::new(t, 0.0, 1.0 - t, 1.0));
        }
    }

    println!("Encoding...");
    let encoder = EncoderV2::new(image.clone()).expect("Encoder creation failed");
    let mut gaussians = encoder.initialize_gaussians(12);  // 12×12 = 144 for gradient
    println!("  Initialized {} Gaussians", gaussians.len());

    println!("Optimizing...");
    let optimizer = OptimizerV2::default();
    let final_loss = optimizer.optimize(&mut gaussians, &image);
    println!("  Final loss: {:.6}", final_loss);

    println!("Rendering...");
    let rendered = RendererV2::render(&gaussians, 256, 256);

    let psnr = compute_psnr(&image, &rendered);
    let improvement = psnr - baseline;

    println!("Results:");
    println!("  Baseline (v1):  {:.2} dB", baseline);
    println!("  Current (v2):   {:.2} dB", psnr);
    println!("  Improvement:    {:+.2} dB", improvement);

    (psnr, improvement)
}

fn test_vertical_edge(baseline: f32) -> (f32, f32) {
    println!("Creating vertical edge (black|white)...");

    let mut image = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let color = if x < 128 { 0.0 } else { 1.0 };
            image.set_pixel(x, y, Color4::new(color, color, color, 1.0));
        }
    }

    println!("Encoding...");
    let encoder = EncoderV2::new(image.clone()).expect("Encoder creation failed");
    let mut gaussians = encoder.initialize_gaussians(10);  // 10×10
    println!("  Initialized {} Gaussians", gaussians.len());
    println!("  (Should be thin across edge, long along edge)");

    println!("Optimizing...");
    let optimizer = OptimizerV2::default();
    let final_loss = optimizer.optimize(&mut gaussians, &image);
    println!("  Final loss: {:.6}", final_loss);

    println!("Rendering...");
    let rendered = RendererV2::render(&gaussians, 256, 256);

    let psnr = compute_psnr(&image, &rendered);
    let improvement = psnr - baseline;

    println!("Results:");
    println!("  Baseline (v1):  {:.2} dB", baseline);
    println!("  Current (v2):   {:.2} dB", psnr);
    println!("  Improvement:    {:+.2} dB", improvement);

    // Check edge specifically
    let edge_psnr = compute_psnr_region(&image, &rendered, 120, 136, 0, 256);
    println!("  Edge region:    {:.2} dB", edge_psnr);

    (psnr, improvement)
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

    if mse < 1e-10 {
        100.0
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}

fn compute_psnr_region(
    original: &ImageBuffer<f32>,
    rendered: &ImageBuffer<f32>,
    x_start: u32,
    x_end: u32,
    y_start: u32,
    y_end: u32,
) -> f32 {
    let mut mse = 0.0;
    let mut count = 0.0;

    for y in y_start..y_end.min(original.height) {
        for x in x_start..x_end.min(original.width) {
            if let (Some(p1), Some(p2)) = (original.get_pixel(x, y), rendered.get_pixel(x, y)) {
                mse += (p1.r - p2.r).powi(2);
                mse += (p1.g - p2.g).powi(2);
                mse += (p1.b - p2.b).powi(2);
                count += 3.0;
            }
        }
    }

    if count > 0.0 {
        mse /= count;
    }

    if mse < 1e-10 {
        100.0
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}
