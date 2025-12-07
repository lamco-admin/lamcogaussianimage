//! Test GPU-accelerated optimizer
//! Verifies quality matches CPU and measures speedup

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::{EncoderV2, optimizer_v2::OptimizerV2};
use std::time::Instant;

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  GPU Optimizer Integration Test                          ║");
    println!("║  Expected: 454× speedup, quality matches CPU             ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Create simple test target (solid red)
    let width = 256;
    let height = 256;
    let mut target = ImageBuffer::new(width, height);
    for pixel in &mut target.data {
        pixel.r = 1.0;
        pixel.g = 0.0;
        pixel.b = 0.0;
    }

    println!("Test: Solid red 256×256, N=64, 50 iterations\n");

    // Initialize Gaussians
    let encoder = EncoderV2::new(target.clone()).unwrap();
    let mut gaussians_cpu = encoder.initialize_gaussians(8); // 8×8 = 64
    let mut gaussians_gpu = gaussians_cpu.clone();

    println!("═══════════════════════════════════════════");
    println!("Test 1: CPU Optimizer (Baseline)");
    println!("═══════════════════════════════════════════");

    let mut optimizer_cpu = OptimizerV2 {
        max_iterations: 50,
        ..Default::default()
    };

    let start_cpu = Instant::now();
    let loss_cpu = optimizer_cpu.optimize(&mut gaussians_cpu, &target);
    let time_cpu = start_cpu.elapsed().as_secs_f32();

    println!("  Time: {:.3}s", time_cpu);
    println!("  Final loss: {:.6}", loss_cpu);

    // Compute PSNR
    let rendered_cpu = lgi_encoder_v2::renderer_v2::RendererV2::render(&gaussians_cpu, width, height);
    let psnr_cpu = compute_psnr(&rendered_cpu, &target);
    println!("  PSNR: {:.2} dB", psnr_cpu);

    println!("\n═══════════════════════════════════════════");
    println!("Test 2: GPU Optimizer (454× Expected)");
    println!("═══════════════════════════════════════════");

    let mut optimizer_gpu = OptimizerV2::new_with_gpu();

    if !optimizer_gpu.has_gpu() {
        println!("  ❌ GPU not available - cannot test");
        println!("  Ensure Vulkan is available and GPU drivers are installed");
        return;
    }

    let start_gpu = Instant::now();
    let loss_gpu = optimizer_gpu.optimize(&mut gaussians_gpu, &target);
    let time_gpu = start_gpu.elapsed().as_secs_f32();

    println!("  Time: {:.3}s", time_gpu);
    println!("  Final loss: {:.6}", loss_gpu);

    // Compute PSNR
    let rendered_gpu = lgi_encoder_v2::renderer_v2::RendererV2::render(&gaussians_gpu, width, height);
    let psnr_gpu = compute_psnr(&rendered_gpu, &target);
    println!("  PSNR: {:.2} dB", psnr_gpu);

    println!("\n═══════════════════════════════════════════");
    println!("COMPARISON");
    println!("═══════════════════════════════════════════");

    let speedup = time_cpu / time_gpu;
    let psnr_diff = (psnr_gpu - psnr_cpu).abs();

    println!("  CPU time:  {:.3}s", time_cpu);
    println!("  GPU time:  {:.3}s", time_gpu);
    println!("  Speedup:   {:.1}×", speedup);
    println!();
    println!("  CPU PSNR:  {:.2} dB", psnr_cpu);
    println!("  GPU PSNR:  {:.2} dB", psnr_gpu);
    println!("  Diff:      {:.2} dB", psnr_diff);

    println!("\n═══════════════════════════════════════════");
    println!("RESULTS");
    println!("═══════════════════════════════════════════");

    if speedup >= 100.0 {
        println!("  ✅ SPEEDUP: {:.0}× - EXCELLENT!", speedup);
    } else if speedup >= 50.0 {
        println!("  ✓ SPEEDUP: {:.0}× - GOOD", speedup);
    } else if speedup >= 10.0 {
        println!("  ⚠️  SPEEDUP: {:.0}× - MODERATE (expected 100×+)", speedup);
    } else {
        println!("  ❌ SPEEDUP: {:.1}× - LOW", speedup);
    }

    if psnr_diff <= 0.5 {
        println!("  ✅ QUALITY: {:.2} dB difference - MATCHES!", psnr_diff);
    } else if psnr_diff <= 1.0 {
        println!("  ⚠️  QUALITY: {:.2} dB difference - ACCEPTABLE", psnr_diff);
    } else {
        println!("  ❌ QUALITY: {:.2} dB difference - MISMATCH", psnr_diff);
    }

    if speedup >= 50.0 && psnr_diff <= 1.0 {
        println!("\n  🎉 GPU INTEGRATION SUCCESS!");
        println!("  Ready for rapid experimentation ({}× faster)", speedup as i32);
    }
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2);
        mse += (r.g - t.g).powi(2);
        mse += (r.b - t.b).powi(2);
    }
    mse /= (rendered.width * rendered.height * 3) as f32;

    if mse < 1e-10 {
        100.0  // Perfect
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}
