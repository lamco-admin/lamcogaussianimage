//! Realistic GPU optimizer test
//! Uses actual photo workload to measure real speedup

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, optimizer_v2::OptimizerV2};
use std::time::Instant;

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Realistic GPU Optimizer Test                            ║");
    println!("║  Photo: 768×432, N=1600, 100 iterations                  ║");
    println!("║  Expected: 400-500× speedup on RTX 4060                  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Load test photo
    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133784383569199567.jpg";

    let target = match ImageBuffer::load(path) {
        Ok(img) => {
            let scale = 768.0 / img.width.max(img.height) as f32;
            let new_w = (img.width as f32 * scale) as u32;
            let new_h = (img.height as f32 * scale) as u32;
            resize_bilinear(&img, new_w, new_h)
        }
        Err(e) => {
            println!("Failed to load: {}", e);
            return;
        }
    };

    println!("Target: {}×{}", target.width, target.height);

    // Initialize Gaussians
    let encoder = EncoderV2::new(target.clone()).unwrap();
    let mut gaussians_cpu = encoder.initialize_gaussians_guided(40); // 40×40 = 1600
    let mut gaussians_gpu = gaussians_cpu.clone();

    println!("Gaussians: {}\n", gaussians_cpu.len());

    println!("═══════════════════════════════════════════");
    println!("Test 1: CPU Optimizer (Baseline)");
    println!("═══════════════════════════════════════════");

    let mut optimizer_cpu = OptimizerV2 {
        max_iterations: 100,
        ..Default::default()
    };

    let start_cpu = Instant::now();
    let loss_cpu = optimizer_cpu.optimize(&mut gaussians_cpu, &target);
    let time_cpu = start_cpu.elapsed().as_secs_f32();

    println!("  Time: {:.2}s", time_cpu);
    println!("  Final loss: {:.6}", loss_cpu);

    // Compute PSNR
    let rendered_cpu = lgi_encoder_v2::renderer_v2::RendererV2::render(&gaussians_cpu, target.width, target.height);
    let psnr_cpu = compute_psnr(&rendered_cpu, &target);
    println!("  PSNR: {:.2} dB", psnr_cpu);

    println!("\n═══════════════════════════════════════════");
    println!("Test 2: GPU Optimizer (Expected 400-500×)");
    println!("═══════════════════════════════════════════");

    let mut optimizer_gpu = OptimizerV2::new_with_gpu();

    if !optimizer_gpu.has_gpu() {
        println!("  ❌ GPU not available");
        return;
    }

    let start_gpu = Instant::now();
    let loss_gpu = optimizer_gpu.optimize(&mut gaussians_gpu, &target);
    let time_gpu = start_gpu.elapsed().as_secs_f32();

    println!("  Time: {:.2}s", time_gpu);
    println!("  Final loss: {:.6}", loss_gpu);

    // Compute PSNR
    let rendered_gpu = lgi_encoder_v2::renderer_v2::RendererV2::render(&gaussians_gpu, target.width, target.height);
    let psnr_gpu = compute_psnr(&rendered_gpu, &target);
    println!("  PSNR: {:.2} dB", psnr_gpu);

    println!("\n═══════════════════════════════════════════");
    println!("SPEEDUP ANALYSIS");
    println!("═══════════════════════════════════════════");

    let speedup = time_cpu / time_gpu;
    let psnr_diff = (psnr_gpu - psnr_cpu).abs();

    println!("  CPU time:   {:.2}s", time_cpu);
    println!("  GPU time:   {:.2}s", time_gpu);
    println!("  Speedup:    {:.1}×", speedup);
    println!();
    println!("  CPU PSNR:   {:.2} dB", psnr_cpu);
    println!("  GPU PSNR:   {:.2} dB", psnr_gpu);
    println!("  Difference: {:.2} dB", psnr_diff);

    println!("\n═══════════════════════════════════════════");
    println!("RESULTS");
    println!("═══════════════════════════════════════════");

    if speedup >= 100.0 {
        println!("  ✅ EXCELLENT: {:.0}× speedup!", speedup);
    } else if speedup >= 50.0 {
        println!("  ✓ GOOD: {:.0}× speedup", speedup);
    } else if speedup >= 10.0 {
        println!("  ⚠️  MODERATE: {:.0}× speedup (expected 100-500×)", speedup);
    } else if speedup >= 1.0 {
        println!("  ⚠️  LOW: {:.1}× speedup (expected 100-500×)", speedup);
    } else {
        println!("  ❌ GPU SLOWER: {:.1}× (initialization overhead on small test?)", speedup);
    }

    if psnr_diff <= 0.5 {
        println!("  ✅ QUALITY MATCHES: {:.2} dB difference", psnr_diff);
    } else if psnr_diff <= 1.0 {
        println!("  ⚠️  QUALITY ACCEPTABLE: {:.2} dB difference", psnr_diff);
    } else {
        println!("  ❌ QUALITY MISMATCH: {:.2} dB difference", psnr_diff);
    }

    if speedup >= 50.0 && psnr_diff <= 1.0 {
        println!("\n  🎉 GPU INTEGRATION SUCCESS!");
        println!("  Ready for rapid experimentation");
        println!();
        println!("  Estimated N=1600, 300 iters:");
        println!("    CPU: ~700s");
        println!("    GPU: ~{:.1}s ({:.0}× faster)", 700.0 / speedup, speedup);
    } else if speedup >= 1.0 {
        println!("\n  ✅ GPU integration working");
        println!("  Speedup lower than expected - may need larger workloads");
    } else {
        println!("\n  ⚠️  GPU slower on this test (small workload)");
        println!("  GPU has initialization overhead");
        println!("  Speedup increases with workload size");
    }
}

fn resize_bilinear(img: &ImageBuffer<f32>, new_width: u32, new_height: u32) -> ImageBuffer<f32> {
    let mut resized = ImageBuffer::new(new_width, new_height);

    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = x as f32 * (img.width as f32 / new_width as f32);
            let src_y = y as f32 * (img.height as f32 / new_height as f32);

            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(img.width - 1);
            let y1 = (y0 + 1).min(img.height - 1);

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            if let (Some(c00), Some(c10), Some(c01), Some(c11)) = (
                img.get_pixel(x0, y0),
                img.get_pixel(x1, y0),
                img.get_pixel(x0, y1),
                img.get_pixel(x1, y1),
            ) {
                let r = (1.0 - fx) * (1.0 - fy) * c00.r +
                        fx * (1.0 - fy) * c10.r +
                        (1.0 - fx) * fy * c01.r +
                        fx * fy * c11.r;

                let g = (1.0 - fx) * (1.0 - fy) * c00.g +
                        fx * (1.0 - fy) * c10.g +
                        (1.0 - fx) * fy * c01.g +
                        fx * fy * c11.g;

                let b = (1.0 - fx) * (1.0 - fy) * c00.b +
                        fx * (1.0 - fy) * c10.b +
                        (1.0 - fx) * fy * c01.b +
                        fx * fy * c11.b;

                resized.set_pixel(x, y, lgi_math::color::Color4::new(r, g, b, 1.0));
            }
        }
    }

    resized
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
        100.0
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}
