//! GPU Speedup Test - Compare CPU vs GPU rendering
//! Expected: 100-1000× speedup on RTX 4060

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, renderer_gpu::GpuRendererV2};
use std::time::Instant;

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  GPU Speedup Test - CPU vs GPU Rendering                ║");
    println!("║  Expected: 100-1000× speedup on RTX 4060                 ║");
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

    println!("Testing: {}×{}", target.width, target.height);

    // Initialize Gaussians
    let encoder = EncoderV2::new(target.clone()).unwrap();
    let gaussians = encoder.initialize_gaussians_guided(40); // N=1600

    println!("Gaussians: {}\n", gaussians.len());

    // Test 1: CPU rendering (baseline)
    println!("═══════════════════════════════════════════");
    println!("CPU Rendering (10 frames)");
    println!("═══════════════════════════════════════════");

    let start_cpu = Instant::now();
    for _ in 0..10 {
        let _ = RendererV2::render(&gaussians, target.width, target.height);
    }
    let cpu_time = start_cpu.elapsed().as_secs_f32();

    println!("  Total: {:.2}s", cpu_time);
    println!("  Per frame: {:.3}s", cpu_time / 10.0);
    println!("  FPS: {:.1}", 10.0 / cpu_time);

    // Test 2: GPU rendering
    println!("\n═══════════════════════════════════════════");
    println!("GPU Rendering (10 frames)");
    println!("═══════════════════════════════════════════");

    let mut gpu_renderer = GpuRendererV2::new_blocking();

    if !gpu_renderer.has_gpu() {
        println!("  ❌ GPU not available - cannot test speedup");
        return;
    }

    println!("  ✅ GPU initialized");

    // Warmup
    let _ = gpu_renderer.render(&gaussians, target.width, target.height);

    let start_gpu = Instant::now();
    for _ in 0..10 {
        let _ = gpu_renderer.render(&gaussians, target.width, target.height);
    }
    let gpu_time = start_gpu.elapsed().as_secs_f32();

    println!("  Total: {:.2}s", gpu_time);
    println!("  Per frame: {:.3}s", gpu_time / 10.0);
    println!("  FPS: {:.1}", 10.0 / gpu_time);

    // Comparison
    println!("\n═══════════════════════════════════════════");
    println!("SPEEDUP ANALYSIS");
    println!("═══════════════════════════════════════════");

    let speedup = cpu_time / gpu_time;

    println!("  CPU time:  {:.2}s", cpu_time);
    println!("  GPU time:  {:.2}s", gpu_time);
    println!("  Speedup:   {:.1}×", speedup);

    if speedup >= 100.0 {
        println!("\n  ✅ EXCELLENT: {:.0}× speedup achieved!", speedup);
    } else if speedup >= 50.0 {
        println!("\n  ✓ GOOD: {:.0}× speedup", speedup);
    } else if speedup >= 10.0 {
        println!("\n  ⚠️  MODERATE: {:.0}× speedup (expected 100×)", speedup);
    } else {
        println!("\n  ❌ POOR: Only {:.1}× speedup", speedup);
    }

    println!("\n  Estimated optimization speedup (300 iters):");
    println!("    CPU: ~700s");
    println!("    GPU: ~{:.1}s ({:.0}× faster)", 700.0 / speedup, speedup);
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
