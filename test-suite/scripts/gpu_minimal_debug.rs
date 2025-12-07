//! GPU Minimal Debug Test
//! 3 Gaussians on simple gradient - compare CPU vs GPU pixel by pixel

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{renderer_v2::RendererV2, renderer_gpu::GpuRendererV2};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

fn main() {
    println!("GPU Minimal Debug - 3 Gaussians on gradient\n");

    // Create simple gradient target
    let width = 64;
    let height = 64;
    let mut target = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let value = x as f32 / width as f32;
            target.set_pixel(x, y, Color4::new(value, value, value, 1.0));
        }
    }

    // Create 3 overlapping Gaussians
    let gaussians = vec![
        Gaussian2D::new(
            Vector2::new(0.3, 0.5),
            Euler::new(0.1, 0.1, 0.0),
            Color4::new(1.0, 0.0, 0.0, 1.0),
            1.0,
        ),
        Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::new(0.1, 0.1, 0.0),
            Color4::new(0.0, 1.0, 0.0, 1.0),
            1.0,
        ),
        Gaussian2D::new(
            Vector2::new(0.7, 0.5),
            Euler::new(0.1, 0.1, 0.0),
            Color4::new(0.0, 0.0, 1.0, 1.0),
            1.0,
        ),
    ];

    println!("Rendering 3 Gaussians with CPU...");
    let rendered_cpu = RendererV2::render(&gaussians, width, height);

    println!("Rendering 3 Gaussians with GPU...");
    let mut gpu_renderer = GpuRendererV2::new_blocking();
    let rendered_gpu = gpu_renderer.render(&gaussians, width, height);

    println!("\nComparing pixel by pixel...\n");

    let mut max_diff = 0.0;
    let mut max_diff_pos = (0, 0);
    let mut diffs = Vec::new();

    for y in 0..height {
        for x in 0..width {
            if let (Some(cpu_pixel), Some(gpu_pixel)) = (rendered_cpu.get_pixel(x, y), rendered_gpu.get_pixel(x, y)) {
                let diff_r = (cpu_pixel.r - gpu_pixel.r).abs();
                let diff_g = (cpu_pixel.g - gpu_pixel.g).abs();
                let diff_b = (cpu_pixel.b - gpu_pixel.b).abs();
                let max_channel_diff = diff_r.max(diff_g).max(diff_b);

                if max_channel_diff > max_diff {
                    max_diff = max_channel_diff;
                    max_diff_pos = (x, y);
                }

                if max_channel_diff > 0.01 {
                    diffs.push((x, y, cpu_pixel.clone(), gpu_pixel.clone(), max_channel_diff));
                }
            }
        }
    }

    println!("Maximum difference: {:.6} at pixel ({}, {})", max_diff, max_diff_pos.0, max_diff_pos.1);
    println!("Pixels with >0.01 difference: {}", diffs.len());

    if diffs.len() > 0 {
        println!("\nFirst 10 significant differences:");
        for (i, (x, y, cpu, gpu, diff)) in diffs.iter().take(10).enumerate() {
            println!("  {}: ({:2},{:2}) CPU:[{:.3},{:.3},{:.3}] GPU:[{:.3},{:.3},{:.3}] diff:{:.3}",
                     i+1, x, y, cpu.r, cpu.g, cpu.b, gpu.r, gpu.g, gpu.b, diff);
        }
    }

    // Calculate PSNR
    let psnr_cpu = compute_psnr(&rendered_cpu, &target);
    let psnr_gpu = compute_psnr(&rendered_gpu, &target);

    println!("\nPSNR:");
    println!("  CPU: {:.2} dB", psnr_cpu);
    println!("  GPU: {:.2} dB", psnr_gpu);
    println!("  Diff: {:.2} dB", (psnr_cpu - psnr_gpu).abs());

    if (psnr_cpu - psnr_gpu).abs() < 0.5 {
        println!("\n✅ GPU matches CPU");
    } else {
        println!("\n❌ GPU BROKEN - investigating discrepancy");
    }
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= (rendered.width * rendered.height * 3) as f32;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
