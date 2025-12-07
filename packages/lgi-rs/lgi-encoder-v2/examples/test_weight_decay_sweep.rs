//! Initial Scale Sensitivity Sweep
//!
//! Gaussian initialization scale affects optimization trajectory:
//! - **Too small**: Gaussians need to grow significantly, slow convergence
//! - **Too large**: Gaussians overlap heavily, hard to differentiate
//! - **Just right**: Matches image feature scale, fast convergence
//!
//! This experiment tests different initial scale values to find the sweet spot.
//! Also compares isotropic vs anisotropic initialization.

use lgi_core::ImageBuffer;
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_gpu::GpuAdamOptimizer;
use lgi_math::{
    gaussian::Gaussian2D,
    parameterization::Euler,
    vec::Vector2,
    color::Color4,
};
use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("   INITIAL SCALE SENSITIVITY SWEEP");
    println!("   Finding optimal initial Gaussian scale");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Initialize GPU
    println!("Initializing GPU...");
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).expect("Failed to find GPU adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("LGI Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    )).expect("Failed to create GPU device");

    println!("  GPU: {}\n", adapter.get_info().name);

    let image_path = "test-data/kodak-dataset/kodim03.png";
    let full_path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), image_path);

    let img = image::open(&full_path).expect("Failed to load image").to_rgb32f();
    let width = img.width();
    let height = img.height();

    let mut target = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            target.set_pixel(x, y, Color4::new(pixel[0], pixel[1], pixel[2], 1.0));
        }
    }

    println!("Image: kodim03 ({}x{})\n", width, height);

    let n_gaussians = 256;
    let iterations = 200;

    // Scale configurations to test
    // (name, scale_x, scale_y) - in normalized coordinates (0-1)
    let configs: Vec<(&str, f32, f32)> = vec![
        ("Tiny (0.01)", 0.01, 0.01),
        ("Small (0.02)", 0.02, 0.02),
        ("Default (0.04)", 0.04, 0.04),
        ("Medium (0.06)", 0.06, 0.06),
        ("Large (0.08)", 0.08, 0.08),
        ("Very Large (0.12)", 0.12, 0.12),
        ("Huge (0.2)", 0.2, 0.2),
        // Anisotropic variations
        ("Wide (0.08x0.02)", 0.08, 0.02),
        ("Tall (0.02x0.08)", 0.02, 0.08),
        ("Grid-matched", 0.0625, 0.0625), // 1/16 for 256 Gaussians on 768x512
    ];

    let mut results = Vec::new();

    for (name, scale_x, scale_y) in &configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Scale: {}", name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let gaussians = init_with_scale(width, height, n_gaussians, &target, *scale_x, *scale_y);
        let rendered = RendererV2::render(&gaussians, width, height);
        let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
        println!("  Initial PSNR: {:.2}dB", init_psnr);

        let mut gaussians = gaussians;
        let start = Instant::now();

        let mut optimizer = GpuAdamOptimizer::new(&device).expect("Failed to create Adam");
        optimizer.max_iterations = iterations;

        // Run with progress reporting
        let mut last_psnr = init_psnr;
        for phase in 0..4 {
            optimizer.max_iterations = 50;
            let _ = optimizer.optimize(&device, &queue, &mut gaussians, &target);

            let mid_render = RendererV2::render(&gaussians, width, height);
            let mid_psnr = lgi_core::error_metrics::compute_psnr(&target, &mid_render);

            let direction = if mid_psnr > last_psnr { "↑" } else { "↓" };
            println!("  [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                     (phase + 1) * 50, iterations, mid_psnr, direction, mid_psnr - last_psnr);
            last_psnr = mid_psnr;
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

        // Save result
        let clean_name = name.to_lowercase()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("x", "_");
        let mut result = TestResult::new(&format!("init_scale_{}", clean_name), image_path);
        result.optimizer = format!("GPU Adam (init={})", name);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f64() as f32;
        result.image_dimensions = (width, height);
        result.set_param("init_scale_x", *scale_x as f64);
        result.set_param("init_scale_y", *scale_y as f64);
        result.complete();
        let _ = result.save_with_image(&final_render, RESULTS_DIR);

        println!("\n  Final: {:.2}dB (Δ={:+.2}dB from init) in {:.1}s\n",
                 final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

        results.push((name.to_string(), init_psnr, final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64()));
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    INITIAL SCALE SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:20} {:>10} {:>12} {:>12} {:>10}", "Config", "Init PSNR", "Final PSNR", "Improvement", "Time");
    println!("  {:->20} {:->10} {:->12} {:->12} {:->10}", "", "", "", "", "");

    for (name, init_psnr, final_psnr, improvement, time) in &results {
        println!("  {:20} {:>8.2}dB {:>10.2}dB {:>+10.2}dB {:>8.1}s",
                 name, init_psnr, final_psnr, improvement, time);
    }

    // Find best final PSNR
    let best = results.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
    let default = results.iter().find(|(n, _, _, _, _)| n.contains("Default"));

    println!("\n  Best config: {} ({:.2}dB final)", best.0, best.2);
    if let Some(d) = default {
        println!("  Improvement over default: {:+.2}dB", best.2 - d.2);
    }

    // Analysis
    println!("\n  Analysis:");

    // Check if better init helps final quality
    let best_init = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!("  → Best initial PSNR: {} ({:.2}dB)", best_init.0, best_init.1);

    // Correlation between init and final?
    let small_scales: Vec<_> = results.iter().filter(|(n, _, _, _, _)|
        n.contains("Tiny") || n.contains("Small")).collect();
    let large_scales: Vec<_> = results.iter().filter(|(n, _, _, _, _)|
        n.contains("Large") || n.contains("Huge")).collect();

    if !small_scales.is_empty() && !large_scales.is_empty() {
        let small_avg: f32 = small_scales.iter().map(|(_, _, f, _, _)| f).sum::<f32>()
            / small_scales.len() as f32;
        let large_avg: f32 = large_scales.iter().map(|(_, _, f, _, _)| f).sum::<f32>()
            / large_scales.len() as f32;

        if small_avg > large_avg + 0.2 {
            println!("  → Smaller initial scales converge to better quality");
        } else if large_avg > small_avg + 0.2 {
            println!("  → Larger initial scales converge to better quality");
        } else {
            println!("  → Scale has moderate impact on final quality");
        }
    }

    // Check anisotropic vs isotropic
    let aniso: Vec<_> = results.iter().filter(|(n, _, _, _, _)|
        n.contains("Wide") || n.contains("Tall")).collect();
    let iso_default = results.iter().find(|(n, _, _, _, _)| n.contains("Default"));

    if !aniso.is_empty() {
        if let Some(iso) = iso_default {
            let aniso_best: f32 = aniso.iter().map(|(_, _, f, _, _)| *f).fold(f32::MIN, f32::max);
            if aniso_best > iso.2 + 0.1 {
                println!("  → Anisotropic initialization can help (+{:.2}dB)", aniso_best - iso.2);
            }
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn init_with_scale(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    scale_x: f32,
    scale_y: f32,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    let cols = (count as f32).sqrt().ceil() as u32;
    let rows = (count as f32 / cols as f32).ceil() as u32;
    let cell_w = width as f32 / cols as f32;
    let cell_h = height as f32 / rows as f32;

    for i in 0..count {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;

        let jitter_x: f32 = rng.gen_range(-0.3..0.3);
        let jitter_y: f32 = rng.gen_range(-0.3..0.3);
        let px = ((col as f32 + 0.5 + jitter_x) * cell_w) as i32;
        let py = ((row as f32 + 0.5 + jitter_y) * cell_h) as i32;
        let px = px.clamp(0, width as i32 - 1) as u32;
        let py = py.clamp(0, height as i32 - 1) as u32;

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        let shape = Euler {
            scale_x,
            scale_y,
            rotation: 0.0,
        };
        gaussians.push(Gaussian2D::new(position, shape, color, 1.0));
    }

    gaussians
}
