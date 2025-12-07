//! Gaussian Count Sweep
//!
//! Tests how quality scales with number of Gaussians.
//! This helps determine optimal count for different quality targets.
//!
//! Key questions:
//! - How does PSNR scale with Gaussian count?
//! - Is there a point of diminishing returns?
//! - What's the relationship between count and convergence speed?

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
    println!("   GAUSSIAN COUNT SWEEP");
    println!("   Testing how quality scales with number of Gaussians");
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

    println!("Image: kodim03 ({}x{}) = {} pixels\n", width, height, width * height);

    // Gaussian counts to test (powers of 2 and some in-between)
    let counts: Vec<usize> = vec![
        32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048
    ];

    let iterations = 200;
    let mut results = Vec::new();

    for n_gaussians in &counts {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Gaussians: {} ({:.4}% of pixels)", n_gaussians,
                 100.0 * (*n_gaussians as f64) / ((width * height) as f64));
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let gaussians = init_jittered_grid(width, height, *n_gaussians, &target);
        let rendered = RendererV2::render(&gaussians, width, height);
        let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
        println!("  Initial PSNR: {:.2}dB", init_psnr);

        let mut gaussians = gaussians;
        let start = Instant::now();

        let mut optimizer = GpuAdamOptimizer::new(&device).expect("Failed to create Adam");
        optimizer.max_iterations = iterations;

        // Run with progress reporting
        let mut last_psnr = init_psnr;
        let checkpoints = [50, 100, 150, 200];
        for (i, &checkpoint) in checkpoints.iter().enumerate() {
            let iters_to_run = if i == 0 { checkpoint } else { checkpoint - checkpoints[i - 1] };
            optimizer.max_iterations = iters_to_run;
            let _ = optimizer.optimize(&device, &queue, &mut gaussians, &target);

            let mid_render = RendererV2::render(&gaussians, width, height);
            let mid_psnr = lgi_core::error_metrics::compute_psnr(&target, &mid_render);

            let direction = if mid_psnr > last_psnr { "↑" } else { "↓" };
            println!("  [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                     checkpoint, iterations, mid_psnr, direction, mid_psnr - last_psnr);
            last_psnr = mid_psnr;
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

        // Calculate bits per Gaussian (rough estimate)
        // Each Gaussian: 2 position + 3 color + 2 scale + 1 rotation + 1 opacity = 9 floats = 36 bytes = 288 bits
        let bits_per_gaussian = 288;
        let total_bits = *n_gaussians * bits_per_gaussian;
        let bpp = total_bits as f64 / (width * height) as f64;

        // Save result
        let mut result = TestResult::new(&format!("gaussian_count_{}", n_gaussians), image_path);
        result.optimizer = format!("GPU Adam (n={})", n_gaussians);
        result.n_gaussians = *n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f64() as f32;
        result.image_dimensions = (width, height);
        result.set_param("gaussian_count", *n_gaussians as f64);
        result.set_param("bits_per_pixel", bpp);
        result.set_param("compression_ratio", (width * height * 24) as f64 / total_bits as f64);
        result.complete();
        let _ = result.save_with_image(&final_render, RESULTS_DIR);

        println!("\n  Final: {:.2}dB (Δ={:+.2}dB) @ {:.3} bpp in {:.1}s\n",
                 final_psnr, final_psnr - init_psnr, bpp, elapsed.as_secs_f64());

        results.push((
            *n_gaussians,
            init_psnr,
            final_psnr,
            final_psnr - init_psnr,
            bpp,
            elapsed.as_secs_f64()
        ));
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    GAUSSIAN COUNT SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:>8} {:>10} {:>10} {:>12} {:>10} {:>10}",
             "Count", "Init PSNR", "Final", "Improvement", "BPP", "Time");
    println!("  {:->8} {:->10} {:->10} {:->12} {:->10} {:->10}", "", "", "", "", "", "");

    for (count, init_psnr, final_psnr, improvement, bpp, time) in &results {
        println!("  {:>8} {:>8.2}dB {:>8.2}dB {:>+10.2}dB {:>9.3} {:>8.1}s",
                 count, init_psnr, final_psnr, improvement, bpp, time);
    }

    // Analysis
    println!("\n  Rate-Distortion Analysis:");
    println!("  {:>8} {:>12} {:>14}",
             "Count", "Final PSNR", "dB per 0.1bpp");
    println!("  {:->8} {:->12} {:->14}", "", "", "");

    for i in 0..results.len() {
        let (count, _, final_psnr, _, bpp, _) = results[i];
        let efficiency = if bpp > 0.0 { final_psnr as f64 / (bpp * 10.0) } else { 0.0 };
        println!("  {:>8} {:>10.2}dB {:>12.2}",
                 count, final_psnr, efficiency);
    }

    // Find knee in the curve
    println!("\n  Diminishing Returns Analysis:");
    if results.len() >= 2 {
        for i in 1..results.len() {
            let (count_prev, _, psnr_prev, _, _, _) = results[i - 1];
            let (count_curr, _, psnr_curr, _, _, _) = results[i];
            let count_ratio = count_curr as f64 / count_prev as f64;
            let psnr_gain = psnr_curr - psnr_prev;
            let gain_per_double = if count_ratio > 0.0 {
                psnr_gain as f64 / count_ratio.log2()
            } else { 0.0 };

            if psnr_gain > 0.0 {
                println!("  {} → {}: +{:.2}dB ({:.2}dB per 2x Gaussians)",
                         count_prev, count_curr, psnr_gain, gain_per_double);
            }
        }
    }

    // Recommend optimal counts for different quality targets
    println!("\n  Quality Target Recommendations:");
    let targets = [20.0, 22.0, 24.0, 26.0];
    for target in &targets {
        let best = results.iter()
            .filter(|(_, _, psnr, _, _, _)| *psnr >= *target)
            .min_by(|a, b| a.0.cmp(&b.0));

        if let Some((count, _, psnr, _, bpp, _)) = best {
            println!("  Target ≥{:.0}dB: {} Gaussians ({:.2}dB @ {:.3}bpp)",
                     target, count, psnr, bpp);
        } else {
            println!("  Target ≥{:.0}dB: Not achieved (need more Gaussians)", target);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn init_jittered_grid(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    let cols = (count as f32).sqrt().ceil() as u32;
    let rows = (count as f32 / cols as f32).ceil() as u32;
    let cell_w = width as f32 / cols as f32;
    let cell_h = height as f32 / rows as f32;

    // Scale initial Gaussian size based on cell coverage
    let init_scale = (0.5 * (cell_w.min(cell_h) / width.max(height) as f32)).max(0.01).min(0.1);

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

        gaussians.push(Gaussian2D::new(position, Euler::isotropic(init_scale), color, 1.0));
    }

    gaussians
}
