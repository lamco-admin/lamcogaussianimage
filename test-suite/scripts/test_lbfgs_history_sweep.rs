//! L-BFGS History Size Sweep
//!
//! L-BFGS maintains a history of gradient pairs to approximate the inverse Hessian.
//! The history size (m) controls memory vs accuracy tradeoff:
//!
//! - Small m (3-5): Faster iterations, less memory, may miss curvature info
//! - Medium m (10-20): Typical sweet spot
//! - Large m (30-50): More accurate Hessian but diminishing returns
//!
//! Hypothesis: For Gaussian splatting with many parameters, larger history
//! might help capture complex parameter interactions.

use lgi_core::ImageBuffer;
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_gpu::GpuLbfgsOptimizer;
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
    println!("   L-BFGS HISTORY SIZE SWEEP");
    println!("   How history size affects convergence quality");
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
    let iterations = 100;

    let base_gaussians = init_jittered_grid(width, height, n_gaussians, &target);
    let rendered = RendererV2::render(&base_gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
    println!("Initial: {} Gaussians, PSNR={:.2}dB\n", n_gaussians, init_psnr);

    // History sizes to test
    let history_sizes = vec![3, 5, 10, 15, 20, 30, 50];

    let mut results = Vec::new();

    for &history_size in &history_sizes {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("History Size: {}", history_size);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let mut gaussians = base_gaussians.clone();
        let start = Instant::now();

        let mut optimizer = GpuLbfgsOptimizer::new(&device).expect("Failed to create L-BFGS");
        optimizer.config.history_size = history_size;
        optimizer.config.max_iterations = iterations;

        // Run optimization with progress reporting
        let mut last_psnr = init_psnr;
        for phase in 0..5 {
            optimizer.config.max_iterations = 20;
            let _ = optimizer.optimize(&device, &queue, &mut gaussians, &target);

            let mid_render = RendererV2::render(&gaussians, width, height);
            let mid_psnr = lgi_core::error_metrics::compute_psnr(&target, &mid_render);

            let direction = if mid_psnr > last_psnr { "↑" } else { "↓" };
            println!("  [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                     (phase + 1) * 20, iterations, mid_psnr, direction, mid_psnr - last_psnr);
            last_psnr = mid_psnr;
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

        // Save result
        let mut result = TestResult::new(&format!("lbfgs_history_{}", history_size), image_path);
        result.optimizer = format!("GPU L-BFGS (m={})", history_size);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f64() as f32;
        result.image_dimensions = (width, height);
        result.set_param("history_size", history_size as f64);
        result.complete();
        let _ = result.save_with_image(&final_render, RESULTS_DIR);

        println!("\n  Final: {:.2}dB (Δ={:+.2}dB) in {:.1}s\n",
                 final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

        results.push((history_size, final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64()));
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    HISTORY SIZE SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  Starting PSNR: {:.2}dB\n", init_psnr);
    println!("  {:>8} {:>12} {:>12} {:>10}", "History", "Final PSNR", "Improvement", "Time");
    println!("  {:->8} {:->12} {:->12} {:->10}", "", "", "", "");

    for (history, final_psnr, improvement, time) in &results {
        println!("  {:>8} {:>10.2}dB {:>+10.2}dB {:>8.1}s", history, final_psnr, improvement, time);
    }

    // Find best
    let best = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let default = results.iter().find(|(h, _, _, _)| *h == 10);

    println!("\n  Best history size: {} ({:.2}dB)", best.0, best.1);
    if let Some(d) = default {
        println!("  Improvement over default (m=10): {:+.2}dB", best.1 - d.1);
    }

    // Analyze trend
    println!("\n  Analysis:");
    let small_avg: f32 = results.iter().filter(|(h, _, _, _)| *h <= 5)
        .map(|(_, _, i, _)| i).sum::<f32>() / 2.0;
    let large_avg: f32 = results.iter().filter(|(h, _, _, _)| *h >= 30)
        .map(|(_, _, i, _)| i).sum::<f32>() / 2.0;

    if large_avg > small_avg + 0.1 {
        println!("  → Larger history beneficial for this problem");
        println!("    (larger: {:+.2}dB vs smaller: {:+.2}dB)", large_avg, small_avg);
    } else if small_avg > large_avg + 0.1 {
        println!("  → Smaller history sufficient - diminishing returns from larger m");
    } else {
        println!("  → History size has minimal impact in tested range");
    }

    // Time analysis
    let time_per_psnr: Vec<_> = results.iter()
        .map(|(h, _, i, t)| (*h, *t / (*i).max(0.01) as f64))
        .collect();

    let most_efficient = time_per_psnr.iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if let Some((h, _)) = most_efficient {
        println!("  → Most time-efficient: m={}", h);
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

        gaussians.push(Gaussian2D::new(position, Euler::isotropic(0.04), color, 1.0));
    }

    gaussians
}
