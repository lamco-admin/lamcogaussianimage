//! Adam Momentum (Beta) Parameter Sweep
//!
//! Adam uses two momentum terms:
//! - beta1 (0.9 default): First moment decay - controls gradient smoothing
//! - beta2 (0.999 default): Second moment decay - controls adaptive learning rate
//!
//! Different settings may work better for Gaussian splatting:
//! - Lower beta1: More responsive to recent gradients (faster adaptation)
//! - Higher beta1: Smoother updates (more stable, slower)
//! - Lower beta2: More adaptive to gradient variance (can help noisy gradients)
//! - Higher beta2: More stable learning rates (better for steady gradients)
//!
//! Hypothesis: Gaussian splatting gradients may benefit from different momentum
//! settings than the defaults, especially given the multi-parameter nature.

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
    println!("   ADAM BETA PARAMETER SWEEP");
    println!("   Exploring momentum settings for Gaussian splatting");
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

    let base_gaussians = init_jittered_grid(width, height, n_gaussians, &target);
    let rendered = RendererV2::render(&base_gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
    println!("Initial: {} Gaussians, PSNR={:.2}dB\n", n_gaussians, init_psnr);

    // Beta configurations to test
    // (name, beta1, beta2)
    let configs: Vec<(&str, f32, f32)> = vec![
        ("Default (0.9/0.999)", 0.9, 0.999),
        ("Low β1 (0.8/0.999)", 0.8, 0.999),
        ("High β1 (0.95/0.999)", 0.95, 0.999),
        ("Low β2 (0.9/0.99)", 0.9, 0.99),
        ("High β2 (0.9/0.9999)", 0.9, 0.9999),
        ("Both Low (0.8/0.99)", 0.8, 0.99),
        ("Both High (0.95/0.9999)", 0.95, 0.9999),
        ("3DGS-like (0.9/0.99)", 0.9, 0.99),  // Common in 3D Gaussian Splatting
        ("Aggressive (0.5/0.9)", 0.5, 0.9),   // Very responsive
    ];

    let mut results = Vec::new();

    for (name, beta1, beta2) in &configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("{}", name);
        println!("  β1={}, β2={}", beta1, beta2);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let mut gaussians = base_gaussians.clone();
        let start = Instant::now();

        let mut optimizer = GpuAdamOptimizer::new(&device).expect("Failed to create Adam");
        optimizer.max_iterations = iterations;
        optimizer.beta1 = *beta1;
        optimizer.beta2 = *beta2;

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
            .replace("/", "_")
            .replace("β", "b");
        let mut result = TestResult::new(&format!("adam_beta_{}", clean_name), image_path);
        result.optimizer = format!("GPU Adam ({})", name);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f64() as f32;
        result.image_dimensions = (width, height);
        result.set_param("beta1", *beta1 as f64);
        result.set_param("beta2", *beta2 as f64);
        result.complete();
        let _ = result.save_with_image(&final_render, RESULTS_DIR);

        println!("\n  Final: {:.2}dB (Δ={:+.2}dB) in {:.1}s\n",
                 final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

        results.push((name.to_string(), *beta1, *beta2, final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64()));
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    BETA PARAMETER SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  Starting PSNR: {:.2}dB\n", init_psnr);
    println!("  {:25} {:>6} {:>8} {:>12} {:>12}", "Config", "β1", "β2", "Final PSNR", "Improvement");
    println!("  {:->25} {:->6} {:->8} {:->12} {:->12}", "", "", "", "", "");

    for (name, beta1, beta2, final_psnr, improvement, _) in &results {
        println!("  {:25} {:>6.2} {:>8.4} {:>10.2}dB {:>+10.2}dB",
                 name, beta1, beta2, final_psnr, improvement);
    }

    // Find best
    let best = results.iter().max_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap();
    let default = results.iter().find(|(n, _, _, _, _, _)| n.contains("Default"));

    println!("\n  Best config: {} ({:.2}dB)", best.0, best.3);
    if let Some(d) = default {
        println!("  Improvement over default: {:+.2}dB", best.3 - d.3);
    }

    // Analysis
    println!("\n  Analysis:");

    // Check if lower beta1 helps
    let low_beta1_avg: f32 = results.iter()
        .filter(|(_, b1, _, _, _, _)| *b1 < 0.9)
        .map(|(_, _, _, _, i, _)| i)
        .sum::<f32>() / results.iter().filter(|(_, b1, _, _, _, _)| *b1 < 0.9).count().max(1) as f32;

    let high_beta1_avg: f32 = results.iter()
        .filter(|(_, b1, _, _, _, _)| *b1 >= 0.9)
        .map(|(_, _, _, _, i, _)| i)
        .sum::<f32>() / results.iter().filter(|(_, b1, _, _, _, _)| *b1 >= 0.9).count().max(1) as f32;

    if low_beta1_avg > high_beta1_avg + 0.1 {
        println!("  → Lower β1 beneficial - more responsive to recent gradients");
    } else if high_beta1_avg > low_beta1_avg + 0.1 {
        println!("  → Higher β1 better - smoother optimization helps");
    }

    // Check if lower beta2 helps (3DGS-style)
    let low_beta2_avg: f32 = results.iter()
        .filter(|(_, _, b2, _, _, _)| *b2 < 0.999)
        .map(|(_, _, _, _, i, _)| i)
        .sum::<f32>() / results.iter().filter(|(_, _, b2, _, _, _)| *b2 < 0.999).count().max(1) as f32;

    let high_beta2_avg: f32 = results.iter()
        .filter(|(_, _, b2, _, _, _)| *b2 >= 0.999)
        .map(|(_, _, _, _, i, _)| i)
        .sum::<f32>() / results.iter().filter(|(_, _, b2, _, _, _)| *b2 >= 0.999).count().max(1) as f32;

    if low_beta2_avg > high_beta2_avg + 0.1 {
        println!("  → Lower β2 helpful - more adaptive learning rates work better");
        println!("    (3DGS-style β2=0.99 validated for Gaussian splatting)");
    } else if high_beta2_avg > low_beta2_avg + 0.1 {
        println!("  → Higher β2 better - stable learning rates preferred");
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
