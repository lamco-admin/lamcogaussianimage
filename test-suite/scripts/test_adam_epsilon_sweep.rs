//! Adam Epsilon Parameter Sweep
//!
//! Adam's epsilon prevents division by zero in the update step:
//! θ = θ - lr * m_hat / (√v_hat + ε)
//!
//! Default ε=1e-8 works for most cases, but:
//! - Too small: Risk of instability when v_hat is very small
//! - Too large: Reduces adaptivity (more like SGD with momentum)
//!
//! For Gaussian splatting, different parameters may benefit from different ε:
//! - Position: Gradients can be very small (large image scale)
//! - Color: Typically well-behaved
//! - Scale: Can vary widely
//!
//! Hypothesis: Slightly larger ε might improve stability for position updates.

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
    println!("   ADAM EPSILON PARAMETER SWEEP");
    println!("   Testing numerical stability parameter for Adam optimizer");
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

    // Epsilon values to test
    let epsilons: Vec<(f32, &str)> = vec![
        (1e-10, "1e-10 (very small)"),
        (1e-8, "1e-8 (default)"),
        (1e-7, "1e-7"),
        (1e-6, "1e-6"),
        (1e-5, "1e-5"),
        (1e-4, "1e-4"),
        (1e-3, "1e-3 (large)"),
    ];

    let mut results = Vec::new();

    for (epsilon, name) in &epsilons {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("ε = {}", name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let mut gaussians = base_gaussians.clone();
        let start = Instant::now();

        let mut optimizer = GpuAdamOptimizer::new(&device).expect("Failed to create Adam");
        optimizer.max_iterations = iterations;
        optimizer.epsilon = *epsilon;

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
            .replace("-", "");
        let mut result = TestResult::new(&format!("adam_epsilon_{}", clean_name), image_path);
        result.optimizer = format!("GPU Adam (ε={})", epsilon);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f64() as f32;
        result.image_dimensions = (width, height);
        result.set_param("epsilon", *epsilon as f64);
        result.complete();
        let _ = result.save_with_image(&final_render, RESULTS_DIR);

        println!("\n  Final: {:.2}dB (Δ={:+.2}dB) in {:.1}s\n",
                 final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

        results.push((name.to_string(), *epsilon, final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64()));
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    EPSILON PARAMETER SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  Starting PSNR: {:.2}dB\n", init_psnr);
    println!("  {:20} {:>12} {:>12} {:>10}", "Epsilon", "Final PSNR", "Improvement", "Time");
    println!("  {:->20} {:->12} {:->12} {:->10}", "", "", "", "");

    for (name, _, final_psnr, improvement, time) in &results {
        println!("  {:20} {:>10.2}dB {:>+10.2}dB {:>8.1}s", name, final_psnr, improvement, time);
    }

    // Find best
    let best = results.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
    let default = results.iter().find(|(n, _, _, _, _)| n.contains("default"));

    println!("\n  Best epsilon: {} ({:.2}dB)", best.0, best.2);
    if let Some(d) = default {
        println!("  Difference from default: {:+.2}dB", best.2 - d.2);
    }

    // Analysis
    println!("\n  Analysis:");

    // Check if there's a clear trend
    let small_eps: Vec<_> = results.iter().filter(|(_, e, _, _, _)| *e < 1e-7).collect();
    let large_eps: Vec<_> = results.iter().filter(|(_, e, _, _, _)| *e > 1e-5).collect();

    let small_avg: f32 = small_eps.iter().map(|(_, _, _, i, _)| i).sum::<f32>()
        / small_eps.len().max(1) as f32;
    let large_avg: f32 = large_eps.iter().map(|(_, _, _, i, _)| i).sum::<f32>()
        / large_eps.len().max(1) as f32;

    if small_avg > large_avg + 0.1 {
        println!("  → Smaller epsilon better - full adaptivity beneficial");
    } else if large_avg > small_avg + 0.1 {
        println!("  → Larger epsilon better - more stability helps");
    } else {
        println!("  → Epsilon has minimal impact in tested range");
        println!("    (default ε=1e-8 is appropriate)");
    }

    // Check for instability with very small epsilon
    let very_small = results.iter().find(|(_, e, _, _, _)| *e < 1e-9);
    if let Some((_, _, psnr, _, _)) = very_small {
        let default_psnr = default.map(|(_, _, p, _, _)| *p).unwrap_or(0.0);
        if *psnr < default_psnr - 0.5 {
            println!("  → Very small ε shows instability - avoid ε < 1e-9");
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
