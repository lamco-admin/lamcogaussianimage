//! Per-Parameter Learning Rate Sweep
//!
//! Different Gaussian parameters have different gradient scales and sensitivities:
//! - **Position**: Small gradients (image scale), needs larger LR
//! - **Color**: Well-bounded (0-1), moderate LR works well
//! - **Scale**: Can vary widely, needs careful tuning
//! - **Opacity**: Bounded (0-1), but affects visibility significantly
//! - **Rotation**: Bounded (-π, π), often under-utilized
//!
//! This experiment tests different LR ratios to find optimal relative scaling.

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
    println!("   PER-PARAMETER LEARNING RATE SWEEP");
    println!("   Testing different LR ratios for Gaussian parameters");
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

    // LR configurations: (name, pos, color, scale, opacity, rotation)
    // Base: pos=0.0002, color=0.02, scale=0.005, opacity=0.05, rotation=0.01
    let configs: Vec<(&str, f32, f32, f32, f32, f32)> = vec![
        ("Default", 0.0002, 0.02, 0.005, 0.05, 0.01),
        ("High Position", 0.001, 0.02, 0.005, 0.05, 0.01),
        ("Low Position", 0.00005, 0.02, 0.005, 0.05, 0.01),
        ("High Color", 0.0002, 0.1, 0.005, 0.05, 0.01),
        ("Low Color", 0.0002, 0.005, 0.005, 0.05, 0.01),
        ("High Scale", 0.0002, 0.02, 0.02, 0.05, 0.01),
        ("Low Scale", 0.0002, 0.02, 0.001, 0.05, 0.01),
        ("High Opacity", 0.0002, 0.02, 0.005, 0.2, 0.01),
        ("Low Opacity", 0.0002, 0.02, 0.005, 0.01, 0.01),
        ("High Rotation", 0.0002, 0.02, 0.005, 0.05, 0.05),
        ("All High", 0.001, 0.1, 0.02, 0.2, 0.05),
        ("All Low", 0.00005, 0.005, 0.001, 0.01, 0.002),
        ("3DGS-like", 0.00016, 0.025, 0.005, 0.05, 0.001),
    ];

    let mut results = Vec::new();

    for (name, lr_pos, lr_color, lr_scale, lr_opacity, lr_rotation) in &configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("{}", name);
        println!("  pos={:.5}, color={:.3}, scale={:.4}, opacity={:.2}, rot={:.3}",
                 lr_pos, lr_color, lr_scale, lr_opacity, lr_rotation);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let mut gaussians = base_gaussians.clone();
        let start = Instant::now();

        let mut optimizer = GpuAdamOptimizer::new(&device).expect("Failed to create Adam");
        optimizer.max_iterations = iterations;
        optimizer.learning_rates.position = *lr_pos;
        optimizer.learning_rates.position_final = lr_pos * 0.1; // 10x decay
        optimizer.learning_rates.color = *lr_color;
        optimizer.learning_rates.scale = *lr_scale;
        optimizer.learning_rates.opacity = *lr_opacity;
        optimizer.learning_rates.rotation = *lr_rotation;

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
        let clean_name = name.to_lowercase().replace(" ", "_").replace("-", "_");
        let mut result = TestResult::new(&format!("param_lr_{}", clean_name), image_path);
        result.optimizer = format!("GPU Adam ({})", name);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f64() as f32;
        result.image_dimensions = (width, height);
        result.set_param("lr_position", *lr_pos as f64);
        result.set_param("lr_color", *lr_color as f64);
        result.set_param("lr_scale", *lr_scale as f64);
        result.set_param("lr_opacity", *lr_opacity as f64);
        result.set_param("lr_rotation", *lr_rotation as f64);
        result.complete();
        let _ = result.save_with_image(&final_render, RESULTS_DIR);

        println!("\n  Final: {:.2}dB (Δ={:+.2}dB) in {:.1}s\n",
                 final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

        results.push((name.to_string(), final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64()));
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    PER-PARAMETER LR SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  Starting PSNR: {:.2}dB\n", init_psnr);
    println!("  {:20} {:>12} {:>12} {:>10}", "Config", "Final PSNR", "Improvement", "Time");
    println!("  {:->20} {:->12} {:->12} {:->10}", "", "", "", "");

    for (name, final_psnr, improvement, time) in &results {
        println!("  {:20} {:>10.2}dB {:>+10.2}dB {:>8.1}s", name, final_psnr, improvement, time);
    }

    // Find best
    let best = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let default = results.iter().find(|(n, _, _, _)| n.contains("Default"));

    println!("\n  Best config: {} ({:.2}dB)", best.0, best.1);
    if let Some(d) = default {
        println!("  Improvement over default: {:+.2}dB", best.1 - d.1);
    }

    // Analysis by parameter
    println!("\n  Per-parameter analysis:");

    // Position LR analysis
    let high_pos = results.iter().find(|(n, _, _, _)| n == "High Position");
    let low_pos = results.iter().find(|(n, _, _, _)| n == "Low Position");
    if let (Some(hp), Some(lp)) = (high_pos, low_pos) {
        if hp.2 > lp.2 + 0.1 {
            println!("  → Higher position LR helps (+{:.2}dB)", hp.2 - lp.2);
        } else if lp.2 > hp.2 + 0.1 {
            println!("  → Lower position LR better (+{:.2}dB)", lp.2 - hp.2);
        }
    }

    // Color LR analysis
    let high_col = results.iter().find(|(n, _, _, _)| n == "High Color");
    let low_col = results.iter().find(|(n, _, _, _)| n == "Low Color");
    if let (Some(hc), Some(lc)) = (high_col, low_col) {
        if hc.2 > lc.2 + 0.1 {
            println!("  → Higher color LR helps (+{:.2}dB)", hc.2 - lc.2);
        } else if lc.2 > hc.2 + 0.1 {
            println!("  → Lower color LR better (+{:.2}dB)", lc.2 - hc.2);
        }
    }

    // Scale LR analysis
    let high_scale = results.iter().find(|(n, _, _, _)| n == "High Scale");
    let low_scale = results.iter().find(|(n, _, _, _)| n == "Low Scale");
    if let (Some(hs), Some(ls)) = (high_scale, low_scale) {
        if hs.2 > ls.2 + 0.1 {
            println!("  → Higher scale LR helps (+{:.2}dB)", hs.2 - ls.2);
        } else if ls.2 > hs.2 + 0.1 {
            println!("  → Lower scale LR better (+{:.2}dB)", ls.2 - hs.2);
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
