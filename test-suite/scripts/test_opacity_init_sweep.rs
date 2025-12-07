//! Initial Opacity Value Sweep
//!
//! Gaussian initial opacity affects blending behavior:
//! - **Too low (0.1)**: Gaussians nearly invisible, need many iterations to increase
//! - **Mid (0.5)**: Semi-transparent, allows gradual refinement
//! - **High (0.9-1.0)**: Opaque, may cause early saturation
//!
//! Hypothesis: Starting with moderate opacity allows both:
//! - Easy blending/compositing to fix errors
//! - Quick visibility of optimization effects

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
    println!("   INITIAL OPACITY SWEEP");
    println!("   Finding optimal starting opacity for Gaussians");
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

    // Opacity values to test
    let opacities: Vec<(f32, &str)> = vec![
        (0.1, "0.1 (very low)"),
        (0.3, "0.3 (low)"),
        (0.5, "0.5 (medium)"),
        (0.7, "0.7 (high-medium)"),
        (0.9, "0.9 (high)"),
        (1.0, "1.0 (full)"),
    ];

    let mut results = Vec::new();

    for (opacity, name) in &opacities {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Initial Opacity: {}", name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let gaussians = init_with_opacity(width, height, n_gaussians, &target, *opacity);
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
            .replace("-", "_")
            .replace(".", "p");
        let mut result = TestResult::new(&format!("init_opacity_{}", clean_name), image_path);
        result.optimizer = format!("GPU Adam (opacity={})", opacity);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f64() as f32;
        result.image_dimensions = (width, height);
        result.set_param("init_opacity", *opacity as f64);
        result.complete();
        let _ = result.save_with_image(&final_render, RESULTS_DIR);

        println!("\n  Final: {:.2}dB (Δ={:+.2}dB from init) in {:.1}s\n",
                 final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

        results.push((name.to_string(), *opacity, init_psnr, final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64()));
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    INITIAL OPACITY SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:20} {:>10} {:>10} {:>12} {:>10}",
             "Config", "Init PSNR", "Final", "Improvement", "Time");
    println!("  {:->20} {:->10} {:->10} {:->12} {:->10}", "", "", "", "", "");

    for (name, _, init_psnr, final_psnr, improvement, time) in &results {
        println!("  {:20} {:>8.2}dB {:>8.2}dB {:>+10.2}dB {:>8.1}s",
                 name, init_psnr, final_psnr, improvement, time);
    }

    // Find best final PSNR
    let best = results.iter().max_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap();
    let default = results.iter().find(|(_, o, _, _, _, _)| (*o - 1.0).abs() < 0.01);

    println!("\n  Best config: {} ({:.2}dB final)", best.0, best.3);
    if let Some(d) = default {
        println!("  Improvement over full opacity: {:+.2}dB", best.3 - d.3);
    }

    // Analysis
    println!("\n  Analysis:");

    // Check trend
    let low_opacity: Vec<_> = results.iter().filter(|(_, o, _, _, _, _)| *o < 0.5).collect();
    let high_opacity: Vec<_> = results.iter().filter(|(_, o, _, _, _, _)| *o >= 0.7).collect();

    if !low_opacity.is_empty() && !high_opacity.is_empty() {
        let low_avg: f32 = low_opacity.iter().map(|(_, _, _, f, _, _)| f).sum::<f32>()
            / low_opacity.len() as f32;
        let high_avg: f32 = high_opacity.iter().map(|(_, _, _, f, _, _)| f).sum::<f32>()
            / high_opacity.len() as f32;

        if low_avg > high_avg + 0.1 {
            println!("  → Lower initial opacity converges better");
            println!("    (allows more blending/composition flexibility)");
        } else if high_avg > low_avg + 0.1 {
            println!("  → Higher initial opacity converges better");
            println!("    (immediate visibility accelerates optimization)");
        } else {
            println!("  → Opacity has moderate impact - mid-range values work well");
        }
    }

    // Check if starting quality correlates with final quality
    let best_init = results.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
    println!("  → Best initial PSNR: {} ({:.2}dB)", best_init.0, best_init.2);
    if best_init.3 >= best.3 - 0.1 {
        println!("  → Better initial quality correlates with better final quality");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn init_with_opacity(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    opacity: f32,
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

        gaussians.push(Gaussian2D::new(position, Euler::isotropic(0.04), color, opacity));
    }

    gaussians
}
