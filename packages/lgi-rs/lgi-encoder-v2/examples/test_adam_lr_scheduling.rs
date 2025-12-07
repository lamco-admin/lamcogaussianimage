//! Adam Learning Rate Scheduling Test
//!
//! Adam's default behavior uses constant learning rates throughout optimization.
//! This test explores different learning rate schedules:
//!
//! 1. **Constant**: Baseline, no decay
//! 2. **Step decay**: Reduce LR by factor every N iterations
//! 3. **Exponential decay**: LR = LR_0 * exp(-λt)
//! 4. **Cosine annealing**: LR follows cosine curve from max to min
//! 5. **Warmup + decay**: Start low, ramp up, then decay
//!
//! Hypothesis: Learning rate scheduling could help:
//! - Larger initial steps for fast progress
//! - Smaller steps near convergence for fine-tuning
//! - Warmup to stabilize early gradients

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
    println!("   ADAM LEARNING RATE SCHEDULING TEST");
    println!("   Comparing different LR decay strategies");
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
    let total_iterations = 200;
    let phase_iterations = 50; // Run in phases to apply scheduling

    let base_gaussians = init_jittered_grid(width, height, n_gaussians, &target);
    let rendered = RendererV2::render(&base_gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
    println!("Initial: {} Gaussians, PSNR={:.2}dB\n", n_gaussians, init_psnr);

    // LR schedule configurations
    let schedules: Vec<(&str, Box<dyn Fn(usize, usize) -> f32>)> = vec![
        ("Constant", Box::new(|_iter, _total| 1.0)),
        ("Step 0.5@100", Box::new(|iter, _total| if iter >= 100 { 0.5 } else { 1.0 })),
        ("Step 0.25@100", Box::new(|iter, _total| if iter >= 100 { 0.25 } else { 1.0 })),
        ("Exponential", Box::new(|iter, total| (-3.0 * iter as f32 / total as f32).exp())),
        ("Cosine Annealing", Box::new(|iter, total| {
            0.5 * (1.0 + (std::f32::consts::PI * iter as f32 / total as f32).cos())
        })),
        ("Linear Decay", Box::new(|iter, total| 1.0 - 0.9 * (iter as f32 / total as f32))),
        ("Warmup+Decay", Box::new(|iter, total| {
            let warmup = 20;
            if iter < warmup {
                iter as f32 / warmup as f32
            } else {
                1.0 - 0.8 * ((iter - warmup) as f32 / (total - warmup) as f32)
            }
        })),
    ];

    let mut results = Vec::new();

    for (schedule_name, schedule_fn) in &schedules {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Schedule: {}", schedule_name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let mut gaussians = base_gaussians.clone();
        let start = Instant::now();

        // Run in phases to apply LR schedule
        let num_phases = total_iterations / phase_iterations;
        for phase in 0..num_phases {
            let phase_start = phase * phase_iterations;
            let lr_multiplier = schedule_fn(phase_start, total_iterations);

            let mut optimizer = GpuAdamOptimizer::new(&device).expect("Failed to create Adam");
            optimizer.max_iterations = phase_iterations;

            // Scale the base learning rates
            optimizer.learning_rates.position *= lr_multiplier;
            optimizer.learning_rates.color *= lr_multiplier;
            optimizer.learning_rates.scale *= lr_multiplier;
            optimizer.learning_rates.opacity *= lr_multiplier;
            optimizer.learning_rates.rotation *= lr_multiplier;

            let _ = optimizer.optimize(&device, &queue, &mut gaussians, &target);

            // Report progress
            let mid_render = RendererV2::render(&gaussians, width, height);
            let mid_psnr = lgi_core::error_metrics::compute_psnr(&target, &mid_render);
            println!("  Phase {} (iter {}-{}): LR×{:.2}, PSNR={:.2}dB",
                     phase + 1, phase_start, phase_start + phase_iterations, lr_multiplier, mid_psnr);
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

        // Save result
        let clean_name = schedule_name.to_lowercase().replace(" ", "_").replace("@", "_").replace("+", "_");
        let mut result = TestResult::new(&format!("lr_schedule_{}", clean_name), image_path);
        result.optimizer = format!("GPU Adam ({})", schedule_name);
        result.n_gaussians = n_gaussians;
        result.iterations = total_iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f64() as f32;
        result.image_dimensions = (width, height);
        result.set_param("lr_schedule", schedule_name.to_string());
        result.complete();
        let _ = result.save_with_image(&final_render, RESULTS_DIR);

        println!("\n  Final: {:.2}dB (Δ={:+.2}dB) in {:.1}s\n",
                 final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

        results.push((schedule_name.to_string(), final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64()));
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    LR SCHEDULING SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  Starting PSNR: {:.2}dB\n", init_psnr);
    println!("  {:20} {:>12} {:>12} {:>10}", "Schedule", "Final PSNR", "Improvement", "Time");
    println!("  {:->20} {:->12} {:->12} {:->10}", "", "", "", "");

    for (name, final_psnr, improvement, time) in &results {
        println!("  {:20} {:>10.2}dB {:>+10.2}dB {:>8.1}s", name, final_psnr, improvement, time);
    }

    // Find best
    let best = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let constant = results.iter().find(|(n, _, _, _)| n == "Constant").unwrap();

    println!("\n  Best schedule: {} ({:.2}dB)", best.0, best.1);
    println!("  Improvement over constant: {:+.2}dB", best.1 - constant.1);

    // Analysis
    println!("\n  Analysis:");
    if best.1 > constant.1 + 0.1 {
        println!("  → Learning rate scheduling provides meaningful improvement!");
        println!("  → {} achieves {:.2}dB better than constant LR", best.0, best.1 - constant.1);
    } else {
        println!("  → Constant LR competitive - Adam's adaptivity handles scheduling implicitly");
    }

    // Check if warmup helps
    let warmup = results.iter().find(|(n, _, _, _)| n.contains("Warmup"));
    if let Some(w) = warmup {
        if w.1 > constant.1 + 0.1 {
            println!("  → Warmup phase beneficial - stabilizes early optimization");
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
