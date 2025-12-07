//! Test Optimizer Staging - Different optimizers for different phases
//!
//! Hypothesis: Different optimizers may excel at different optimization stages:
//! - Adam: Good at large-scale moves in early iterations (adaptive LR handles varied gradients)
//! - L-BFGS: Good at fine-tuning once in a good basin (uses curvature info)
//! - L-M: Good when near solution (Gauss-Newton convergence)
//!
//! This test compares:
//! 1. Adam only (200 iter)
//! 2. L-BFGS only (100 iter)
//! 3. Adam (100) → L-BFGS (50) - warm start then polish
//! 4. Adam (150) → L-BFGS (25) - longer warm start
//! 5. Adam (50) → L-BFGS (75) - quick start, longer polish

use lgi_core::ImageBuffer;
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_gpu::{GpuAdamOptimizer, GpuLbfgsOptimizer, LbfgsConfig};
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
    println!("   OPTIMIZER STAGING TEST");
    println!("   Testing if combining optimizers beats single-optimizer approaches");
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
    let base_gaussians = init_jittered_grid(width, height, n_gaussians, &target);
    let rendered = RendererV2::render(&base_gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
    println!("Base initialization: {} Gaussians, PSNR={:.2}dB\n", n_gaussians, init_psnr);

    // Test configurations
    let configs = vec![
        ("Adam only (200)", OptimizerConfig::Adam { iters: 200 }),
        ("L-BFGS only (100)", OptimizerConfig::Lbfgs { iters: 100 }),
        ("Adam(100) → L-BFGS(50)", OptimizerConfig::Staged { adam_iters: 100, lbfgs_iters: 50 }),
        ("Adam(150) → L-BFGS(25)", OptimizerConfig::Staged { adam_iters: 150, lbfgs_iters: 25 }),
        ("Adam(50) → L-BFGS(75)", OptimizerConfig::Staged { adam_iters: 50, lbfgs_iters: 75 }),
        ("Adam(75) → L-BFGS(50)", OptimizerConfig::Staged { adam_iters: 75, lbfgs_iters: 50 }),
    ];

    let mut results = Vec::new();

    for (name, config) in configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Testing: {}", name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let mut gaussians = base_gaussians.clone();
        let result = run_config(
            name,
            config,
            &mut gaussians,
            &target,
            &device,
            &queue,
            image_path,
            init_psnr,
        );
        results.push((name, result));
        println!();
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    OPTIMIZER STAGING SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  Starting PSNR: {:.2}dB\n", init_psnr);
    println!("  {:25} {:>12} {:>12} {:>10}", "Configuration", "Final PSNR", "Improvement", "Time");
    println!("  {:->25} {:->12} {:->12} {:->10}", "", "", "", "");

    for (name, (final_psnr, improvement, time)) in &results {
        println!("  {:25} {:>10.2}dB {:>+10.2}dB {:>8.1}s", name, final_psnr, improvement, time);
    }

    // Find best
    let best = results.iter().max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap()).unwrap();
    let adam_only = results.iter().find(|(n, _)| n.contains("Adam only")).unwrap();
    let lbfgs_only = results.iter().find(|(n, _)| n.contains("L-BFGS only")).unwrap();

    println!("\n  Best configuration: {} ({:.2}dB)", best.0, best.1.0);

    // Analysis
    let staging_beat_adam = results.iter()
        .filter(|(n, _)| n.contains("→"))
        .any(|(_, (psnr, _, _))| *psnr > adam_only.1.0);

    let staging_beat_lbfgs = results.iter()
        .filter(|(n, _)| n.contains("→"))
        .any(|(_, (psnr, _, _))| *psnr > lbfgs_only.1.0);

    println!("\n  Analysis:");
    if staging_beat_adam && staging_beat_lbfgs {
        println!("  ✓ Staged optimization outperforms both single optimizers!");
        println!("  → Adam warm-up + L-BFGS polish is a winning combination");
    } else if staging_beat_adam {
        println!("  ~ Staging beats Adam alone but not L-BFGS");
    } else if staging_beat_lbfgs {
        println!("  ~ Staging beats L-BFGS alone but not Adam");
    } else {
        println!("  ~ Single optimizer approaches competitive with staging");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

#[derive(Clone)]
enum OptimizerConfig {
    Adam { iters: usize },
    Lbfgs { iters: usize },
    Staged { adam_iters: usize, lbfgs_iters: usize },
}

fn run_config(
    name: &str,
    config: OptimizerConfig,
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image_path: &str,
    init_psnr: f32,
) -> (f32, f32, f64) {
    let width = target.width;
    let height = target.height;

    let clean_name = name.replace("→", "_to_").replace("(", "").replace(")", "").replace(" ", "_");
    let mut result = TestResult::new(&format!("staging_{}", clean_name), image_path);
    result.optimizer = name.to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.initial_psnr = init_psnr;

    let start = Instant::now();

    match config {
        OptimizerConfig::Adam { iters } => {
            println!("  Running Adam ({} iterations)...", iters);
            let mut optimizer = GpuAdamOptimizer::new(device).expect("Failed to create Adam");
            optimizer.max_iterations = iters;
            let _ = optimizer.optimize(device, queue, gaussians, target);
            result.iterations = iters;
        }
        OptimizerConfig::Lbfgs { iters } => {
            println!("  Running L-BFGS ({} iterations)...", iters);
            let mut optimizer = GpuLbfgsOptimizer::new(device).expect("Failed to create L-BFGS");
            optimizer.config.max_iterations = iters;
            let _ = optimizer.optimize(device, queue, gaussians, target);
            result.iterations = iters;
        }
        OptimizerConfig::Staged { adam_iters, lbfgs_iters } => {
            println!("  Stage 1: Adam ({} iterations)...", adam_iters);
            let mut adam = GpuAdamOptimizer::new(device).expect("Failed to create Adam");
            adam.max_iterations = adam_iters;
            let _ = adam.optimize(device, queue, gaussians, target);

            let mid_render = RendererV2::render(gaussians, width, height);
            let mid_psnr = lgi_core::error_metrics::compute_psnr(target, &mid_render);
            println!("    After Adam: {:.2}dB", mid_psnr);

            println!("  Stage 2: L-BFGS ({} iterations)...", lbfgs_iters);
            let mut lbfgs = GpuLbfgsOptimizer::new(device).expect("Failed to create L-BFGS");
            lbfgs.config.max_iterations = lbfgs_iters;
            let _ = lbfgs.optimize(device, queue, gaussians, target);

            result.iterations = adam_iters + lbfgs_iters;
            result.set_param("adam_iters", adam_iters as f64);
            result.set_param("lbfgs_iters", lbfgs_iters as f64);
        }
    }

    let elapsed = start.elapsed();
    let final_render = RendererV2::render(gaussians, width, height);
    let final_psnr = lgi_core::error_metrics::compute_psnr(target, &final_render);

    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f64() as f32;
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    println!("  Result: PSNR={:.2}dB (improvement: {:+.2}dB) in {:.1}s",
             final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

    (final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64())
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
