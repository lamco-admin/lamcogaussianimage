//! GPU Optimizer Comparison Test
//!
//! Compares all three GPU optimizers on the same initialization:
//! 1. GPU Adam - First-order adaptive learning rates
//! 2. GPU L-BFGS - Quasi-Newton with limited memory
//! 3. GPU L-M - Levenberg-Marquardt (Gauss-Newton + regularization)
//!
//! This test uses identical initial conditions to ensure fair comparison.

use lgi_core::ImageBuffer;
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_gpu::{GpuAdamOptimizer, GpuLbfgsOptimizer, GpuLMOptimizer, LbfgsConfig, LMConfig};
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
    println!("   GPU OPTIMIZER COMPARISON TEST");
    println!("   Comparing Adam, L-BFGS, and L-M on identical initialization");
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

    // Load image
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

    // Create identical initialization (we'll clone from this)
    let base_gaussians = init_jittered_grid(width, height, n_gaussians, &target);
    let rendered = RendererV2::render(&base_gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
    println!("Base initialization: {} Gaussians, PSNR={:.2}dB\n", n_gaussians, init_psnr);

    // Test 1: GPU Adam
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: GPU ADAM - Adaptive learning rate optimizer");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let adam_result = {
        let mut gaussians = base_gaussians.clone();
        run_adam_optimization(
            "gpu_adam",
            &mut gaussians,
            &target,
            &device,
            &queue,
            image_path,
            init_psnr,
        )
    };

    // Test 2: GPU L-BFGS
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: GPU L-BFGS - Quasi-Newton with limited memory");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let lbfgs_result = {
        let mut gaussians = base_gaussians.clone();
        run_lbfgs_optimization(
            "gpu_lbfgs",
            &mut gaussians,
            &target,
            &device,
            &queue,
            image_path,
            init_psnr,
        )
    };

    // Test 3: GPU L-M
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: GPU L-M - Levenberg-Marquardt (Gauss-Newton)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let lm_result = {
        let mut gaussians = base_gaussians.clone();
        run_lm_optimization(
            "gpu_lm",
            &mut gaussians,
            &target,
            &device,
            &queue,
            image_path,
            init_psnr,
        )
    };

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    GPU OPTIMIZER COMPARISON SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  Starting PSNR: {:.2}dB (identical for all)\n", init_psnr);
    println!("  {:20} {:>12} {:>12} {:>10}", "Optimizer", "Final PSNR", "Improvement", "Time");
    println!("  {:->20} {:->12} {:->12} {:->10}", "", "", "", "");

    for (name, (final_psnr, improvement, time)) in [
        ("GPU Adam", adam_result),
        ("GPU L-BFGS", lbfgs_result),
        ("GPU L-M", lm_result),
    ] {
        println!("  {:20} {:>10.2}dB {:>+10.2}dB {:>8.1}s", name, final_psnr, improvement, time);
    }

    // Find best
    let results = [
        ("Adam", adam_result.0),
        ("L-BFGS", lbfgs_result.0),
        ("L-M", lm_result.0),
    ];
    let best = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();

    println!("\n  Best optimizer: {} ({:.2}dB)", best.0, best.1);

    // Analysis
    println!("\n  Analysis:");
    if adam_result.0 > lbfgs_result.0 && adam_result.0 > lm_result.0 {
        println!("  → Adam outperforms quasi-Newton methods on this problem");
        println!("  → First-order methods with per-parameter LR work well for Gaussian splatting");
    } else if lbfgs_result.0 > adam_result.0 {
        println!("  → L-BFGS beats Adam - good Hessian approximation benefits convergence");
    }

    if lm_result.0 < adam_result.0 - 1.0 {
        println!("  → L-M significantly underperforms - block-diagonal Hessian approximation insufficient");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn run_adam_optimization(
    name: &str,
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image_path: &str,
    init_psnr: f32,
) -> (f32, f32, f64) {
    let width = target.width;
    let height = target.height;
    let n_gaussians = gaussians.len();

    let mut result = TestResult::new(&format!("optimizer_comparison_{}", name), image_path);
    result.optimizer = "GPU Adam".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.initial_psnr = init_psnr;

    println!("  Creating GPU Adam optimizer...");
    let mut optimizer = GpuAdamOptimizer::new(device).expect("Failed to create GPU Adam optimizer");
    optimizer.max_iterations = 200;
    result.iterations = optimizer.max_iterations;

    println!("  Running optimization ({} iterations)...", optimizer.max_iterations);
    let start = Instant::now();
    let _ = optimizer.optimize(device, queue, gaussians, target);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(gaussians, width, height);
    let final_psnr = lgi_core::error_metrics::compute_psnr(target, &final_render);

    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f64() as f32;
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    println!("  Result: PSNR={:.2}dB (improvement: {:+.2}dB)", final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.1}s", elapsed.as_secs_f64());

    (final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64())
}

fn run_lbfgs_optimization(
    name: &str,
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image_path: &str,
    init_psnr: f32,
) -> (f32, f32, f64) {
    let width = target.width;
    let height = target.height;
    let n_gaussians = gaussians.len();

    let mut result = TestResult::new(&format!("optimizer_comparison_{}", name), image_path);
    result.optimizer = "GPU L-BFGS".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.initial_psnr = init_psnr;

    println!("  Creating GPU L-BFGS optimizer...");
    let mut optimizer = GpuLbfgsOptimizer::new(device).expect("Failed to create GPU L-BFGS optimizer");
    optimizer.config = LbfgsConfig {
        max_iterations: 100,  // L-BFGS typically needs fewer iterations
        history_size: 10,
        tolerance: 1e-6,
        c1: 1e-4,
        max_line_search_iters: 20,
        initial_step: 1.0,
    };
    result.iterations = optimizer.config.max_iterations;
    result.set_param("history_size", optimizer.config.history_size as f64);

    println!("  Running optimization ({} iterations, m={})...",
             optimizer.config.max_iterations, optimizer.config.history_size);
    let start = Instant::now();
    let _ = optimizer.optimize(device, queue, gaussians, target);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(gaussians, width, height);
    let final_psnr = lgi_core::error_metrics::compute_psnr(target, &final_render);

    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f64() as f32;
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    println!("  Result: PSNR={:.2}dB (improvement: {:+.2}dB)", final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.1}s", elapsed.as_secs_f64());

    (final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64())
}

fn run_lm_optimization(
    name: &str,
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image_path: &str,
    init_psnr: f32,
) -> (f32, f32, f64) {
    let width = target.width;
    let height = target.height;
    let n_gaussians = gaussians.len();

    let mut result = TestResult::new(&format!("optimizer_comparison_{}", name), image_path);
    result.optimizer = "GPU L-M (Block-diagonal)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.initial_psnr = init_psnr;

    println!("  Creating GPU L-M optimizer...");
    let mut optimizer = GpuLMOptimizer::new(device).expect("Failed to create GPU L-M optimizer");
    // Note: L-M config is set via optimizer.config if needed
    result.iterations = 50;  // L-M default

    println!("  Running optimization (max 50 iterations)...");
    let start = Instant::now();
    let _ = optimizer.optimize(device, queue, gaussians, target);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(gaussians, width, height);
    let final_psnr = lgi_core::error_metrics::compute_psnr(target, &final_render);

    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f64() as f32;
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    println!("  Result: PSNR={:.2}dB (improvement: {:+.2}dB)", final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.1}s", elapsed.as_secs_f64());

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
