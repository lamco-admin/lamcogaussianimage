//! Test Optimizer Performance at Different Scales
//!
//! Hypothesis: Different optimizers may perform differently at different Gaussian counts:
//! - Low count (64-128): Fewer interactions, cleaner gradients, L-M/L-BFGS might shine
//! - Medium count (256-512): Standard regime, Adam typically wins
//! - High count (1024+): Many overlapping Gaussians, complex landscape
//!
//! This helps identify when to use each optimizer based on target quality/count.

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
    println!("   OPTIMIZER SCALE SWEEP TEST");
    println!("   Comparing optimizers at different Gaussian counts");
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

    // Test scales
    let scales = vec![64, 128, 256, 512];

    let mut adam_results = Vec::new();
    let mut lbfgs_results = Vec::new();

    for &n_gaussians in &scales {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Testing with {} Gaussians", n_gaussians);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let base_gaussians = init_jittered_grid(width, height, n_gaussians, &target);
        let rendered = RendererV2::render(&base_gaussians, width, height);
        let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
        println!("  Initial: PSNR={:.2}dB\n", init_psnr);

        // Adam
        println!("  Adam (200 iter):");
        let mut adam_gaussians = base_gaussians.clone();
        let adam_result = run_adam(
            &mut adam_gaussians,
            &target,
            &device,
            &queue,
            image_path,
            init_psnr,
            n_gaussians,
        );
        adam_results.push((n_gaussians, adam_result));

        // L-BFGS
        println!("  L-BFGS (100 iter):");
        let mut lbfgs_gaussians = base_gaussians.clone();
        let lbfgs_result = run_lbfgs(
            &mut lbfgs_gaussians,
            &target,
            &device,
            &queue,
            image_path,
            init_psnr,
            n_gaussians,
        );
        lbfgs_results.push((n_gaussians, lbfgs_result));

        println!();
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    SCALE SWEEP SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:>8} │ {:>14} │ {:>14} │ {:>10}", "N", "Adam PSNR", "L-BFGS PSNR", "Winner");
    println!("  {:->8}─┼─{:->14}─┼─{:->14}─┼─{:->10}", "", "", "", "");

    for i in 0..scales.len() {
        let n = scales[i];
        let adam_psnr = adam_results[i].1.0;
        let lbfgs_psnr = lbfgs_results[i].1.0;
        let winner = if adam_psnr > lbfgs_psnr + 0.1 {
            "Adam"
        } else if lbfgs_psnr > adam_psnr + 0.1 {
            "L-BFGS"
        } else {
            "Tie"
        };
        println!("  {:>8} │ {:>12.2}dB │ {:>12.2}dB │ {:>10}",
                 n, adam_psnr, lbfgs_psnr, winner);
    }

    // Analysis
    println!("\n  Time comparison:");
    println!("  {:>8} │ {:>12} │ {:>12} │ {:>12}", "N", "Adam", "L-BFGS", "Ratio");
    println!("  {:->8}─┼─{:->12}─┼─{:->12}─┼─{:->12}", "", "", "", "");

    for i in 0..scales.len() {
        let n = scales[i];
        let adam_time = adam_results[i].1.2;
        let lbfgs_time = lbfgs_results[i].1.2;
        let ratio = lbfgs_time / adam_time;
        println!("  {:>8} │ {:>10.1}s │ {:>10.1}s │ {:>10.1}×",
                 n, adam_time, lbfgs_time, ratio);
    }

    // Overall analysis
    let adam_wins = adam_results.iter().zip(&lbfgs_results)
        .filter(|((_, a), (_, b))| a.0 > b.0 + 0.1).count();
    let lbfgs_wins = adam_results.iter().zip(&lbfgs_results)
        .filter(|((_, a), (_, b))| b.0 > a.0 + 0.1).count();

    println!("\n  Overall: Adam wins {} scales, L-BFGS wins {} scales",
             adam_wins, lbfgs_wins);

    // Find if L-BFGS has any regime where it clearly outperforms
    let lbfgs_sweet_spot = lbfgs_results.iter().zip(&adam_results)
        .max_by(|((_, a), (_, _)), ((_, b), (_, _))| {
            let diff_a = a.0 - adam_results.iter().find(|(n, _)| *n == scales[0]).unwrap().1.0;
            let diff_b = b.0 - adam_results.iter().find(|(n, _)| *n == scales[0]).unwrap().1.0;
            diff_a.partial_cmp(&diff_b).unwrap()
        });

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn run_adam(
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image_path: &str,
    init_psnr: f32,
    n: usize,
) -> (f32, f32, f64) {
    let width = target.width;
    let height = target.height;

    let mut result = TestResult::new(&format!("scale_sweep_adam_{}", n), image_path);
    result.optimizer = "GPU Adam".to_string();
    result.n_gaussians = n;
    result.initial_psnr = init_psnr;
    result.image_dimensions = (width, height);

    let mut optimizer = GpuAdamOptimizer::new(device).expect("Failed to create Adam");
    optimizer.max_iterations = 200;
    result.iterations = 200;

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

    println!("    Final={:.2}dB (Δ={:+.2}dB) in {:.1}s",
             final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

    (final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64())
}

fn run_lbfgs(
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image_path: &str,
    init_psnr: f32,
    n: usize,
) -> (f32, f32, f64) {
    let width = target.width;
    let height = target.height;

    let mut result = TestResult::new(&format!("scale_sweep_lbfgs_{}", n), image_path);
    result.optimizer = "GPU L-BFGS".to_string();
    result.n_gaussians = n;
    result.initial_psnr = init_psnr;
    result.image_dimensions = (width, height);

    let mut optimizer = GpuLbfgsOptimizer::new(device).expect("Failed to create L-BFGS");
    optimizer.config.max_iterations = 100;
    result.iterations = 100;
    result.set_param("history_size", optimizer.config.history_size as f64);

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

    println!("    Final={:.2}dB (Δ={:+.2}dB) in {:.1}s",
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

    // Adaptive sigma based on Gaussian count
    let base_sigma = 0.04 * (256.0 / count as f32).sqrt().clamp(0.5, 2.0);

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

        gaussians.push(Gaussian2D::new(position, Euler::isotropic(base_sigma), color, 1.0));
    }

    gaussians
}
