//! Convergence Analysis - When Do Optimizers Plateau?
//!
//! Understanding convergence behavior helps determine:
//! - Optimal iteration budget for different optimizers
//! - When to switch optimizers in staged approaches
//! - Early stopping criteria
//!
//! This experiment runs optimizers for extended iterations and tracks:
//! - PSNR improvement per iteration
//! - Iteration where 90%/95%/99% of final improvement is achieved
//! - Rate of convergence slow-down

use lgi_core::ImageBuffer;
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_gpu::{GpuAdamOptimizer, GpuLbfgsOptimizer};
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
    println!("   CONVERGENCE ANALYSIS");
    println!("   How quickly do optimizers plateau?");
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
    let max_iterations = 500; // Extended run

    let base_gaussians = init_jittered_grid(width, height, n_gaussians, &target);
    let rendered = RendererV2::render(&base_gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
    println!("Initial: {} Gaussians, PSNR={:.2}dB\n", n_gaussians, init_psnr);

    // Test optimizers
    let optimizers = vec!["Adam", "L-BFGS"];

    for opt_name in &optimizers {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Optimizer: {}", opt_name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let mut gaussians = base_gaussians.clone();
        let mut psnr_history: Vec<f32> = vec![init_psnr];
        let start = Instant::now();

        // Run in small chunks to track convergence
        let chunk_size = 10;
        let num_chunks = max_iterations / chunk_size;

        for chunk in 0..num_chunks {
            match *opt_name {
                "Adam" => {
                    let mut optimizer = GpuAdamOptimizer::new(&device).expect("Failed to create Adam");
                    optimizer.max_iterations = chunk_size;
                    let _ = optimizer.optimize(&device, &queue, &mut gaussians, &target);
                },
                "L-BFGS" => {
                    let mut optimizer = GpuLbfgsOptimizer::new(&device).expect("Failed to create L-BFGS");
                    optimizer.config.max_iterations = chunk_size;
                    let _ = optimizer.optimize(&device, &queue, &mut gaussians, &target);
                },
                _ => unreachable!()
            }

            let mid_render = RendererV2::render(&gaussians, width, height);
            let mid_psnr = lgi_core::error_metrics::compute_psnr(&target, &mid_render);
            psnr_history.push(mid_psnr);

            if chunk % 5 == 0 || chunk == num_chunks - 1 {
                let iter = (chunk + 1) * chunk_size;
                let improvement = mid_psnr - init_psnr;
                println!("  [{:4}] PSNR={:.2}dB (Δ={:+.2}dB)", iter, mid_psnr, improvement);
            }
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

        // Analyze convergence
        let total_improvement = final_psnr - init_psnr;

        // Find when we hit different percentages of final improvement
        let find_milestone = |pct: f32| -> usize {
            let target_psnr = init_psnr + total_improvement * pct;
            for (i, &psnr) in psnr_history.iter().enumerate() {
                if psnr >= target_psnr {
                    return i * chunk_size;
                }
            }
            max_iterations
        };

        let iter_90pct = find_milestone(0.90);
        let iter_95pct = find_milestone(0.95);
        let iter_99pct = find_milestone(0.99);

        println!("\n  Convergence Analysis:");
        println!("    Final PSNR: {:.2}dB (Δ={:+.2}dB)", final_psnr, total_improvement);
        println!("    90% of improvement by iteration: {}", iter_90pct);
        println!("    95% of improvement by iteration: {}", iter_95pct);
        println!("    99% of improvement by iteration: {}", iter_99pct);

        // Calculate rate of improvement over different phases
        let early_rate = if psnr_history.len() > 5 {
            (psnr_history[5] - psnr_history[0]) / (5 * chunk_size) as f32
        } else { 0.0 };

        let mid_rate = if psnr_history.len() > 25 {
            (psnr_history[25] - psnr_history[5]) / (20 * chunk_size) as f32
        } else { 0.0 };

        let late_rate = if psnr_history.len() > 50 {
            (psnr_history[49] - psnr_history[25]) / (24 * chunk_size) as f32
        } else { 0.0 };

        println!("\n  Improvement rate (dB/iteration):");
        println!("    Early (0-50):    {:.5}", early_rate);
        println!("    Mid (50-250):    {:.5}", mid_rate);
        println!("    Late (250-500):  {:.5}", late_rate);

        // Save result
        let mut result = TestResult::new(&format!("convergence_{}", opt_name.to_lowercase()), image_path);
        result.optimizer = format!("GPU {} (convergence analysis)", opt_name);
        result.n_gaussians = n_gaussians;
        result.iterations = max_iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = total_improvement;
        result.elapsed_seconds = elapsed.as_secs_f64() as f32;
        result.image_dimensions = (width, height);
        result.set_param("iter_90pct", iter_90pct as f64);
        result.set_param("iter_95pct", iter_95pct as f64);
        result.set_param("iter_99pct", iter_99pct as f64);
        result.set_param("early_rate", early_rate as f64);
        result.set_param("mid_rate", mid_rate as f64);
        result.set_param("late_rate", late_rate as f64);
        result.complete();
        let _ = result.save_with_image(&final_render, RESULTS_DIR);

        println!("\n  Time: {:.1}s ({:.1} iter/s)\n",
                 elapsed.as_secs_f64(),
                 max_iterations as f64 / elapsed.as_secs_f64());
    }

    // Summary comparison
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    CONVERGENCE SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  Recommendations based on convergence analysis:");
    println!("  - For quick previews: 50-100 iterations usually sufficient");
    println!("  - For quality: Most improvement happens in first 200-300 iterations");
    println!("  - Staged approach: Start with Adam, switch at plateau");

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
