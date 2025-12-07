//! JtJ Strategy Sweep for L-M Optimization
//!
//! The Levenberg-Marquardt optimizer can use different strategies for computing
//! the J^T J matrix (Gauss-Newton Hessian approximation):
//!
//! 1. Block-Diagonal: Only diagonal blocks (fast, ignores interactions)
//! 2. Full Dense: All cross-Gaussian interactions (slow, accurate)
//! 3. Sparse: Only overlapping Gaussian pairs (middle ground)
//! 4. Hierarchical: Multi-scale sparse passes
//!
//! Hypothesis: Different strategies may excel for different scenarios:
//! - Block-diagonal: Good when Gaussians don't overlap much
//! - Sparse: Best balance when there's moderate overlap
//! - Dense/Hierarchical: Better for highly overlapping cases
//!
//! This test compares strategies to find the best quality/performance trade-off.

use lgi_core::ImageBuffer;
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_gpu::{GpuLMOptimizer, JtJStrategyRegistry};
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
    println!("   JtJ STRATEGY SWEEP FOR L-M OPTIMIZATION");
    println!("   Comparing different Hessian approximation strategies");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // List available strategies
    println!("Available JtJ strategies:");
    for (name, desc, captures_interactions, complexity) in JtJStrategyRegistry::list_strategies() {
        println!("  - {:15} {} (interactions: {}, {})",
                 name, desc, captures_interactions, complexity);
    }
    println!();

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

    // Test with different Gaussian counts to see how strategies scale
    let test_configs = vec![
        (128, "Low density (128 Gaussians) - less overlap"),
        (256, "Standard density (256 Gaussians)"),
    ];

    // Strategies to test (skip matrix-free as it's placeholder)
    let strategies_to_test = vec![
        "block-diagonal",
        "sparse",
        "full-dense",
        "hierarchical",
    ];

    let mut all_results: Vec<(usize, String, f32, f32, f64)> = Vec::new();

    for (n_gaussians, config_desc) in &test_configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("{}", config_desc);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // Create base initialization
        let base_gaussians = init_jittered_grid(width, height, *n_gaussians, &target);
        let rendered = RendererV2::render(&base_gaussians, width, height);
        let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
        println!("  Initial PSNR: {:.2}dB\n", init_psnr);

        for strategy_name in &strategies_to_test {
            println!("  Testing {} strategy...", strategy_name);

            // Create L-M optimizer with specified strategy
            let mut gaussians = base_gaussians.clone();
            let start = Instant::now();

            // Create optimizer and set strategy
            match run_lm_with_strategy(
                &device,
                &queue,
                &mut gaussians,
                &target,
                strategy_name,
            ) {
                Ok(()) => {
                    let elapsed = start.elapsed();
                    let final_render = RendererV2::render(&gaussians, width, height);
                    let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

                    let mut result = TestResult::new(
                        &format!("jtj_{}_{}", strategy_name.replace("-", "_"), n_gaussians),
                        image_path
                    );
                    result.optimizer = format!("GPU L-M ({})", strategy_name);
                    result.n_gaussians = *n_gaussians;
                    result.iterations = 50;
                    result.initial_psnr = init_psnr;
                    result.final_psnr = final_psnr;
                    result.improvement_db = final_psnr - init_psnr;
                    result.elapsed_seconds = elapsed.as_secs_f64() as f32;
                    result.image_dimensions = (width, height);
                    result.set_param("jtj_strategy", strategy_name.to_string());
                    result.complete();
                    let _ = result.save_with_image(&final_render, RESULTS_DIR);

                    println!("    Result: {:.2}dB (Δ={:+.2}dB) in {:.1}s",
                             final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

                    all_results.push((*n_gaussians, strategy_name.to_string(),
                                     final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64()));
                }
                Err(e) => {
                    println!("    FAILED: {}", e);
                }
            }
        }
        println!();
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    JtJ STRATEGY SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    for (n_gaussians, _) in &test_configs {
        println!("  {} Gaussians:", n_gaussians);
        println!("  {:15} {:>12} {:>12} {:>10}", "Strategy", "Final PSNR", "Improvement", "Time");
        println!("  {:->15} {:->12} {:->12} {:->10}", "", "", "", "");

        let config_results: Vec<_> = all_results.iter()
            .filter(|(n, _, _, _, _)| *n == *n_gaussians)
            .collect();

        for (_, strategy, final_psnr, improvement, time) in &config_results {
            println!("  {:15} {:>10.2}dB {:>+10.2}dB {:>8.1}s",
                     strategy, final_psnr, improvement, time);
        }

        // Find best for this config
        if let Some(best) = config_results.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()) {
            println!("  Best: {} ({:.2}dB)", best.1, best.2);
        }
        println!();
    }

    // Analysis
    println!("  Analysis:");

    // Check if capturing interactions helps
    let block_diag_results: Vec<_> = all_results.iter()
        .filter(|(_, s, _, _, _)| s == "block-diagonal")
        .collect();
    let interaction_results: Vec<_> = all_results.iter()
        .filter(|(_, s, _, _, _)| s != "block-diagonal")
        .collect();

    let block_avg: f32 = if !block_diag_results.is_empty() {
        block_diag_results.iter().map(|(_, _, _, imp, _)| imp).sum::<f32>()
            / block_diag_results.len() as f32
    } else { 0.0 };

    let interaction_avg: f32 = if !interaction_results.is_empty() {
        interaction_results.iter().map(|(_, _, _, imp, _)| imp).sum::<f32>()
            / interaction_results.len() as f32
    } else { 0.0 };

    if interaction_avg > block_avg + 0.1 {
        println!("  → Capturing cross-Gaussian interactions improves results");
        println!("    (interaction-aware: {:+.2}dB vs block-diagonal: {:+.2}dB)",
                 interaction_avg, block_avg);
    } else {
        println!("  → Block-diagonal competitive with interaction-aware strategies");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn run_lm_with_strategy(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    strategy_name: &str,
) -> Result<(), String> {
    // Create strategy
    let strategy = JtJStrategyRegistry::create(device, strategy_name)
        .map_err(|e| format!("Failed to create strategy: {:?}", e))?;

    // Create optimizer with strategy
    let mut optimizer = GpuLMOptimizer::with_strategy(device, strategy)
        .map_err(|e| format!("Failed to create L-M: {:?}", e))?;

    // Run optimization
    let _ = optimizer.optimize(device, queue, gaussians, target);
    Ok(())
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
