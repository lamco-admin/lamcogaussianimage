//! Comprehensive L-M JtJ Strategy Comparison
//!
//! Compares optimization quality across:
//! 1. CPU Adam (baseline - our best performer)
//! 2. GPU Adam
//! 3. L-M with Block-Diagonal JtJ (default)
//! 4. L-M with Sparse JtJ (3σ overlap)
//! 5. L-M with Hierarchical JtJ
//!
//! Measures: Final PSNR, convergence rate, total time

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{renderer_v2::RendererV2, test_results::TestResult, adam_optimizer::AdamOptimizer};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, color::Color4, vec::Vector2};
use lgi_gpu::{GpuAdamOptimizer, GpuLMOptimizer, BlockDiagonalJtJ, SparseJtJ, HierarchicalJtJ};
use std::time::Instant;

const TEST_IMAGE: &str = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

// Test configuration
const N_GAUSSIANS: usize = 64;  // 8x8 grid - reasonable for comparison

fn load_test_image() -> ImageBuffer<f32> {
    ImageBuffer::load(TEST_IMAGE).expect("Failed to load test image")
}

fn init_gaussians(n: usize, target: &ImageBuffer<f32>) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut gaussians = Vec::with_capacity(n);
    let grid_size = (n as f32).sqrt() as usize;

    for i in 0..grid_size {
        for j in 0..grid_size {
            if gaussians.len() >= n {
                break;
            }

            let x = (i as f32 + 0.5) / grid_size as f32;
            let y = (j as f32 + 0.5) / grid_size as f32;

            // Sample color from target image
            let px = (x * target.width as f32) as usize;
            let py = (y * target.height as f32) as usize;
            let idx = py.min(target.height as usize - 1) * target.width as usize
                    + px.min(target.width as usize - 1);
            let pixel = &target.data[idx];

            gaussians.push(Gaussian2D {
                position: Vector2 { x, y },
                shape: Euler {
                    scale_x: 0.04,
                    scale_y: 0.04,
                    rotation: 0.0,
                },
                color: Color4::new(pixel.r, pixel.g, pixel.b, 1.0),
                opacity: 1.0,
                weight: None,
            });
        }
    }

    gaussians
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0f32;
    let n = (target.width * target.height) as f32;

    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2);
        mse += (r.g - t.g).powi(2);
        mse += (r.b - t.b).powi(2);
    }
    mse /= n * 3.0;

    if mse < 1e-10 {
        100.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

/// Run CPU Adam optimizer (our proven baseline)
fn run_cpu_adam(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
) -> (f32, f32, f32, std::time::Duration) {
    println!("\n[CPU Adam Optimizer]");

    let mut gaussians = gaussians.to_vec();
    let init_rendered = RendererV2::render(&gaussians, target.width, target.height);
    let init_psnr = compute_psnr(&init_rendered, target);
    println!("  Initial PSNR: {:.2} dB", init_psnr);

    let mut optimizer = AdamOptimizer::default();

    let start = Instant::now();
    let _final_loss = optimizer.optimize(&mut gaussians, target);
    let elapsed = start.elapsed();

    let final_rendered = RendererV2::render(&gaussians, target.width, target.height);
    let final_psnr = compute_psnr(&final_rendered, target);

    println!("  Final PSNR: {:.2} dB (+{:.2} dB)", final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.2}s", elapsed.as_secs_f32());

    (init_psnr, final_psnr, final_psnr - init_psnr, elapsed)
}

/// Run GPU Adam optimizer
fn run_gpu_adam(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(f32, f32, f32, std::time::Duration), Box<dyn std::error::Error>> {
    println!("\n[GPU Adam Optimizer]");

    let mut gaussians = gaussians.to_vec();
    let init_rendered = RendererV2::render(&gaussians, target.width, target.height);
    let init_psnr = compute_psnr(&init_rendered, target);
    println!("  Initial PSNR: {:.2} dB", init_psnr);

    let optimizer = GpuAdamOptimizer::new(device)?;

    let start = Instant::now();
    let _final_loss = optimizer.optimize(device, queue, &mut gaussians, target)?;
    let elapsed = start.elapsed();

    let final_rendered = RendererV2::render(&gaussians, target.width, target.height);
    let final_psnr = compute_psnr(&final_rendered, target);

    println!("  Final PSNR: {:.2} dB (+{:.2} dB)", final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.2}s", elapsed.as_secs_f32());

    Ok((init_psnr, final_psnr, final_psnr - init_psnr, elapsed))
}

/// Run GPU L-M optimizer with Block-Diagonal JtJ
fn run_lm_block_diagonal(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(f32, f32, f32, std::time::Duration), Box<dyn std::error::Error>> {
    println!("\n[GPU L-M Optimizer - Block-Diagonal JtJ]");

    let mut gaussians = gaussians.to_vec();
    let init_rendered = RendererV2::render(&gaussians, target.width, target.height);
    let init_psnr = compute_psnr(&init_rendered, target);
    println!("  Initial PSNR: {:.2} dB", init_psnr);

    // Default uses block-diagonal
    let mut optimizer = GpuLMOptimizer::new(device)?;

    let start = Instant::now();
    let _final_loss = optimizer.optimize(device, queue, &mut gaussians, target)?;
    let elapsed = start.elapsed();

    let final_rendered = RendererV2::render(&gaussians, target.width, target.height);
    let final_psnr = compute_psnr(&final_rendered, target);

    println!("  Final PSNR: {:.2} dB (+{:.2} dB)", final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.2}s", elapsed.as_secs_f32());

    Ok((init_psnr, final_psnr, final_psnr - init_psnr, elapsed))
}

/// Run GPU L-M optimizer with Sparse JtJ (3σ overlap)
fn run_lm_sparse(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(f32, f32, f32, std::time::Duration), Box<dyn std::error::Error>> {
    println!("\n[GPU L-M Optimizer - Sparse JtJ (3σ)]");

    let mut gaussians = gaussians.to_vec();
    let init_rendered = RendererV2::render(&gaussians, target.width, target.height);
    let init_psnr = compute_psnr(&init_rendered, target);
    println!("  Initial PSNR: {:.2} dB", init_psnr);

    let strategy = SparseJtJ::new(device, 3.0)?;
    let mut optimizer = GpuLMOptimizer::with_strategy(device, Box::new(strategy))?;

    let start = Instant::now();
    let _final_loss = optimizer.optimize(device, queue, &mut gaussians, target)?;
    let elapsed = start.elapsed();

    let final_rendered = RendererV2::render(&gaussians, target.width, target.height);
    let final_psnr = compute_psnr(&final_rendered, target);

    println!("  Final PSNR: {:.2} dB (+{:.2} dB)", final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.2}s", elapsed.as_secs_f32());

    Ok((init_psnr, final_psnr, final_psnr - init_psnr, elapsed))
}

/// Run GPU L-M optimizer with Hierarchical JtJ
fn run_lm_hierarchical(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(f32, f32, f32, std::time::Duration), Box<dyn std::error::Error>> {
    println!("\n[GPU L-M Optimizer - Hierarchical JtJ]");

    let mut gaussians = gaussians.to_vec();
    let init_rendered = RendererV2::render(&gaussians, target.width, target.height);
    let init_psnr = compute_psnr(&init_rendered, target);
    println!("  Initial PSNR: {:.2} dB", init_psnr);

    let strategy = HierarchicalJtJ::new(device, 3)?;
    let mut optimizer = GpuLMOptimizer::with_strategy(device, Box::new(strategy))?;

    let start = Instant::now();
    let _final_loss = optimizer.optimize(device, queue, &mut gaussians, target)?;
    let elapsed = start.elapsed();

    let final_rendered = RendererV2::render(&gaussians, target.width, target.height);
    let final_psnr = compute_psnr(&final_rendered, target);

    println!("  Final PSNR: {:.2} dB (+{:.2} dB)", final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.2}s", elapsed.as_secs_f32());

    Ok((init_psnr, final_psnr, final_psnr - init_psnr, elapsed))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("======================================================================");
    println!("       L-M JtJ Strategy Comparison vs Adam Baseline");
    println!("======================================================================");
    println!();

    // Initialize GPU
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).map_err(|e| format!("No GPU adapter found: {:?}", e))?;

    println!("GPU: \"{}\"", adapter.get_info().name);

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor::default(),
    ))?;

    // Load image
    let target = load_test_image();
    println!("Image: {}x{}", target.width, target.height);
    println!("Gaussians: {} ({} params)", N_GAUSSIANS, N_GAUSSIANS * 10);

    // Initialize identical Gaussians for all tests
    let base_gaussians = init_gaussians(N_GAUSSIANS, &target);

    // Store results for comparison
    let mut results: Vec<(&str, f32, f32, f32, f32)> = Vec::new();  // (name, init, final, improvement, time_s)

    // 1. CPU Adam (proven baseline)
    {
        let (init, final_psnr, improvement, elapsed) = run_cpu_adam(&base_gaussians, &target);
        results.push(("CPU Adam", init, final_psnr, improvement, elapsed.as_secs_f32()));
    }

    // 2. GPU Adam (default iterations internally)
    {
        let (init, final_psnr, improvement, elapsed) = run_gpu_adam(&base_gaussians, &target, &device, &queue)?;
        results.push(("GPU Adam", init, final_psnr, improvement, elapsed.as_secs_f32()));
    }

    // 3. L-M with Block-Diagonal (fast but less accurate Hessian)
    {
        let (init, final_psnr, improvement, elapsed) = run_lm_block_diagonal(&base_gaussians, &target, &device, &queue)?;
        results.push(("L-M Block-Diag", init, final_psnr, improvement, elapsed.as_secs_f32()));
    }

    // 4. L-M with Sparse (3σ overlap)
    {
        let (init, final_psnr, improvement, elapsed) = run_lm_sparse(&base_gaussians, &target, &device, &queue)?;
        results.push(("L-M Sparse", init, final_psnr, improvement, elapsed.as_secs_f32()));
    }

    // 5. L-M with Hierarchical
    {
        let (init, final_psnr, improvement, elapsed) = run_lm_hierarchical(&base_gaussians, &target, &device, &queue)?;
        results.push(("L-M Hierarchical", init, final_psnr, improvement, elapsed.as_secs_f32()));
    }

    // Print comparison table
    println!();
    println!("======================================================================");
    println!("                         RESULTS SUMMARY");
    println!("======================================================================");
    println!();
    println!("{:<18} {:>10} {:>10} {:>12} {:>10}", "Optimizer", "Init PSNR", "Final PSNR", "Improvement", "Time");
    println!("{:-<18} {:-^10} {:-^10} {:-^12} {:-^10}", "", "", "", "", "");

    let best_improvement = results.iter().map(|r| r.3).fold(f32::NEG_INFINITY, f32::max);
    let best_psnr = results.iter().map(|r| r.2).fold(f32::NEG_INFINITY, f32::max);

    for (name, init, final_psnr, improvement, time) in &results {
        let best_marker = if *improvement == best_improvement { " *" } else { "" };
        println!("{:<18} {:>10.2} {:>10.2} {:>+11.2}{} {:>9.2}s",
                 name, init, final_psnr, improvement, best_marker, time);
    }

    println!();
    println!("* = Best improvement");
    println!();

    // Analysis
    let adam_improvement = results.iter().find(|r| r.0 == "GPU Adam").map(|r| r.3).unwrap_or(0.0);

    println!("Analysis vs GPU Adam baseline:");
    for (name, _, _, improvement, time) in &results {
        if *name != "GPU Adam" {
            let delta = improvement - adam_improvement;
            let gpu_time = results.iter().find(|r| r.0 == "GPU Adam").map(|r| r.4).unwrap_or(1.0);
            let faster = if gpu_time / time > 1.0 {
                format!("{:.1}x faster", gpu_time / time)
            } else {
                format!("{:.1}x slower", time / gpu_time)
            };
            println!("  {}: {:+.2} dB vs Adam, {}", name, delta, faster);
        }
    }

    // Save results
    let mut result = TestResult::new("lm_strategy_comparison", TEST_IMAGE);
    result.optimizer = "L-M JtJ Strategy Comparison".to_string();
    result.n_gaussians = N_GAUSSIANS;
    result.iterations = 0;  // Varies per optimizer
    result.image_dimensions = (target.width, target.height);

    for (i, (_name, init, final_psnr, improvement, time)) in results.iter().enumerate() {
        result.set_param(&format!("{}_init_psnr", i), *init as f64);
        result.set_param(&format!("{}_final_psnr", i), *final_psnr as f64);
        result.set_param(&format!("{}_improvement", i), *improvement as f64);
        result.set_param(&format!("{}_time_s", i), *time as f64);
    }

    result.final_psnr = best_psnr;
    result.improvement_db = best_improvement;
    result.complete();
    let _ = result.save_partial(RESULTS_DIR);

    println!();
    println!("Results saved to {}", RESULTS_DIR);

    Ok(())
}
