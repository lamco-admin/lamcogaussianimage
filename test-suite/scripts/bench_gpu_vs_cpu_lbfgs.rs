//! Benchmark GPU vs CPU L-BFGS optimization
//!
//! Compares:
//! 1. CPU L-BFGS (from lgi-core)
//! 2. GPU L-BFGS (hybrid: GPU gradients + CPU two-loop recursion)
//!
//! Key metrics: Final PSNR, convergence speed, quality parity

use lgi_core::{ImageBuffer, lbfgs::LBFGS};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, color::Color4, vec::Vector2};
use lgi_gpu::{GpuLbfgsOptimizer, LbfgsConfig};
use std::time::Instant;

const TEST_IMAGE: &str = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";

fn load_test_image() -> ImageBuffer<f32> {
    ImageBuffer::load(TEST_IMAGE).expect("Failed to load test image")
}

fn init_gaussians(n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    // Jittered grid initialization (matches Adam benchmark)
    let mut gaussians = Vec::with_capacity(n);
    let grid_size = (n as f32).sqrt() as usize;

    for i in 0..grid_size {
        for j in 0..grid_size {
            if gaussians.len() >= n {
                break;
            }

            // Add small deterministic jitter
            let jitter_x = (i as f32 * 0.7) % 1.0 * 0.02 - 0.01;
            let jitter_y = (j as f32 * 0.3) % 1.0 * 0.02 - 0.01;

            let x = (i as f32 + 0.5) / grid_size as f32 + jitter_x;
            let y = (j as f32 + 0.5) / grid_size as f32 + jitter_y;

            gaussians.push(Gaussian2D {
                position: Vector2 {
                    x: x.clamp(0.0, 1.0),
                    y: y.clamp(0.0, 1.0),
                },
                shape: Euler {
                    scale_x: 0.03,
                    scale_y: 0.03,
                    rotation: 0.0,
                },
                color: Color4::new(0.5, 0.5, 0.5, 1.0),
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

fn run_cpu_lbfgs(
    gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    max_iterations: usize,
) -> (f32, f32, std::time::Duration) {
    println!("[CPU L-BFGS]");

    // Flatten gaussians to parameter vector
    let n = gaussians.len();
    let params_per_gaussian = 7; // x, y, r, g, b, scale_x, scale_y
    let mut params: Vec<f32> = Vec::with_capacity(n * params_per_gaussian);

    for g in gaussians.iter() {
        params.push(g.position.x);
        params.push(g.position.y);
        params.push(g.color.r);
        params.push(g.color.g);
        params.push(g.color.b);
        params.push(g.shape.scale_x);
        params.push(g.shape.scale_y);
    }

    // Loss function
    let loss_fn = |p: &[f32]| -> f32 {
        // Unflatten to gaussians
        let mut temp_g = gaussians.to_vec();
        for i in 0..n {
            let base = i * params_per_gaussian;
            temp_g[i].position.x = p[base].clamp(0.0, 1.0);
            temp_g[i].position.y = p[base + 1].clamp(0.0, 1.0);
            temp_g[i].color.r = p[base + 2].clamp(0.0, 1.0);
            temp_g[i].color.g = p[base + 3].clamp(0.0, 1.0);
            temp_g[i].color.b = p[base + 4].clamp(0.0, 1.0);
            temp_g[i].shape.scale_x = p[base + 5].clamp(0.001, 0.5);
            temp_g[i].shape.scale_y = p[base + 6].clamp(0.001, 0.5);
        }

        let rendered = RendererV2::render(&temp_g, target.width, target.height);
        let mut mse = 0.0f32;
        for (r, t) in rendered.data.iter().zip(target.data.iter()) {
            mse += (r.r - t.r).powi(2);
            mse += (r.g - t.g).powi(2);
            mse += (r.b - t.b).powi(2);
        }
        mse / (target.width * target.height * 3) as f32
    };

    // Gradient function (finite differences for CPU)
    let grad_fn = |p: &[f32]| -> Vec<f32> {
        let eps = 1e-5;
        let f_x = loss_fn(p);
        let mut grad = vec![0.0f32; p.len()];

        for i in 0..p.len() {
            let mut p_plus = p.to_vec();
            p_plus[i] += eps;
            let f_plus = loss_fn(&p_plus);
            grad[i] = (f_plus - f_x) / eps;
        }

        grad
    };

    let start = Instant::now();
    let init_rendered = RendererV2::render(gaussians, target.width, target.height);
    let init_psnr = compute_psnr(&init_rendered, target);
    println!("  Initial PSNR: {:.2} dB", init_psnr);

    // Run L-BFGS
    let mut optimizer = LBFGS::new(10);
    let (optimized_params, final_loss, iterations) = optimizer.optimize(
        params,
        loss_fn,
        grad_fn,
        max_iterations,
        1e-6,
    );

    let elapsed = start.elapsed();

    // Unflatten back to gaussians
    for i in 0..n {
        let base = i * params_per_gaussian;
        gaussians[i].position.x = optimized_params[base].clamp(0.0, 1.0);
        gaussians[i].position.y = optimized_params[base + 1].clamp(0.0, 1.0);
        gaussians[i].color.r = optimized_params[base + 2].clamp(0.0, 1.0);
        gaussians[i].color.g = optimized_params[base + 3].clamp(0.0, 1.0);
        gaussians[i].color.b = optimized_params[base + 4].clamp(0.0, 1.0);
        gaussians[i].shape.scale_x = optimized_params[base + 5].clamp(0.001, 0.5);
        gaussians[i].shape.scale_y = optimized_params[base + 6].clamp(0.001, 0.5);
    }

    let final_rendered = RendererV2::render(gaussians, target.width, target.height);
    let final_psnr = compute_psnr(&final_rendered, target);

    println!("  Final loss: {:.6}", final_loss);
    println!("  Final PSNR: {:.2} dB", final_psnr);
    println!("  Iterations: {}", iterations);
    println!("  Time: {:.2}s", elapsed.as_secs_f32());

    (init_psnr, final_psnr, elapsed)
}

fn run_gpu_lbfgs(
    gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    max_iterations: usize,
) -> Result<(f32, f32, std::time::Duration), Box<dyn std::error::Error>> {
    println!("[GPU L-BFGS]");

    // Initialize wgpu
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    println!("  GPU: \"{}\"", adapter.get_info().name);

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor::default(),
    ))?;

    let start = Instant::now();

    // Create optimizer
    let mut optimizer = GpuLbfgsOptimizer::new(&device)?;
    optimizer.config = LbfgsConfig {
        max_iterations,
        history_size: 10,
        tolerance: 1e-6,
        c1: 1e-4,
        max_line_search_iters: 20,
        initial_step: 1.0,
    };

    let init_time = start.elapsed();
    println!("  GPU init time: {:.2}ms", init_time.as_secs_f64() * 1000.0);

    let init_rendered = RendererV2::render(gaussians, target.width, target.height);
    let init_psnr = compute_psnr(&init_rendered, target);
    println!("  Initial PSNR: {:.2} dB", init_psnr);

    let opt_start = Instant::now();
    let final_loss = optimizer.optimize(&device, &queue, gaussians, target)?;
    let opt_elapsed = opt_start.elapsed();

    let final_rendered = RendererV2::render(gaussians, target.width, target.height);
    let final_psnr = compute_psnr(&final_rendered, target);

    let total_elapsed = start.elapsed();

    println!("  Final loss: {:.6}", final_loss);
    println!("  Final PSNR: {:.2} dB", final_psnr);
    println!("  Optimization time: {:.2}s", opt_elapsed.as_secs_f32());
    println!("  Total time (incl init): {:.2}s", total_elapsed.as_secs_f32());

    Ok((init_psnr, final_psnr, total_elapsed))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("======================================================================");
    println!("             GPU vs CPU L-BFGS Optimization Benchmark");
    println!("======================================================================\n");

    let target = load_test_image();
    println!("Loaded image: {}x{}", target.width, target.height);

    let configs = [
        (64, 50),   // Small: 64 Gaussians, 50 iterations
        (128, 50),  // Medium: 128 Gaussians, 50 iterations
    ];

    for (n_gaussians, max_iterations) in configs {
        println!("\n------------------------------------------------------------");
        println!("Configuration: {} Gaussians, {} iterations", n_gaussians, max_iterations);
        println!("------------------------------------------------------------\n");

        // Initialize same Gaussians for both
        let gaussians_base = init_gaussians(n_gaussians);

        // CPU L-BFGS
        let mut cpu_gaussians = gaussians_base.clone();
        let (cpu_init, cpu_final, cpu_time) = run_cpu_lbfgs(&mut cpu_gaussians, &target, max_iterations);

        println!();

        // GPU L-BFGS
        let mut gpu_gaussians = gaussians_base.clone();
        match run_gpu_lbfgs(&mut gpu_gaussians, &target, max_iterations) {
            Ok((gpu_init, gpu_final, gpu_time)) => {
                println!("\n[Comparison]");
                let speedup = cpu_time.as_secs_f32() / gpu_time.as_secs_f32();
                let psnr_diff = gpu_final - cpu_final;
                println!("  Speedup: {:.1}x", speedup);
                println!("  PSNR diff: {:+.2} dB", psnr_diff);

                if psnr_diff.abs() < 0.5 {
                    println!("  Quality: MATCHED (within 0.5 dB)");
                } else if psnr_diff > 0.0 {
                    println!("  Quality: GPU BETTER by {:.2} dB", psnr_diff);
                } else {
                    println!("  Quality: GPU WORSE by {:.2} dB", -psnr_diff);
                }
            }
            Err(e) => {
                println!("GPU L-BFGS failed: {:?}", e);
            }
        }
    }

    println!("\n======================================================================");
    println!("Benchmark complete");
    println!("======================================================================");

    Ok(())
}
