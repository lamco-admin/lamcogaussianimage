//! L-BFGS GPU/CPU Consistency Test
//!
//! This test verifies that GPU and CPU L-BFGS implementations produce
//! consistent results when given identical:
//! - Initial parameters
//! - Gradient computation method
//! - Parameterization (10 params per Gaussian)
//!
//! Key check: Both should take the same steps and converge to similar solutions.

use lgi_core::{ImageBuffer, lbfgs::LBFGS};
use lgi_encoder_v2::{renderer_v2::RendererV2, test_results::TestResult};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, color::Color4, vec::Vector2};
use lgi_gpu::{GpuLbfgsOptimizer, LbfgsConfig};
use std::time::Instant;

const TEST_IMAGE: &str = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

const PARAMS_PER_GAUSSIAN: usize = 10;  // x, y, scale_x, scale_y, rotation, r, g, b, a, opacity

fn load_test_image() -> ImageBuffer<f32> {
    ImageBuffer::load(TEST_IMAGE).expect("Failed to load test image")
}

fn init_gaussians(n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut gaussians = Vec::with_capacity(n);
    let grid_size = (n as f32).sqrt() as usize;

    for i in 0..grid_size {
        for j in 0..grid_size {
            if gaussians.len() >= n {
                break;
            }

            let x = (i as f32 + 0.5) / grid_size as f32;
            let y = (j as f32 + 0.5) / grid_size as f32;

            gaussians.push(Gaussian2D {
                position: Vector2 { x, y },
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

/// Flatten gaussians to 10-param vector (matching GPU format)
fn flatten_gaussians(gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Vec<f32> {
    let mut params = Vec::with_capacity(gaussians.len() * PARAMS_PER_GAUSSIAN);
    for g in gaussians {
        params.push(g.position.x);
        params.push(g.position.y);
        params.push(g.shape.scale_x);
        params.push(g.shape.scale_y);
        params.push(g.shape.rotation);
        params.push(g.color.r);
        params.push(g.color.g);
        params.push(g.color.b);
        params.push(g.color.a);
        params.push(g.opacity);
    }
    params
}

/// Unflatten 10-param vector to gaussians
fn unflatten_gaussians(params: &[f32], n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut gaussians = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * PARAMS_PER_GAUSSIAN;
        gaussians.push(Gaussian2D {
            position: Vector2 {
                x: params[base].clamp(0.0, 1.0),
                y: params[base + 1].clamp(0.0, 1.0),
            },
            shape: Euler {
                scale_x: params[base + 2].clamp(0.001, 0.5),
                scale_y: params[base + 3].clamp(0.001, 0.5),
                rotation: params[base + 4],
            },
            color: Color4::new(
                params[base + 5].clamp(0.0, 1.0),
                params[base + 6].clamp(0.0, 1.0),
                params[base + 7].clamp(0.0, 1.0),
                params[base + 8].clamp(0.0, 1.0),
            ),
            opacity: params[base + 9].clamp(0.01, 1.0),
            weight: None,
        });
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

/// Run CPU L-BFGS with 10-param format (matching GPU)
fn run_cpu_lbfgs_10param(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    max_iterations: usize,
) -> (Vec<Gaussian2D<f32, Euler<f32>>>, f32, f32, std::time::Duration, Vec<(usize, f32, f32)>) {
    println!("\n[CPU L-BFGS - 10 param]");

    let n = gaussians.len();
    let mut params = flatten_gaussians(gaussians);
    let mut history: Vec<(usize, f32, f32)> = Vec::new();  // (iter, loss, psnr)

    let init_gaussians = unflatten_gaussians(&params, n);
    let init_rendered = RendererV2::render(&init_gaussians, target.width, target.height);
    let init_psnr = compute_psnr(&init_rendered, target);
    println!("  Initial PSNR: {:.2} dB", init_psnr);

    // Loss function using same rendering as GPU
    let loss_fn = |p: &[f32]| -> f32 {
        let temp_g = unflatten_gaussians(p, n);
        let rendered = RendererV2::render(&temp_g, target.width, target.height);
        let mut mse = 0.0f32;
        for (r, t) in rendered.data.iter().zip(target.data.iter()) {
            mse += (r.r - t.r).powi(2);
            mse += (r.g - t.g).powi(2);
            mse += (r.b - t.b).powi(2);
        }
        mse / (target.width * target.height * 3) as f32
    };

    // Gradient function (finite differences - slow but matches GPU numerically)
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

    // Run L-BFGS with iteration logging
    let mut optimizer = LBFGS::new(10);
    let mut gradient = grad_fn(&params);
    let mut f_x = loss_fn(&params);

    for iter in 0..max_iterations {
        // Log every 5 iterations
        if iter % 5 == 0 {
            let temp_g = unflatten_gaussians(&params, n);
            let rendered = RendererV2::render(&temp_g, target.width, target.height);
            let psnr = compute_psnr(&rendered, target);
            history.push((iter, f_x, psnr));
            println!("    [CPU {:3}] loss={:.6}, PSNR={:.2} dB", iter, f_x, psnr);
        }

        // Check convergence
        let grad_norm: f32 = gradient.iter().map(|&g| g * g).sum::<f32>().sqrt();
        if grad_norm < 1e-6 {
            println!("  Converged at iteration {} (grad_norm: {:.2e})", iter, grad_norm);
            break;
        }

        // Compute direction
        let direction = optimizer.compute_direction(&gradient);

        // Line search
        let alpha = optimizer.line_search(&params, &direction, &gradient, &mut |p: &[f32]| loss_fn(p), f_x);

        // Update
        let x_prev = params.clone();
        let grad_prev = gradient.clone();

        for i in 0..params.len() {
            params[i] += alpha * direction[i];
        }

        // New gradient and loss
        gradient = grad_fn(&params);
        f_x = loss_fn(&params);

        // Update history
        optimizer.update(&x_prev, &params, &grad_prev, &gradient);
    }

    let elapsed = start.elapsed();

    let final_gaussians = unflatten_gaussians(&params, n);
    let final_rendered = RendererV2::render(&final_gaussians, target.width, target.height);
    let final_psnr = compute_psnr(&final_rendered, target);

    println!("  Final loss: {:.6}", f_x);
    println!("  Final PSNR: {:.2} dB", final_psnr);
    println!("  Time: {:.2}s", elapsed.as_secs_f32());

    (final_gaussians, init_psnr, final_psnr, elapsed, history)
}

fn run_gpu_lbfgs(
    gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    max_iterations: usize,
) -> Result<(f32, f32, std::time::Duration), Box<dyn std::error::Error>> {
    println!("\n[GPU L-BFGS]");

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

    let init_rendered = RendererV2::render(gaussians, target.width, target.height);
    let init_psnr = compute_psnr(&init_rendered, target);
    println!("  Initial PSNR: {:.2} dB", init_psnr);

    let mut optimizer = GpuLbfgsOptimizer::new(&device)?;
    optimizer.config = LbfgsConfig {
        max_iterations,
        history_size: 10,
        tolerance: 1e-6,
        c1: 1e-4,
        max_line_search_iters: 20,
        initial_step: 1.0,
    };

    let start = Instant::now();
    let final_loss = optimizer.optimize(&device, &queue, gaussians, target)?;
    let elapsed = start.elapsed();

    let final_rendered = RendererV2::render(gaussians, target.width, target.height);
    let final_psnr = compute_psnr(&final_rendered, target);

    println!("  Final loss: {:.6}", final_loss);
    println!("  Final PSNR: {:.2} dB", final_psnr);
    println!("  Time: {:.2}s", elapsed.as_secs_f32());

    Ok((init_psnr, final_psnr, elapsed))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("======================================================================");
    println!("           L-BFGS GPU/CPU Consistency Test (10-param format)");
    println!("======================================================================");
    println!();
    println!("This test uses the SAME parameterization for both GPU and CPU:");
    println!("  10 params per Gaussian: x, y, scale_x, scale_y, rotation, r, g, b, a, opacity");
    println!();

    let target = load_test_image();
    println!("Loaded image: {}x{}", target.width, target.height);

    // Use small config for testing
    let n_gaussians = 16;  // 4x4 grid - small for fast testing
    let max_iterations = 20;  // Short run

    println!("\nConfiguration:");
    println!("  Gaussians: {} ({}x{} grid)", n_gaussians, 4, 4);
    println!("  Max iterations: {}", max_iterations);
    println!("  Total parameters: {}", n_gaussians * PARAMS_PER_GAUSSIAN);

    // Initialize identical Gaussians for both
    let gaussians_base = init_gaussians(n_gaussians);

    // Run GPU L-BFGS first (faster)
    let mut gpu_gaussians = gaussians_base.clone();
    let (gpu_init, gpu_final, gpu_time) = run_gpu_lbfgs(&mut gpu_gaussians, &target, max_iterations)?;

    // Run CPU L-BFGS with 10-param format
    let (cpu_gaussians, cpu_init, cpu_final, cpu_time, cpu_history) =
        run_cpu_lbfgs_10param(&gaussians_base, &target, max_iterations);

    // Comparison
    println!("\n======================================================================");
    println!("                           COMPARISON");
    println!("======================================================================");
    println!();
    println!("Initial PSNR:  CPU={:.2} dB, GPU={:.2} dB (diff: {:+.3} dB)",
             cpu_init, gpu_init, gpu_init - cpu_init);
    println!("Final PSNR:    CPU={:.2} dB, GPU={:.2} dB (diff: {:+.3} dB)",
             cpu_final, gpu_final, gpu_final - cpu_final);
    println!();
    println!("Improvement:   CPU={:+.2} dB, GPU={:+.2} dB",
             cpu_final - cpu_init, gpu_final - gpu_init);
    println!();
    println!("Time:          CPU={:.2}s, GPU={:.2}s (speedup: {:.1}x)",
             cpu_time.as_secs_f32(), gpu_time.as_secs_f32(),
             cpu_time.as_secs_f32() / gpu_time.as_secs_f32());

    let psnr_diff = (gpu_final - cpu_final).abs();
    println!();
    if psnr_diff < 0.5 {
        println!("RESULT: CONSISTENT - GPU and CPU within 0.5 dB ({:.2} dB difference)", psnr_diff);
    } else if psnr_diff < 1.0 {
        println!("RESULT: CLOSE - GPU and CPU within 1.0 dB ({:.2} dB difference)", psnr_diff);
    } else {
        println!("RESULT: DIVERGENT - GPU and CPU differ by {:.2} dB", psnr_diff);
    }

    // Save results with TestResult framework
    let rendered = RendererV2::render(&gpu_gaussians, target.width, target.height);

    let mut result = TestResult::new("lbfgs_gpu_cpu_consistency", TEST_IMAGE);
    result.optimizer = "L-BFGS GPU vs CPU Consistency Test".to_string();
    result.n_gaussians = n_gaussians;
    result.iterations = max_iterations;
    result.initial_psnr = gpu_init;
    result.final_psnr = gpu_final;
    result.improvement_db = gpu_final - gpu_init;
    result.elapsed_seconds = gpu_time.as_secs_f32();
    result.image_dimensions = (target.width, target.height);

    result.set_param("cpu_final_psnr", cpu_final as f64);
    result.set_param("cpu_time_seconds", cpu_time.as_secs_f64());
    result.set_param("psnr_difference", (gpu_final - cpu_final) as f64);
    result.set_param("history_size", 10.0);
    result.set_param("params_per_gaussian", PARAMS_PER_GAUSSIAN as f64);

    result.complete();
    let _ = result.save_with_image(&rendered, RESULTS_DIR);

    println!("\nResults saved to {}", RESULTS_DIR);

    Ok(())
}
