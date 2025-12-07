//! Benchmark: GPU Adam vs CPU Adam optimizer
//!
//! Compares performance and quality between GPU-accelerated and CPU-based
//! Adam optimization for Gaussian image encoding.

use std::time::Instant;

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

const TEST_IMAGE: &str = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn create_initial_gaussians(n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut gaussians = Vec::with_capacity(n);
    let grid_size = (n as f32).sqrt() as usize;

    for i in 0..grid_size {
        for j in 0..grid_size {
            if gaussians.len() >= n {
                break;
            }

            // Add small deterministic jitter for better coverage
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

fn compute_psnr(mse: f32) -> f32 {
    if mse <= 0.0 {
        return 100.0;
    }
    10.0 * (1.0 / mse).log10()
}

fn print_separator(ch: char, width: usize) {
    println!("{}", std::iter::repeat(ch).take(width).collect::<String>());
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    print_separator('=', 70);
    println!("GPU vs CPU Adam Benchmark");
    print_separator('=', 70);

    // Load test image
    println!("Loading image: {}", TEST_IMAGE);

    let target = match ImageBuffer::load(TEST_IMAGE) {
        Ok(img) => {
            println!("  Loaded: {}x{}", img.width, img.height);
            img
        }
        Err(e) => {
            eprintln!("ERROR: Could not load image: {}", e);
            return Err(format!("Failed to load test image: {}", e).into());
        }
    };

    let _width = target.width;
    let _height = target.height;

    // Test configurations
    let configs = [
        (256, 100),  // Standard: 256 Gaussians, 100 iterations
        (256, 200),  // Extended: 256 Gaussians, 200 iterations
    ];

    for (n_gaussians, iterations) in configs {
        println!();
        print_separator('-', 60);
        println!("Configuration: {} Gaussians, {} iterations", n_gaussians, iterations);
        print_separator('-', 60);

        // ============ CPU Adam ============
        println!("\n[CPU Adam]");
        let mut cpu_gaussians = create_initial_gaussians(n_gaussians);

        let mut cpu_optimizer = AdamOptimizer::new();
        cpu_optimizer.max_iterations = iterations;

        let cpu_start = Instant::now();
        let cpu_loss = cpu_optimizer.optimize(&mut cpu_gaussians, &target);
        let cpu_elapsed = cpu_start.elapsed();

        let cpu_psnr = compute_psnr(cpu_loss);
        println!("  Final loss: {:.6}", cpu_loss);
        println!("  Final PSNR: {:.2} dB", cpu_psnr);
        println!("  Time: {:.2}s", cpu_elapsed.as_secs_f32());
        println!("  Iter/sec: {:.1}", iterations as f32 / cpu_elapsed.as_secs_f32());

        // Save CPU result
        let mut cpu_result = TestResult::new("bench_cpu_adam", TEST_IMAGE);
        cpu_result.optimizer = format!("CPU Adam ({} iters)", iterations);
        cpu_result.n_gaussians = n_gaussians;
        cpu_result.iterations = iterations;
        cpu_result.final_psnr = cpu_psnr;
        cpu_result.elapsed_seconds = cpu_elapsed.as_secs_f32();
        cpu_result.set_param("type", "cpu");
        cpu_result.complete();
        let _ = cpu_result.save(RESULTS_DIR);

        // ============ GPU Adam ============
        println!("\n[GPU Adam]");
        let mut gpu_gaussians = create_initial_gaussians(n_gaussians);

        // Initialize GPU
        let gpu_start = Instant::now();

        match pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|e| format!("Adapter request failed: {:?}", e))?;

            println!("  GPU: {:?}", adapter.get_info().name);

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .map_err(|e| format!("Device request failed: {:?}", e))?;

            // Create GPU optimizer
            let mut optimizer = lgi_gpu::GpuAdamOptimizer::new(&device)
                .map_err(|e| format!("Optimizer creation failed: {:?}", e))?;
            optimizer.max_iterations = iterations;

            let init_time = gpu_start.elapsed();
            println!("  GPU init time: {:.2}ms", init_time.as_secs_f32() * 1000.0);

            // Run optimization
            let opt_start = Instant::now();
            let gpu_loss = optimizer
                .optimize(&device, &queue, &mut gpu_gaussians, &target)
                .map_err(|e| format!("Optimization failed: {:?}", e))?;
            let opt_time = opt_start.elapsed();

            Ok::<(f32, std::time::Duration, std::time::Duration), String>((gpu_loss, init_time, opt_time))
        }) {
            Ok((gpu_loss, init_time, opt_time)) => {
                let gpu_elapsed = gpu_start.elapsed();
                let gpu_psnr = compute_psnr(gpu_loss);

                println!("  Final loss: {:.6}", gpu_loss);
                println!("  Final PSNR: {:.2} dB", gpu_psnr);
                println!("  Optimization time: {:.2}s", opt_time.as_secs_f32());
                println!("  Total time (incl init): {:.2}s", gpu_elapsed.as_secs_f32());
                println!("  Iter/sec: {:.1}", iterations as f32 / opt_time.as_secs_f32());

                // Comparison
                println!("\n[Comparison]");
                let speedup = cpu_elapsed.as_secs_f32() / opt_time.as_secs_f32();
                let psnr_diff = gpu_psnr - cpu_psnr;
                println!("  Speedup: {:.1}x", speedup);
                println!("  PSNR diff: {:+.2} dB", psnr_diff);

                if speedup >= 1.0 {
                    println!("  GPU is {:.1}x faster", speedup);
                } else {
                    println!("  CPU is {:.1}x faster", 1.0 / speedup);
                }

                // Save GPU result
                let mut gpu_result = TestResult::new("bench_gpu_adam", TEST_IMAGE);
                gpu_result.optimizer = format!("GPU Adam ({} iters)", iterations);
                gpu_result.n_gaussians = n_gaussians;
                gpu_result.iterations = iterations;
                gpu_result.final_psnr = gpu_psnr;
                gpu_result.elapsed_seconds = opt_time.as_secs_f32();
                gpu_result.set_param("type", "gpu");
                gpu_result.set_param("init_time_ms", (init_time.as_secs_f32() * 1000.0) as f64);
                gpu_result.set_param("speedup", speedup as f64);
                gpu_result.complete();
                let _ = gpu_result.save(RESULTS_DIR);
            }
            Err(e) => {
                println!("  GPU optimization failed: {}", e);
                println!("  (This is expected if no GPU is available)");
            }
        }
    }

    println!();
    print_separator('=', 70);
    println!("Benchmark complete");
    print_separator('=', 70);

    Ok(())
}
