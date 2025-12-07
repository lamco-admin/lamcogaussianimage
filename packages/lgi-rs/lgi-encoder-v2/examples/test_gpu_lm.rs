//! Test: GPU Levenberg-Marquardt optimizer
//!
//! Tests the GPU L-M optimizer with implicit Jacobian computation.

use std::time::Instant;

use lgi_core::ImageBuffer;
use lgi_gpu::lm_optimizer::{GpuLMOptimizer, LMConfig};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

const TEST_IMAGE: &str = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║            GPU Levenberg-Marquardt Optimizer Test                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Load test image
    println!("Loading image: {}", TEST_IMAGE);
    let target = ImageBuffer::load(TEST_IMAGE)?;
    println!("  Image: {}×{}", target.width, target.height);

    // Test configuration
    // L-M is slow but should be more accurate per-iteration than Adam
    let n_gaussians = 64;  // Start small for L-M (Jacobian is O(n²) in params)
    let iterations = 100;  // More iterations since L-M converges slowly initially

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Configuration: {} Gaussians, {} iterations", n_gaussians, iterations);
    println!("  Parameters: {} ({}×10)", n_gaussians * 10, n_gaussians);
    println!("  J^T J size: {}×{} ({:.2} KB)",
             n_gaussians * 10, n_gaussians * 10,
             (n_gaussians * 10 * n_gaussians * 10 * 4) as f32 / 1024.0);
    println!("═══════════════════════════════════════════════════════════════");

    // Create initial Gaussians
    let mut gaussians = create_initial_gaussians(n_gaussians);

    // Initialize GPU and run optimizer
    let result = pollster::block_on(async {
        // Initialize GPU
        println!("\nInitializing GPU...");
        let gpu_start = Instant::now();

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("Failed to get GPU adapter: {:?}", e))?;

        println!("  GPU: {}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| format!("Device request failed: {:?}", e))?;

        println!("  GPU initialized in {:.2}s", gpu_start.elapsed().as_secs_f32());

        // Create L-M optimizer with custom config
        // Note: Higher initial lambda works better for this problem - the Hessian
        // approximation (block-diagonal J^T J) needs more regularization initially
        let config = LMConfig {
            max_iterations: iterations,
            initial_lambda: 1e4,  // Start higher to avoid wasted iterations
            lambda_up: 10.0,
            lambda_down: 0.1,
            tolerance: 1e-8,
            ..Default::default()
        };

        let mut optimizer = GpuLMOptimizer::new(&device)
            .map_err(|e| format!("Failed to create L-M optimizer: {:?}", e))?;
        optimizer.config = config;

        // Run optimization
        println!("\nRunning GPU L-M optimization...");
        let start = Instant::now();

        let final_loss = optimizer.optimize(
            &device,
            &queue,
            &mut gaussians,
            &target,
        ).map_err(|e| format!("Optimization failed: {:?}", e))?;

        let elapsed = start.elapsed();

        Ok::<(f32, std::time::Duration), String>((final_loss, elapsed))
    });

    match result {
        Ok((final_loss, elapsed)) => {
            let psnr = compute_psnr(final_loss);

            println!("\n╔══════════════════════════════════════════════════════════════════╗");
            println!("║                          RESULTS                                 ║");
            println!("╠══════════════════════════════════════════════════════════════════╣");
            println!("║  Final Loss:    {:8.6}                                       ║", final_loss);
            println!("║  Final PSNR:    {:6.2} dB                                       ║", psnr);
            println!("║  Time:          {:6.2} s                                        ║", elapsed.as_secs_f32());
            println!("║  Iter/sec:      {:6.1}                                          ║", iterations as f32 / elapsed.as_secs_f32());
            println!("╚══════════════════════════════════════════════════════════════════╝");
        }
        Err(e) => {
            println!("\n⚠ ERROR: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
