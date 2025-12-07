//! Test: Compare L-M convergence with different J^T J strategies
//!
//! This test compares the actual optimization quality between:
//! - Block-diagonal (fast, ignores Gaussian interactions)
//! - Full-dense (slow, captures all interactions)
//!
//! The hypothesis: Full-dense should converge faster/better because
//! 67% of the Hessian energy is in off-diagonal blocks.

use std::time::Instant;

use lgi_core::ImageBuffer;
use lgi_gpu::{
    GpuLMOptimizer, LMConfig,
    BlockDiagonalJtJ, FullDenseJtJ,
};
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

            let x = (i as f32 + 0.5) / grid_size as f32;
            let y = (j as f32 + 0.5) / grid_size as f32;

            gaussians.push(Gaussian2D {
                position: Vector2 { x, y },
                shape: Euler {
                    scale_x: 0.05,  // Larger for more overlap
                    scale_y: 0.05,
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

fn compute_psnr(target: &ImageBuffer<f32>, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> f32 {
    use lgi_encoder_v2::renderer_v2::RendererV2;

    let rendered = RendererV2::render(gaussians, target.width, target.height);

    let mut mse = 0.0f32;
    for (t, r) in target.data.iter().zip(rendered.data.iter()) {
        mse += (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
    }
    mse /= (target.data.len() * 3) as f32;

    if mse < 1e-10 {
        100.0
    } else {
        -10.0 * mse.log10()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================");
    println!("   L-M Strategy Convergence Test");
    println!("================================");
    println!();

    // Load test image
    println!("Loading image: {}", TEST_IMAGE);
    let target = ImageBuffer::load(TEST_IMAGE)?;
    println!("  Image: {}x{}", target.width, target.height);

    // Use small Gaussian count (full-dense is O(G^2) per pixel!)
    let n_gaussians = 16;  // 4x4 grid

    println!("\nConfiguration:");
    println!("  Gaussians: {} ({}x{} grid)", n_gaussians, 4, 4);
    println!("  Image: {}x{} = {} pixels", target.width, target.height,
             target.width * target.height);
    println!();

    // Initialize GPU
    let result = pollster::block_on(async {
        println!("Initializing GPU...");
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

        // L-M configuration (same for both)
        let config = LMConfig {
            max_iterations: 20,  // Limited for speed
            tolerance: 1e-8,
            initial_lambda: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.1,
            lambda_min: 1e-10,
            lambda_max: 1e10,
        };

        // ===========================
        // Test 1: Block-Diagonal
        // ===========================
        println!("\n========================================");
        println!("Strategy 1: Block-Diagonal");
        println!("========================================");

        let mut gaussians_bd = create_initial_gaussians(n_gaussians);
        let init_psnr = compute_psnr(&target, &gaussians_bd);
        println!("  Initial PSNR: {:.2} dB", init_psnr);

        let strategy_bd = BlockDiagonalJtJ::new(&device)
            .map_err(|e| format!("Block-diagonal strategy failed: {:?}", e))?;
        let mut optimizer_bd = GpuLMOptimizer::with_strategy(&device, Box::new(strategy_bd))
            .map_err(|e| format!("Optimizer creation failed: {:?}", e))?;
        optimizer_bd.config = config.clone();

        let start_bd = Instant::now();
        let final_loss_bd = optimizer_bd.optimize(&device, &queue, &mut gaussians_bd, &target)
            .map_err(|e| format!("Optimization failed: {:?}", e))?;
        let elapsed_bd = start_bd.elapsed();

        let final_psnr_bd = compute_psnr(&target, &gaussians_bd);
        println!("\n  Block-Diagonal Results:");
        println!("    Final PSNR: {:.2} dB", final_psnr_bd);
        println!("    Improvement: {:+.2} dB", final_psnr_bd - init_psnr);
        println!("    Final loss: {:.6}", final_loss_bd);
        println!("    Time: {:.2}s", elapsed_bd.as_secs_f64());

        // ===========================
        // Test 2: Full-Dense
        // ===========================
        println!("\n========================================");
        println!("Strategy 2: Full-Dense (captures interactions)");
        println!("========================================");

        let mut gaussians_fd = create_initial_gaussians(n_gaussians);
        println!("  Initial PSNR: {:.2} dB", init_psnr);

        let strategy_fd = FullDenseJtJ::new(&device)
            .map_err(|e| format!("Full-dense strategy failed: {:?}", e))?;
        let mut optimizer_fd = GpuLMOptimizer::with_strategy(&device, Box::new(strategy_fd))
            .map_err(|e| format!("Optimizer creation failed: {:?}", e))?;
        optimizer_fd.config = config.clone();

        let start_fd = Instant::now();
        let final_loss_fd = optimizer_fd.optimize(&device, &queue, &mut gaussians_fd, &target)
            .map_err(|e| format!("Optimization failed: {:?}", e))?;
        let elapsed_fd = start_fd.elapsed();

        let final_psnr_fd = compute_psnr(&target, &gaussians_fd);
        println!("\n  Full-Dense Results:");
        println!("    Final PSNR: {:.2} dB", final_psnr_fd);
        println!("    Improvement: {:+.2} dB", final_psnr_fd - init_psnr);
        println!("    Final loss: {:.6}", final_loss_fd);
        println!("    Time: {:.2}s", elapsed_fd.as_secs_f64());

        // ===========================
        // Comparison
        // ===========================
        println!("\n========================================");
        println!("COMPARISON");
        println!("========================================");
        println!("  Block-Diagonal: {:.2} dB in {:.2}s", final_psnr_bd, elapsed_bd.as_secs_f64());
        println!("  Full-Dense:     {:.2} dB in {:.2}s", final_psnr_fd, elapsed_fd.as_secs_f64());
        println!();
        println!("  Quality difference: {:+.2} dB (full-dense - block-diagonal)",
                 final_psnr_fd - final_psnr_bd);
        println!("  Speedup: {:.1}x (block-diagonal / full-dense)",
                 elapsed_fd.as_secs_f64() / elapsed_bd.as_secs_f64());

        if final_psnr_fd > final_psnr_bd {
            println!("\n  RESULT: Full-dense is {:+.2} dB better, but {:.1}x slower",
                     final_psnr_fd - final_psnr_bd,
                     elapsed_fd.as_secs_f64() / elapsed_bd.as_secs_f64());
        } else if final_psnr_bd > final_psnr_fd {
            println!("\n  RESULT: Block-diagonal is {:+.2} dB better and {:.1}x faster!",
                     final_psnr_bd - final_psnr_fd,
                     elapsed_fd.as_secs_f64() / elapsed_bd.as_secs_f64());
        } else {
            println!("\n  RESULT: Same quality, but block-diagonal is {:.1}x faster",
                     elapsed_fd.as_secs_f64() / elapsed_bd.as_secs_f64());
        }

        Ok::<(), String>(())
    });

    match result {
        Ok(()) => {
            println!("\nTest completed successfully.");
        }
        Err(e) => {
            println!("\nERROR: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
