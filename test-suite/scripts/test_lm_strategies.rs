//! Test: Compare L-M J^T J strategies (block-diagonal vs full-dense)
//!
//! This tests the new strategy system for the GPU L-M optimizer.
//! Compares:
//! - Block-diagonal (fast, ignores Gaussian interactions)
//! - Full-dense (slow, captures all interactions)

use std::time::Instant;

use bytemuck;
use lgi_core::ImageBuffer;
use lgi_gpu::jtj_strategy::{
    JtJStrategy, JtJContext, BlockDiagonalJtJ, FullDenseJtJ,
    JtJStrategyRegistry,
};
use lgi_gpu::adam_optimizer::ShaderGaussian;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_encoder_v2::renderer_v2::RendererV2;
use wgpu::util::DeviceExt;

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

fn render_gaussians(gaussians: &[Gaussian2D<f32, Euler<f32>>], width: u32, height: u32) -> Vec<[f32; 4]> {
    let rendered = RendererV2::render(gaussians, width, height);
    rendered.data.iter().map(|c| [c.r, c.g, c.b, c.a]).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        L-M J^T J Strategy Comparison Test                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Load test image
    println!("Loading image: {}", TEST_IMAGE);
    let target = ImageBuffer::load(TEST_IMAGE)?;
    println!("  Image: {}×{}", target.width, target.height);

    // Use small Gaussian count for testing (full-dense is O(G²))
    let n_gaussians = 16;  // 16×16 = 256 param pairs × 100 = 25,600 blocks
    let num_params = n_gaussians * 10;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Configuration: {} Gaussians, {} parameters", n_gaussians, num_params);
    println!("  Block-diagonal: {} diagonal 10×10 blocks", n_gaussians);
    println!("  Full-dense: {} total 10×10 blocks", n_gaussians * n_gaussians);
    println!("═══════════════════════════════════════════════════════════════");

    // Create initial Gaussians
    let gaussians = create_initial_gaussians(n_gaussians);

    // Render current state
    let rendered_data = render_gaussians(&gaussians, target.width, target.height);

    // Convert to GPU format
    let gpu_gaussians: Vec<ShaderGaussian> = gaussians.iter()
        .map(ShaderGaussian::from_gaussian)
        .collect();

    // Initialize GPU
    let result = pollster::block_on(async {
        println!("\nInitializing GPU...");
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

        // Create buffers
        let gaussian_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Gaussian Buffer"),
            contents: bytemuck::cast_slice(&gpu_gaussians),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let target_data: Vec<[f32; 4]> = target.data.iter()
            .map(|c| [c.r, c.g, c.b, c.a])
            .collect();
        let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Target Buffer"),
            contents: bytemuck::cast_slice(&target_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let rendered_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Rendered Buffer"),
            contents: bytemuck::cast_slice(&rendered_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create context
        let ctx = JtJContext {
            gaussians: &gpu_gaussians,
            gaussian_buffer: &gaussian_buffer,
            target_buffer: &target_buffer,
            rendered_buffer: &rendered_buffer,
            width: target.width,
            height: target.height,
            num_params,
        };

        // List available strategies
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("Available Strategies:");
        for (name, desc, captures, complexity) in JtJStrategyRegistry::list_strategies() {
            println!("  {} [{}] - {}",
                     name,
                     if captures { "interactions" } else { "no interactions" },
                     desc);
            println!("    Complexity: {}", complexity);
        }
        println!("═══════════════════════════════════════════════════════════════");

        // Test Block-Diagonal
        println!("\n▶ Testing Block-Diagonal Strategy...");
        let block_diag = BlockDiagonalJtJ::new(&device)
            .map_err(|e| format!("Failed to create block-diagonal strategy: {:?}", e))?;

        let start = Instant::now();
        let block_result = block_diag.compute(&device, &queue, &ctx)
            .map_err(|e| format!("Block-diagonal compute failed: {:?}", e))?;
        let block_time = start.elapsed();

        println!("  Strategy: {}", block_diag.name());
        println!("  Captures interactions: {}", block_diag.captures_interactions());
        if let Some(ref info) = block_result.sparsity_info {
            println!("  Sparsity: {:.2}% dense ({} / {} elements)",
                     info.density() * 100.0, info.nnz, info.total);
        }
        println!("  Time: {:.3}ms", block_time.as_secs_f64() * 1000.0);

        // Analyze J^T J structure
        let jtj = &block_result.jtj;
        let jtr = &block_result.jtr;
        let jtj_norm: f32 = jtj.iter().map(|x| x * x).sum::<f32>().sqrt();
        let jtr_norm: f32 = jtr.iter().map(|x| x * x).sum::<f32>().sqrt();
        let jtj_max = jtj.iter().cloned().fold(0.0f32, f32::max);
        let jtr_max = jtr.iter().cloned().fold(0.0f32, f32::max);

        println!("  ||J^T J||_F = {:.4}, max = {:.4}", jtj_norm, jtj_max);
        println!("  ||J^T r|| = {:.4}, max = {:.4}", jtr_norm, jtr_max);

        // Test Full-Dense
        println!("\n▶ Testing Full-Dense Strategy...");
        let full_dense = FullDenseJtJ::new(&device)
            .map_err(|e| format!("Failed to create full-dense strategy: {:?}", e))?;

        let start = Instant::now();
        let dense_result = full_dense.compute(&device, &queue, &ctx)
            .map_err(|e| format!("Full-dense compute failed: {:?}", e))?;
        let dense_time = start.elapsed();

        println!("  Strategy: {}", full_dense.name());
        println!("  Captures interactions: {}", full_dense.captures_interactions());
        if let Some(ref info) = dense_result.sparsity_info {
            println!("  Sparsity: {:.2}% dense ({} / {} elements)",
                     info.density() * 100.0, info.nnz, info.total);
        }
        println!("  Time: {:.3}ms", dense_time.as_secs_f64() * 1000.0);

        // Analyze J^T J structure
        let jtj = &dense_result.jtj;
        let jtr = &dense_result.jtr;
        let jtj_norm: f32 = jtj.iter().map(|x| x * x).sum::<f32>().sqrt();
        let jtr_norm: f32 = jtr.iter().map(|x| x * x).sum::<f32>().sqrt();
        let jtj_max = jtj.iter().cloned().fold(0.0f32, f32::max);
        let jtr_max = jtr.iter().cloned().fold(0.0f32, f32::max);

        println!("  ||J^T J||_F = {:.4}, max = {:.4}", jtj_norm, jtj_max);
        println!("  ||J^T r|| = {:.4}, max = {:.4}", jtr_norm, jtr_max);

        // Compare off-diagonal content
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("Comparison:");
        println!("  Speedup: {:.1}x faster for block-diagonal",
                 dense_time.as_secs_f64() / block_time.as_secs_f64());

        // Count non-zero off-diagonal blocks in dense result
        let mut off_diag_energy = 0.0f32;
        let mut diag_energy = 0.0f32;
        for gi in 0..n_gaussians {
            for gj in 0..n_gaussians {
                let base_i = gi * 10;
                let base_j = gj * 10;
                let mut block_energy = 0.0f32;
                for i in 0..10 {
                    for j in 0..10 {
                        let val = dense_result.jtj[(base_i + i) * num_params + (base_j + j)];
                        block_energy += val * val;
                    }
                }
                if gi == gj {
                    diag_energy += block_energy;
                } else {
                    off_diag_energy += block_energy;
                }
            }
        }
        let total_energy = diag_energy + off_diag_energy;
        if total_energy > 0.0 {
            println!("  Energy distribution:");
            println!("    Diagonal blocks: {:.2}%", diag_energy / total_energy * 100.0);
            println!("    Off-diagonal blocks: {:.2}%", off_diag_energy / total_energy * 100.0);
        }

        println!("═══════════════════════════════════════════════════════════════");

        Ok::<(), String>(())
    });

    match result {
        Ok(()) => {
            println!("\n✓ Strategy comparison complete");
        }
        Err(e) => {
            println!("\n⚠ ERROR: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
