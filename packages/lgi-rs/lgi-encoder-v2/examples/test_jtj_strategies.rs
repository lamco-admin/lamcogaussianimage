//! Test for J^T J computation strategies
//!
//! Compares block-diagonal, sparse, hierarchical, and full-dense strategies
//! by measuring sparsity patterns and computational characteristics.

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{renderer_v2::RendererV2, test_results::TestResult};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, color::Color4, vec::Vector2};
use lgi_gpu::{
    JtJStrategyRegistry, JtJContext, JtJStrategy,
    BlockDiagonalJtJ, SparseJtJ, HierarchicalJtJ, FullDenseJtJ,
    adam_optimizer::ShaderGaussian,
};
use std::time::Instant;

const TEST_IMAGE: &str = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("======================================================================");
    println!("           J^T J Strategy Comparison Test");
    println!("======================================================================");
    println!();

    // List available strategies
    println!("Available strategies:");
    for (name, desc, interactions, complexity) in JtJStrategyRegistry::list_strategies() {
        println!("  - {}: {} [interactions: {}, complexity: {}]",
                 name, desc, interactions, complexity);
    }
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

    // Load image and initialize Gaussians
    let target = load_test_image();
    println!("Image: {}x{}", target.width, target.height);

    // Use 16 Gaussians for quick testing (4x4 grid)
    let n_gaussians = 16;
    let gaussians = init_gaussians(n_gaussians);
    println!("Gaussians: {} ({} total params)", n_gaussians, n_gaussians * 10);
    println!();

    // Convert to GPU format and create buffers
    use wgpu::util::DeviceExt;

    let gpu_gaussians: Vec<ShaderGaussian> = gaussians.iter().map(|g| ShaderGaussian {
        position: [g.position.x, g.position.y],
        scale_x: g.shape.scale_x,
        scale_y: g.shape.scale_y,
        rotation: g.shape.rotation,
        color: [g.color.r, g.color.g, g.color.b, g.color.a],
        opacity: g.opacity,
        _padding: [0; 2],  // u32 padding
    }).collect();

    let gaussian_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Gaussian Buffer"),
        contents: bytemuck::cast_slice(&gpu_gaussians),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Create target image buffer
    let target_data: Vec<[f32; 4]> = target.data.iter()
        .map(|p| [p.r, p.g, p.b, p.a])
        .collect();

    let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Target Buffer"),
        contents: bytemuck::cast_slice(&target_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Render and create rendered buffer
    let rendered = RendererV2::render(&gaussians, target.width, target.height);
    let rendered_data: Vec<[f32; 4]> = rendered.data.iter()
        .map(|p| [p.r, p.g, p.b, p.a])
        .collect();

    let rendered_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Rendered Buffer"),
        contents: bytemuck::cast_slice(&rendered_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let ctx = JtJContext {
        gaussians: &gpu_gaussians,
        gaussian_buffer: &gaussian_buffer,
        target_buffer: &target_buffer,
        rendered_buffer: &rendered_buffer,
        width: target.width,
        height: target.height,
        num_params: n_gaussians * 10,
    };

    // Test each strategy
    println!("Testing strategies...");
    println!("----------------------------------------------------------------------");

    // 1. Block-Diagonal
    {
        print!("Block-Diagonal:  ");
        let strategy = BlockDiagonalJtJ::new(&device)?;
        let start = Instant::now();
        let result = strategy.compute(&device, &queue, &ctx)?;
        let elapsed = start.elapsed();

        let info = result.sparsity_info.as_ref().unwrap();
        println!("nnz={:6}, density={:5.1}%, time={:.3}ms",
                 info.nnz, info.density() * 100.0, elapsed.as_secs_f64() * 1000.0);
    }

    // 2. Sparse (3σ)
    {
        print!("Sparse (3σ):     ");
        let strategy = SparseJtJ::new(&device, 3.0)?;
        let start = Instant::now();
        let result = strategy.compute(&device, &queue, &ctx)?;
        let elapsed = start.elapsed();

        let info = result.sparsity_info.as_ref().unwrap();
        println!("nnz={:6}, density={:5.1}%, time={:.3}ms",
                 info.nnz, info.density() * 100.0, elapsed.as_secs_f64() * 1000.0);
    }

    // 3. Sparse (6σ - more overlap)
    {
        print!("Sparse (6σ):     ");
        let strategy = SparseJtJ::new(&device, 6.0)?;
        let start = Instant::now();
        let result = strategy.compute(&device, &queue, &ctx)?;
        let elapsed = start.elapsed();

        let info = result.sparsity_info.as_ref().unwrap();
        println!("nnz={:6}, density={:5.1}%, time={:.3}ms",
                 info.nnz, info.density() * 100.0, elapsed.as_secs_f64() * 1000.0);
    }

    // 4. Hierarchical (3 levels)
    {
        print!("Hierarchical:    ");
        let strategy = HierarchicalJtJ::new(&device, 3)?;
        let start = Instant::now();
        let result = strategy.compute(&device, &queue, &ctx)?;
        let elapsed = start.elapsed();

        let info = result.sparsity_info.as_ref().unwrap();
        println!("nnz={:6}, density={:5.1}%, time={:.3}ms (σ={:?})",
                 info.nnz, info.density() * 100.0, elapsed.as_secs_f64() * 1000.0,
                 strategy.sigma_values());
    }

    // 5. Full-Dense
    {
        print!("Full-Dense:      ");
        let strategy = FullDenseJtJ::new(&device)?;
        let start = Instant::now();
        let result = strategy.compute(&device, &queue, &ctx)?;
        let elapsed = start.elapsed();

        let info = result.sparsity_info.as_ref().unwrap();
        println!("nnz={:6}, density={:5.1}%, time={:.3}ms",
                 info.nnz, info.density() * 100.0, elapsed.as_secs_f64() * 1000.0);
    }

    println!("----------------------------------------------------------------------");
    println!();

    // Summary
    println!("Strategy Comparison Summary:");
    println!("  - Block-diagonal: Fastest, but ignores cross-Gaussian interactions");
    println!("  - Sparse (3σ): Captures nearby interactions, good balance");
    println!("  - Sparse (6σ): More interactions, higher density");
    println!("  - Hierarchical: Multi-scale approach, weighted combination");
    println!("  - Full-Dense: All interactions, slowest but most complete");
    println!();

    // Save result
    let mut result = TestResult::new("jtj_strategy_comparison", TEST_IMAGE);
    result.optimizer = "JtJ Strategy Comparison".to_string();
    result.n_gaussians = n_gaussians;
    result.image_dimensions = (target.width, target.height);
    result.set_param("strategies_tested", 5.0);
    result.set_param("total_params", (n_gaussians * 10) as f64);
    result.complete();
    let _ = result.save_partial(RESULTS_DIR);

    println!("Results saved to {}", RESULTS_DIR);

    Ok(())
}
