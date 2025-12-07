//! GPU-accelerated Levenberg-Marquardt optimizer
//!
//! # Architecture
//!
//! This implements a hybrid GPU/CPU L-M optimizer:
//! - **GPU**: Computes J^T J (Gauss-Newton Hessian) and J^T r (gradient) in parallel
//! - **CPU**: Solves the linear system (J^T J + λI)δ = -J^T r
//! - **GPU**: Applies parameter updates
//!
//! ## Why This Approach?
//!
//! The full Jacobian J has shape (n_residuals × n_params):
//! - For 768×512 image, 256 Gaussians: 1.18M × 1.79K = 2.1 billion elements (8GB)
//! - Too large to store explicitly
//!
//! Instead, we compute J^T J and J^T r directly:
//! - J^T J: (n_params × n_params) = 1792 × 1792 = 12MB
//! - J^T r: (n_params,) = 1792 floats = 7KB
//!
//! ## Current Limitations
//!
//! **Block-diagonal approximation**: The current implementation only computes
//! within-Gaussian J^T J blocks, ignoring cross-Gaussian interactions. This
//! causes the Hessian approximation to be poor when Gaussians overlap significantly.
//!
//! In practice, this means:
//! - Lambda stays high (1e6-1e7) for regularization
//! - Convergence is slow: ~0.4% improvement in 100 iterations
//! - Adam significantly outperforms L-M for this problem
//!
//! ## Alternative Approaches (not implemented)
//!
//! 1. **Sparse Jacobian**: Each Gaussian only affects nearby pixels. Could use sparse
//!    matrix format to reduce memory. Complexity: moderate.
//!
//! 2. **Full dense J^T J**: Compute all off-diagonal blocks. Would improve convergence
//!    but requires O(n²) compute per pixel.
//!
//! 3. **Stochastic L-M**: Sample a subset of pixels per iteration. Trades accuracy
//!    for speed. Good for large images.
//!
//! 4. **Jacobian-free methods**: Use matrix-vector products J*v and J^T*v without
//!    forming J. Requires iterative solver (CG, GMRES).
//!
//! ## References
//!
//! - Nocedal & Wright, "Numerical Optimization", Chapter 10
//! - 3DGS-LM (ICCV 2025): L-M for 3D Gaussian Splatting

use wgpu::{Device, Queue};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, color::Color4};

use crate::error::Result;
use crate::adam_optimizer::ShaderGaussian;
use crate::buffer::RenderParams;
use crate::jtj_strategy::{JtJStrategy, JtJContext, BlockDiagonalJtJ};

/// Parameters per Gaussian in the optimization
/// Order: x, y, scale_x, scale_y, rotation, r, g, b, a, opacity
pub const PARAMS_PER_GAUSSIAN: usize = 10;

/// Levenberg-Marquardt configuration
#[derive(Clone, Debug)]
pub struct LMConfig {
    /// Maximum iterations (default: 50)
    pub max_iterations: usize,
    /// Convergence tolerance on gradient norm (default: 1e-6)
    pub tolerance: f32,
    /// Initial damping factor λ (default: 1e-3)
    pub initial_lambda: f32,
    /// Lambda increase factor when step rejected (default: 10.0)
    pub lambda_up: f32,
    /// Lambda decrease factor when step accepted (default: 0.1)
    pub lambda_down: f32,
    /// Minimum lambda (default: 1e-7)
    pub lambda_min: f32,
    /// Maximum lambda (default: 1e7)
    pub lambda_max: f32,
}

impl Default for LMConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            initial_lambda: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.1,
            lambda_min: 1e-7,
            lambda_max: 1e7,
        }
    }
}

/// GPU parameters for JtJ/Jtr computation
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct JacobianParams {
    width: u32,
    height: u32,
    num_gaussians: u32,
    num_params: u32,
}

/// GPU Levenberg-Marquardt optimizer
///
/// Supports pluggable J^T J computation strategies via the `JtJStrategy` trait.
/// Default is block-diagonal (fast, ignores cross-Gaussian interactions).
/// Use `set_strategy()` to switch to full-dense or other strategies.
pub struct GpuLMOptimizer {
    /// Configuration
    pub config: LMConfig,

    /// J^T J computation strategy (default: block-diagonal)
    strategy: Box<dyn JtJStrategy>,

    // Compute pipelines
    render_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    render_layout: wgpu::BindGroupLayout,
    update_layout: wgpu::BindGroupLayout,
}

impl GpuLMOptimizer {
    /// Create a new GPU L-M optimizer with the default block-diagonal strategy
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_strategy(device, Box::new(BlockDiagonalJtJ::new(device)?))
    }

    /// Create a new GPU L-M optimizer with a custom J^T J strategy
    pub fn with_strategy(device: &Device, strategy: Box<dyn JtJStrategy>) -> Result<Self> {
        // Render shader (same as other optimizers)
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("L-M Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gaussian_render.wgsl").into()),
        });

        // Update shader
        let update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("L-M Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lm_update.wgsl").into()),
        });

        // Render bind group layout
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("L-M Render Layout"),
            entries: &[
                // Gaussians (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Rendered image (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Render params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Update bind group layout
        let update_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("L-M Update Layout"),
            entries: &[
                // Gaussians (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Delta (parameter updates)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipelines
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("L-M Render Pipeline Layout"),
            bind_group_layouts: &[&render_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("L-M Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            module: &render_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("L-M Update Pipeline Layout"),
            bind_group_layouts: &[&update_layout],
            push_constant_ranges: &[],
        });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("L-M Update Pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &update_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            config: LMConfig::default(),
            strategy,
            render_pipeline,
            update_pipeline,
            render_layout,
            update_layout,
        })
    }

    /// Set a new J^T J computation strategy
    ///
    /// Available strategies:
    /// - `BlockDiagonalJtJ`: Fast, O(G), ignores cross-Gaussian interactions
    /// - `FullDenseJtJ`: Slow, O(G²), captures all interactions
    /// - `SparseJtJ`: Middle ground, only overlapping pairs
    /// - `MatrixFreeJtJ`: Memory-efficient iterative solver
    pub fn set_strategy(&mut self, strategy: Box<dyn JtJStrategy>) {
        self.strategy = strategy;
    }

    /// Get the current strategy name
    pub fn strategy_name(&self) -> &'static str {
        self.strategy.name()
    }

    /// Run L-M optimization
    ///
    /// Returns the final loss value
    pub fn optimize(
        &mut self,
        device: &Device,
        queue: &Queue,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> Result<f32> {
        let num_gaussians = gaussians.len();
        let num_params = num_gaussians * PARAMS_PER_GAUSSIAN;
        let width = target.width;
        let height = target.height;

        println!("=== GPU Levenberg-Marquardt Optimizer ===");
        println!("  Strategy: {} ({})",
                 self.strategy.name(),
                 if self.strategy.captures_interactions() { "with interactions" } else { "no interactions" });
        println!("  Gaussians: {}", num_gaussians);
        println!("  Parameters: {} ({}×{})", num_params, num_gaussians, PARAMS_PER_GAUSSIAN);
        println!("  Image: {}×{}", width, height);
        println!("  J^T J size: {}×{} ({:.2} MB)", num_params, num_params,
                 (num_params * num_params * 4) as f32 / 1_000_000.0);

        // Convert Gaussians to GPU format
        let mut gpu_gaussians: Vec<ShaderGaussian> = gaussians.iter()
            .map(ShaderGaussian::from_gaussian)
            .collect();

        // Create GPU buffers
        let gaussian_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("L-M Gaussian Buffer"),
            contents: bytemuck::cast_slice(&gpu_gaussians),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // Target buffer
        let target_data: Vec<[f32; 4]> = target.data.iter()
            .map(|c| [c.r, c.g, c.b, c.a])
            .collect();
        let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("L-M Target Buffer"),
            contents: bytemuck::cast_slice(&target_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Rendered image buffer
        let rendered_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("L-M Rendered Buffer"),
            size: (width * height * 16) as u64, // 4 floats per pixel
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Render params
        let render_params = RenderParams {
            width,
            height,
            gaussian_count: num_gaussians as u32,
            render_mode: 0, // AlphaComposite
            cutoff_threshold: 0.0,
            n_sigma: 3.5,
            _padding: [0; 2],
        };
        let render_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("L-M Render Params Buffer"),
            contents: bytemuck::bytes_of(&render_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Initial render and loss
        self.render_gpu(device, queue, &gaussian_buffer, &rendered_buffer, &render_params_buffer, width, height);
        let initial_loss = self.compute_loss_gpu(device, queue, &rendered_buffer, &target_buffer, width, height)?;
        println!("  Initial loss: {:.6}", initial_loss);

        let mut lambda = self.config.initial_lambda;
        let mut current_loss = initial_loss;
        let mut best_loss = initial_loss;
        let mut best_gaussians = gpu_gaussians.clone();

        for iter in 1..=self.config.max_iterations {
            // Update gpu_gaussians from buffer for context
            gpu_gaussians = self.read_gaussians(device, queue, &gaussian_buffer, num_gaussians)?
                .iter()
                .map(ShaderGaussian::from_gaussian)
                .collect();

            // Step 1: Compute J^T J and J^T r using the strategy
            let ctx = JtJContext {
                gaussians: &gpu_gaussians,
                gaussian_buffer: &gaussian_buffer,
                target_buffer: &target_buffer,
                rendered_buffer: &rendered_buffer,
                width,
                height,
                num_params,
            };

            let jtj_result = self.strategy.compute(device, queue, &ctx)?;
            let jtj = jtj_result.jtj;
            let jtr = jtj_result.jtr;

            // Check gradient norm for convergence
            let grad_norm: f32 = jtr.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if grad_norm < self.config.tolerance {
                println!("  Converged at iteration {} (grad_norm: {:.2e})", iter, grad_norm);
                break;
            }

            // Step 3: Solve (J^T J + λI) δ = -J^T r on CPU
            let delta = self.solve_damped_system(&jtj, &jtr, lambda, num_params);

            // Step 4: Try the step
            let delta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("L-M Delta Buffer"),
                contents: bytemuck::cast_slice(&delta),
                usage: wgpu::BufferUsages::STORAGE,
            });

            // Save current state
            let saved_gaussians = self.read_gaussians(device, queue, &gaussian_buffer, num_gaussians)?;

            // Apply update
            self.apply_update_gpu(device, queue, &gaussian_buffer, &delta_buffer, num_gaussians)?;

            // Render and compute new loss
            self.render_gpu(device, queue, &gaussian_buffer, &rendered_buffer, &render_params_buffer, width, height);
            let new_loss = self.compute_loss_gpu(device, queue, &rendered_buffer, &target_buffer, width, height)?;

            // Step 5: Accept or reject
            if new_loss < current_loss {
                // Accept step, decrease lambda
                current_loss = new_loss;
                lambda = (lambda * self.config.lambda_down).max(self.config.lambda_min);

                if new_loss < best_loss {
                    best_loss = new_loss;
                    best_gaussians = self.read_gaussians(device, queue, &gaussian_buffer, num_gaussians)?
                        .iter()
                        .map(ShaderGaussian::from_gaussian)
                        .collect();
                }

                if iter % 5 == 0 {
                    println!("  Iter {}: loss={:.6} λ={:.2e} ✓", iter, new_loss, lambda);
                }
            } else {
                // Reject step, restore state, increase lambda
                let gpu_saved: Vec<ShaderGaussian> = saved_gaussians.iter()
                    .map(ShaderGaussian::from_gaussian)
                    .collect();
                queue.write_buffer(&gaussian_buffer, 0, bytemuck::cast_slice(&gpu_saved));
                lambda = (lambda * self.config.lambda_up).min(self.config.lambda_max);

                if iter % 5 == 0 {
                    println!("  Iter {}: loss={:.6} λ={:.2e} ✗ (rejected)", iter, current_loss, lambda);
                }
            }
        }

        // Restore best state
        queue.write_buffer(&gaussian_buffer, 0, bytemuck::cast_slice(&best_gaussians));

        // Copy back to gaussians
        let final_gaussians = self.read_gaussians(device, queue, &gaussian_buffer, num_gaussians)?;
        for (i, g) in final_gaussians.into_iter().enumerate() {
            gaussians[i] = g;
        }

        println!("  Final loss: {:.6}", best_loss);
        println!("  Improvement: {:.2}%", (initial_loss - best_loss) / initial_loss * 100.0);

        Ok(best_loss)
    }

    /// Render Gaussians on GPU
    fn render_gpu(
        &self,
        device: &Device,
        queue: &Queue,
        gaussian_buffer: &wgpu::Buffer,
        rendered_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("L-M Render Bind Group"),
            layout: &self.render_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gaussian_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rendered_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("L-M Render Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("L-M Render Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
        }

        queue.submit(Some(encoder.finish()));
    }

    /// Apply parameter update on GPU
    fn apply_update_gpu(
        &self,
        device: &Device,
        queue: &Queue,
        gaussian_buffer: &wgpu::Buffer,
        delta_buffer: &wgpu::Buffer,
        num_gaussians: usize,
    ) -> Result<()> {
        let params = JacobianParams {
            width: 0,
            height: 0,
            num_gaussians: num_gaussians as u32,
            num_params: (num_gaussians * PARAMS_PER_GAUSSIAN) as u32,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("L-M Update Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("L-M Update Bind Group"),
            layout: &self.update_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gaussian_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: delta_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("L-M Update Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("L-M Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((num_gaussians as u32 + 63) / 64, 1, 1);
        }

        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Compute loss on GPU
    fn compute_loss_gpu(
        &self,
        device: &Device,
        queue: &Queue,
        rendered_buffer: &wgpu::Buffer,
        target_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> Result<f32> {
        let n_pixels = (width * height) as usize;

        // Read rendered and target
        let rendered = self.read_image(device, queue, rendered_buffer, n_pixels)?;
        let target = self.read_image(device, queue, target_buffer, n_pixels)?;

        // Compute MSE
        let mut mse = 0.0f32;
        for i in 0..n_pixels {
            mse += (rendered[i * 4] - target[i * 4]).powi(2);     // R
            mse += (rendered[i * 4 + 1] - target[i * 4 + 1]).powi(2); // G
            mse += (rendered[i * 4 + 2] - target[i * 4 + 2]).powi(2); // B
        }
        mse /= (n_pixels * 3) as f32;

        Ok(mse)
    }

    /// Solve (J^T J + λI) δ = -J^T r using Cholesky decomposition
    fn solve_damped_system(&self, jtj: &[f32], jtr: &[f32], lambda: f32, n: usize) -> Vec<f32> {
        // Symmetrize J^T J: A = (J^T J + (J^T J)^T) / 2
        // This is theoretically unnecessary but helps with numerical stability
        // when accumulating from many GPU threads
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let val_ij = jtj[i * n + j];
                let val_ji = jtj[j * n + i];
                a[i * n + j] = (val_ij + val_ji) * 0.5;
            }
        }

        // Add damping: A = J^T J + λI
        for i in 0..n {
            a[i * n + i] += lambda;
        }

        // Solve using simple Cholesky (for now, could use LAPACK for production)
        // For simplicity, use Gaussian elimination with partial pivoting
        let b: Vec<f32> = jtr.iter().map(|&x| -x).collect();

        self.solve_linear_system(&a, &b, n)
    }

    /// Simple linear system solver (Gaussian elimination)
    fn solve_linear_system(&self, a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
        let mut a = a.to_vec();
        let mut b = b.to_vec();

        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_val = a[k * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                if a[i * n + k].abs() > max_val {
                    max_val = a[i * n + k].abs();
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                for j in 0..n {
                    a.swap(k * n + j, max_row * n + j);
                }
                b.swap(k, max_row);
            }

            // Check for singularity
            if a[k * n + k].abs() < 1e-10 {
                continue;
            }

            // Eliminate
            for i in (k + 1)..n {
                let factor = a[i * n + k] / a[k * n + k];
                for j in k..n {
                    a[i * n + j] -= factor * a[k * n + j];
                }
                b[i] -= factor * b[k];
            }
        }

        // Back substitution
        let mut x = vec![0.0f32; n];
        for i in (0..n).rev() {
            if a[i * n + i].abs() < 1e-10 {
                continue;
            }
            x[i] = b[i];
            for j in (i + 1)..n {
                x[i] -= a[i * n + j] * x[j];
            }
            x[i] /= a[i * n + i];
        }

        x
    }

    /// Read image from GPU buffer
    fn read_image(&self, device: &Device, queue: &Queue, buffer: &wgpu::Buffer, n_pixels: usize) -> Result<Vec<f32>> {
        let size = (n_pixels * 16) as u64; // 4 floats per pixel
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("L-M Image Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        let submission_index = queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(std::time::Duration::from_secs(10)),
        });

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Read Gaussians from GPU buffer
    fn read_gaussians(
        &self,
        device: &Device,
        queue: &Queue,
        buffer: &wgpu::Buffer,
        num_gaussians: usize,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        let size = (num_gaussians * std::mem::size_of::<ShaderGaussian>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("L-M Gaussian Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        let submission_index = queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(std::time::Duration::from_secs(10)),
        });

        let data = slice.get_mapped_range();
        let gpu_gaussians: Vec<ShaderGaussian> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(gpu_gaussians.into_iter().map(|g| g.to_gaussian()).collect())
    }
}
