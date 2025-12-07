//! GPU-accelerated L-BFGS optimizer for Gaussian parameters
//! Copyright (c) 2025 Lamco Development
//!
//! Hybrid implementation:
//! - GPU: Gradient computation, rendering (line search), parameter updates
//! - CPU: Two-loop recursion (inherently sequential, history size m is small)
//!
//! Key lesson from Adam GPU implementation: WGSL array stride alignment is critical.
//! All structs in storage buffer arrays must be padded to 8-byte alignment (vec2).

use crate::{GpuError, Result};
use crate::adam_optimizer::ShaderGaussian;
use crate::gradient::GpuGradient;
use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, color::Color4};
use wgpu::{Device, Queue};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::collections::VecDeque;

/// L-BFGS optimizer configuration
#[derive(Clone, Debug)]
pub struct LbfgsConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// History size (m parameter, typically 5-20)
    pub history_size: usize,
    /// Convergence tolerance (gradient norm)
    pub tolerance: f32,
    /// Armijo line search constant (typically 1e-4)
    pub c1: f32,
    /// Maximum line search iterations
    pub max_line_search_iters: usize,
    /// Initial step size for line search
    pub initial_step: f32,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            history_size: 10,
            tolerance: 1e-6,
            c1: 1e-4,
            max_line_search_iters: 20,
            initial_step: 1.0,
        }
    }
}

/// Search direction struct for GPU update
/// CRITICAL: Must match WGSL layout with 48-byte stride (8-byte alignment)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SearchDirection {
    pub d_position: [f32; 2],   // 8 bytes @ offset 0
    pub d_scale_x: f32,         // 4 bytes @ offset 8
    pub d_scale_y: f32,         // 4 bytes @ offset 12
    pub d_rotation: f32,        // 4 bytes @ offset 16
    pub d_color: [f32; 4],      // 16 bytes @ offset 20
    pub d_opacity: f32,         // 4 bytes @ offset 36
    pub _padding: [f32; 2],     // 8 bytes @ offset 40 - CRITICAL: 48 bytes total!
}

/// L-BFGS update params for GPU shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LbfgsUpdateParams {
    pub step_size: f32,
    pub num_gaussians: u32,
    pub position_min: f32,
    pub position_max: f32,
    pub color_min: f32,
    pub color_max: f32,
    pub scale_min: f32,
    pub scale_max: f32,
    pub opacity_min: f32,
    pub opacity_max: f32,
    pub _padding: [u32; 2],     // Pad to 48 bytes
}

/// Compute params for gradient shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ComputeParams {
    width: u32,
    height: u32,
    num_gaussians: u32,
    _padding: u32,
}

/// GPU L-BFGS optimizer
pub struct GpuLbfgsOptimizer {
    // Pipelines
    render_pipeline: wgpu::ComputePipeline,
    gradient_pipeline: wgpu::ComputePipeline,
    lbfgs_update_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    render_layout: wgpu::BindGroupLayout,
    gradient_layout: wgpu::BindGroupLayout,
    lbfgs_layout: wgpu::BindGroupLayout,

    // L-BFGS state (on CPU for two-loop recursion)
    s_history: VecDeque<Vec<f32>>,  // Position differences
    y_history: VecDeque<Vec<f32>>,  // Gradient differences
    prev_params: Option<Vec<f32>>,
    prev_gradient: Option<Vec<f32>>,

    // Configuration
    pub config: LbfgsConfig,
}

impl GpuLbfgsOptimizer {
    /// Number of parameters per Gaussian (must match flattening)
    const PARAMS_PER_GAUSSIAN: usize = 10;  // pos(2) + scale(2) + rot(1) + color(4) + opacity(1)

    /// Create new GPU L-BFGS optimizer
    pub fn new(device: &Device) -> Result<Self> {
        // Load shaders
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("L-BFGS Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gaussian_render.wgsl").into()),
        });

        let gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("L-BFGS Gradient Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gradient_compute.wgsl").into()),
        });

        let lbfgs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("L-BFGS Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lbfgs_update.wgsl").into()),
        });

        // Create bind group layouts (reuse Adam's layouts where possible)
        let render_layout = Self::create_render_layout(device);
        let gradient_layout = Self::create_gradient_layout(device);
        let lbfgs_layout = Self::create_lbfgs_layout(device);

        // Create pipelines
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("L-BFGS Render Pipeline Layout"),
            bind_group_layouts: &[&render_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("L-BFGS Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            module: &render_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let gradient_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("L-BFGS Gradient Pipeline Layout"),
            bind_group_layouts: &[&gradient_layout],
            push_constant_ranges: &[],
        });

        let gradient_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("L-BFGS Gradient Pipeline"),
            layout: Some(&gradient_pipeline_layout),
            module: &gradient_shader,
            entry_point: Some("compute_gradients_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let lbfgs_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("L-BFGS Update Pipeline Layout"),
            bind_group_layouts: &[&lbfgs_layout],
            push_constant_ranges: &[],
        });

        let lbfgs_update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("L-BFGS Update Pipeline"),
            layout: Some(&lbfgs_pipeline_layout),
            module: &lbfgs_shader,
            entry_point: Some("lbfgs_update_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            render_pipeline,
            gradient_pipeline,
            lbfgs_update_pipeline,
            render_layout,
            gradient_layout,
            lbfgs_layout,
            s_history: VecDeque::with_capacity(10),
            y_history: VecDeque::with_capacity(10),
            prev_params: None,
            prev_gradient: None,
            config: LbfgsConfig::default(),
        })
    }

    fn create_render_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("L-BFGS Render Bind Group Layout"),
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
                // Output image (write)
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
                // Params (uniform)
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
        })
    }

    fn create_gradient_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("L-BFGS Gradient Bind Group Layout"),
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
                // Rendered image (read)
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
                // Target image (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Gradients (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_lbfgs_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("L-BFGS Update Bind Group Layout"),
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
                // Search direction (read)
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
                // Params (uniform)
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
        })
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
        self.prev_params = None;
        self.prev_gradient = None;
    }

    /// Flatten Gaussians to parameter vector
    fn flatten_gaussians(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Vec<f32> {
        let mut params = Vec::with_capacity(gaussians.len() * Self::PARAMS_PER_GAUSSIAN);
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

    /// Flatten GPU gradients to vector
    fn flatten_gradients(&self, gradients: &[GpuGradient]) -> Vec<f32> {
        let mut params = Vec::with_capacity(gradients.len() * Self::PARAMS_PER_GAUSSIAN);
        for g in gradients {
            params.push(g.d_position[0]);
            params.push(g.d_position[1]);
            params.push(g.d_scale_x);
            params.push(g.d_scale_y);
            params.push(g.d_rotation);
            params.push(g.d_color[0]);
            params.push(g.d_color[1]);
            params.push(g.d_color[2]);
            params.push(g.d_color[3]);
            params.push(g.d_opacity);
        }
        params
    }

    /// Convert search direction to GPU format
    fn direction_to_gpu(&self, direction: &[f32], num_gaussians: usize) -> Vec<SearchDirection> {
        let mut result = Vec::with_capacity(num_gaussians);
        for i in 0..num_gaussians {
            let base = i * Self::PARAMS_PER_GAUSSIAN;
            result.push(SearchDirection {
                d_position: [direction[base], direction[base + 1]],
                d_scale_x: direction[base + 2],
                d_scale_y: direction[base + 3],
                d_rotation: direction[base + 4],
                d_color: [direction[base + 5], direction[base + 6], direction[base + 7], direction[base + 8]],
                d_opacity: direction[base + 9],
                _padding: [0.0; 2],
            });
        }
        result
    }

    /// L-BFGS two-loop recursion (CPU, as it's inherently sequential)
    fn compute_direction(&self, gradient: &[f32]) -> Vec<f32> {
        let n = gradient.len();
        let m = self.s_history.len();

        if m == 0 {
            // First iteration: use steepest descent
            return gradient.iter().map(|&g| -g).collect();
        }

        // Two-loop recursion algorithm
        let mut q = gradient.to_vec();
        let mut alpha = vec![0.0f32; m];

        // First loop (backward)
        for i in (0..m).rev() {
            let rho_i = 1.0 / Self::dot_product(&self.y_history[i], &self.s_history[i]);
            if rho_i.is_finite() {
                alpha[i] = rho_i * Self::dot_product(&self.s_history[i], &q);
                for j in 0..n {
                    q[j] -= alpha[i] * self.y_history[i][j];
                }
            }
        }

        // Initial Hessian approximation: H_0 = γI
        // γ = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
        let gamma = {
            let last_s = &self.s_history[m - 1];
            let last_y = &self.y_history[m - 1];
            let sy = Self::dot_product(last_s, last_y);
            let yy = Self::dot_product(last_y, last_y);
            if yy > 1e-10 { sy / yy } else { 1.0 }
        };

        let mut r: Vec<f32> = q.iter().map(|&qi| gamma * qi).collect();

        // Second loop (forward)
        for i in 0..m {
            let rho_i = 1.0 / Self::dot_product(&self.y_history[i], &self.s_history[i]);
            if rho_i.is_finite() {
                let beta = rho_i * Self::dot_product(&self.y_history[i], &r);
                for j in 0..n {
                    r[j] += self.s_history[i][j] * (alpha[i] - beta);
                }
            }
        }

        // Return search direction (negative because we're minimizing)
        r.iter().map(|&ri| -ri).collect()
    }

    /// Dot product helper
    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
    }

    /// Update L-BFGS history
    fn update_history(&mut self, params: &[f32], gradient: &[f32]) {
        if let (Some(prev_p), Some(prev_g)) = (&self.prev_params, &self.prev_gradient) {
            let n = params.len();

            // s_k = x_{k+1} - x_k
            let s: Vec<f32> = (0..n).map(|i| params[i] - prev_p[i]).collect();

            // y_k = g_{k+1} - g_k
            let y: Vec<f32> = (0..n).map(|i| gradient[i] - prev_g[i]).collect();

            // Curvature condition: s^T y > 0
            let sy = Self::dot_product(&s, &y);
            if sy > 1e-10 {
                if self.s_history.len() >= self.config.history_size {
                    self.s_history.pop_front();
                    self.y_history.pop_front();
                }
                self.s_history.push_back(s);
                self.y_history.push_back(y);
            }
        }

        self.prev_params = Some(params.to_vec());
        self.prev_gradient = Some(gradient.to_vec());
    }

    /// Run optimization
    pub fn optimize(
        &mut self,
        device: &Device,
        queue: &Queue,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> Result<f32> {
        self.reset();

        let num_gaussians = gaussians.len();
        let width = target.width;
        let height = target.height;

        // Convert to GPU format
        let gpu_gaussians: Vec<ShaderGaussian> = gaussians
            .iter()
            .map(ShaderGaussian::from_gaussian)
            .collect();

        // Create GPU buffers
        let gaussian_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("L-BFGS Gaussian Buffer"),
            contents: bytemuck::cast_slice(&gpu_gaussians),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let rendered_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("L-BFGS Rendered Buffer"),
            size: (width * height * 4 * std::mem::size_of::<f32>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Upload target image
        let target_data: Vec<[f32; 4]> = (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                if let Some(p) = target.get_pixel(x, y) {
                    [p.r, p.g, p.b, p.a]
                } else {
                    [0.0; 4]
                }
            })
            .collect();

        let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("L-BFGS Target Buffer"),
            contents: bytemuck::cast_slice(&target_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let gradient_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("L-BFGS Gradient Buffer"),
            size: (num_gaussians * std::mem::size_of::<GpuGradient>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Render params
        let render_params = crate::buffer::RenderParams {
            width,
            height,
            gaussian_count: num_gaussians as u32,
            render_mode: 1,  // AccumulatedSum
            cutoff_threshold: 0.001,
            n_sigma: 3.5,
            _padding: [0; 2],
        };

        let render_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("L-BFGS Render Params Buffer"),
            contents: bytemuck::bytes_of(&render_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let gradient_params = ComputeParams {
            width,
            height,
            num_gaussians: num_gaussians as u32,
            _padding: 0,
        };

        let gradient_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("L-BFGS Gradient Params Buffer"),
            contents: bytemuck::bytes_of(&gradient_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Best state tracking
        let mut best_loss = f32::INFINITY;
        let mut best_gaussians = gpu_gaussians.clone();
        let mut best_iteration = 0;

        // Compute initial loss
        let initial_loss = self.compute_loss(gaussians, target);
        println!("  GPU L-BFGS initial loss: {:.6}", initial_loss);
        best_loss = initial_loss;

        // CRITICAL FIX: Initialize prev_params/prev_gradient BEFORE the loop
        // This ensures the first (s,y) pair is captured correctly.
        // Without this, the first iteration skips history update because prev_params is None,
        // causing GPU L-BFGS to start using quasi-Newton directions one iteration later than CPU.
        {
            let initial_grads = self.compute_gradients_gpu(
                device, queue,
                &gaussian_buffer, &rendered_buffer, &target_buffer,
                &gradient_buffer, &render_params_buffer, &gradient_params_buffer,
                num_gaussians, width, height,
            )?;
            let initial_gaussians = self.read_gaussians(device, queue, &gaussian_buffer, num_gaussians)?;
            let initial_params = self.flatten_gaussians(&initial_gaussians);
            let initial_grad_flat = self.flatten_gradients(&initial_grads);

            self.prev_params = Some(initial_params);
            self.prev_gradient = Some(initial_grad_flat);
        }

        for iter in 1..=self.config.max_iterations {
            // Step 1: Render and compute gradients on GPU
            let gradients = self.compute_gradients_gpu(
                device, queue,
                &gaussian_buffer, &rendered_buffer, &target_buffer,
                &gradient_buffer, &render_params_buffer, &gradient_params_buffer,
                num_gaussians, width, height,
            )?;

            // Step 2: Flatten current state and gradients
            let current_gaussians = self.read_gaussians(device, queue, &gaussian_buffer, num_gaussians)?;
            let params = self.flatten_gaussians(&current_gaussians);
            let grad_flat = self.flatten_gradients(&gradients);

            // Check convergence
            let grad_norm: f32 = grad_flat.iter().map(|&g| g * g).sum::<f32>().sqrt();
            if grad_norm < self.config.tolerance {
                println!("  L-BFGS converged at iteration {} (grad_norm: {:.2e})", iter, grad_norm);
                break;
            }

            // Step 3: Compute search direction (CPU, two-loop recursion)
            let direction = self.compute_direction(&grad_flat);

            // Step 4: Line search (using GPU for loss evaluation)
            let directional_deriv = Self::dot_product(&grad_flat, &direction);

            // Match CPU behavior: if not descent direction, use tiny step rather than resetting
            // CPU line_search returns 1e-8 in this case, it doesn't reset the optimizer
            let step_size = if directional_deriv >= 0.0 {
                1e-8  // Tiny step, preserves history
            } else {
                // CRITICAL: Use current loss, not initial_loss!
                // Armijo condition must compare against f(x), not f(x_0)
                let current_loss = self.compute_loss(&current_gaussians, target);

                self.line_search_gpu(
                    device, queue,
                    &gaussian_buffer, &rendered_buffer, &target_buffer,
                    &render_params_buffer, &gradient_buffer, &gradient_params_buffer,
                    &direction, num_gaussians, width, height,
                    current_loss, directional_deriv, target,
                )?
            };

            // Step 5: Update parameters on GPU
            self.apply_update_gpu(
                device, queue,
                &gaussian_buffer,
                &direction, step_size, num_gaussians,
            )?;

            // Step 6: Update L-BFGS history
            let new_gaussians = self.read_gaussians(device, queue, &gaussian_buffer, num_gaussians)?;
            let new_params = self.flatten_gaussians(&new_gaussians);

            // Compute new gradients for history update
            let new_grads = self.compute_gradients_gpu(
                device, queue,
                &gaussian_buffer, &rendered_buffer, &target_buffer,
                &gradient_buffer, &render_params_buffer, &gradient_params_buffer,
                num_gaussians, width, height,
            )?;
            let new_grad_flat = self.flatten_gradients(&new_grads);

            self.update_history(&new_params, &new_grad_flat);

            // Compute loss and track best
            let loss = self.compute_loss(&new_gaussians, target);

            if loss < best_loss {
                best_loss = loss;
                best_gaussians = new_gaussians.iter().map(ShaderGaussian::from_gaussian).collect();
                best_iteration = iter;
            }

            if iter % 10 == 0 {
                println!("  GPU L-BFGS iteration {}/{}: loss = {:.6} (step = {:.4})",
                         iter, self.config.max_iterations, loss, step_size);
            }
        }

        // Restore best state
        println!("  Restored best from iteration {}/{} (loss: {:.6})", best_iteration, self.config.max_iterations, best_loss);

        for (i, gpu_g) in best_gaussians.iter().enumerate() {
            gaussians[i] = gpu_g.to_gaussian();
        }

        Ok(best_loss)
    }

    /// Compute gradients on GPU
    fn compute_gradients_gpu(
        &self,
        device: &Device,
        queue: &Queue,
        gaussian_buffer: &wgpu::Buffer,
        rendered_buffer: &wgpu::Buffer,
        target_buffer: &wgpu::Buffer,
        gradient_buffer: &wgpu::Buffer,
        render_params_buffer: &wgpu::Buffer,
        gradient_params_buffer: &wgpu::Buffer,
        num_gaussians: usize,
        width: u32,
        height: u32,
    ) -> Result<Vec<GpuGradient>> {
        // Create bind groups
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("L-BFGS Render Bind Group"),
            layout: &self.render_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gaussian_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rendered_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: render_params_buffer.as_entire_binding() },
            ],
        });

        let gradient_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("L-BFGS Gradient Bind Group"),
            layout: &self.gradient_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gaussian_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rendered_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: target_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gradient_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gradient_params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("L-BFGS Gradient Encoder"),
        });

        // Render pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("L-BFGS Render Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &render_bind_group, &[]);
            pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
        }

        // Gradient pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("L-BFGS Gradient Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gradient_pipeline);
            pass.set_bind_group(0, &gradient_bind_group, &[]);
            pass.dispatch_workgroups((num_gaussians as u32 + 255) / 256, 1, 1);
        }

        // Read back gradients
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("L-BFGS Gradient Staging"),
            size: gradient_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(gradient_buffer, 0, &staging_buffer, 0, gradient_buffer.size());
        let submission_index = queue.submit(std::iter::once(encoder.finish()));

        // Wait and read
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        if let Err(_) = device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(std::time::Duration::from_secs(5)),
        }) {
            return Err(GpuError::BufferMapping);
        }

        pollster::block_on(async {
            receiver.receive().await.ok_or(GpuError::BufferMapping)?
                .map_err(|_| GpuError::BufferMapping)
        })?;

        let data = buffer_slice.get_mapped_range();
        let gradients: Vec<GpuGradient> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(gradients)
    }

    /// Read Gaussians from GPU
    fn read_gaussians(
        &self,
        device: &Device,
        queue: &Queue,
        gaussian_buffer: &wgpu::Buffer,
        num_gaussians: usize,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("L-BFGS Gaussian Staging"),
            size: gaussian_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("L-BFGS Read Gaussians Encoder"),
        });
        encoder.copy_buffer_to_buffer(gaussian_buffer, 0, &staging_buffer, 0, gaussian_buffer.size());
        let submission_index = queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        if let Err(_) = device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(std::time::Duration::from_secs(5)),
        }) {
            return Err(GpuError::BufferMapping);
        }

        pollster::block_on(async {
            receiver.receive().await.ok_or(GpuError::BufferMapping)?
                .map_err(|_| GpuError::BufferMapping)
        })?;

        let data = buffer_slice.get_mapped_range();
        let gpu_gaussians: &[ShaderGaussian] = bytemuck::cast_slice(&data);
        let gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = gpu_gaussians.iter().map(|g| g.to_gaussian()).collect();
        drop(data);
        staging_buffer.unmap();

        Ok(gaussians)
    }

    /// Line search using GPU for loss evaluation
    fn line_search_gpu(
        &self,
        device: &Device,
        queue: &Queue,
        gaussian_buffer: &wgpu::Buffer,
        rendered_buffer: &wgpu::Buffer,
        target_buffer: &wgpu::Buffer,
        render_params_buffer: &wgpu::Buffer,
        gradient_buffer: &wgpu::Buffer,
        gradient_params_buffer: &wgpu::Buffer,
        direction: &[f32],
        num_gaussians: usize,
        width: u32,
        height: u32,
        f_x: f32,
        directional_deriv: f32,
        target: &ImageBuffer<f32>,
    ) -> Result<f32> {
        let mut alpha = self.config.initial_step;

        // Save original state
        let original_gaussians = self.read_gaussians(device, queue, gaussian_buffer, num_gaussians)?;

        for _ in 0..self.config.max_line_search_iters {
            // Apply tentative step
            self.apply_update_gpu(device, queue, gaussian_buffer, direction, alpha, num_gaussians)?;

            // Compute loss at new position using actual target
            let trial_gaussians = self.read_gaussians(device, queue, gaussian_buffer, num_gaussians)?;
            let f_new = self.compute_loss(&trial_gaussians, target);

            // Restore original and check Armijo condition
            let gpu_original: Vec<ShaderGaussian> = original_gaussians.iter()
                .map(ShaderGaussian::from_gaussian)
                .collect();
            queue.write_buffer(gaussian_buffer, 0, bytemuck::cast_slice(&gpu_original));

            // Armijo condition: f(x + αp) ≤ f(x) + c1·α·∇f^T·p
            if f_new <= f_x + self.config.c1 * alpha * directional_deriv {
                return Ok(alpha);
            }

            alpha *= 0.5;
        }

        // Return small step if line search fails
        Ok(alpha)
    }

    /// Apply L-BFGS update on GPU
    fn apply_update_gpu(
        &self,
        device: &Device,
        queue: &Queue,
        gaussian_buffer: &wgpu::Buffer,
        direction: &[f32],
        step_size: f32,
        num_gaussians: usize,
    ) -> Result<()> {
        // Convert direction to GPU format
        let gpu_direction = self.direction_to_gpu(direction, num_gaussians);

        let direction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("L-BFGS Direction Buffer"),
            contents: bytemuck::cast_slice(&gpu_direction),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let update_params = LbfgsUpdateParams {
            step_size,
            num_gaussians: num_gaussians as u32,
            position_min: 0.0,
            position_max: 1.0,
            color_min: 0.0,
            color_max: 1.0,
            scale_min: 0.001,
            scale_max: 0.5,
            opacity_min: 0.01,
            opacity_max: 1.0,
            _padding: [0; 2],
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("L-BFGS Update Params Buffer"),
            contents: bytemuck::bytes_of(&update_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("L-BFGS Update Bind Group"),
            layout: &self.lbfgs_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gaussian_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: direction_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("L-BFGS Update Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("L-BFGS Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.lbfgs_update_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((num_gaussians as u32 + 255) / 256, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    /// Compute loss (CPU, matching GPU render)
    fn compute_loss(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) -> f32 {
        let width = target.width;
        let height = target.height;
        let mut loss = 0.0;

        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / width as f32;
                let py = y as f32 / height as f32;

                let mut weight_sum = 0.0f32;
                let mut color_sum = Color4::new(0.0f32, 0.0, 0.0, 0.0);

                for gaussian in gaussians {
                    let dx = px - gaussian.position.x;
                    let dy = py - gaussian.position.y;

                    let cos_t = gaussian.shape.rotation.cos();
                    let sin_t = gaussian.shape.rotation.sin();
                    let dx_rot = dx * cos_t + dy * sin_t;
                    let dy_rot = -dx * sin_t + dy * cos_t;

                    let sx = gaussian.shape.scale_x.max(0.001);
                    let sy = gaussian.shape.scale_y.max(0.001);
                    let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);

                    if dist_sq > 12.25 { continue; }

                    let gaussian_val = (-0.5 * dist_sq).exp();
                    let weight = gaussian.opacity * gaussian_val;

                    weight_sum += weight;
                    color_sum.r += weight * gaussian.color.r;
                    color_sum.g += weight * gaussian.color.g;
                    color_sum.b += weight * gaussian.color.b;
                }

                let rendered = if weight_sum > 1e-10 {
                    Color4::new(color_sum.r / weight_sum, color_sum.g / weight_sum, color_sum.b / weight_sum, 1.0)
                } else {
                    Color4::new(0.0, 0.0, 0.0, 1.0)
                };

                if let Some(target_color) = target.get_pixel(x, y) {
                    loss += (rendered.r - target_color.r).powi(2);
                    loss += (rendered.g - target_color.g).powi(2);
                    loss += (rendered.b - target_color.b).powi(2);
                }
            }
        }

        loss / (width * height * 3) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        // CRITICAL: Verify struct sizes match shader expectations
        assert_eq!(std::mem::size_of::<SearchDirection>(), 48, "SearchDirection must be 48 bytes");
        assert_eq!(std::mem::size_of::<LbfgsUpdateParams>(), 48, "LbfgsUpdateParams must be 48 bytes");
    }
}
