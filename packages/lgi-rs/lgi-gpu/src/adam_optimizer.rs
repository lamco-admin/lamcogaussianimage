//! GPU-accelerated Adam optimizer for Gaussian parameters
//! Copyright (c) 2025 Lamco Development
//!
//! Performs full optimization loop on GPU without CPU round-trips.
//! This eliminates the GPU→CPU→GPU memory transfer bottleneck.

use crate::{GpuError, Result};
use crate::buffer::GpuGaussian;
use crate::gradient::GpuGradient;
use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};
use wgpu::{Device, Queue};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

/// Per-parameter learning rates (matches CPU AdamOptimizer)
#[derive(Clone, Debug)]
pub struct GpuLearningRates {
    pub position: f32,
    pub position_final: f32,  // Final position LR after decay
    pub color: f32,
    pub scale: f32,
    pub opacity: f32,
    pub rotation: f32,
}

impl Default for GpuLearningRates {
    fn default() -> Self {
        Self {
            position: 0.0002,
            position_final: 0.00002,  // 10× decay over max_iterations
            color: 0.02,
            scale: 0.005,
            opacity: 0.05,
            rotation: 0.01,
        }
    }
}

impl GpuLearningRates {
    /// Calculate exponentially decayed position learning rate
    /// Decay from initial to final over max_iterations (matches CPU AdamOptimizer)
    pub fn position_lr_at_iteration(&self, t: usize, max_iterations: usize) -> f32 {
        let progress = t as f32 / max_iterations as f32;
        let log_initial = self.position.ln();
        let log_final = self.position_final.ln();
        (log_initial + progress * (log_final - log_initial)).exp()
    }
}

/// Adam optimizer parameters for GPU shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct AdamParams {
    // Per-parameter learning rates
    pub lr_position: f32,
    pub lr_color: f32,
    pub lr_scale: f32,
    pub lr_opacity: f32,
    pub lr_rotation: f32,

    // Adam hyperparameters
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,

    // Bias correction factors
    pub bias_correction1: f32,
    pub bias_correction2: f32,

    // Clamp ranges
    pub position_min: f32,
    pub position_max: f32,
    pub color_min: f32,
    pub color_max: f32,
    pub scale_min: f32,
    pub scale_max: f32,
    pub opacity_min: f32,
    pub opacity_max: f32,

    pub num_gaussians: u32,
    // WGSL vec3<u32> has 16-byte alignment, so we need explicit padding
    pub _pad1: u32,          // Aligns next field to 16 bytes (offset 80)
    pub _padding: [u32; 3],  // Matches vec3<u32> (12 bytes, offset 80-92)
    pub _pad2: u32,          // Rounds struct to 16 bytes (total: 96)
}

/// Adam first moment (must match shader layout)
/// CRITICAL: WGSL pads structs in arrays to alignment of largest field (vec2 = 8 bytes)
/// Size: 44 bytes content + 4 bytes padding = 48 bytes total stride
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct AdamMoment {
    pub m_position: [f32; 2],   // 8 bytes @ offset 0
    pub m_scale_x: f32,         // 4 bytes @ offset 8
    pub m_scale_y: f32,         // 4 bytes @ offset 12
    pub m_rotation: f32,        // 4 bytes @ offset 16
    pub m_color: [f32; 4],      // 16 bytes @ offset 20
    pub m_opacity: f32,         // 4 bytes @ offset 36
    pub _padding: [f32; 2],     // 8 bytes @ offset 40 - CRITICAL: 48 bytes total!
}

/// Adam second moment (must match shader layout)
/// CRITICAL: WGSL pads structs in arrays to alignment of largest field (vec2 = 8 bytes)
/// Size: 44 bytes content + 4 bytes padding = 48 bytes total stride
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct AdamVariance {
    pub v_position: [f32; 2],   // 8 bytes @ offset 0
    pub v_scale_x: f32,         // 4 bytes @ offset 8
    pub v_scale_y: f32,         // 4 bytes @ offset 12
    pub v_rotation: f32,        // 4 bytes @ offset 16
    pub v_color: [f32; 4],      // 16 bytes @ offset 20
    pub v_opacity: f32,         // 4 bytes @ offset 36
    pub _padding: [f32; 2],     // 8 bytes @ offset 40 - CRITICAL: 48 bytes total!
}

/// Gaussian struct matching gradient_compute.wgsl layout
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ShaderGaussian {
    pub position: [f32; 2],
    pub scale_x: f32,
    pub scale_y: f32,
    pub rotation: f32,
    pub color: [f32; 4],
    pub opacity: f32,
    pub _padding: [u32; 2],
}

impl ShaderGaussian {
    /// Convert from Gaussian2D
    pub fn from_gaussian(g: &Gaussian2D<f32, Euler<f32>>) -> Self {
        Self {
            position: [g.position.x, g.position.y],
            scale_x: g.shape.scale_x,
            scale_y: g.shape.scale_y,
            rotation: g.shape.rotation,
            color: [g.color.r, g.color.g, g.color.b, g.color.a],
            opacity: g.opacity,
            _padding: [0, 0],
        }
    }

    /// Convert to Gaussian2D
    pub fn to_gaussian(&self) -> Gaussian2D<f32, Euler<f32>> {
        Gaussian2D {
            position: lgi_math::vec::Vector2 { x: self.position[0], y: self.position[1] },
            shape: Euler {
                scale_x: self.scale_x,
                scale_y: self.scale_y,
                rotation: self.rotation,
            },
            color: lgi_math::color::Color4::new(
                self.color[0],
                self.color[1],
                self.color[2],
                self.color[3],
            ),
            opacity: self.opacity,
            weight: None,
        }
    }
}

/// Compute parameters for gradient shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ComputeParams {
    width: u32,
    height: u32,
    num_gaussians: u32,
    _padding: u32,
}

/// GPU Adam optimizer with persistent buffers
pub struct GpuAdamOptimizer {
    // Pipelines
    render_pipeline: wgpu::ComputePipeline,
    gradient_pipeline: wgpu::ComputePipeline,
    adam_pipeline: wgpu::ComputePipeline,
    reset_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    render_layout: wgpu::BindGroupLayout,
    gradient_layout: wgpu::BindGroupLayout,
    adam_layout: wgpu::BindGroupLayout,

    // Hyperparameters
    pub learning_rates: GpuLearningRates,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub max_iterations: usize,
}

impl GpuAdamOptimizer {
    /// Create new GPU Adam optimizer
    pub fn new(device: &Device) -> Result<Self> {
        // Load shaders
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gaussian_render.wgsl").into()),
        });

        let gradient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gradient Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gradient_compute.wgsl").into()),
        });

        let adam_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Adam Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/adam_update.wgsl").into()),
        });

        // Create bind group layouts
        let render_layout = Self::create_render_layout(device);
        let gradient_layout = Self::create_gradient_layout(device);
        let adam_layout = Self::create_adam_layout(device);

        // Create pipelines
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&render_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            module: &render_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let gradient_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gradient Pipeline Layout"),
            bind_group_layouts: &[&gradient_layout],
            push_constant_ranges: &[],
        });

        let gradient_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gradient Pipeline"),
            layout: Some(&gradient_pipeline_layout),
            module: &gradient_shader,
            entry_point: Some("compute_gradients_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let adam_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Adam Pipeline Layout"),
            bind_group_layouts: &[&adam_layout],
            push_constant_ranges: &[],
        });

        let adam_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Adam Update Pipeline"),
            layout: Some(&adam_pipeline_layout),
            module: &adam_shader,
            entry_point: Some("adam_update_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let reset_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reset Moments Pipeline"),
            layout: Some(&adam_pipeline_layout),
            module: &adam_shader,
            entry_point: Some("reset_moments"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            render_pipeline,
            gradient_pipeline,
            adam_pipeline,
            reset_pipeline,
            render_layout,
            gradient_layout,
            adam_layout,
            learning_rates: GpuLearningRates::default(),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            max_iterations: 200,
        })
    }

    fn create_render_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
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
            label: Some("Gradient Bind Group Layout"),
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

    fn create_adam_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Adam Bind Group Layout"),
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
                // Gradients (read)
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
                // First moments m (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Second moments v (read-write)
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
                // Adam params (uniform)
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

    /// Run full optimization on GPU
    ///
    /// This keeps all data on GPU for the entire optimization loop,
    /// only reading back final results.
    pub fn optimize(
        &self,
        device: &Device,
        queue: &Queue,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> Result<f32> {
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
            label: Some("Gaussian Buffer"),
            contents: bytemuck::cast_slice(&gpu_gaussians),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let rendered_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rendered Buffer"),
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
            label: Some("Target Buffer"),
            contents: bytemuck::cast_slice(&target_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let gradient_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gradient Buffer"),
            size: (num_gaussians * std::mem::size_of::<GpuGradient>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Initialize moment buffers to zero
        let zero_moments: Vec<AdamMoment> = vec![AdamMoment {
            m_position: [0.0, 0.0],
            m_scale_x: 0.0,
            m_scale_y: 0.0,
            m_rotation: 0.0,
            m_color: [0.0; 4],
            m_opacity: 0.0,
            _padding: [0.0; 2],  // 48 bytes total
        }; num_gaussians];

        let zero_variances: Vec<AdamVariance> = vec![AdamVariance {
            v_position: [0.0, 0.0],
            v_scale_x: 0.0,
            v_scale_y: 0.0,
            v_rotation: 0.0,
            v_color: [0.0; 4],
            v_opacity: 0.0,
            _padding: [0.0; 2],  // 48 bytes total
        }; num_gaussians];

        let moment_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Moment M Buffer"),
            contents: bytemuck::cast_slice(&zero_moments),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let variance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Variance V Buffer"),
            contents: bytemuck::cast_slice(&zero_variances),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Render params - use AccumulatedSum (1) to match CPU RendererV2
        let render_params = crate::buffer::RenderParams {
            width,
            height,
            gaussian_count: num_gaussians as u32,
            render_mode: 1,  // 1 = AccumulatedSum (weighted normalization, matches CPU RendererV2)
            cutoff_threshold: 0.001,
            n_sigma: 3.5,
            _padding: [0; 2],
        };

        let render_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Render Params Buffer"),
            contents: bytemuck::bytes_of(&render_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Gradient compute params
        let gradient_params = ComputeParams {
            width,
            height,
            num_gaussians: num_gaussians as u32,
            _padding: 0,
        };

        let gradient_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gradient Params Buffer"),
            contents: bytemuck::bytes_of(&gradient_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Best-state tracking
        let mut best_loss = f32::INFINITY;
        let mut best_gaussians: Vec<ShaderGaussian> = gpu_gaussians.clone();
        let mut best_iteration = 0;
        let loss_check_interval = 10;  // Check loss every N iterations

        // Debug: compute initial loss before any optimization
        let initial_loss = self.compute_final_loss(gaussians, target);
        println!("  GPU Adam initial loss (before opt): {:.6}", initial_loss);

        // Run optimization loop
        for t in 1..=self.max_iterations {
            // Update Adam params with current bias corrections and decayed position LR
            let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(t as i32);
            let position_lr = self.learning_rates.position_lr_at_iteration(t, self.max_iterations);

            let adam_params = AdamParams {
                lr_position: position_lr,
                lr_color: self.learning_rates.color,
                lr_scale: self.learning_rates.scale,
                lr_opacity: self.learning_rates.opacity,
                lr_rotation: self.learning_rates.rotation,
                beta1: self.beta1,
                beta2: self.beta2,
                epsilon: self.epsilon,
                bias_correction1,
                bias_correction2,
                position_min: 0.0,
                position_max: 1.0,
                color_min: 0.0,
                color_max: 1.0,
                scale_min: 0.001,
                scale_max: 0.5,
                opacity_min: 0.01,
                opacity_max: 1.0,
                num_gaussians: num_gaussians as u32,
                _pad1: 0,
                _padding: [0; 3],
                _pad2: 0,
            };

            let adam_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Adam Params Buffer"),
                contents: bytemuck::bytes_of(&adam_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // Create bind groups
            let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Render Bind Group"),
                layout: &self.render_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gaussian_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rendered_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: render_params_buffer.as_entire_binding(),
                    },
                ],
            });

            let gradient_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Gradient Bind Group"),
                layout: &self.gradient_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gaussian_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rendered_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: target_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: gradient_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: gradient_params_buffer.as_entire_binding(),
                    },
                ],
            });

            let adam_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Adam Bind Group"),
                layout: &self.adam_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gaussian_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gradient_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: moment_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: variance_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: adam_params_buffer.as_entire_binding(),
                    },
                ],
            });

            // Create command encoder for this iteration
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Adam Iteration {} Encoder", t)),
            });

            // Pass 1: Render
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Render Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.render_pipeline);
                pass.set_bind_group(0, &render_bind_group, &[]);
                let wg_x = (width + 15) / 16;
                let wg_y = (height + 15) / 16;
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            // Pass 2: Compute gradients
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Gradient Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.gradient_pipeline);
                pass.set_bind_group(0, &gradient_bind_group, &[]);
                let wg_count = (num_gaussians as u32 + 255) / 256;
                pass.dispatch_workgroups(wg_count, 1, 1);
            }

            // Pass 3: Adam update
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Adam Update Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.adam_pipeline);
                pass.set_bind_group(0, &adam_bind_group, &[]);
                let wg_count = (num_gaussians as u32 + 255) / 256;
                pass.dispatch_workgroups(wg_count, 1, 1);
            }

            // Submit
            queue.submit(std::iter::once(encoder.finish()));

            // Periodic loss checking and best-state tracking
            if t % loss_check_interval == 0 || t == self.max_iterations {
                // Read back current Gaussians to check loss
                let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Loss Check Staging Buffer"),
                    size: gaussian_buffer.size(),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Loss Check Encoder"),
                });
                encoder.copy_buffer_to_buffer(&gaussian_buffer, 0, &staging_buffer, 0, gaussian_buffer.size());
                let submission_index = queue.submit(std::iter::once(encoder.finish()));

                // Wait for copy
                let buffer_slice = staging_buffer.slice(..);
                let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).ok();
                });

                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: Some(submission_index),
                    timeout: Some(std::time::Duration::from_secs(5)),
                });

                if let Ok(Some(Ok(()))) = pollster::block_on(async {
                    Ok::<_, ()>(receiver.receive().await)
                }) {
                    let data = buffer_slice.get_mapped_range();
                    let current_gaussians: &[ShaderGaussian] = bytemuck::cast_slice(&data);

                    // Compute loss
                    let temp_gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = current_gaussians
                        .iter()
                        .map(|g| g.to_gaussian())
                        .collect();
                    let loss = self.compute_final_loss(&temp_gaussians, target);

                    // Track best state
                    if loss < best_loss {
                        best_loss = loss;
                        best_gaussians.copy_from_slice(current_gaussians);
                        best_iteration = t;
                    }

                    println!("  GPU Adam iteration {}/{}: loss = {:.6} (pos_lr = {:.6})",
                             t, self.max_iterations, loss, position_lr);

                    drop(data);
                }
                staging_buffer.unmap();
            }
        }

        // Restore best state (use tracked best, not final GPU state which may have diverged)
        println!("  Restored best from iteration {}/{} (loss: {:.6})", best_iteration, self.max_iterations, best_loss);

        // Copy best_gaussians (tracked on CPU) back to output
        for (i, gpu_g) in best_gaussians.iter().enumerate() {
            gaussians[i] = gpu_g.to_gaussian();
        }

        // Return the best loss we found
        Ok(best_loss)
    }

    /// Compute loss using weighted-normalization renderer (matches GPU shader render_mode: 1)
    ///
    /// This must match what the GPU shader produces: C = Σ(w×c) / Σ(w)
    /// Note: lgi_core::Renderer::AccumulatedSum does NOT use normalization,
    /// so we implement it directly here to match the GPU shader.
    fn compute_final_loss(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) -> f32 {
        use lgi_math::color::Color4;

        let width = target.width;
        let height = target.height;
        let mut loss = 0.0;

        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / width as f32;
                let py = y as f32 / height as f32;

                // Weighted normalization: C = Σ(w×c) / Σ(w)
                let mut weight_sum = 0.0f32;
                let mut color_sum = Color4::new(0.0f32, 0.0, 0.0, 0.0);

                for gaussian in gaussians {
                    let dx = px - gaussian.position.x;
                    let dy = py - gaussian.position.y;

                    // Use ANISOTROPIC formula to match GPU render shader
                    // This ensures loss measurement matches actual GPU rendering
                    let cos_t = gaussian.shape.rotation.cos();
                    let sin_t = gaussian.shape.rotation.sin();
                    let dx_rot = dx * cos_t + dy * sin_t;
                    let dy_rot = -dx * sin_t + dy * cos_t;

                    let sx = gaussian.shape.scale_x.max(0.001);
                    let sy = gaussian.shape.scale_y.max(0.001);
                    let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);

                    // Cutoff
                    if dist_sq > 12.25 {  // 3.5 sigma
                        continue;
                    }

                    // Gaussian weight (matching GPU render shader and RendererV2)
                    let gaussian_val = (-0.5 * dist_sq).exp();
                    let weight = gaussian.opacity * gaussian_val;

                    weight_sum += weight;
                    color_sum.r += weight * gaussian.color.r;
                    color_sum.g += weight * gaussian.color.g;
                    color_sum.b += weight * gaussian.color.b;
                }

                // Normalized color
                let rendered = if weight_sum > 1e-10 {
                    Color4::new(
                        color_sum.r / weight_sum,
                        color_sum.g / weight_sum,
                        color_sum.b / weight_sum,
                        1.0,
                    )
                } else {
                    Color4::new(0.0, 0.0, 0.0, 1.0)
                };

                // Get target pixel
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
        // Verify struct sizes match shader expectations
        assert_eq!(std::mem::size_of::<ShaderGaussian>(), 48, "ShaderGaussian must be 48 bytes");
        assert_eq!(std::mem::size_of::<AdamMoment>(), 48, "AdamMoment must be 48 bytes");
        assert_eq!(std::mem::size_of::<AdamVariance>(), 48, "AdamVariance must be 48 bytes");
    }
}
