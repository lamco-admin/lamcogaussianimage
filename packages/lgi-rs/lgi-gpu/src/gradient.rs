//! GPU-accelerated gradient computation
//! Copyright (c) 2025 Lamco Development

use crate::{GpuError, Result};
use crate::buffer::GpuGaussian;
use lgi_core::ImageBuffer;
use wgpu::util::DeviceExt;
use lgi_math::gaussian::Gaussian2D;
use lgi_math::parameterization::Euler;
use wgpu::{Device, Queue};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuGradient {
    pub d_position: [f32; 2],
    pub d_scale_x: f32,
    pub d_scale_y: f32,
    pub d_rotation: f32,
    pub d_color: [f32; 4],
    pub d_opacity: f32,
    pub _padding: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ComputeParams {
    width: u32,
    height: u32,
    num_gaussians: u32,
    _padding: u32,
}

pub struct GpuGradientComputer {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuGradientComputer {
    pub fn new(device: &Device) -> Result<Self> {
        // Load shader
        let shader_source = include_str!("shaders/gradient_compute.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gradient Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gradient Compute Bind Group Layout"),
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
                // Gradients (read-write)
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
        });

        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gradient Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gradient Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_gradients_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    pub fn compute_gradients(
        &self,
        device: &Device,
        queue: &Queue,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> Result<Vec<GpuGradient>> {
        let num_gaussians = gaussians.len();
        let width = rendered.width;
        let height = rendered.height;

        // Convert Gaussians to GPU format
        let gpu_gaussians: Vec<GpuGaussian> = gaussians
            .iter()
            .map(|g| GpuGaussian::from_gaussian(g))
            .collect();

        // Create buffers
        let gaussian_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian Buffer"),
            contents: bytemuck::cast_slice(&gpu_gaussians),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Convert images to GPU format
        let rendered_data: Vec<[f32; 4]> = (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                if let Some(p) = rendered.get_pixel(x, y) {
                    [p.r, p.g, p.b, p.a]
                } else {
                    [0.0; 4]
                }
            })
            .collect();

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

        let rendered_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rendered Image Buffer"),
            contents: bytemuck::cast_slice(&rendered_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let target_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Target Image Buffer"),
            contents: bytemuck::cast_slice(&target_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create gradient output buffer
        let gradient_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gradient Buffer"),
            size: (num_gaussians * std::mem::size_of::<GpuGradient>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params = ComputeParams {
            width,
            height,
            num_gaussians: num_gaussians as u32,
            _padding: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gradient Compute Bind Group"),
            layout: &self.bind_group_layout,
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
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gradient Compute Encoder"),
        });

        // Dispatch compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gradient Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_count = (num_gaussians as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Read back results
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gradient Staging Buffer"),
            size: gradient_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &gradient_buffer,
            0,
            &staging_buffer,
            0,
            gradient_buffer.size(),
        );

        // Submit and track submission index
        let submission_index = queue.submit(std::iter::once(encoder.finish()));

        // Map and read (synchronous using pollster)
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        // Wait for THIS SPECIFIC submission (v27 API with submission_index)
        if let Err(e) = device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(std::time::Duration::from_secs(10)),
        }) {
            log::error!("Gradient GPU polling error: {:?}", e);
            return Err(GpuError::BufferMapping);
        }

        // Block on receiving result
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
}
