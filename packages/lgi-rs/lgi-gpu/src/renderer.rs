//! GPU renderer implementation

use crate::{
    backend::BackendSelector,
    buffer::{GpuBufferManager, GpuGaussian, RenderParams},
    capabilities::{BackendType, GpuCapabilities},
    pipeline::GaussianPipeline,
    GpuError, Result,
};
use lgi_core::{ImageBuffer, RenderMode};
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler};
use std::time::Instant;
use wgpu;

/// GPU-accelerated Gaussian renderer
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: GaussianPipeline,
    buffer_manager: GpuBufferManager,
    capabilities: GpuCapabilities,

    // Performance tracking
    last_render_time: f32,
    render_count: u64,
}

impl GpuRenderer {
    /// Create new GPU renderer with auto-detected backend
    pub async fn new() -> Result<Self> {
        let selector = BackendSelector::new();

        // Select best adapter
        let adapter = selector.select_adapter().await?;
        let adapter_info = adapter.get_info();

        log::info!("Selected GPU: {} ({:?})", adapter_info.name, adapter_info.backend);

        // Request device
        let (device, queue) = selector.request_device(&adapter).await?;

        // Detect capabilities
        let capabilities = GpuCapabilities::detect(
            &adapter_info,
            device.features(),
            &device.limits(),
        );

        capabilities.print_info();

        // Create pipeline
        let pipeline = GaussianPipeline::new(&device)?;

        // Create buffer manager
        let buffer_manager = GpuBufferManager::new();

        Ok(Self {
            device,
            queue,
            pipeline,
            buffer_manager,
            capabilities,
            last_render_time: 0.0,
            render_count: 0,
        })
    }

    /// Create with specific backend
    pub async fn with_backend(backend: wgpu::Backend) -> Result<Self> {
        let selector = BackendSelector::with_backends(backend.into());
        let adapter = selector.select_adapter().await?;
        let adapter_info = adapter.get_info();

        if adapter_info.backend != backend {
            return Err(GpuError::UnsupportedFeature(
                format!("Requested {:?} but got {:?}", backend, adapter_info.backend)
            ));
        }

        let (device, queue) = selector.request_device(&adapter).await?;
        let capabilities = GpuCapabilities::detect(&adapter_info, device.features(), &device.limits());
        let pipeline = GaussianPipeline::new(&device)?;
        let buffer_manager = GpuBufferManager::new();

        Ok(Self {
            device,
            queue,
            pipeline,
            buffer_manager,
            capabilities,
            last_render_time: 0.0,
            render_count: 0,
        })
    }

    /// Render Gaussians to image buffer
    pub fn render(
        &mut self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        width: u32,
        height: u32,
        mode: RenderMode,
    ) -> Result<ImageBuffer<f32>> {
        let start = Instant::now();

        // Convert Gaussians to GPU format
        let gpu_gaussians = GpuGaussian::from_gaussians(gaussians);

        // Update buffers
        self.buffer_manager.update_gaussian_buffer(&self.device, &gpu_gaussians)?;
        self.buffer_manager.create_output_buffer(&self.device, width, height)?;

        // Create render params
        let params = RenderParams {
            width,
            height,
            gaussian_count: gaussians.len() as u32,
            render_mode: match mode {
                RenderMode::AlphaComposite => 0,
                RenderMode::AccumulatedSum => 1,
            },
            cutoff_threshold: 1e-5,
            n_sigma: 3.5,
            _padding: [0, 0],
        };

        self.buffer_manager.update_params_buffer(&self.device, &params)?;

        // Create bind group
        let bind_group = self.pipeline.create_bind_group(
            &self.device,
            self.buffer_manager.gaussian_buffer.as_ref().unwrap(),
            self.buffer_manager.output_buffer.as_ref().unwrap(),
            self.buffer_manager.params_buffer.as_ref().unwrap(),
        );

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gaussian Render Encoder"),
        });

        // Dispatch compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gaussian Render Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (16×16 workgroup size from shader)
            let workgroups_x = (width + 15) / 16;
            let workgroups_y = (height + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy output to staging buffer for readback
        encoder.copy_buffer_to_buffer(
            self.buffer_manager.output_buffer.as_ref().unwrap(),
            0,
            self.buffer_manager.staging_buffer.as_ref().unwrap(),
            0,
            (width * height * 4 * std::mem::size_of::<f32>() as u32) as u64,
        );

        // Submit commands and track submission index
        let submission_index = self.queue.submit(Some(encoder.finish()));

        // Read back results (synchronous for now, async requires runtime integration)
        let output = self.read_output_buffer(width, height, submission_index)?;

        // Track performance
        self.last_render_time = start.elapsed().as_secs_f32() * 1000.0;
        self.render_count += 1;

        Ok(output)
    }

    /// Read output buffer from GPU
    fn read_output_buffer(
        &self,
        width: u32,
        height: u32,
        submission_index: wgpu::SubmissionIndex,
    ) -> Result<ImageBuffer<f32>> {
        let staging_buffer = self.buffer_manager.staging_buffer.as_ref()
            .ok_or(GpuError::BufferMapping)?;

        // Map buffer for reading
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        // Wait for THIS SPECIFIC submission (v27 API with submission_index)
        if let Err(e) = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(std::time::Duration::from_secs(10)),
        }) {
            log::error!("GPU polling error: {:?}", e);
            return Err(GpuError::BufferMapping);
        }

        pollster::block_on(async {
            receiver.receive().await
                .ok_or(GpuError::BufferMapping)?
                .map_err(|_| GpuError::BufferMapping)?;
            Ok::<(), GpuError>(())
        })?;

        // Read data
        let data = buffer_slice.get_mapped_range();
        let pixels: &[f32] = bytemuck::cast_slice(&data);

        // Convert to ImageBuffer
        let mut image = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                if idx + 3 < pixels.len() {
                    image.set_pixel(x, y, Color4::new(
                        pixels[idx],
                        pixels[idx + 1],
                        pixels[idx + 2],
                        pixels[idx + 3],
                    ));
                }
            }
        }

        // Unmap buffer
        drop(data);
        staging_buffer.unmap();

        Ok(image)
    }

    /// Get adapter name
    pub fn adapter_name(&self) -> &str {
        &self.capabilities.backend_info.adapter_name
    }

    /// Get backend type
    pub fn backend(&self) -> BackendType {
        self.capabilities.backend_info.backend
    }

    /// Get last render time in milliseconds
    pub fn last_render_time_ms(&self) -> f32 {
        self.last_render_time
    }

    /// Get FPS from last render
    pub fn fps(&self) -> f32 {
        if self.last_render_time > 0.0 {
            1000.0 / self.last_render_time
        } else {
            0.0
        }
    }

    /// Get GPU capabilities
    pub fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }

    /// Get device reference for gradient computation
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get queue reference for gradient computation
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}
