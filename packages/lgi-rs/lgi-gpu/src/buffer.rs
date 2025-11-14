//! GPU buffer management for Gaussian data

use crate::{GpuError, Result};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

/// GPU-friendly Gaussian representation (aligned for shader)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuGaussian {
    pub position: [f32; 2],    // μx, μy
    pub scale: [f32; 2],       // σx, σy
    pub rotation: f32,         // θ
    pub _padding1: f32,        // Alignment padding
    pub _padding2: [f32; 2],   // CRITICAL: vec3 in WGSL has 16-byte alignment!
    pub color: [f32; 3],       // R, G, B (offset 32, aligned to 16)
    pub opacity: f32,          // α
    // Total: 12 floats × 4 bytes = 48 bytes (correct WGSL alignment)
}

impl GpuGaussian {
    /// Convert from Gaussian2D
    pub fn from_gaussian(g: &Gaussian2D<f32, Euler<f32>>) -> Self {
        Self {
            position: [g.position.x, g.position.y],
            scale: [g.shape.scale_x, g.shape.scale_y],
            rotation: g.shape.rotation,
            _padding1: 0.0,
            _padding2: [0.0, 0.0],  // Align color to 16-byte boundary for WGSL
            color: [g.color.r, g.color.g, g.color.b],
            opacity: g.opacity,
        }
    }

    /// Convert batch of Gaussians
    pub fn from_gaussians(gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Vec<Self> {
        gaussians.iter().map(Self::from_gaussian).collect()
    }
}

/// Render parameters for shader
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RenderParams {
    pub width: u32,
    pub height: u32,
    pub gaussian_count: u32,
    pub render_mode: u32,      // 0=AlphaComposite, 1=AccumulatedSum
    pub cutoff_threshold: f32,
    pub n_sigma: f32,
    pub _padding: [u32; 2],    // Alignment to 32 bytes
}

/// GPU buffer manager
pub struct GpuBufferManager {
    // Don't store device, use reference when needed
    /// Gaussian storage buffer
    pub gaussian_buffer: Option<wgpu::Buffer>,

    /// Output image buffer
    pub output_buffer: Option<wgpu::Buffer>,

    /// Staging buffer for readback
    pub staging_buffer: Option<wgpu::Buffer>,

    /// Uniform buffer for render params
    pub params_buffer: Option<wgpu::Buffer>,
}

impl GpuBufferManager {
    /// Create new buffer manager
    pub fn new() -> Self {
        Self {
            gaussian_buffer: None,
            output_buffer: None,
            staging_buffer: None,
            params_buffer: None,
        }
    }

    /// Create or update Gaussian buffer
    pub fn update_gaussian_buffer(&mut self, device: &wgpu::Device, gaussians: &[GpuGaussian]) -> Result<()> {
        let data = bytemuck::cast_slice(gaussians);

        self.gaussian_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian Storage Buffer"),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }));

        Ok(())
    }

    /// Create or resize output buffer
    pub fn create_output_buffer(&mut self, device: &wgpu::Device, width: u32, height: u32) -> Result<()> {
        let size = (width * height * 4 * std::mem::size_of::<f32>() as u32) as u64;

        self.output_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Image Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Create staging buffer for readback
        self.staging_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        Ok(())
    }

    /// Create or update params buffer
    pub fn update_params_buffer(&mut self, device: &wgpu::Device, params: &RenderParams) -> Result<()> {
        let data = bytemuck::bytes_of(params);

        self.params_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Render Params Buffer"),
            contents: data,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        Ok(())
    }
}
