//! GPU buffer management for Gaussian data

use crate::{GpuError, Result};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

/// GPU-friendly Gaussian representation (aligned for shader)
/// MUST match WGSL struct layout EXACTLY:
///   offset 0-8:   position (vec2<f32>)
///   offset 8-12:  scale_x (f32)
///   offset 12-16: scale_y (f32)
///   offset 16-20: rotation (f32)
///   offset 20-24: color_r (f32)
///   offset 24-28: color_g (f32)
///   offset 28-32: color_b (f32)
///   offset 32-36: color_a (f32)
///   offset 36-40: opacity (f32)
///   offset 40-48: _padding (vec2<u32>)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuGaussian {
    pub position: [f32; 2],    // μx, μy - 8 bytes @ offset 0
    pub scale_x: f32,          // σx - 4 bytes @ offset 8
    pub scale_y: f32,          // σy - 4 bytes @ offset 12
    pub rotation: f32,         // θ - 4 bytes @ offset 16
    pub color_r: f32,          // R - 4 bytes @ offset 20
    pub color_g: f32,          // G - 4 bytes @ offset 24
    pub color_b: f32,          // B - 4 bytes @ offset 28
    pub color_a: f32,          // A - 4 bytes @ offset 32
    pub opacity: f32,          // α - 4 bytes @ offset 36
    pub _padding: [u32; 2],    // Alignment - 8 bytes @ offset 40
    // Total: 48 bytes (matches WGSL exactly)
}

impl GpuGaussian {
    /// Convert from Gaussian2D
    pub fn from_gaussian(g: &Gaussian2D<f32, Euler<f32>>) -> Self {
        Self {
            position: [g.position.x, g.position.y],
            scale_x: g.shape.scale_x,
            scale_y: g.shape.scale_y,
            rotation: g.shape.rotation,
            color_r: g.color.r,
            color_g: g.color.g,
            color_b: g.color.b,
            color_a: g.color.a,
            opacity: g.opacity,
            _padding: [0, 0],
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
