//! J^T J computation strategies for Levenberg-Marquardt optimization
//!
//! # Background
//!
//! The L-M algorithm solves (J^T J + λI)δ = -J^T r at each iteration.
//! Computing J^T J directly is expensive - the Jacobian J has shape
//! (n_residuals × n_params), which for a 768×512 image with 256 Gaussians
//! would be 1.18M × 2560 = 3 billion elements.
//!
//! Different strategies trade off accuracy vs. compute cost:
//!
//! 1. **Block-diagonal**: Only compute within-Gaussian blocks. O(P) per pixel
//!    where P = params per Gaussian (10). Fast but ignores interactions.
//!
//! 2. **Full dense**: Compute all cross-Gaussian interactions. O(G²) per pixel
//!    where G = number of Gaussians. Slow but captures everything.
//!
//! 3. **Sparse**: Only compute for spatially-overlapping Gaussian pairs.
//!    Complexity depends on overlap pattern. Good middle ground.
//!
//! 4. **Matrix-free/CG**: Never form J^T J explicitly. Use J*v and J^T*v
//!    products in an iterative solver. Memory-efficient for huge problems.
//!
//! # Usage
//!
//! ```ignore
//! // Use block-diagonal (fast, less accurate)
//! let strategy = BlockDiagonalJtJ::new(&device)?;
//! optimizer.set_jtj_strategy(Box::new(strategy));
//!
//! // Use full dense (slow, accurate)
//! let strategy = FullDenseJtJ::new(&device)?;
//! optimizer.set_jtj_strategy(Box::new(strategy));
//! ```

use wgpu::{Device, Queue};
use crate::error::Result;
use crate::adam_optimizer::ShaderGaussian;

/// Context passed to J^T J computation
pub struct JtJContext<'a> {
    /// Current Gaussian parameters (GPU format)
    pub gaussians: &'a [ShaderGaussian],
    /// GPU buffer containing Gaussians
    pub gaussian_buffer: &'a wgpu::Buffer,
    /// Target image data
    pub target_buffer: &'a wgpu::Buffer,
    /// Rendered image buffer
    pub rendered_buffer: &'a wgpu::Buffer,
    /// Image dimensions
    pub width: u32,
    pub height: u32,
    /// Number of parameters total (gaussians × params_per_gaussian)
    pub num_params: usize,
}

/// Result of J^T J computation
pub struct JtJResult {
    /// J^T J matrix (num_params × num_params), row-major
    pub jtj: Vec<f32>,
    /// J^T r vector (num_params,)
    pub jtr: Vec<f32>,
    /// Sparsity info (for diagnostics)
    pub sparsity_info: Option<SparsityInfo>,
}

/// Information about J^T J sparsity pattern
#[derive(Debug, Clone)]
pub struct SparsityInfo {
    /// Number of non-zero elements in J^T J
    pub nnz: usize,
    /// Total elements in J^T J
    pub total: usize,
    /// Whether interactions were captured
    pub captures_interactions: bool,
}

impl SparsityInfo {
    /// Compute sparsity ratio (0.0 = all zeros, 1.0 = fully dense)
    pub fn density(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.nnz as f32 / self.total as f32
        }
    }
}

/// Strategy for computing J^T J (Gauss-Newton Hessian approximation)
///
/// Different strategies provide different trade-offs:
/// - Block-diagonal: Fast but misses Gaussian interactions
/// - Full dense: Slow but captures all interactions
/// - Sparse: Middle ground for overlapping Gaussians
/// - Matrix-free: Uses iterative solver, never forms matrix
pub trait JtJStrategy: Send + Sync {
    /// Compute J^T J and J^T r for the current state
    ///
    /// # Arguments
    /// * `device` - wgpu Device
    /// * `queue` - wgpu Queue
    /// * `ctx` - Computation context with buffers and dimensions
    ///
    /// # Returns
    /// J^T J matrix and J^T r vector
    fn compute(
        &self,
        device: &Device,
        queue: &Queue,
        ctx: &JtJContext,
    ) -> Result<JtJResult>;

    /// Human-readable name for logging
    fn name(&self) -> &'static str;

    /// Description of this strategy's trade-offs
    fn description(&self) -> &'static str;

    /// Whether this strategy captures cross-Gaussian interactions
    fn captures_interactions(&self) -> bool;

    /// Expected computational complexity class
    fn complexity(&self) -> ComplexityClass;
}

/// Computational complexity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityClass {
    /// O(P) per pixel where P = params per Gaussian (block-diagonal)
    LinearPerGaussian,
    /// O(G²) per pixel where G = number of Gaussians (full dense)
    QuadraticInGaussians,
    /// Between linear and quadratic, depends on overlap (sparse)
    Sparse,
    /// Memory-efficient but requires iterative solver (matrix-free)
    MatrixFree,
}

impl std::fmt::Display for ComplexityClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LinearPerGaussian => write!(f, "O(P·pixels) - linear per Gaussian"),
            Self::QuadraticInGaussians => write!(f, "O(G²·pixels) - quadratic in Gaussians"),
            Self::Sparse => write!(f, "O(overlap·pixels) - depends on spatial overlap"),
            Self::MatrixFree => write!(f, "O(iter·G·pixels) - iterative solver"),
        }
    }
}

// ============================================================================
// Block-Diagonal Strategy (current implementation)
// ============================================================================

/// Block-diagonal J^T J computation
///
/// Only computes the 10×10 diagonal blocks for each Gaussian.
/// Ignores all cross-Gaussian interactions.
///
/// **Pros**: Very fast, minimal memory
/// **Cons**: Poor Hessian approximation when Gaussians overlap
pub struct BlockDiagonalJtJ {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl BlockDiagonalJtJ {
    /// Create a new block-diagonal strategy
    pub fn new(device: &Device) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Block-Diagonal JtJ Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lm_jacobian.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Block-Diagonal JtJ Layout"),
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
                // Target image (read)
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
                // Rendered image (read)
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
                // J^T J output (read-write for atomics)
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
                // J^T r output (read-write for atomics)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Block-Diagonal JtJ Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Block-Diagonal JtJ Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    /// Get the bind group layout for external use
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the compute pipeline for external use
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Read fixed-point matrix from GPU buffer
    fn read_matrix(device: &Device, queue: &Queue, buffer: &wgpu::Buffer, n: usize) -> Result<Vec<f32>> {
        let size = (n * n * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block-Diagonal Matrix Staging"),
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
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let result: Vec<f32> = u32_data.iter()
            .map(|&bits| {
                let int_val = bits as i32;
                int_val as f32 / FIXED_SCALE
            })
            .collect();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Read fixed-point vector from GPU buffer
    fn read_vector(device: &Device, queue: &Queue, buffer: &wgpu::Buffer, n: usize) -> Result<Vec<f32>> {
        let size = (n * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block-Diagonal Vector Staging"),
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
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let result: Vec<f32> = u32_data.iter()
            .map(|&bits| {
                let int_val = bits as i32;
                int_val as f32 / FIXED_SCALE
            })
            .collect();
        drop(data);
        staging.unmap();

        Ok(result)
    }
}

impl JtJStrategy for BlockDiagonalJtJ {
    fn compute(
        &self,
        device: &Device,
        queue: &Queue,
        ctx: &JtJContext,
    ) -> Result<JtJResult> {
        use wgpu::util::DeviceExt;

        let num_gaussians = ctx.num_params / 10;

        // Create uniform params buffer
        let params = JacobianParams {
            width: ctx.width,
            height: ctx.height,
            num_gaussians: num_gaussians as u32,
            num_params: ctx.num_params as u32,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Block-Diagonal JtJ Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create output buffers
        let jtj_size = (ctx.num_params * ctx.num_params * 4) as u64;
        let jtj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block-Diagonal JtJ Buffer"),
            size: jtj_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let jtr_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Block-Diagonal Jtr Buffer"),
            size: (ctx.num_params * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Zero out buffers
        let zeros_jtj = vec![0u32; ctx.num_params * ctx.num_params];
        let zeros_jtr = vec![0u32; ctx.num_params];
        queue.write_buffer(&jtj_buffer, 0, bytemuck::cast_slice(&zeros_jtj));
        queue.write_buffer(&jtr_buffer, 0, bytemuck::cast_slice(&zeros_jtr));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Block-Diagonal JtJ Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ctx.gaussian_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ctx.target_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ctx.rendered_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: jtj_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: jtr_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Block-Diagonal JtJ Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Block-Diagonal JtJ Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((ctx.width + 15) / 16, (ctx.height + 15) / 16, 1);
        }

        queue.submit(Some(encoder.finish()));

        // Read results back using shared helper
        let jtj = Self::read_matrix(device, queue, &jtj_buffer, ctx.num_params)?;
        let jtr = Self::read_vector(device, queue, &jtr_buffer, ctx.num_params)?;

        let nnz = num_gaussians * 10 * 10; // Only diagonal blocks have values

        Ok(JtJResult {
            jtj,
            jtr,
            sparsity_info: Some(SparsityInfo {
                nnz,
                total: ctx.num_params * ctx.num_params,
                captures_interactions: false,
            }),
        })
    }

    fn name(&self) -> &'static str {
        "Block-Diagonal"
    }

    fn description(&self) -> &'static str {
        "Computes only within-Gaussian 10×10 blocks. Fast but ignores cross-Gaussian interactions."
    }

    fn captures_interactions(&self) -> bool {
        false
    }

    fn complexity(&self) -> ComplexityClass {
        ComplexityClass::LinearPerGaussian
    }
}

// ============================================================================
// Full Dense Strategy
// ============================================================================

/// Fixed-point scale for GPU atomic accumulation (matches WGSL shader)
const FIXED_SCALE: f32 = 1_000_000.0;

/// GPU parameters for JtJ computation (block-diagonal and full-dense)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct JacobianParams {
    width: u32,
    height: u32,
    num_gaussians: u32,
    num_params: u32,
}

/// GPU parameters for sparse JtJ computation (with overlap threshold)
///
/// Layout matches WGSL struct with vec3<f32> padding:
/// - width, height, num_gaussians, num_params: 16 bytes
/// - overlap_sigma: 4 bytes (offset 16)
/// - _pad0: 12 bytes alignment padding (offset 20)
/// - _padding vec3<f32>: 12 bytes + 4 bytes padding to 16-byte boundary = 16 bytes
/// Total: 48 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SparseJacobianParams {
    width: u32,
    height: u32,
    num_gaussians: u32,
    num_params: u32,
    overlap_sigma: f32,      // How many sigma to consider overlap (default: 3.0)
    _padding: [f32; 7],      // Pad to 48 bytes total (WGSL vec3 has 16-byte alignment)
}

/// Full dense J^T J computation
///
/// Computes all G² cross-Gaussian blocks, capturing every Gaussian interaction.
///
/// **Pros**: Most accurate Hessian approximation
/// **Cons**: O(G²) per pixel, expensive for many Gaussians
pub struct FullDenseJtJ {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl FullDenseJtJ {
    /// Create a new full dense strategy
    pub fn new(device: &Device) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Full-Dense JtJ Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lm_jacobian_dense.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Full-Dense JtJ Layout"),
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
                // Target image (read)
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
                // Rendered image (read)
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
                // J^T J output (read-write for atomics)
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
                // J^T r output (read-write for atomics)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Full-Dense JtJ Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Full-Dense JtJ Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    /// Get the bind group layout for external use
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the compute pipeline for external use
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Read fixed-point matrix from GPU buffer
    fn read_matrix(device: &Device, queue: &Queue, buffer: &wgpu::Buffer, n: usize) -> Result<Vec<f32>> {
        use wgpu::util::DeviceExt;
        let size = (n * n * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Full-Dense Matrix Staging"),
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
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let result: Vec<f32> = u32_data.iter()
            .map(|&bits| {
                let int_val = bits as i32;
                int_val as f32 / FIXED_SCALE
            })
            .collect();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Read fixed-point vector from GPU buffer
    fn read_vector(device: &Device, queue: &Queue, buffer: &wgpu::Buffer, n: usize) -> Result<Vec<f32>> {
        let size = (n * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Full-Dense Vector Staging"),
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
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let result: Vec<f32> = u32_data.iter()
            .map(|&bits| {
                let int_val = bits as i32;
                int_val as f32 / FIXED_SCALE
            })
            .collect();
        drop(data);
        staging.unmap();

        Ok(result)
    }
}

impl JtJStrategy for FullDenseJtJ {
    fn compute(
        &self,
        device: &Device,
        queue: &Queue,
        ctx: &JtJContext,
    ) -> Result<JtJResult> {
        use wgpu::util::DeviceExt;

        let num_gaussians = ctx.num_params / 10;

        // Create uniform params buffer
        let params = JacobianParams {
            width: ctx.width,
            height: ctx.height,
            num_gaussians: num_gaussians as u32,
            num_params: ctx.num_params as u32,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Full-Dense JtJ Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create output buffers
        let jtj_size = (ctx.num_params * ctx.num_params * 4) as u64;
        let jtj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Full-Dense JtJ Buffer"),
            size: jtj_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let jtr_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Full-Dense Jtr Buffer"),
            size: (ctx.num_params * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Zero out buffers
        let zeros_jtj = vec![0u32; ctx.num_params * ctx.num_params];
        let zeros_jtr = vec![0u32; ctx.num_params];
        queue.write_buffer(&jtj_buffer, 0, bytemuck::cast_slice(&zeros_jtj));
        queue.write_buffer(&jtr_buffer, 0, bytemuck::cast_slice(&zeros_jtr));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Full-Dense JtJ Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ctx.gaussian_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ctx.target_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ctx.rendered_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: jtj_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: jtr_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Full-Dense JtJ Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Full-Dense JtJ Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((ctx.width + 15) / 16, (ctx.height + 15) / 16, 1);
        }

        queue.submit(Some(encoder.finish()));

        // Read results back
        let jtj = Self::read_matrix(device, queue, &jtj_buffer, ctx.num_params)?;
        let jtr = Self::read_vector(device, queue, &jtr_buffer, ctx.num_params)?;

        Ok(JtJResult {
            jtj,
            jtr,
            sparsity_info: Some(SparsityInfo {
                nnz: ctx.num_params * ctx.num_params,
                total: ctx.num_params * ctx.num_params,
                captures_interactions: true,
            }),
        })
    }

    fn name(&self) -> &'static str {
        "Full Dense"
    }

    fn description(&self) -> &'static str {
        "Computes all G² cross-Gaussian blocks. Slow but captures all interactions."
    }

    fn captures_interactions(&self) -> bool {
        true
    }

    fn complexity(&self) -> ComplexityClass {
        ComplexityClass::QuadraticInGaussians
    }
}

// ============================================================================
// Sparse Strategy (Overlap-Based)
// ============================================================================

/// Sparse J^T J computation based on spatial overlap
///
/// Only computes blocks for Gaussian pairs that actually overlap spatially.
/// Two Gaussians overlap if `dist² < (r_i + r_j)²` where each radius is
/// `max(scale_x, scale_y) × overlap_sigma`.
///
/// **Pros**: Captures meaningful interactions, skips distant pairs
/// **Cons**: Still O(G²) worst-case check, but O(G×k) in practice
///
/// Typical overlap_sigma values:
/// - 3.0: Standard (99.7% of Gaussian mass)
/// - 2.0: Aggressive (95% of mass)
/// - 4.0: Conservative (more interactions captured)
pub struct SparseJtJ {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// How many sigma to consider for overlap (default: 3.0)
    pub overlap_sigma: f32,
}

impl SparseJtJ {
    /// Create a new sparse strategy
    ///
    /// # Arguments
    /// * `device` - wgpu Device
    /// * `overlap_sigma` - Number of standard deviations to consider for overlap
    pub fn new(device: &Device, overlap_sigma: f32) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sparse JtJ Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lm_jacobian_sparse.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sparse JtJ Layout"),
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
                // Target image (read)
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
                // Rendered image (read)
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
                // J^T J output (read-write for atomics)
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
                // J^T r output (read-write for atomics)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params (includes overlap_sigma)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sparse JtJ Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sparse JtJ Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            overlap_sigma,
        })
    }

    /// Get the bind group layout for external use
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the compute pipeline for external use
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Read fixed-point matrix from GPU buffer
    fn read_matrix(device: &Device, queue: &Queue, buffer: &wgpu::Buffer, n: usize) -> Result<Vec<f32>> {
        let size = (n * n * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Matrix Staging"),
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
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let result: Vec<f32> = u32_data.iter()
            .map(|&bits| {
                let int_val = bits as i32;
                int_val as f32 / FIXED_SCALE
            })
            .collect();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Read fixed-point vector from GPU buffer
    fn read_vector(device: &Device, queue: &Queue, buffer: &wgpu::Buffer, n: usize) -> Result<Vec<f32>> {
        let size = (n * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Vector Staging"),
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
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let result: Vec<f32> = u32_data.iter()
            .map(|&bits| {
                let int_val = bits as i32;
                int_val as f32 / FIXED_SCALE
            })
            .collect();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Count non-zero entries in the matrix for sparsity reporting
    fn count_nnz(jtj: &[f32]) -> usize {
        jtj.iter().filter(|&&v| v.abs() > 1e-10).count()
    }
}

impl JtJStrategy for SparseJtJ {
    fn compute(
        &self,
        device: &Device,
        queue: &Queue,
        ctx: &JtJContext,
    ) -> Result<JtJResult> {
        use wgpu::util::DeviceExt;

        let num_gaussians = ctx.num_params / 10;

        // Create uniform params buffer with overlap_sigma
        let params = SparseJacobianParams {
            width: ctx.width,
            height: ctx.height,
            num_gaussians: num_gaussians as u32,
            num_params: ctx.num_params as u32,
            overlap_sigma: self.overlap_sigma,
            _padding: [0.0; 7],
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sparse JtJ Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create output buffers
        let jtj_size = (ctx.num_params * ctx.num_params * 4) as u64;
        let jtj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse JtJ Buffer"),
            size: jtj_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let jtr_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Jtr Buffer"),
            size: (ctx.num_params * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Zero out buffers
        let zeros_jtj = vec![0u32; ctx.num_params * ctx.num_params];
        let zeros_jtr = vec![0u32; ctx.num_params];
        queue.write_buffer(&jtj_buffer, 0, bytemuck::cast_slice(&zeros_jtj));
        queue.write_buffer(&jtr_buffer, 0, bytemuck::cast_slice(&zeros_jtr));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sparse JtJ Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ctx.gaussian_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ctx.target_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ctx.rendered_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: jtj_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: jtr_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sparse JtJ Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sparse JtJ Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((ctx.width + 15) / 16, (ctx.height + 15) / 16, 1);
        }

        queue.submit(Some(encoder.finish()));

        // Read results back
        let jtj = Self::read_matrix(device, queue, &jtj_buffer, ctx.num_params)?;
        let jtr = Self::read_vector(device, queue, &jtr_buffer, ctx.num_params)?;

        // Count actual non-zeros for sparsity reporting
        let nnz = Self::count_nnz(&jtj);

        Ok(JtJResult {
            jtj,
            jtr,
            sparsity_info: Some(SparsityInfo {
                nnz,
                total: ctx.num_params * ctx.num_params,
                captures_interactions: true,
            }),
        })
    }

    fn name(&self) -> &'static str {
        "Sparse (Overlap-Based)"
    }

    fn description(&self) -> &'static str {
        "Computes blocks only for spatially-overlapping Gaussian pairs. \
         Uses center distance vs combined effective radius check."
    }

    fn captures_interactions(&self) -> bool {
        true
    }

    fn complexity(&self) -> ComplexityClass {
        ComplexityClass::Sparse
    }
}

// ============================================================================
// Hierarchical Strategy (Multi-Scale)
// ============================================================================

/// Hierarchical J^T J computation using multi-scale sparse passes
///
/// Runs the sparse shader multiple times at different overlap_sigma values:
/// - Coarse level: Large sigma captures global/distant interactions
/// - Fine level: Small sigma captures close/local interactions
///
/// Results from all levels are accumulated, providing a multi-scale
/// approximation to the full-dense matrix without the O(G²) cost.
///
/// **Pros**: Captures both local and global interactions efficiently
/// **Cons**: Multiple GPU dispatches per iteration
///
/// Typical configuration:
/// - 3 levels: σ = [6.0, 3.0, 1.5]
/// - Coarse level captures ~99.99% of distant pairs
/// - Fine level captures precise local gradients
pub struct HierarchicalJtJ {
    /// Sparse strategy instances for each level
    levels: Vec<SparseJtJ>,
    /// Number of hierarchy levels
    num_levels: usize,
}

impl HierarchicalJtJ {
    /// Create a new hierarchical strategy
    ///
    /// # Arguments
    /// * `device` - wgpu Device
    /// * `num_levels` - Number of hierarchy levels (2-4 recommended)
    pub fn new(device: &Device, num_levels: usize) -> Result<Self> {
        let num_levels = num_levels.clamp(2, 4);

        // Generate sigma values for each level (geometric progression)
        // From coarse (large sigma) to fine (small sigma)
        let sigma_values: Vec<f32> = match num_levels {
            2 => vec![6.0, 2.0],
            3 => vec![6.0, 3.0, 1.5],
            4 => vec![8.0, 4.0, 2.0, 1.0],
            _ => vec![6.0, 3.0, 1.5],  // Default to 3 levels
        };

        let mut levels = Vec::with_capacity(num_levels);
        for &sigma in &sigma_values {
            levels.push(SparseJtJ::new(device, sigma)?);
        }

        Ok(Self {
            levels,
            num_levels,
        })
    }

    /// Get sigma values used at each level
    pub fn sigma_values(&self) -> Vec<f32> {
        self.levels.iter().map(|l| l.overlap_sigma).collect()
    }
}

impl JtJStrategy for HierarchicalJtJ {
    fn compute(
        &self,
        device: &Device,
        queue: &Queue,
        ctx: &JtJContext,
    ) -> Result<JtJResult> {
        // Initialize accumulator
        let n = ctx.num_params;
        let mut accumulated_jtj = vec![0.0f32; n * n];
        let mut accumulated_jtr = vec![0.0f32; n];
        let mut total_nnz = 0usize;

        // Run each level and accumulate results
        for (level_idx, sparse) in self.levels.iter().enumerate() {
            let result = sparse.compute(device, queue, ctx)?;

            // Weight factor: finer levels contribute more to local accuracy
            // Coarse levels contribute more to global structure
            let weight = 1.0f32 / (level_idx + 1) as f32;

            // Accumulate weighted results
            for i in 0..n * n {
                accumulated_jtj[i] += result.jtj[i] * weight;
            }
            for i in 0..n {
                accumulated_jtr[i] += result.jtr[i] * weight;
            }

            if let Some(info) = &result.sparsity_info {
                total_nnz = total_nnz.max(info.nnz);  // Track max density across levels
            }
        }

        // Normalize by total weight
        let total_weight: f32 = (1..=self.num_levels).map(|l| 1.0 / l as f32).sum();
        for v in accumulated_jtj.iter_mut() {
            *v /= total_weight;
        }
        for v in accumulated_jtr.iter_mut() {
            *v /= total_weight;
        }

        Ok(JtJResult {
            jtj: accumulated_jtj,
            jtr: accumulated_jtr,
            sparsity_info: Some(SparsityInfo {
                nnz: total_nnz,
                total: n * n,
                captures_interactions: true,
            }),
        })
    }

    fn name(&self) -> &'static str {
        "Hierarchical (Multi-Scale)"
    }

    fn description(&self) -> &'static str {
        "Multi-scale sparse computation with weighted accumulation. \
         Captures both local and global Gaussian interactions."
    }

    fn captures_interactions(&self) -> bool {
        true
    }

    fn complexity(&self) -> ComplexityClass {
        ComplexityClass::Sparse
    }
}

// ============================================================================
// Matrix-Free Strategy (to be implemented)
// ============================================================================

/// Placeholder for matrix-free L-M (CG/GMRES based)
///
/// Never forms J^T J explicitly. Uses J*v and J^T*v products in an
/// iterative solver (Conjugate Gradient or GMRES).
///
/// **Pros**: Memory efficient, scales to huge problems
/// **Cons**: Requires iterative solver, may need more iterations
pub struct MatrixFreeJtJ {
    /// Maximum CG iterations
    pub max_cg_iterations: usize,
    /// CG convergence tolerance
    pub cg_tolerance: f32,
}

impl MatrixFreeJtJ {
    /// Create a new matrix-free strategy
    pub fn new(_device: &Device, max_cg_iterations: usize, cg_tolerance: f32) -> Result<Self> {
        Ok(Self {
            max_cg_iterations,
            cg_tolerance,
        })
    }
}

impl JtJStrategy for MatrixFreeJtJ {
    fn compute(
        &self,
        _device: &Device,
        _queue: &Queue,
        ctx: &JtJContext,
    ) -> Result<JtJResult> {
        // Matrix-free doesn't return a matrix - it would integrate with
        // the solver differently. For now, return zeros to match interface.

        Ok(JtJResult {
            jtj: vec![], // Empty - matrix-free doesn't form matrix
            jtr: vec![0.0; ctx.num_params],
            sparsity_info: None, // Not applicable for matrix-free
        })
    }

    fn name(&self) -> &'static str {
        "Matrix-Free (CG)"
    }

    fn description(&self) -> &'static str {
        "Uses J*v products in iterative solver. Never forms J^T J matrix."
    }

    fn captures_interactions(&self) -> bool {
        true // Implicitly captures all interactions through J*v products
    }

    fn complexity(&self) -> ComplexityClass {
        ComplexityClass::MatrixFree
    }
}

// ============================================================================
// Strategy Registry
// ============================================================================

/// Registry of available J^T J strategies
pub struct JtJStrategyRegistry;

impl JtJStrategyRegistry {
    /// List all available strategies with their properties
    pub fn list_strategies() -> Vec<(&'static str, &'static str, bool, ComplexityClass)> {
        vec![
            (
                "block-diagonal",
                "Only diagonal blocks, ignores interactions",
                false,
                ComplexityClass::LinearPerGaussian,
            ),
            (
                "full-dense",
                "All G² blocks, captures all interactions",
                true,
                ComplexityClass::QuadraticInGaussians,
            ),
            (
                "sparse",
                "Overlapping pairs only, middle ground",
                true,
                ComplexityClass::Sparse,
            ),
            (
                "hierarchical",
                "Multi-scale coarse-to-fine approach",
                true,
                ComplexityClass::Sparse,
            ),
            (
                "matrix-free",
                "Iterative solver, memory efficient",
                true,
                ComplexityClass::MatrixFree,
            ),
        ]
    }

    /// Create a strategy by name
    pub fn create(device: &Device, name: &str) -> Result<Box<dyn JtJStrategy>> {
        match name {
            "block-diagonal" => Ok(Box::new(BlockDiagonalJtJ::new(device)?)),
            "full-dense" => Ok(Box::new(FullDenseJtJ::new(device)?)),
            "sparse" => Ok(Box::new(SparseJtJ::new(device, 3.0)?)),  // 3σ = 99.7% of mass
            "hierarchical" => Ok(Box::new(HierarchicalJtJ::new(device, 3)?)),  // 3 levels
            "matrix-free" => Ok(Box::new(MatrixFreeJtJ::new(device, 100, 1e-6)?)),
            _ => Err(crate::error::GpuError::Other(
                format!("Unknown JtJ strategy: {}", name)
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_registry() {
        let strategies = JtJStrategyRegistry::list_strategies();
        assert_eq!(strategies.len(), 5);

        // Block-diagonal should not capture interactions
        assert!(!strategies[0].2);

        // Others should capture interactions
        assert!(strategies[1].2);  // full-dense
        assert!(strategies[2].2);  // sparse
        assert!(strategies[3].2);  // hierarchical
        assert!(strategies[4].2);  // matrix-free
    }

    #[test]
    fn test_sparsity_info() {
        let info = SparsityInfo {
            nnz: 1000,
            total: 10000,
            captures_interactions: true,
        };
        assert!((info.density() - 0.1).abs() < 1e-6);
    }
}
