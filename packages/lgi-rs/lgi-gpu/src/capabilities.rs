//! GPU capabilities detection and feature level management

use wgpu;

/// Backend type information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Vulkan (Linux, Windows, Android)
    Vulkan,
    /// DirectX 12 (Windows)
    Dx12,
    /// Metal (macOS, iOS)
    Metal,
    /// WebGPU (browsers)
    WebGpu,
    /// OpenGL (fallback)
    OpenGl,
}

impl From<wgpu::Backend> for BackendType {
    fn from(backend: wgpu::Backend) -> Self {
        match backend {
            wgpu::Backend::Vulkan => BackendType::Vulkan,
            wgpu::Backend::Dx12 => BackendType::Dx12,
            wgpu::Backend::Metal => BackendType::Metal,
            wgpu::Backend::BrowserWebGpu => BackendType::WebGpu,
            wgpu::Backend::Gl => BackendType::OpenGl,
            _ => BackendType::Vulkan, // Default fallback
        }
    }
}

/// GPU adapter information
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Backend type
    pub backend: BackendType,

    /// Adapter name (e.g., "NVIDIA GeForce RTX 4090")
    pub adapter_name: String,

    /// Device type (Discrete, Integrated, Cpu, etc.)
    pub device_type: wgpu::DeviceType,

    /// Driver name and version
    pub driver_info: String,
}

impl BackendInfo {
    /// Create from wgpu adapter info
    pub fn from_adapter_info(info: &wgpu::AdapterInfo) -> Self {
        Self {
            backend: info.backend.into(),
            adapter_name: info.name.clone(),
            device_type: info.device_type,
            driver_info: format!("{} - Driver: {}", info.backend, info.driver_info),
        }
    }
}

/// GPU capabilities and supported features
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// Backend information
    pub backend_info: BackendInfo,

    /// Feature level (core, advanced, cutting-edge)
    pub feature_level: FeatureLevel,

    /// Maximum workgroup size (x, y, z)
    pub max_workgroup_size: (u32, u32, u32),

    /// Maximum buffer size
    pub max_buffer_size: u64,

    /// Maximum texture dimension
    pub max_texture_dimension_2d: u32,

    /// Supported features
    pub features: SupportedFeatures,
}

/// Feature level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FeatureLevel {
    /// Core features only (WebGPU baseline)
    Core,
    /// Common features (most modern GPUs)
    Advanced,
    /// Cutting-edge features (latest hardware)
    CuttingEdge,
}

/// Detailed feature support
#[derive(Debug, Clone)]
pub struct SupportedFeatures {
    /// Compute shaders (required)
    pub compute_shaders: bool,

    /// Timestamp queries (for profiling)
    pub timestamp_query: bool,

    /// Shader f16 support (half-precision)
    pub shader_f16: bool,

    /// Subgroup operations (wave intrinsics)
    pub subgroup_operations: bool,

    /// Push constants
    pub push_constants: bool,

    /// Multi-queue support
    pub multi_queue: bool,

    /// Storage buffer array dynamic indexing
    pub dynamic_indexing: bool,
}

impl GpuCapabilities {
    /// Detect capabilities from adapter and device
    pub fn detect(
        adapter_info: &wgpu::AdapterInfo,
        device_features: wgpu::Features,
        device_limits: &wgpu::Limits,
    ) -> Self {
        let backend_info = BackendInfo::from_adapter_info(adapter_info);

        // Detect supported features
        let features = SupportedFeatures {
            compute_shaders: true, // Required for wgpu compute
            timestamp_query: device_features.contains(wgpu::Features::TIMESTAMP_QUERY),
            shader_f16: device_features.contains(wgpu::Features::SHADER_F16),
            subgroup_operations: device_features.contains(wgpu::Features::SUBGROUP),
            push_constants: device_features.contains(wgpu::Features::PUSH_CONSTANTS),
            multi_queue: true,  // v27: Multi-draw indirect now unconditionally supported
            dynamic_indexing: true, // Core WebGPU feature
        };

        // Classify feature level
        let feature_level = if features.shader_f16 && features.subgroup_operations {
            FeatureLevel::CuttingEdge
        } else if features.timestamp_query && features.push_constants {
            FeatureLevel::Advanced
        } else {
            FeatureLevel::Core
        };

        Self {
            backend_info,
            feature_level,
            max_workgroup_size: (
                device_limits.max_compute_workgroup_size_x,
                device_limits.max_compute_workgroup_size_y,
                device_limits.max_compute_workgroup_size_z,
            ),
            max_buffer_size: device_limits.max_buffer_size,
            max_texture_dimension_2d: device_limits.max_texture_dimension_2d,
            features,
        }
    }

    /// Print capabilities summary
    pub fn print_info(&self) {
        println!("GPU Capabilities:");
        println!("  Backend: {:?}", self.backend_info.backend);
        println!("  Adapter: {}", self.backend_info.adapter_name);
        println!("  Device Type: {:?}", self.backend_info.device_type);
        println!("  Feature Level: {:?}", self.feature_level);
        println!("  Max Workgroup: {:?}", self.max_workgroup_size);
        println!("  Max Buffer: {} MB", self.max_buffer_size / 1024 / 1024);
        println!("  Max Texture 2D: {}×{}", self.max_texture_dimension_2d, self.max_texture_dimension_2d);

        println!("  Advanced Features:");
        println!("    Timestamp Query: {}", self.features.timestamp_query);
        println!("    Shader F16: {}", self.features.shader_f16);
        println!("    Subgroup Ops: {}", self.features.subgroup_operations);
        println!("    Push Constants: {}", self.features.push_constants);
    }
}
