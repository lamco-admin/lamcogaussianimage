//! GPU backend selection and initialization

use crate::{GpuError, Result, BackendInfo, GpuCapabilities};
use wgpu;

pub use crate::capabilities::BackendType;

/// GPU backend selector with auto-detection
pub struct BackendSelector {
    instance: wgpu::Instance,
}

impl BackendSelector {
    /// Create new selector (tries all available backends)
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        Self { instance }
    }

    /// Create with specific backends
    pub fn with_backends(backends: wgpu::Backends) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        Self { instance }
    }

    /// Select best available adapter
    pub async fn select_adapter(&self) -> Result<wgpu::Adapter> {
        // v27 API: request_adapter now returns Result instead of Option
        // Try high performance first (discrete GPU)
        match self.instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await {
            Ok(adapter) => {
                log::info!("✅ Selected high-performance adapter: {}", adapter.get_info().name);
                return Ok(adapter);
            }
            Err(_) => {
                // Try any available adapter
                match self.instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::None,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }).await {
                    Ok(adapter) => {
                        log::info!("✅ Selected available adapter: {}", adapter.get_info().name);
                        return Ok(adapter);
                    }
                    Err(_) => return Err(GpuError::NoAdapter),
                }
            }
        }
    }

    /// Request device with core features only (avoiding experimental)
    pub async fn request_device(&self, adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue)> {
        // Use only core features we need (compute shaders are always available)
        // v27 experimental features require unsafe, so we avoid them for now
        let features = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::PUSH_CONSTANTS;

        adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("LGI GPU Renderer"),
                required_features: features & adapter.features(), // Only request what's available
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::default(),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| GpuError::DeviceRequest(e.to_string()))
    }

    /// Enumerate all available adapters
    pub fn enumerate_adapters(&self) -> Vec<wgpu::Adapter> {
        self.instance.enumerate_adapters(wgpu::Backends::all())
    }

    /// Print all available adapters
    pub fn print_available_adapters(&self) {
        println!("Available GPU Adapters:");
        for (idx, adapter) in self.enumerate_adapters().iter().enumerate() {
            let info = adapter.get_info();
            println!("  [{}] {} - {:?} ({:?})",
                idx,
                info.name,
                info.backend,
                info.device_type
            );
        }
    }
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self::new()
    }
}
