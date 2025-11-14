//! GPU rendering error types

use thiserror::Error;

/// Result type for GPU operations
pub type Result<T> = std::result::Result<T, GpuError>;

/// GPU rendering errors
#[derive(Error, Debug)]
pub enum GpuError {
    /// No compatible GPU adapter found
    #[error("No compatible GPU adapter found")]
    NoAdapter,

    /// Device request failed
    #[error("Failed to request GPU device: {0}")]
    DeviceRequest(String),

    /// Shader compilation failed
    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// Buffer creation failed
    #[error("Buffer creation failed: {0}")]
    BufferCreation(String),

    /// Pipeline creation failed
    #[error("Pipeline creation failed: {0}")]
    PipelineCreation(String),
    
    /// GPU not initialized
    #[error("GPU not initialized - call GpuManager::initialize() first")]
    NotInitialized,

    /// Rendering failed
    #[error("Rendering failed: {0}")]
    RenderFailed(String),

    /// Buffer mapping failed
    #[error("Buffer mapping failed")]
    BufferMapping,

    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// LGI core error
    #[error("LGI error: {0}")]
    LgiCore(#[from] lgi_core::LgiError),

    /// Generic error
    #[error("{0}")]
    Other(String),
}
