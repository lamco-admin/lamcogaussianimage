//! Error types for LGI core operations

use thiserror::Error;

/// Result type for LGI operations
pub type Result<T> = std::result::Result<T, LgiError>;

/// Errors that can occur during LGI encoding/decoding
#[derive(Error, Debug)]
pub enum LgiError {
    /// Invalid Gaussian parameters
    #[error("Invalid Gaussian: {0}")]
    InvalidGaussian(String),

    /// Image processing error
    #[error("Image error: {0}")]
    ImageError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Optimization failed
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}
