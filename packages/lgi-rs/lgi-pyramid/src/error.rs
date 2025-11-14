//! Pyramid error types

use thiserror::Error;

/// Result type for pyramid operations
pub type Result<T> = std::result::Result<T, PyramidError>;

/// Pyramid errors
#[derive(Error, Debug)]
pub enum PyramidError {
    /// LGI core error
    #[error("LGI error: {0}")]
    LgiCore(#[from] lgi_core::LgiError),

    /// Invalid pyramid level
    #[error("Invalid pyramid level: {0}")]
    InvalidLevel(usize),

    /// Pyramid not built
    #[error("Pyramid not built yet")]
    NotBuilt,

    /// Invalid viewport
    #[error("Invalid viewport: {0}")]
    InvalidViewport(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}
