//! Error types for LGI format I/O

use thiserror::Error;
use std::io;

/// Result type for LGI format operations
pub type Result<T> = std::result::Result<T, LgiFormatError>;

/// Errors that can occur during LGI file I/O
#[derive(Error, Debug)]
pub enum LgiFormatError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Invalid magic number
    #[error("Invalid magic number: expected 'LGI\\0', got {0:?}")]
    InvalidMagicNumber([u8; 4]),

    /// Unsupported format version
    #[error("Unsupported format version: {0} (expected {1})")]
    UnsupportedVersion(u32, u32),

    /// Invalid chunk type
    #[error("Invalid chunk type: {0:?}")]
    InvalidChunkType([u8; 4]),

    /// CRC32 mismatch
    #[error("CRC32 mismatch: expected {expected:08x}, got {actual:08x}")]
    CrcMismatch {
        /// Expected CRC32
        expected: u32,
        /// Actual CRC32
        actual: u32,
    },

    /// Missing required chunk
    #[error("Missing required chunk: {0}")]
    MissingChunk(String),

    /// Invalid chunk data
    #[error("Invalid chunk data: {0}")]
    InvalidChunkData(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Compression error
    #[error("Compression error: {0}")]
    Compression(String),

    /// Decompression error
    #[error("Decompression error: {0}")]
    Decompression(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl From<serde_json::Error> for LgiFormatError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

impl From<bincode::Error> for LgiFormatError {
    fn from(err: bincode::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}
