//! # LGI File Format
//!
//! Chunk-based binary format for Gaussian image/video compression.
//!
//! ## Format Structure
//!
//! ```text
//! LGI File:
//!   Magic Number: "LGI\0" (4 bytes)
//!
//!   Chunks:
//!     HEAD - Header/metadata
//!     GAUS - Gaussian data (VQ codebook + indices)
//!     meta - JSON metadata (optional)
//!     INDE - Index for random access (optional)
//!     IEND - End marker
//!
//! Chunk Structure:
//!   Length: u32 (4 bytes)
//!   Type:   [u8; 4] (4 bytes, e.g., "HEAD")
//!   Data:   [u8; length]
//!   CRC32:  u32 (4 bytes)
//! ```

#![warn(missing_docs)]

pub mod error;
pub mod chunk;
pub mod header;
pub mod gaussian_data;
pub mod metadata;
pub mod writer;
pub mod reader;
pub mod validation;
pub mod quantization;
pub mod compression;

pub use error::{LgiFormatError, Result};
pub use chunk::{Chunk, ChunkType};
pub use header::LgiHeader;
pub use gaussian_data::GaussianData;
pub use metadata::LgiMetadata;
pub use writer::LgiWriter;
pub use reader::LgiReader;
pub use quantization::{QuantizationProfile, Quantizer, QuantizedGaussian};
pub use compression::CompressionConfig;

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

/// Magic number for LGI files: "LGI\0"
pub const MAGIC_NUMBER: [u8; 4] = [b'L', b'G', b'I', 0x00];

/// LGI file format version
pub const FORMAT_VERSION: u32 = 1;

/// Complete LGI file representation
#[derive(Debug, Clone)]
pub struct LgiFile {
    /// File header
    pub header: LgiHeader,

    /// Gaussian data
    pub gaussian_data: GaussianData,

    /// Optional metadata
    pub metadata: Option<LgiMetadata>,
}

impl LgiFile {
    /// Create new LGI file from Gaussians
    pub fn new(
        gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
        width: u32,
        height: u32,
    ) -> Self {
        let header = LgiHeader::new(width, height, gaussians.len());
        let gaussian_data = GaussianData::from_gaussians(gaussians);

        Self {
            header,
            gaussian_data,
            metadata: None,
        }
    }

    /// Create with compression configuration
    pub fn with_compression(
        gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
        width: u32,
        height: u32,
        config: CompressionConfig,
    ) -> Self {
        let mut header = LgiHeader::new(width, height, gaussians.len());

        let gaussian_data = if config.enable_vq {
            // VQ compression mode
            header = header.with_vq(config.vq_codebook_size as u16);
            if config.enable_zstd {
                header = header.with_zstd();
                GaussianData::from_gaussians_vq_zstd(gaussians, config.vq_codebook_size, config.zstd_level)
            } else {
                GaussianData::from_gaussians_vq(gaussians, config.vq_codebook_size)
            }
        } else {
            // Quantization profile mode
            if config.enable_zstd {
                header = header.with_zstd();
            }
            GaussianData::from_gaussians_quantized(
                gaussians,
                config.quantization,
                config.enable_zstd,
                config.zstd_level,
            )
        };

        Self {
            header,
            gaussian_data,
            metadata: None,
        }
    }

    /// Create with VQ compression (legacy API for compatibility)
    pub fn with_vq(
        gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
        width: u32,
        height: u32,
        codebook_size: usize,
    ) -> Self {
        Self::with_compression(
            gaussians,
            width,
            height,
            CompressionConfig::balanced(),
        )
    }

    /// Add metadata
    pub fn with_metadata(mut self, metadata: LgiMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get Gaussians
    pub fn gaussians(&self) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        self.gaussian_data.to_gaussians()
    }

    /// Get image dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.header.width, self.header.height)
    }

    /// Get Gaussian count
    pub fn gaussian_count(&self) -> usize {
        self.header.gaussian_count as usize
    }

    /// Check if file uses VQ compression
    pub fn is_compressed(&self) -> bool {
        self.gaussian_data.is_vq_compressed()
    }

    /// Get compression ratio (if VQ compressed)
    pub fn compression_ratio(&self) -> f32 {
        self.gaussian_data.compression_ratio()
    }
}
