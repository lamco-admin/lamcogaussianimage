//! LGI file header (HEAD chunk)

use crate::FORMAT_VERSION;
use serde::{Deserialize, Serialize};

/// LGI file header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LgiHeader {
    /// Format version
    pub version: u32,

    /// Image width in pixels
    pub width: u32,

    /// Image height in pixels
    pub height: u32,

    /// Number of Gaussians
    pub gaussian_count: u32,

    /// Compression flags
    pub compression_flags: CompressionFlags,

    /// Color space (0 = sRGB, 1 = Linear, 2 = Display-P3, etc.)
    pub color_space: u8,

    /// Bit depth per channel
    pub bit_depth: u8,

    /// Reserved for future use
    pub reserved: [u8; 16],
}

/// Compression flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionFlags {
    /// Uses VQ compression
    pub vq_compressed: bool,

    /// Uses zstd compression on top
    pub zstd_compressed: bool,

    /// VQ codebook size (if VQ compressed)
    pub vq_codebook_size: u16,

    /// Reserved flags
    pub reserved: u16,
}

impl LgiHeader {
    /// Create new header
    pub fn new(width: u32, height: u32, gaussian_count: usize) -> Self {
        Self {
            version: FORMAT_VERSION,
            width,
            height,
            gaussian_count: gaussian_count as u32,
            compression_flags: CompressionFlags::default(),
            color_space: 0, // sRGB
            bit_depth: 8,
            reserved: [0; 16],
        }
    }

    /// Enable VQ compression
    pub fn with_vq(mut self, codebook_size: u16) -> Self {
        self.compression_flags.vq_compressed = true;
        self.compression_flags.vq_codebook_size = codebook_size;
        self
    }

    /// Enable zstd compression
    pub fn with_zstd(mut self) -> Self {
        self.compression_flags.zstd_compressed = true;
        self
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Failed to serialize header")
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

impl Default for CompressionFlags {
    fn default() -> Self {
        Self {
            vq_compressed: false,
            zstd_compressed: false,
            vq_codebook_size: 0,
            reserved: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = LgiHeader::new(1920, 1080, 5000);

        let bytes = header.to_bytes();
        let decoded = LgiHeader::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.width, 1920);
        assert_eq!(decoded.height, 1080);
        assert_eq!(decoded.gaussian_count, 5000);
        assert_eq!(decoded.version, FORMAT_VERSION);
    }

    #[test]
    fn test_header_with_compression() {
        let header = LgiHeader::new(640, 480, 1000)
            .with_vq(256)
            .with_zstd();

        assert!(header.compression_flags.vq_compressed);
        assert!(header.compression_flags.zstd_compressed);
        assert_eq!(header.compression_flags.vq_codebook_size, 256);

        let bytes = header.to_bytes();
        let decoded = LgiHeader::from_bytes(&bytes).unwrap();

        assert!(decoded.compression_flags.vq_compressed);
        assert_eq!(decoded.compression_flags.vq_codebook_size, 256);
    }
}
