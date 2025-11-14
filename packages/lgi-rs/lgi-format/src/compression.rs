//! Comprehensive compression configuration
//!
//! Integrates: Quantization + VQ + zstd

use crate::quantization::QuantizationProfile;
use serde::{Deserialize, Serialize};

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Quantization profile
    pub quantization: QuantizationProfile,

    /// Enable VQ compression
    pub enable_vq: bool,

    /// VQ codebook size (if enabled)
    pub vq_codebook_size: usize,

    /// Enable zstd compression
    pub enable_zstd: bool,

    /// zstd compression level (0-22, 9 is good default)
    pub zstd_level: i32,
}

impl CompressionConfig {
    /// Create new compression config
    pub fn new() -> Self {
        Self::balanced()
    }

    /// Balanced preset: Good quality/size tradeoff
    pub fn balanced() -> Self {
        Self {
            quantization: QuantizationProfile::LGIQ_B,
            enable_vq: true,
            vq_codebook_size: 256,
            enable_zstd: true,
            zstd_level: 9,
        }
    }

    /// Small preset: Maximum compression
    pub fn small() -> Self {
        Self {
            quantization: QuantizationProfile::LGIQ_B,
            enable_vq: true,
            vq_codebook_size: 128,  // Smaller codebook
            enable_zstd: true,
            zstd_level: 19,  // Higher compression
        }
    }

    /// High quality preset: Better quality
    pub fn high_quality() -> Self {
        Self {
            quantization: QuantizationProfile::LGIQ_H,
            enable_vq: false,  // No VQ for highest quality
            vq_codebook_size: 0,
            enable_zstd: true,
            zstd_level: 9,
        }
    }

    /// Lossless preset: Bit-exact reconstruction
    pub fn lossless() -> Self {
        Self {
            quantization: QuantizationProfile::LGIQ_X,
            enable_vq: false,  // No VQ for lossless
            vq_codebook_size: 0,
            enable_zstd: true,
            zstd_level: 19,  // Max zstd compression
        }
    }

    /// No compression (for testing)
    pub fn uncompressed() -> Self {
        Self {
            quantization: QuantizationProfile::LGIQ_X,
            enable_vq: false,
            vq_codebook_size: 0,
            enable_zstd: false,
            zstd_level: 0,
        }
    }

    /// Custom configuration
    pub fn custom(
        quantization: QuantizationProfile,
        enable_vq: bool,
        vq_codebook_size: usize,
        enable_zstd: bool,
        zstd_level: i32,
    ) -> Self {
        Self {
            quantization,
            enable_vq,
            vq_codebook_size,
            enable_zstd,
            zstd_level,
        }
    }

    /// Get expected compression ratio
    pub fn expected_compression_ratio(&self, gaussian_count: usize) -> f32 {
        let uncompressed = (gaussian_count * 48) as f32;

        let quantized = if self.enable_vq && gaussian_count > 500 {
            // VQ: codebook + indices
            (self.vq_codebook_size * 36) as f32 + gaussian_count as f32
        } else {
            // Just quantization
            (gaussian_count * self.quantization.bytes_per_gaussian()) as f32
        };

        let with_zstd = if self.enable_zstd {
            quantized * 0.7  // zstd typically achieves 30% reduction
        } else {
            quantized
        };

        uncompressed / with_zstd
    }

    /// Get expected quality range (PSNR)
    pub fn expected_psnr_range(&self) -> (f32, f32) {
        let (min, max) = self.quantization.expected_psnr_range();

        // VQ typically reduces PSNR by 0.5-1 dB (with QA training)
        if self.enable_vq {
            (min - 1.0, max - 0.5)
        } else {
            (min, max)
        }
    }

    /// Check if lossy
    pub fn is_lossy(&self) -> bool {
        !self.quantization.is_lossless() || self.enable_vq
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Uncompressed size (bytes)
    pub uncompressed_size: usize,

    /// Compressed size (bytes)
    pub compressed_size: usize,

    /// Compression ratio
    pub ratio: f32,

    /// Quality loss (PSNR difference)
    pub quality_loss_db: f32,

    /// Compression breakdown
    pub breakdown: CompressionBreakdown,
}

/// Detailed compression breakdown
#[derive(Debug, Clone)]
pub struct CompressionBreakdown {
    /// Size after quantization
    pub after_quantization: usize,

    /// Size after VQ (if enabled)
    pub after_vq: Option<usize>,

    /// Size after zstd (if enabled)
    pub after_zstd: Option<usize>,

    /// Final size
    pub final_size: usize,
}

impl CompressionStats {
    /// Calculate compression ratio
    pub fn ratio(&self) -> f32 {
        self.uncompressed_size as f32 / self.compressed_size as f32
    }

    /// Pretty print statistics
    pub fn print(&self) {
        println!("Compression Statistics:");
        println!("  Uncompressed: {} KB", self.uncompressed_size / 1024);
        println!("  Compressed:   {} KB", self.compressed_size / 1024);
        println!("  Ratio:        {:.2}×", self.ratio);
        println!("  Quality loss: {:.2} dB", self.quality_loss_db);

        println!("\n  Breakdown:");
        println!("    After quantization: {} KB", self.breakdown.after_quantization / 1024);
        if let Some(vq) = self.breakdown.after_vq {
            println!("    After VQ:           {} KB", vq / 1024);
        }
        if let Some(zstd) = self.breakdown.after_zstd {
            println!("    After zstd:         {} KB", zstd / 1024);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_configs() {
        let balanced = CompressionConfig::balanced();
        assert_eq!(balanced.quantization, QuantizationProfile::LGIQ_B);
        assert!(balanced.enable_vq);
        assert!(balanced.enable_zstd);

        let lossless = CompressionConfig::lossless();
        assert_eq!(lossless.quantization, QuantizationProfile::LGIQ_X);
        assert!(!lossless.enable_vq);
        assert!(lossless.enable_zstd);
    }

    #[test]
    fn test_expected_compression_ratios() {
        let balanced = CompressionConfig::balanced();
        let ratio = balanced.expected_compression_ratio(1000);
        println!("Balanced (1000G): {:.2}×", ratio);
        assert!(ratio > 4.0 && ratio < 10.0);

        let lossless = CompressionConfig::lossless();
        let ratio = lossless.expected_compression_ratio(1000);
        println!("Lossless (1000G): {:.2}×", ratio);
        assert!(ratio > 1.5 && ratio < 3.5);
    }
}
