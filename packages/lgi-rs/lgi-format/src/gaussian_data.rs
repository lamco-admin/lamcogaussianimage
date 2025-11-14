//! Gaussian data storage (GAUS chunk)
//!
//! Supports two modes:
//! 1. Uncompressed: Full 32-bit floats for all parameters
//! 2. VQ Compressed: Codebook + 8-bit indices

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};
use lgi_encoder::vector_quantization::{VectorQuantizer, GaussianVector};
use crate::quantization::{QuantizationProfile, Quantizer, QuantizedGaussian};
use serde::{Deserialize, Serialize};

/// Gaussian data storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GaussianData {
    /// Uncompressed: Full precision
    Uncompressed(Vec<GaussianVector>),

    /// Quantized: Using LGIQ profiles
    Quantized {
        /// Quantization profile
        profile: QuantizationProfile,

        /// Quantized Gaussians
        quantized: Vec<QuantizedGaussian>,

        /// zstd compressed data (if enabled)
        zstd_compressed: Option<Vec<u8>>,
    },

    /// VQ Compressed: Codebook + indices (legacy, for compatibility)
    VqCompressed {
        /// VQ codebook
        codebook: Vec<GaussianVector>,

        /// Indices into codebook
        indices: Vec<u8>,

        /// zstd compressed indices (if enabled)
        zstd_compressed: Option<Vec<u8>>,
    },
}

impl GaussianData {
    /// Create from Gaussians (uncompressed)
    pub fn from_gaussians(gaussians: Vec<Gaussian2D<f32, Euler<f32>>>) -> Self {
        let vectors: Vec<GaussianVector> = gaussians
            .iter()
            .map(|g| GaussianVector::from_gaussian(g))
            .collect();

        GaussianData::Uncompressed(vectors)
    }

    /// Create with quantization profile
    pub fn from_gaussians_quantized(
        gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
        profile: QuantizationProfile,
        enable_zstd: bool,
        zstd_level: i32,
    ) -> Self {
        let quantizer = Quantizer::new(profile);
        let quantized: Vec<QuantizedGaussian> = gaussians
            .iter()
            .map(|g| quantizer.quantize(g))
            .collect();

        let zstd_compressed = if enable_zstd {
            // Serialize quantized data
            let serialized = bincode::serialize(&quantized).expect("Serialization failed");

            // Compress with zstd
            zstd::encode_all(&serialized[..], zstd_level).ok()
        } else {
            None
        };

        GaussianData::Quantized {
            profile,
            quantized,
            zstd_compressed,
        }
    }

    /// Create from Gaussians with VQ compression (legacy)
    pub fn from_gaussians_vq(
        gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
        codebook_size: usize,
    ) -> Self {
        // Train VQ codebook
        let mut vq = VectorQuantizer::new(codebook_size);
        vq.train(&gaussians, 100);

        // Quantize
        let indices = vq.quantize_all(&gaussians);

        GaussianData::VqCompressed {
            codebook: vq.codebook.clone(),
            indices,
            zstd_compressed: None,
        }
    }

    /// Create VQ with zstd compression
    pub fn from_gaussians_vq_zstd(
        gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
        codebook_size: usize,
        zstd_level: i32,
    ) -> Self {
        let mut vq = VectorQuantizer::new(codebook_size);
        vq.train(&gaussians, 100);

        let indices = vq.quantize_all(&gaussians);

        // Compress indices with zstd
        let compressed = zstd::encode_all(&indices[..], zstd_level).ok();

        GaussianData::VqCompressed {
            codebook: vq.codebook.clone(),
            indices,
            zstd_compressed: compressed,
        }
    }

    /// Convert to Gaussians
    pub fn to_gaussians(&self) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        match self {
            GaussianData::Uncompressed(vectors) => {
                vectors.iter().map(|v| v.to_gaussian()).collect()
            }
            GaussianData::Quantized { profile, quantized, zstd_compressed } => {
                // If zstd compressed, decompress first
                let quant_data = if let Some(compressed) = zstd_compressed {
                    let decompressed = zstd::decode_all(&compressed[..])
                        .expect("zstd decompression failed");
                    bincode::deserialize(&decompressed).expect("Deserialization failed")
                } else {
                    quantized.clone()
                };

                // Dequantize
                let quantizer = Quantizer::new(*profile);
                quant_data.iter().map(|q| quantizer.dequantize(q)).collect()
            }
            GaussianData::VqCompressed { codebook, indices, zstd_compressed } => {
                // If zstd compressed, decompress indices first
                let idx_data = if let Some(compressed) = zstd_compressed {
                    zstd::decode_all(&compressed[..]).expect("zstd decompression failed")
                } else {
                    indices.clone()
                };

                // Dequantize using codebook
                idx_data
                    .iter()
                    .map(|&idx| {
                        let vector = &codebook[idx as usize];
                        vector.to_gaussian()
                    })
                    .collect()
            }
        }
    }

    /// Check if VQ compressed
    pub fn is_vq_compressed(&self) -> bool {
        matches!(self, GaussianData::VqCompressed { .. })
    }

    /// Check if quantized
    pub fn is_quantized(&self) -> bool {
        matches!(self, GaussianData::Quantized { .. })
    }

    /// Check if zstd compressed
    pub fn is_zstd_compressed(&self) -> bool {
        match self {
            GaussianData::Quantized { zstd_compressed, .. } => zstd_compressed.is_some(),
            GaussianData::VqCompressed { zstd_compressed, .. } => zstd_compressed.is_some(),
            _ => false,
        }
    }

    /// Get Gaussian count
    pub fn gaussian_count(&self) -> usize {
        match self {
            GaussianData::Uncompressed(vectors) => vectors.len(),
            GaussianData::Quantized { quantized, .. } => quantized.len(),
            GaussianData::VqCompressed { indices, .. } => indices.len(),
        }
    }

    /// Get uncompressed size in bytes
    pub fn uncompressed_size(&self) -> usize {
        self.gaussian_count() * std::mem::size_of::<GaussianVector>()
    }

    /// Get compressed size in bytes
    pub fn compressed_size(&self) -> usize {
        match self {
            GaussianData::Uncompressed(vectors) => {
                vectors.len() * std::mem::size_of::<GaussianVector>()
            }
            GaussianData::Quantized { profile, quantized, zstd_compressed } => {
                if let Some(compressed) = zstd_compressed {
                    compressed.len()
                } else {
                    quantized.len() * profile.bytes_per_gaussian()
                }
            }
            GaussianData::VqCompressed { codebook, indices, zstd_compressed } => {
                let codebook_size = codebook.len() * std::mem::size_of::<GaussianVector>();
                let indices_size = if let Some(compressed) = zstd_compressed {
                    compressed.len()
                } else {
                    indices.len()
                };
                codebook_size + indices_size
            }
        }
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let uncompressed = self.uncompressed_size() as f32;
        let compressed = self.compressed_size() as f32;

        if compressed > 0.0 {
            uncompressed / compressed
        } else {
            1.0
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Failed to serialize Gaussian data")
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::{vec::Vector2, color::Color4};

    fn create_test_gaussians(count: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        (0..count)
            .map(|i| {
                let x = (i as f32) / (count as f32);
                Gaussian2D::new(
                    Vector2::new(x, 0.5),
                    Euler::isotropic(0.01),
                    Color4::rgb(x, 0.5, 0.5),
                    0.8,
                )
            })
            .collect()
    }

    #[test]
    fn test_uncompressed_roundtrip() {
        let gaussians = create_test_gaussians(100);
        let original_count = gaussians.len();

        let gaussian_data = GaussianData::from_gaussians(gaussians.clone());
        let reconstructed = gaussian_data.to_gaussians();

        assert_eq!(reconstructed.len(), original_count);

        // Check first Gaussian
        assert!((reconstructed[0].position.x - gaussians[0].position.x).abs() < 1e-6);
        assert!((reconstructed[0].opacity - gaussians[0].opacity).abs() < 1e-6);
    }

    #[test]
    fn test_vq_compression() {
        let gaussians = create_test_gaussians(500);
        let original_count = gaussians.len();

        let gaussian_data = GaussianData::from_gaussians_vq(gaussians.clone(), 256);

        assert!(gaussian_data.is_vq_compressed());
        assert_eq!(gaussian_data.gaussian_count(), original_count);

        // Check compression ratio
        let ratio = gaussian_data.compression_ratio();
        println!("VQ Compression ratio: {:.2}×", ratio);
        // Note: Actual compression varies based on serialization overhead
        // For 500 Gaussians with 256 codebook: expect ~1.8-2.5× from VQ alone
        // (Full file with zstd achieves 5-10×)
        assert!(ratio > 1.5, "Expected >1.5× compression, got {:.2}×", ratio);

        // Reconstruct
        let reconstructed = gaussian_data.to_gaussians();
        assert_eq!(reconstructed.len(), original_count);

        // Quality should be reasonable (VQ has some loss)
        let diff = (reconstructed[0].position.x - gaussians[0].position.x).abs();
        assert!(diff < 0.1, "Position error too large: {}", diff);
    }

    #[test]
    fn test_serialization() {
        let gaussians = create_test_gaussians(100);
        let gaussian_data = GaussianData::from_gaussians_vq(gaussians, 64);

        let bytes = gaussian_data.to_bytes();
        let decoded = GaussianData::from_bytes(&bytes).unwrap();

        assert!(decoded.is_vq_compressed());
        assert_eq!(decoded.gaussian_count(), 100);
    }
}
