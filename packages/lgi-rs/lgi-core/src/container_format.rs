//! LGI container format - Full specification implementation
//! Supports lossless/lossy, compressed/uncompressed storage

use std::io::{Write, Read};

pub const LGI_MAGIC: &[u8; 4] = b"LGI\0";

/// Parameter encoding modes
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamEncoding {
    Euler = 0,      // σx, σy, θ (rotation)
    Cholesky = 1,   // L11, L21, L22 (Cholesky factors)
    LogRadii = 2,   // log(σx), log(σy), θ
    InvCov = 3,     // Inverse covariance elements
    Raw = 255,      // Uncompressed f32 (lossless)
}

/// Compression methods
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionMethod {
    None = 0,
    Zstd = 1,       // zstd compression
    Lz4 = 2,        // LZ4 compression
    Brotli = 3,     // Brotli compression
    Delta = 16,     // Delta coding (positions)
    Predictive = 32, // Predictive coding (scales)
    VectorQuant = 64, // Vector quantization
}

/// Bit depth for quantized storage
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BitDepth {
    Bit8 = 8,
    Bit10 = 10,
    Bit12 = 12,
    Bit14 = 14,
    Bit16 = 16,
    Float32 = 32,  // Lossless
}

/// Feature flags
pub mod FeatureFlags {
    pub const HAS_TEXTURES: u32 = 1 << 0;
    pub const HAS_LOD: u32 = 1 << 1;
    pub const HAS_SPATIAL_INDEX: u32 = 1 << 2;
    pub const HAS_PROGRESSIVE: u32 = 1 << 3;
    pub const HAS_TILING: u32 = 1 << 4;
    pub const HAS_METADATA: u32 = 1 << 5;
    pub const IS_HDR: u32 = 1 << 6;
    pub const HAS_BLUE_NOISE: u32 = 1 << 7;
}

#[repr(C)]
pub struct LGIHeader {
    pub version_major: u16,
    pub version_minor: u16,
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub colorspace: u16,        // 0=sRGB, 1=Linear, 2=P3, 3=BT.2020
    pub bitdepth: u8,           // BitDepth enum
    pub alpha_mode: u8,         // 0=None, 1=Straight, 2=Premultiplied
    pub param_encoding: u16,    // ParamEncoding enum
    pub compression_flags: u16, // CompressionMethod flags (can combine)
    pub gaussian_count: u32,
    pub feature_flags: u32,     // FeatureFlags bits
    pub background_color: u32,  // RGBA8
    pub index_offset: u64,      // Offset to spatial index (0 = none)
}

impl Default for LGIHeader {
    fn default() -> Self {
        Self {
            version_major: 1,
            version_minor: 0,
            canvas_width: 1920,
            canvas_height: 1080,
            colorspace: 0,
            bitdepth: BitDepth::Bit8 as u8,
            alpha_mode: 1,
            param_encoding: ParamEncoding::Euler as u16,
            compression_flags: CompressionMethod::None as u16,
            gaussian_count: 0,
            feature_flags: 0,
            background_color: 0,
            index_offset: 0,
        }
    }
}

impl LGIHeader {
    /// Create lossless uncompressed configuration
    pub fn lossless_uncompressed() -> Self {
        Self {
            bitdepth: BitDepth::Float32 as u8,
            param_encoding: ParamEncoding::Raw as u16,
            compression_flags: CompressionMethod::None as u16,
            ..Default::default()
        }
    }

    /// Create lossless compressed configuration (zstd)
    pub fn lossless_compressed() -> Self {
        Self {
            bitdepth: BitDepth::Float32 as u8,
            param_encoding: ParamEncoding::Raw as u16,
            compression_flags: CompressionMethod::Zstd as u16,
            ..Default::default()
        }
    }

    /// Create lossy baseline configuration (LGIQ-B)
    pub fn lossy_baseline() -> Self {
        Self {
            bitdepth: BitDepth::Bit8 as u8,
            param_encoding: ParamEncoding::Euler as u16,
            compression_flags: CompressionMethod::None as u16,
            ..Default::default()
        }
    }

    /// Create lossy standard configuration (LGIQ-S, with compression)
    pub fn lossy_standard() -> Self {
        Self {
            bitdepth: BitDepth::Bit10 as u8,
            param_encoding: ParamEncoding::Euler as u16,
            compression_flags: (CompressionMethod::Delta as u16) | (CompressionMethod::Zstd as u16),
            ..Default::default()
        }
    }

    /// Create lossy high-fidelity configuration (LGIQ-H)
    pub fn lossy_high_fidelity() -> Self {
        Self {
            bitdepth: BitDepth::Bit14 as u8,
            param_encoding: ParamEncoding::Cholesky as u16,
            compression_flags: (CompressionMethod::Delta as u16)
                             | (CompressionMethod::Predictive as u16)
                             | (CompressionMethod::Zstd as u16),
            ..Default::default()
        }
    }
}

pub struct LGIChunk {
    pub chunk_type: [u8; 4],
    pub data: Vec<u8>,
}

impl LGIChunk {
    pub fn new(chunk_type: &[u8; 4], data: Vec<u8>) -> Self {
        Self {
            chunk_type: *chunk_type,
            data,
        }
    }

    pub fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let length = self.data.len() as u32;
        writer.write_all(&length.to_le_bytes())?;
        writer.write_all(&self.chunk_type)?;
        writer.write_all(&self.data)?;

        let crc = crc32(&self.chunk_type, &self.data);
        writer.write_all(&crc.to_le_bytes())?;

        Ok(())
    }

    pub fn read_from<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];

        // Read length
        reader.read_exact(&mut buf4)?;
        let length = u32::from_le_bytes(buf4);

        // Read chunk type
        let mut chunk_type = [0u8; 4];
        reader.read_exact(&mut chunk_type)?;

        // Read data
        let mut data = vec![0u8; length as usize];
        reader.read_exact(&mut data)?;

        // Read and verify CRC
        reader.read_exact(&mut buf4)?;
        let stored_crc = u32::from_le_bytes(buf4);
        let computed_crc = crc32(&chunk_type, &data);

        if stored_crc != computed_crc {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "CRC mismatch"
            ));
        }

        Ok(Self { chunk_type, data })
    }
}

fn crc32(chunk_type: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in chunk_type.iter().chain(data.iter()) {
        crc = crc ^ (byte as u32);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB88320
            } else {
                crc >> 1
            };
        }
    }
    crc ^ 0xFFFFFFFF
}

/// Chunk types
pub mod ChunkTypes {
    pub const HEAD: &[u8; 4] = b"HEAD";  // Header
    pub const GAUS: &[u8; 4] = b"GAUS";  // Gaussian data
    pub const TILE: &[u8; 4] = b"TILE";  // Tile boundaries
    pub const LODC: &[u8; 4] = b"LODC";  // LOD bands
    pub const PRGS: &[u8; 4] = b"PRGS";  // Progressive ordering
    pub const INDE: &[u8; 4] = b"INDE";  // Spatial index
    pub const META: &[u8; 4] = b"meta";  // Metadata
    pub const ICCP: &[u8; 4] = b"iCCP";  // ICC profile
    pub const EXIF: &[u8; 4] = b"eXIf";  // EXIF data
    pub const TEXT: &[u8; 4] = b"tEXt";  // Text comments
}
