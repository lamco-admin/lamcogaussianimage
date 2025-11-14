//! LGI File Reader
//! Reads Gaussian data from .lgi files following the LGI format specification

use std::fs::File;
use std::io::{Read, BufReader};
use std::path::Path;
use lgi_core::container_format::{LGI_MAGIC, LGIHeader, ParamEncoding, CompressionMethod, FeatureFlags};
use lgi_core::quantization::QuantizedGaussian;
use lgi_core::compression_utils::{delta_decode_positions};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use zstd;

pub struct LGIReader;

/// Result structure for progressive files
pub struct ProgressiveResult {
    pub header: LGIHeader,
    pub gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
    pub importance_order: Option<Vec<usize>>,
}

impl LGIReader {
    /// Read Gaussians from .lgi file
    pub fn read_file<P: AsRef<Path>>(
        path: P,
    ) -> std::io::Result<(LGIHeader, Vec<Gaussian2D<f32, Euler<f32>>>)> {
        let result = Self::read_file_progressive(path)?;
        Ok((result.header, result.gaussians))
    }

    /// Read Gaussians from .lgi file with progressive metadata
    pub fn read_file_progressive<P: AsRef<Path>>(
        path: P,
    ) -> std::io::Result<ProgressiveResult> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != LGI_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid LGI magic number",
            ));
        }

        // Read header
        let header = Self::read_header(&mut reader)?;

        // Check if progressive
        let is_progressive = (header.feature_flags & FeatureFlags::HAS_PROGRESSIVE) != 0;

        // Read PRGS chunk if progressive
        let importance_order = if is_progressive {
            Some(Self::read_prgs_chunk(&mut reader, header.gaussian_count)?)
        } else {
            None
        };

        // Read GAUS chunk based on encoding type and compression
        let gaussians = if header.param_encoding == ParamEncoding::Raw as u16 {
            Self::read_gaus_chunk_lossless(&mut reader, header.gaussian_count)?
        } else if header.compression_flags != 0 {
            Self::read_gaus_chunk_compressed(&mut reader, header.gaussian_count, header.compression_flags)?
        } else {
            Self::read_gaus_chunk(&mut reader, header.gaussian_count)?
        };

        Ok(ProgressiveResult {
            header,
            gaussians,
            importance_order,
        })
    }

    fn read_header<R: Read>(reader: &mut R) -> std::io::Result<LGIHeader> {
        let mut buf2 = [0u8; 2];
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        let mut buf1 = [0u8; 1];

        reader.read_exact(&mut buf2)?;
        let version_major = u16::from_le_bytes(buf2);

        reader.read_exact(&mut buf2)?;
        let version_minor = u16::from_le_bytes(buf2);

        reader.read_exact(&mut buf4)?;
        let canvas_width = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let canvas_height = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf2)?;
        let colorspace = u16::from_le_bytes(buf2);

        reader.read_exact(&mut buf1)?;
        let bitdepth = buf1[0];

        reader.read_exact(&mut buf1)?;
        let alpha_mode = buf1[0];

        reader.read_exact(&mut buf2)?;
        let param_encoding = u16::from_le_bytes(buf2);

        reader.read_exact(&mut buf2)?;
        let compression_flags = u16::from_le_bytes(buf2);

        reader.read_exact(&mut buf4)?;
        let gaussian_count = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let feature_flags = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let background_color = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf8)?;
        let index_offset = u64::from_le_bytes(buf8);

        Ok(LGIHeader {
            version_major,
            version_minor,
            canvas_width,
            canvas_height,
            colorspace,
            bitdepth,
            alpha_mode,
            param_encoding,
            compression_flags,
            gaussian_count,
            feature_flags,
            background_color,
            index_offset,
        })
    }

    fn read_gaus_chunk<R: Read>(
        reader: &mut R,
        gaussian_count: u32,
    ) -> std::io::Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        // Read chunk length
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let _chunk_length = u32::from_le_bytes(buf4);

        // Read chunk type
        let mut chunk_type = [0u8; 4];
        reader.read_exact(&mut chunk_type)?;
        if &chunk_type != b"GAUS" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected GAUS chunk",
            ));
        }

        // Read Gaussian data (11 bytes per Gaussian for baseline profile)
        let mut gaussians = Vec::with_capacity(gaussian_count as usize);

        for _ in 0..gaussian_count {
            let mut buf2 = [0u8; 2];
            let mut buf1 = [0u8; 1];

            // Read quantized data
            reader.read_exact(&mut buf2)?;
            let pos_x = u16::from_le_bytes(buf2);

            reader.read_exact(&mut buf2)?;
            let pos_y = u16::from_le_bytes(buf2);

            reader.read_exact(&mut buf2)?;
            let scale_x = u16::from_le_bytes(buf2);

            reader.read_exact(&mut buf2)?;
            let scale_y = u16::from_le_bytes(buf2);

            reader.read_exact(&mut buf2)?;
            let rotation = u16::from_le_bytes(buf2);

            reader.read_exact(&mut buf1)?;
            let color_r = buf1[0] as u16;

            reader.read_exact(&mut buf1)?;
            let color_g = buf1[0] as u16;

            reader.read_exact(&mut buf1)?;
            let color_b = buf1[0] as u16;

            reader.read_exact(&mut buf1)?;
            let opacity = buf1[0] as u16;

            // Dequantize (assuming Baseline profile for now)
            let quantized = QuantizedGaussian {
                position: [pos_x, pos_y],
                scale: [scale_x, scale_y],
                rotation,
                color: [color_r, color_g, color_b],
                opacity,
                profile: lgi_core::quantization::LGIQProfile::Baseline,
            };

            let (px, py, sx, sy, rot, cr, cg, cb, op) = quantized.dequantize();

            gaussians.push(Gaussian2D::new(
                Vector2::new(px, py),
                Euler::new(sx, sy, rot),
                Color4::new(cr, cg, cb, 1.0),
                op,
            ));
        }

        // Read and verify CRC32
        reader.read_exact(&mut buf4)?;
        let _crc = u32::from_le_bytes(buf4);
        // TODO: Verify CRC

        Ok(gaussians)
    }

    /// Read compressed Gaussian data (with decompression and delta decoding)
    fn read_gaus_chunk_compressed<R: Read>(
        reader: &mut R,
        gaussian_count: u32,
        compression_flags: u16,
    ) -> std::io::Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        // Read chunk length
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let chunk_length = u32::from_le_bytes(buf4);

        // Read chunk type
        let mut chunk_type = [0u8; 4];
        reader.read_exact(&mut chunk_type)?;
        if &chunk_type != b"GAUS" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected GAUS chunk",
            ));
        }

        // Read compressed data
        let mut compressed_data = vec![0u8; chunk_length as usize];
        reader.read_exact(&mut compressed_data)?;

        // Decompress if zstd flag is set
        let decompressed = if compression_flags & (CompressionMethod::Zstd as u16) != 0 {
            zstd::decode_all(&compressed_data[..])?
        } else {
            compressed_data
        };

        // Read delta-encoded quantized positions and reconstruct
        let mut quantized_positions: Vec<[u16; 2]> = Vec::with_capacity(gaussian_count as usize);
        let mut gaussians = Vec::with_capacity(gaussian_count as usize);

        let mut offset = 0;

        // Decode delta positions if flag is set
        if compression_flags & (CompressionMethod::Delta as u16) != 0 {
            for i in 0..gaussian_count {
                // Read delta position (i16 deltas)
                let dx = i16::from_le_bytes([decompressed[offset], decompressed[offset + 1]]);
                let dy = i16::from_le_bytes([decompressed[offset + 2], decompressed[offset + 3]]);
                offset += 4;

                // Reconstruct absolute position
                if i == 0 {
                    // First is absolute
                    quantized_positions.push([dx as u16, dy as u16]);
                } else {
                    // Others are deltas
                    let prev_x = quantized_positions[i as usize - 1][0] as i32;
                    let prev_y = quantized_positions[i as usize - 1][1] as i32;
                    let abs_x = (prev_x + dx as i32) as u16;
                    let abs_y = (prev_y + dy as i32) as u16;
                    quantized_positions.push([abs_x, abs_y]);
                }

                // Read other parameters (7 bytes)
                let scale_x = u16::from_le_bytes([decompressed[offset], decompressed[offset + 1]]);
                let scale_y = u16::from_le_bytes([decompressed[offset + 2], decompressed[offset + 3]]);
                let rotation = u16::from_le_bytes([decompressed[offset + 4], decompressed[offset + 5]]);
                let color_r = decompressed[offset + 6] as u16;
                let color_g = decompressed[offset + 7] as u16;
                let color_b = decompressed[offset + 8] as u16;
                let opacity = decompressed[offset + 9] as u16;
                offset += 10;

                // Dequantize (assuming Baseline for compressed)
                let quantized = QuantizedGaussian {
                    position: quantized_positions[i as usize],
                    scale: [scale_x, scale_y],
                    rotation,
                    color: [color_r, color_g, color_b],
                    opacity,
                    profile: lgi_core::quantization::LGIQProfile::Baseline,
                };

                let (px, py, sx, sy, rot, cr, cg, cb, op) = quantized.dequantize();
                gaussians.push(Gaussian2D::new(
                    Vector2::new(px, py),
                    Euler::new(sx, sy, rot),
                    Color4::new(cr, cg, cb, 1.0),
                    op,
                ));
            }
        } else {
            // No delta encoding, read absolute positions
            for _ in 0..gaussian_count {
                let pos_x = u16::from_le_bytes([decompressed[offset], decompressed[offset + 1]]);
                let pos_y = u16::from_le_bytes([decompressed[offset + 2], decompressed[offset + 3]]);
                let scale_x = u16::from_le_bytes([decompressed[offset + 4], decompressed[offset + 5]]);
                let scale_y = u16::from_le_bytes([decompressed[offset + 6], decompressed[offset + 7]]);
                let rotation = u16::from_le_bytes([decompressed[offset + 8], decompressed[offset + 9]]);
                let color_r = decompressed[offset + 10] as u16;
                let color_g = decompressed[offset + 11] as u16;
                let color_b = decompressed[offset + 12] as u16;
                let opacity = decompressed[offset + 13] as u16;
                offset += 14;

                let quantized = QuantizedGaussian {
                    position: [pos_x, pos_y],
                    scale: [scale_x, scale_y],
                    rotation,
                    color: [color_r, color_g, color_b],
                    opacity,
                    profile: lgi_core::quantization::LGIQProfile::Baseline,
                };

                let (px, py, sx, sy, rot, cr, cg, cb, op) = quantized.dequantize();
                gaussians.push(Gaussian2D::new(
                    Vector2::new(px, py),
                    Euler::new(sx, sy, rot),
                    Color4::new(cr, cg, cb, 1.0),
                    op,
                ));
            }
        }

        // Read and verify CRC32
        reader.read_exact(&mut buf4)?;
        let _crc = u32::from_le_bytes(buf4);
        // TODO: Verify CRC

        Ok(gaussians)
    }

    /// Read lossless f32 Gaussian data (no quantization)
    fn read_gaus_chunk_lossless<R: Read>(
        reader: &mut R,
        gaussian_count: u32,
    ) -> std::io::Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        // Read chunk length
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let _chunk_length = u32::from_le_bytes(buf4);

        // Read chunk type
        let mut chunk_type = [0u8; 4];
        reader.read_exact(&mut chunk_type)?;
        if &chunk_type != b"GAUS" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected GAUS chunk",
            ));
        }

        // Read Gaussian data (36 bytes per Gaussian for lossless)
        let mut gaussians = Vec::with_capacity(gaussian_count as usize);

        for _ in 0..gaussian_count {
            let mut buf4 = [0u8; 4];

            // Read position (8 bytes)
            reader.read_exact(&mut buf4)?;
            let pos_x = f32::from_le_bytes(buf4);

            reader.read_exact(&mut buf4)?;
            let pos_y = f32::from_le_bytes(buf4);

            // Read scale (8 bytes)
            reader.read_exact(&mut buf4)?;
            let scale_x = f32::from_le_bytes(buf4);

            reader.read_exact(&mut buf4)?;
            let scale_y = f32::from_le_bytes(buf4);

            // Read rotation (4 bytes)
            reader.read_exact(&mut buf4)?;
            let rotation = f32::from_le_bytes(buf4);

            // Read color (12 bytes)
            reader.read_exact(&mut buf4)?;
            let color_r = f32::from_le_bytes(buf4);

            reader.read_exact(&mut buf4)?;
            let color_g = f32::from_le_bytes(buf4);

            reader.read_exact(&mut buf4)?;
            let color_b = f32::from_le_bytes(buf4);

            // Read opacity (4 bytes)
            reader.read_exact(&mut buf4)?;
            let opacity = f32::from_le_bytes(buf4);

            gaussians.push(Gaussian2D::new(
                Vector2::new(pos_x, pos_y),
                Euler::new(scale_x, scale_y, rotation),
                Color4::new(color_r, color_g, color_b, 1.0),
                opacity,
            ));
        }

        // Read and verify CRC32
        reader.read_exact(&mut buf4)?;
        let _crc = u32::from_le_bytes(buf4);
        // TODO: Verify CRC

        Ok(gaussians)
    }

    /// Read PRGS chunk (progressive loading metadata)
    fn read_prgs_chunk<R: Read>(
        reader: &mut R,
        gaussian_count: u32,
    ) -> std::io::Result<Vec<usize>> {
        // Read chunk length
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let _chunk_length = u32::from_le_bytes(buf4);

        // Read chunk type
        let mut chunk_type = [0u8; 4];
        reader.read_exact(&mut chunk_type)?;
        if &chunk_type != b"PRGS" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected PRGS chunk",
            ));
        }

        // Read importance order indices (u32, 4 bytes per index)
        let mut order = Vec::with_capacity(gaussian_count as usize);
        for _ in 0..gaussian_count {
            reader.read_exact(&mut buf4)?;
            let idx = u32::from_le_bytes(buf4);
            order.push(idx as usize);
        }

        // Read and verify CRC32
        reader.read_exact(&mut buf4)?;
        let _crc = u32::from_le_bytes(buf4);
        // TODO: Verify CRC

        Ok(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        use crate::file_writer::LGIWriter;

        let gaussians_orig = vec![
            Gaussian2D::new(
                Vector2::new(0.5, 0.5),
                Euler::isotropic(0.1),
                Color4::new(1.0, 0.0, 0.0, 1.0),
                1.0,
            ),
        ];

        let writer = LGIWriter::new(256, 256, 1);
        writer.write_file("/tmp/test_roundtrip.lgi", &gaussians_orig).unwrap();

        let (header, gaussians_read) = LGIReader::read_file("/tmp/test_roundtrip.lgi").unwrap();

        assert_eq!(header.canvas_width, 256);
        assert_eq!(header.canvas_height, 256);
        assert_eq!(header.gaussian_count, 1);
        assert_eq!(gaussians_read.len(), 1);

        // Check that data roundtrips (with quantization error)
        let g_orig = &gaussians_orig[0];
        let g_read = &gaussians_read[0];

        assert!((g_orig.position.x - g_read.position.x).abs() < 0.01);
        assert!((g_orig.position.y - g_read.position.y).abs() < 0.01);
    }
}
