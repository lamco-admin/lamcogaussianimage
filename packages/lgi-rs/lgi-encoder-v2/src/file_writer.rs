//! LGI File Writer
//! Writes Gaussian data to .lgi files following the LGI format specification

use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;
use lgi_core::container_format::{LGI_MAGIC, LGIHeader, LGIChunk, ParamEncoding, CompressionMethod, FeatureFlags};
use lgi_core::quantization::QuantizedGaussian;
use lgi_core::progressive;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};
use zstd;

pub struct LGIWriter {
    header: LGIHeader,
    use_progressive: bool,
}

impl LGIWriter {
    pub fn new(width: u32, height: u32, gaussian_count: u32) -> Self {
        let mut header = LGIHeader::default();
        header.canvas_width = width;
        header.canvas_height = height;
        header.gaussian_count = gaussian_count;

        Self {
            header,
            use_progressive: false,
        }
    }

    /// Create a lossless writer (f32, no quantization)
    pub fn new_lossless(width: u32, height: u32, gaussian_count: u32) -> Self {
        let mut header = LGIHeader::lossless_uncompressed();
        header.canvas_width = width;
        header.canvas_height = height;
        header.gaussian_count = gaussian_count;

        Self {
            header,
            use_progressive: false,
        }
    }

    /// Create a compressed writer (LGIQ-B + zstd + delta)
    pub fn new_compressed(width: u32, height: u32, gaussian_count: u32) -> Self {
        let mut header = LGIHeader::default();
        header.canvas_width = width;
        header.canvas_height = height;
        header.gaussian_count = gaussian_count;
        header.compression_flags = (CompressionMethod::Zstd as u16) | (CompressionMethod::Delta as u16);

        Self {
            header,
            use_progressive: false,
        }
    }

    /// Create a progressive writer (importance-ordered for streaming)
    pub fn new_progressive(width: u32, height: u32, gaussian_count: u32) -> Self {
        let mut header = LGIHeader::default();
        header.canvas_width = width;
        header.canvas_height = height;
        header.gaussian_count = gaussian_count;
        header.compression_flags = (CompressionMethod::Zstd as u16) | (CompressionMethod::Delta as u16);
        header.feature_flags |= FeatureFlags::HAS_PROGRESSIVE;

        Self {
            header,
            use_progressive: true,
        }
    }

    /// Write Gaussians to .lgi file
    pub fn write_file<P: AsRef<Path>>(
        &self,
        path: P,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
    ) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write magic number
        writer.write_all(LGI_MAGIC)?;

        // Write header
        self.write_header(&mut writer)?;

        // If progressive, reorder by importance and write PRGS chunk
        let (reordered_gaussians, importance_order): (Vec<Gaussian2D<f32, Euler<f32>>>, Option<Vec<usize>>) =
            if self.use_progressive {
                let order = progressive::order_by_importance(
                    gaussians,
                    self.header.canvas_width,
                    self.header.canvas_height
                );
                let reordered = progressive::reorder_gaussians(gaussians, &order);
                (reordered, Some(order))
            } else {
                (gaussians.to_vec(), None)
            };

        // Write PRGS chunk if progressive (importance order metadata)
        if let Some(order) = &importance_order {
            self.write_prgs_chunk(&mut writer, order)?;
        }

        // Write GAUS chunk (Gaussian data)
        if self.header.param_encoding == ParamEncoding::Raw as u16 {
            self.write_gaus_chunk_lossless(&mut writer, &reordered_gaussians)?;
        } else if self.header.compression_flags != 0 {
            self.write_gaus_chunk_compressed(&mut writer, &reordered_gaussians)?;
        } else {
            self.write_gaus_chunk(&mut writer, &reordered_gaussians)?;
        }

        writer.flush()?;
        Ok(())
    }

    fn write_header<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.header.version_major.to_le_bytes())?;
        writer.write_all(&self.header.version_minor.to_le_bytes())?;
        writer.write_all(&self.header.canvas_width.to_le_bytes())?;
        writer.write_all(&self.header.canvas_height.to_le_bytes())?;
        writer.write_all(&self.header.colorspace.to_le_bytes())?;
        writer.write_all(&[self.header.bitdepth])?;
        writer.write_all(&[self.header.alpha_mode])?;
        writer.write_all(&self.header.param_encoding.to_le_bytes())?;
        writer.write_all(&self.header.compression_flags.to_le_bytes())?;
        writer.write_all(&self.header.gaussian_count.to_le_bytes())?;
        writer.write_all(&self.header.feature_flags.to_le_bytes())?;
        writer.write_all(&self.header.background_color.to_le_bytes())?;
        writer.write_all(&self.header.index_offset.to_le_bytes())?;
        Ok(())
    }

    fn write_gaus_chunk<W: Write>(
        &self,
        writer: &mut W,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
    ) -> std::io::Result<()> {
        // Quantize and serialize Gaussian data
        let mut data = Vec::new();

        for gaussian in gaussians {
            let quantized = QuantizedGaussian::quantize_baseline(
                (gaussian.position.x, gaussian.position.y),
                (gaussian.shape.scale_x, gaussian.shape.scale_y),
                gaussian.shape.rotation,
                (gaussian.color.r, gaussian.color.g, gaussian.color.b),
                gaussian.opacity,
            );

            // Write quantized data (11 bytes per Gaussian for baseline profile)
            data.extend_from_slice(&quantized.position[0].to_le_bytes());
            data.extend_from_slice(&quantized.position[1].to_le_bytes());
            data.extend_from_slice(&quantized.scale[0].to_le_bytes());
            data.extend_from_slice(&quantized.scale[1].to_le_bytes());
            data.extend_from_slice(&quantized.rotation.to_le_bytes());
            data.push(quantized.color[0] as u8);  // 8-bit R (values are 0-255)
            data.push(quantized.color[1] as u8);  // 8-bit G
            data.push(quantized.color[2] as u8);  // 8-bit B
            data.push(quantized.opacity as u8);   // 8-bit α
        }

        let chunk = LGIChunk::new(b"GAUS", data);
        chunk.write_to(writer)?;

        Ok(())
    }

    /// Write compressed Gaussian data (quantized + zstd + delta)
    fn write_gaus_chunk_compressed<W: Write>(
        &self,
        writer: &mut W,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
    ) -> std::io::Result<()> {
        // First, quantize all Gaussians to get discrete values
        let mut quantized_positions: Vec<[u16; 2]> = Vec::with_capacity(gaussians.len());
        let mut other_params: Vec<Vec<u8>> = Vec::with_capacity(gaussians.len());

        for gaussian in gaussians.iter() {
            let quantized = QuantizedGaussian::quantize_baseline(
                (gaussian.position.x, gaussian.position.y),
                (gaussian.shape.scale_x, gaussian.shape.scale_y),
                gaussian.shape.rotation,
                (gaussian.color.r, gaussian.color.g, gaussian.color.b),
                gaussian.opacity,
            );

            quantized_positions.push(quantized.position);

            // Store other parameters (non-position)
            let mut params = Vec::new();
            params.extend_from_slice(&quantized.scale[0].to_le_bytes());
            params.extend_from_slice(&quantized.scale[1].to_le_bytes());
            params.extend_from_slice(&quantized.rotation.to_le_bytes());
            params.push(quantized.color[0] as u8);
            params.push(quantized.color[1] as u8);
            params.push(quantized.color[2] as u8);
            params.push(quantized.opacity as u8);
            other_params.push(params);
        }

        // Delta encode the quantized positions (no cumulative error)
        let mut delta_positions: Vec<[i32; 2]> = Vec::with_capacity(gaussians.len());
        if !quantized_positions.is_empty() {
            // First position is absolute
            delta_positions.push([quantized_positions[0][0] as i32, quantized_positions[0][1] as i32]);

            // Rest are deltas
            for i in 1..quantized_positions.len() {
                let dx = quantized_positions[i][0] as i32 - quantized_positions[i-1][0] as i32;
                let dy = quantized_positions[i][1] as i32 - quantized_positions[i-1][1] as i32;
                delta_positions.push([dx, dy]);
            }
        }

        // Serialize: delta positions + other params
        let mut data = Vec::new();
        for i in 0..gaussians.len() {
            // Write delta-encoded position (4 bytes each for i32)
            data.extend_from_slice(&(delta_positions[i][0] as i16).to_le_bytes());
            data.extend_from_slice(&(delta_positions[i][1] as i16).to_le_bytes());
            // Write other parameters (7 bytes)
            data.extend_from_slice(&other_params[i]);
        }

        // Compress with zstd
        let compressed_data = zstd::encode_all(&data[..], 3)?; // Level 3 for good balance

        // Write as GAUS chunk with compressed data
        let chunk = LGIChunk::new(b"GAUS", compressed_data);
        chunk.write_to(writer)?;

        Ok(())
    }

    /// Write lossless f32 Gaussian data (no quantization)
    fn write_gaus_chunk_lossless<W: Write>(
        &self,
        writer: &mut W,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
    ) -> std::io::Result<()> {
        // Direct f32 serialization (36 bytes per Gaussian)
        let mut data = Vec::new();

        for gaussian in gaussians {
            // Position (8 bytes)
            data.extend_from_slice(&gaussian.position.x.to_le_bytes());
            data.extend_from_slice(&gaussian.position.y.to_le_bytes());

            // Scale (8 bytes)
            data.extend_from_slice(&gaussian.shape.scale_x.to_le_bytes());
            data.extend_from_slice(&gaussian.shape.scale_y.to_le_bytes());

            // Rotation (4 bytes)
            data.extend_from_slice(&gaussian.shape.rotation.to_le_bytes());

            // Color (12 bytes)
            data.extend_from_slice(&gaussian.color.r.to_le_bytes());
            data.extend_from_slice(&gaussian.color.g.to_le_bytes());
            data.extend_from_slice(&gaussian.color.b.to_le_bytes());

            // Opacity (4 bytes)
            data.extend_from_slice(&gaussian.opacity.to_le_bytes());
        }

        let chunk = LGIChunk::new(b"GAUS", data);
        chunk.write_to(writer)?;

        Ok(())
    }

    /// Write PRGS chunk (progressive loading metadata)
    /// Stores the original indices of Gaussians after importance-based reordering
    fn write_prgs_chunk<W: Write>(
        &self,
        writer: &mut W,
        order: &[usize],
    ) -> std::io::Result<()> {
        // Serialize indices as u32 (4 bytes per index)
        let mut data = Vec::new();
        for &idx in order {
            data.extend_from_slice(&(idx as u32).to_le_bytes());
        }

        let chunk = LGIChunk::new(b"PRGS", data);
        chunk.write_to(writer)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::color::Color4;
    use lgi_math::vec::Vector2;

    #[test]
    fn test_write_simple_file() {
        let gaussians = vec![
            Gaussian2D::new(
                Vector2::new(0.5, 0.5),
                Euler::isotropic(0.1),
                Color4::new(1.0, 0.0, 0.0, 1.0),
                1.0,
            ),
        ];

        let writer = LGIWriter::new(256, 256, 1);
        let result = writer.write_file("/tmp/test.lgi", &gaussians);

        assert!(result.is_ok());
    }
}
