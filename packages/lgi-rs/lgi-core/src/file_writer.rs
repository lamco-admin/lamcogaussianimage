//! LGI file writing

use crate::{ImageBuffer, container_format::{LGIHeader, LGIChunk, LGI_MAGIC}};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};
use std::io::Write;

pub struct LGIWriter;

impl LGIWriter {
    pub fn write_file(
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        width: u32,
        height: u32,
        path: &str,
    ) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        // Write magic number
        file.write_all(LGI_MAGIC)?;

        // Create header
        let header = LGIHeader {
            version_major: 1,
            version_minor: 0,
            canvas_width: width,
            canvas_height: height,
            colorspace: 0,  // sRGB
            bitdepth: 8,
            alpha_mode: 1,
            param_encoding: 0,  // EULER
            compression_flags: 0,
            gaussian_count: gaussians.len() as u32,
            feature_flags: 0,
            background_color: 0,
            index_offset: 0,
        };

        // Write HEAD chunk
        let header_data = unsafe {
            std::slice::from_raw_parts(
                &header as *const _ as *const u8,
                std::mem::size_of::<LGIHeader>(),
            )
        }.to_vec();

        let head_chunk = LGIChunk::new(b"HEAD", header_data);
        head_chunk.write_to(&mut file)?;

        // Write GAUS chunk (uncompressed for now)
        let mut gaus_data = Vec::new();
        for g in gaussians {
            gaus_data.extend_from_slice(&g.position.x.to_le_bytes());
            gaus_data.extend_from_slice(&g.position.y.to_le_bytes());
            gaus_data.extend_from_slice(&g.shape.scale_x.to_le_bytes());
            gaus_data.extend_from_slice(&g.shape.scale_y.to_le_bytes());
            gaus_data.extend_from_slice(&g.shape.rotation.to_le_bytes());
            gaus_data.extend_from_slice(&g.color.r.to_le_bytes());
            gaus_data.extend_from_slice(&g.color.g.to_le_bytes());
            gaus_data.extend_from_slice(&g.color.b.to_le_bytes());
            gaus_data.extend_from_slice(&g.opacity.to_le_bytes());
        }

        let gaus_chunk = LGIChunk::new(b"GAUS", gaus_data);
        gaus_chunk.write_to(&mut file)?;

        Ok(())
    }
}
