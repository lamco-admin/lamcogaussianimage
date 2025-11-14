//! LGI file reading

use crate::{container_format::{LGIHeader, LGI_MAGIC}, Result, LgiError};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::io::Read;

pub struct LGIReader;

impl LGIReader {
    pub fn read_file(path: &str) -> Result<(Vec<Gaussian2D<f32, Euler<f32>>>, u32, u32)> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| LgiError::Io(format!("Failed to open: {}", e)))?;

        // Read magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e| LgiError::Io(format!("Read magic failed: {}", e)))?;

        if &magic != LGI_MAGIC {
            return Err(LgiError::InvalidFormat("Bad magic number".to_string()));
        }

        // Read chunks
        let mut gaussians = Vec::new();
        let mut width = 0u32;
        let mut height = 0u32;

        loop {
            let mut len_bytes = [0u8; 4];
            if file.read_exact(&mut len_bytes).is_err() {
                break;  // EOF
            }

            let chunk_len = u32::from_le_bytes(len_bytes);

            let mut chunk_type = [0u8; 4];
            file.read_exact(&mut chunk_type)
                .map_err(|e| LgiError::Io(format!("Read chunk type failed: {}", e)))?;

            let mut chunk_data = vec![0u8; chunk_len as usize];
            file.read_exact(&mut chunk_data)
                .map_err(|e| LgiError::Io(format!("Read chunk data failed: {}", e)))?;

            let mut crc_bytes = [0u8; 4];
            file.read_exact(&mut crc_bytes)
                .map_err(|e| LgiError::Io(format!("Read CRC failed: {}", e)))?;

            // Parse chunks
            match &chunk_type {
                b"HEAD" => {
                    if chunk_data.len() >= std::mem::size_of::<LGIHeader>() {
                        let header: LGIHeader = unsafe {
                            std::ptr::read(chunk_data.as_ptr() as *const LGIHeader)
                        };
                        width = header.canvas_width;
                        height = header.canvas_height;
                    }
                }
                b"GAUS" => {
                    let floats_per_gaussian = 9;
                    let bytes_per_gaussian = floats_per_gaussian * 4;
                    let num_gaussians = chunk_data.len() / bytes_per_gaussian;

                    for i in 0..num_gaussians {
                        let offset = i * bytes_per_gaussian;
                        let mut read_f32 = |idx: usize| -> f32 {
                            let bytes = &chunk_data[offset + idx * 4..offset + (idx + 1) * 4];
                            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                        };

                        let gaussian = Gaussian2D::new(
                            Vector2::new(read_f32(0), read_f32(1)),
                            Euler::new(read_f32(2), read_f32(3), read_f32(4)),
                            Color4::new(read_f32(5), read_f32(6), read_f32(7), 1.0),
                            read_f32(8),
                        );
                        gaussians.push(gaussian);
                    }
                }
                _ => {}  // Skip unknown chunks
            }
        }

        Ok((gaussians, width, height))
    }
}
