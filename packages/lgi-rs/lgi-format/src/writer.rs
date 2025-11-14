//! LGI file writer

use crate::{
    chunk::{Chunk, ChunkType},
    error::Result,
    validation::validate_file,
    LgiFile, MAGIC_NUMBER,
};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// LGI file writer
pub struct LgiWriter;

impl LgiWriter {
    /// Write LGI file to path
    pub fn write_file<P: AsRef<Path>>(file: &LgiFile, path: P) -> Result<()> {
        // Validate before writing
        validate_file(file)?;

        let f = File::create(path)?;
        let mut writer = BufWriter::new(f);

        Self::write(&mut writer, file)?;

        Ok(())
    }

    /// Write LGI file to writer
    pub fn write<W: Write>(writer: &mut W, file: &LgiFile) -> Result<()> {
        // Magic number
        writer.write_all(&MAGIC_NUMBER)?;

        // HEAD chunk
        let head_data = file.header.to_bytes();
        let head_chunk = Chunk::new(ChunkType::Head, head_data);
        head_chunk.write(writer)?;

        // GAUS chunk
        let gaus_data = file.gaussian_data.to_bytes();
        let gaus_chunk = Chunk::new(ChunkType::Gaus, gaus_data);
        gaus_chunk.write(writer)?;

        // meta chunk (optional)
        if let Some(ref metadata) = file.metadata {
            let meta_data = metadata.to_bytes();
            let meta_chunk = Chunk::new(ChunkType::Meta, meta_data);
            meta_chunk.write(writer)?;
        }

        // IEND chunk (end marker)
        let iend_chunk = Chunk::new(ChunkType::Iend, vec![]);
        iend_chunk.write(writer)?;

        writer.flush()?;

        Ok(())
    }

    /// Get estimated file size
    pub fn estimated_size(file: &LgiFile) -> usize {
        let mut size = 4; // Magic number

        // HEAD chunk
        let head_data = file.header.to_bytes();
        size += 4 + 4 + head_data.len() + 4; // length + type + data + crc

        // GAUS chunk
        size += 4 + 4 + file.gaussian_data.compressed_size() + 4;

        // meta chunk
        if let Some(ref metadata) = file.metadata {
            let meta_data = metadata.to_bytes();
            size += 4 + 4 + meta_data.len() + 4;
        }

        // IEND chunk
        size += 4 + 4 + 0 + 4;

        size
    }
}
