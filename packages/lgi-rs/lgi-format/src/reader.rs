//! LGI file reader

use crate::{
    chunk::{Chunk, ChunkType},
    error::{LgiFormatError, Result},
    validation::{validate_file, validate_magic_number},
    GaussianData, LgiFile, LgiHeader, LgiMetadata, MAGIC_NUMBER,
};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// LGI file reader
pub struct LgiReader;

impl LgiReader {
    /// Read LGI file from path
    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<LgiFile> {
        let f = File::open(path)?;
        let mut reader = BufReader::new(f);

        Self::read(&mut reader)
    }

    /// Read LGI file from reader
    pub fn read<R: Read>(reader: &mut R) -> Result<LgiFile> {
        // Read and validate magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        validate_magic_number(magic)?;

        let mut header: Option<LgiHeader> = None;
        let mut gaussian_data: Option<GaussianData> = None;
        let mut metadata: Option<LgiMetadata> = None;

        // Read chunks until IEND
        loop {
            let chunk = Chunk::read(reader)?;

            match chunk.chunk_type {
                ChunkType::Head => {
                    header = Some(LgiHeader::from_bytes(&chunk.data)?);
                }
                ChunkType::Gaus => {
                    gaussian_data = Some(GaussianData::from_bytes(&chunk.data)?);
                }
                ChunkType::Meta => {
                    metadata = Some(LgiMetadata::from_bytes(&chunk.data)?);
                }
                ChunkType::Inde => {
                    // Index chunk - skip for now (future: random access)
                }
                ChunkType::Iend => {
                    // End marker
                    break;
                }
            }
        }

        // Ensure required chunks are present
        let header = header.ok_or_else(|| LgiFormatError::MissingChunk("HEAD".to_string()))?;
        let gaussian_data =
            gaussian_data.ok_or_else(|| LgiFormatError::MissingChunk("GAUS".to_string()))?;

        let file = LgiFile {
            header,
            gaussian_data,
            metadata,
        };

        // Validate file consistency
        validate_file(&file)?;

        Ok(file)
    }

    /// Read only header (for quick inspection)
    pub fn read_header<R: Read>(reader: &mut R) -> Result<LgiHeader> {
        // Read magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        validate_magic_number(magic)?;

        // Read first chunk (should be HEAD)
        let chunk = Chunk::read(reader)?;

        if chunk.chunk_type != ChunkType::Head {
            return Err(LgiFormatError::InvalidChunkData(
                "First chunk must be HEAD".to_string(),
            ));
        }

        LgiHeader::from_bytes(&chunk.data).map_err(|e| e.into())
    }

    /// Read header from file
    pub fn read_header_file<P: AsRef<Path>>(path: P) -> Result<LgiHeader> {
        let f = File::open(path)?;
        let mut reader = BufReader::new(f);
        Self::read_header(&mut reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::LgiWriter;
    use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
    use std::io::Cursor;

    fn create_test_gaussians(count: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        (0..count)
            .map(|i| {
                Gaussian2D::new(
                    Vector2::new((i as f32) / (count as f32), 0.5),
                    Euler::isotropic(0.01),
                    Color4::rgb(0.5, 0.5, 0.5),
                    0.8,
                )
            })
            .collect()
    }

    #[test]
    fn test_write_read_roundtrip_uncompressed() {
        let gaussians = create_test_gaussians(100);
        let file = LgiFile::new(gaussians, 256, 256);

        // Write to buffer
        let mut buffer = Vec::new();
        LgiWriter::write(&mut buffer, &file).unwrap();

        println!("Uncompressed file size: {} bytes", buffer.len());

        // Read from buffer
        let mut cursor = Cursor::new(buffer);
        let loaded_file = LgiReader::read(&mut cursor).unwrap();

        // Verify
        assert_eq!(loaded_file.header.width, 256);
        assert_eq!(loaded_file.header.height, 256);
        assert_eq!(loaded_file.gaussian_count(), 100);
        assert!(!loaded_file.is_compressed());

        let reconstructed = loaded_file.gaussians();
        assert_eq!(reconstructed.len(), 100);
    }

    #[test]
    fn test_write_read_roundtrip_vq() {
        let gaussians = create_test_gaussians(500);
        let file = LgiFile::with_vq(gaussians, 256, 256, 256);

        // Write to buffer
        let mut buffer = Vec::new();
        LgiWriter::write(&mut buffer, &file).unwrap();

        println!("VQ compressed file size: {} bytes", buffer.len());
        println!("Compression ratio: {:.2}×", file.compression_ratio());

        // Read from buffer
        let mut cursor = Cursor::new(buffer);
        let loaded_file = LgiReader::read(&mut cursor).unwrap();

        // Verify
        assert_eq!(loaded_file.header.width, 256);
        assert_eq!(loaded_file.gaussian_count(), 500);
        assert!(loaded_file.is_compressed());

        let reconstructed = loaded_file.gaussians();
        assert_eq!(reconstructed.len(), 500);
    }

    #[test]
    fn test_read_header_only() {
        let gaussians = create_test_gaussians(100);
        let file = LgiFile::new(gaussians, 1920, 1080);

        let mut buffer = Vec::new();
        LgiWriter::write(&mut buffer, &file).unwrap();

        let mut cursor = Cursor::new(buffer);
        let header = LgiReader::read_header(&mut cursor).unwrap();

        assert_eq!(header.width, 1920);
        assert_eq!(header.height, 1080);
        assert_eq!(header.gaussian_count, 100);
    }

    #[test]
    fn test_corrupted_magic_number() {
        let mut buffer = vec![b'X', b'X', b'X', b'X']; // Invalid magic

        let mut cursor = Cursor::new(buffer);
        let result = LgiReader::read(&mut cursor);

        assert!(matches!(result, Err(LgiFormatError::InvalidMagicNumber(_))));
    }
}
