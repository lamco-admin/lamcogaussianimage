//! Chunk-based I/O infrastructure
//!
//! All LGI data is stored in chunks (similar to PNG format):
//! - Length (4 bytes): u32 big-endian
//! - Type (4 bytes): ASCII fourCC code
//! - Data (length bytes): chunk payload
//! - CRC32 (4 bytes): u32 big-endian (of type + data)

use crate::error::{LgiFormatError, Result};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;
use std::io::{Read, Write};

/// Chunk type identifier (4-byte ASCII)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkType {
    /// HEAD - File header
    Head,
    /// GAUS - Gaussian data
    Gaus,
    /// meta - JSON metadata
    Meta,
    /// INDE - Index for random access
    Inde,
    /// IEND - End marker
    Iend,
}

impl ChunkType {
    /// Convert to 4-byte array
    pub fn as_bytes(&self) -> [u8; 4] {
        match self {
            ChunkType::Head => *b"HEAD",
            ChunkType::Gaus => *b"GAUS",
            ChunkType::Meta => *b"meta",
            ChunkType::Inde => *b"INDE",
            ChunkType::Iend => *b"IEND",
        }
    }

    /// Parse from 4-byte array
    pub fn from_bytes(bytes: [u8; 4]) -> Result<Self> {
        match &bytes {
            b"HEAD" => Ok(ChunkType::Head),
            b"GAUS" => Ok(ChunkType::Gaus),
            b"meta" => Ok(ChunkType::Meta),
            b"INDE" => Ok(ChunkType::Inde),
            b"IEND" => Ok(ChunkType::Iend),
            _ => Err(LgiFormatError::InvalidChunkType(bytes)),
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            ChunkType::Head => "HEAD",
            ChunkType::Gaus => "GAUS",
            ChunkType::Meta => "meta",
            ChunkType::Inde => "INDE",
            ChunkType::Iend => "IEND",
        }
    }
}

/// A chunk of data
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Chunk type
    pub chunk_type: ChunkType,

    /// Chunk data
    pub data: Vec<u8>,
}

impl Chunk {
    /// Create new chunk
    pub fn new(chunk_type: ChunkType, data: Vec<u8>) -> Self {
        Self { chunk_type, data }
    }

    /// Write chunk to writer
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Length (4 bytes, big-endian)
        writer.write_u32::<BigEndian>(self.data.len() as u32)?;

        // Type (4 bytes)
        let type_bytes = self.chunk_type.as_bytes();
        writer.write_all(&type_bytes)?;

        // Data
        writer.write_all(&self.data)?;

        // CRC32 (type + data)
        let crc = self.compute_crc();
        writer.write_u32::<BigEndian>(crc)?;

        Ok(())
    }

    /// Read chunk from reader
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        // Length
        let length = reader.read_u32::<BigEndian>()?;

        // Type
        let mut type_bytes = [0u8; 4];
        reader.read_exact(&mut type_bytes)?;
        let chunk_type = ChunkType::from_bytes(type_bytes)?;

        // Data
        let mut data = vec![0u8; length as usize];
        reader.read_exact(&mut data)?;

        // CRC32
        let expected_crc = reader.read_u32::<BigEndian>()?;

        // Verify CRC
        let chunk = Self { chunk_type, data };
        let actual_crc = chunk.compute_crc();

        if expected_crc != actual_crc {
            return Err(LgiFormatError::CrcMismatch {
                expected: expected_crc,
                actual: actual_crc,
            });
        }

        Ok(chunk)
    }

    /// Compute CRC32 of type + data
    fn compute_crc(&self) -> u32 {
        let mut hasher = Hasher::new();
        hasher.update(&self.chunk_type.as_bytes());
        hasher.update(&self.data);
        hasher.finalize()
    }

    /// Get chunk size in bytes (including header and CRC)
    pub fn size(&self) -> usize {
        4 + // length
        4 + // type
        self.data.len() + // data
        4 // crc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_chunk_roundtrip() {
        let data = vec![1, 2, 3, 4, 5];
        let chunk = Chunk::new(ChunkType::Head, data.clone());

        let mut buffer = Vec::new();
        chunk.write(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_chunk = Chunk::read(&mut cursor).unwrap();

        assert_eq!(read_chunk.chunk_type, ChunkType::Head);
        assert_eq!(read_chunk.data, data);
    }

    #[test]
    fn test_crc_validation() {
        let chunk = Chunk::new(ChunkType::Gaus, vec![1, 2, 3]);

        let mut buffer = Vec::new();
        chunk.write(&mut buffer).unwrap();

        // Corrupt data
        buffer[12] ^= 0xFF;

        let mut cursor = Cursor::new(buffer);
        let result = Chunk::read(&mut cursor);

        assert!(matches!(result, Err(LgiFormatError::CrcMismatch { .. })));
    }

    #[test]
    fn test_chunk_types() {
        assert_eq!(ChunkType::Head.as_bytes(), *b"HEAD");
        assert_eq!(ChunkType::Gaus.as_bytes(), *b"GAUS");
        assert_eq!(ChunkType::Meta.as_bytes(), *b"meta");
        assert_eq!(ChunkType::Inde.as_bytes(), *b"INDE");
        assert_eq!(ChunkType::Iend.as_bytes(), *b"IEND");

        assert_eq!(ChunkType::from_bytes(*b"HEAD").unwrap(), ChunkType::Head);
        assert!(ChunkType::from_bytes(*b"XXXX").is_err());
    }
}
