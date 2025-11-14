//! Format validation utilities

use crate::{LgiFile, LgiHeader, FORMAT_VERSION, MAGIC_NUMBER};
use crate::error::{LgiFormatError, Result};

/// Validate LGI file structure
pub fn validate_file(file: &LgiFile) -> Result<()> {
    validate_header(&file.header)?;
    validate_gaussian_data_consistency(file)?;
    Ok(())
}

/// Validate header
pub fn validate_header(header: &LgiHeader) -> Result<()> {
    // Check version
    if header.version != FORMAT_VERSION {
        return Err(LgiFormatError::UnsupportedVersion(
            header.version,
            FORMAT_VERSION,
        ));
    }

    // Check dimensions
    if header.width == 0 || header.height == 0 {
        return Err(LgiFormatError::InvalidChunkData(
            "Invalid dimensions: width and height must be > 0".to_string(),
        ));
    }

    // Check Gaussian count
    if header.gaussian_count == 0 {
        return Err(LgiFormatError::InvalidChunkData(
            "Invalid Gaussian count: must be > 0".to_string(),
        ));
    }

    // Check VQ codebook size if VQ compressed
    if header.compression_flags.vq_compressed {
        if header.compression_flags.vq_codebook_size == 0 {
            return Err(LgiFormatError::InvalidChunkData(
                "VQ compressed but codebook size is 0".to_string(),
            ));
        }
    }

    Ok(())
}

/// Validate Gaussian data matches header
fn validate_gaussian_data_consistency(file: &LgiFile) -> Result<()> {
    let gaussian_count = file.gaussian_data.gaussian_count();

    if gaussian_count != file.header.gaussian_count as usize {
        return Err(LgiFormatError::InvalidChunkData(format!(
            "Gaussian count mismatch: header says {}, data has {}",
            file.header.gaussian_count, gaussian_count
        )));
    }

    // Check VQ compression consistency
    let is_vq = file.gaussian_data.is_vq_compressed();
    if is_vq != file.header.compression_flags.vq_compressed {
        return Err(LgiFormatError::InvalidChunkData(
            "VQ compression flag mismatch between header and data".to_string(),
        ));
    }

    Ok(())
}

/// Validate magic number
pub fn validate_magic_number(magic: [u8; 4]) -> Result<()> {
    if magic != MAGIC_NUMBER {
        return Err(LgiFormatError::InvalidMagicNumber(magic));
    }
    Ok(())
}
