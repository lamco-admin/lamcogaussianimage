//! Preprocessing Output Loader
//!
//! Loads analysis maps and metadata from Python preprocessing pipeline
//!
//! ## Workflow
//!
//! 1. **Python preprocessing** (once per image):
//! ```bash
//! source .venv/bin/activate
//! python tools/preprocess_image_v2.py kodim02.png
//! # Creates: kodim02.json + kodim02_*.npy
//! ```
//!
//! 2. **Rust encoding** (uses preprocessing):
//! ```rust
//! let preprocessing = PreprocessingData::load("kodim02.json")?;
//! let encoder = EncoderV2::new(image)?;
//!
//! // Sample Gaussian positions from placement map
//! let positions = preprocessing.sample_positions(500)?;
//! let gaussians = encoder.initialize_from_positions(&positions, &preprocessing);
//! ```
//!
//! ## File Format
//!
//! **kodim02.json**: Metadata + references to .npy files
//! **kodim02_placement.npy**: Combined probability map [HxW] f32
//! **kodim02_entropy.npy**: Local complexity [HxW] f32
//! **kodim02_gradient.npy**: Edge strength [HxW] f32
//! **kodim02_texture.npy**: Texture classification [HxW] f32
//! **kodim02_saliency.npy**: Visual importance [HxW] f32
//! **kodim02_distance.npy**: Distance to boundaries [HxW] f32
//! **kodim02_skeleton.npy**: Medial axis [HxW] f32
//! **kodim02_segments.npy**: SLIC labels [HxW] i32

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use ndarray::Array2;
use rand::Rng;

/// Source image metadata
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SourceInfo {
    pub filename: String,
    pub path: String,
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub file_size_bytes: u64,
    pub checksum_sha256: String,
}

/// Preprocessing run metadata
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PreprocessingInfo {
    pub version: String,
    pub timestamp: String,
    pub libraries: HashMap<String, String>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub gpu_used: bool,
}

/// Analysis map file references
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnalysisMaps {
    pub placement_map: String,
    pub entropy_map: String,
    pub gradient_map: String,
    pub texture_map: String,
    pub saliency_map: String,
    pub distance_map: String,
    pub skeleton: String,
    pub segments: String,
}

/// Analysis statistics
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Statistics {
    pub global_entropy: f32,
    pub mean_gradient: f32,
    pub texture_percentage: f32,
    pub actual_segments: usize,
}

/// Complete preprocessing metadata
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PreprocessingMetadata {
    pub source: SourceInfo,
    pub preprocessing: PreprocessingInfo,
    pub analysis_maps: AnalysisMaps,
    pub statistics: Statistics,
}

/// Loaded preprocessing data (metadata + maps)
pub struct PreprocessingData {
    /// Metadata from JSON
    pub metadata: PreprocessingMetadata,

    /// Base directory (where JSON and .npy files are)
    pub base_dir: PathBuf,

    /// Loaded maps (lazy-loaded on demand)
    placement_map: Option<Array2<f32>>,
    entropy_map: Option<Array2<f32>>,
    gradient_map: Option<Array2<f32>>,
    texture_map: Option<Array2<f32>>,
    saliency_map: Option<Array2<f32>>,
    distance_map: Option<Array2<f32>>,
    skeleton: Option<Array2<f32>>,
    segments: Option<Array2<i32>>,
}

impl PreprocessingData {
    /// Load preprocessing metadata from JSON
    ///
    /// # Arguments
    /// * `json_path` - Path to preprocessing JSON (e.g., "kodim02.json")
    ///
    /// # Example
    /// ```no_run
    /// let preprocessing = PreprocessingData::load("kodim02.json")?;
    /// println!("Image: {}×{}", preprocessing.metadata.source.width, preprocessing.metadata.source.height);
    /// ```
    pub fn load<P: AsRef<Path>>(json_path: P) -> Result<Self, String> {
        let json_path = json_path.as_ref();
        let base_dir = json_path.parent()
            .ok_or("Invalid JSON path")?
            .to_path_buf();

        // Load JSON metadata
        let json_str = std::fs::read_to_string(json_path)
            .map_err(|e| format!("Failed to read {}: {}", json_path.display(), e))?;

        let metadata: PreprocessingMetadata = serde_json::from_str(&json_str)
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;

        log::info!("📦 Loaded preprocessing metadata: {}", metadata.source.filename);
        log::info!("   Image: {}×{}, {} segments, {:.1}% texture",
            metadata.source.width,
            metadata.source.height,
            metadata.statistics.actual_segments,
            metadata.statistics.texture_percentage);

        Ok(Self {
            metadata,
            base_dir,
            placement_map: None,
            entropy_map: None,
            gradient_map: None,
            texture_map: None,
            saliency_map: None,
            distance_map: None,
            skeleton: None,
            segments: None,
        })
    }

    /// Get placement probability map (lazy load)
    ///
    /// Returns HxW array where each value is probability of placing Gaussian at that pixel
    /// Values sum to 1.0 (valid probability distribution)
    pub fn placement_map(&mut self) -> Result<&Array2<f32>, String> {
        if self.placement_map.is_none() {
            let path = self.base_dir.join(&self.metadata.analysis_maps.placement_map);
            self.placement_map = Some(self.load_npy_f32(&path)?);
            log::debug!("Loaded placement map: {:?}", self.placement_map.as_ref().unwrap().shape());
        }
        Ok(self.placement_map.as_ref().unwrap())
    }

    /// Sample N positions from placement probability map
    ///
    /// Uses importance sampling - positions sampled proportional to probability
    ///
    /// # Arguments
    /// * `n` - Number of positions to sample
    ///
    /// # Returns
    /// Vector of (x, y) pixel coordinates for Gaussian placement
    pub fn sample_positions(&mut self, n: usize) -> Result<Vec<(u32, u32)>, String> {
        let placement = self.placement_map()?;
        let (height, width) = placement.dim();

        // Build CDF for importance sampling
        let mut cdf = Vec::with_capacity(width * height);
        let mut cumsum = 0.0;

        for row in placement.rows() {
            for &prob in row {
                cumsum += prob as f64;
                cdf.push(cumsum);
            }
        }

        // Sample N positions
        let mut rng = rand::thread_rng();
        let mut positions = Vec::with_capacity(n);

        for _ in 0..n {
            let r: f64 = rng.gen();

            // Binary search in CDF
            let idx = match cdf.binary_search_by(|probe| {
                probe.partial_cmp(&r).unwrap()
            }) {
                Ok(i) => i,
                Err(i) => i.min(cdf.len() - 1),
            };

            let x = (idx % width) as u32;
            let y = (idx / width) as u32;

            positions.push((x, y));
        }

        log::info!("Sampled {} positions from placement map", n);

        Ok(positions)
    }

    /// Get entropy map (lazy load)
    pub fn entropy_map(&mut self) -> Result<&Array2<f32>, String> {
        if self.entropy_map.is_none() {
            let path = self.base_dir.join(&self.metadata.analysis_maps.entropy_map);
            self.entropy_map = Some(self.load_npy_f32(&path)?);
        }
        Ok(self.entropy_map.as_ref().unwrap())
    }

    /// Get gradient map (lazy load)
    pub fn gradient_map(&mut self) -> Result<&Array2<f32>, String> {
        if self.gradient_map.is_none() {
            let path = self.base_dir.join(&self.metadata.analysis_maps.gradient_map);
            self.gradient_map = Some(self.load_npy_f32(&path)?);
        }
        Ok(self.gradient_map.as_ref().unwrap())
    }

    /// Get SLIC segments (lazy load)
    pub fn segments(&mut self) -> Result<&Array2<i32>, String> {
        if self.segments.is_none() {
            let path = self.base_dir.join(&self.metadata.analysis_maps.segments);
            self.segments = Some(self.load_npy_i32(&path)?);
        }
        Ok(self.segments.as_ref().unwrap())
    }

    /// Helper: Load .npy file as f32 array
    fn load_npy_f32(&self, path: &Path) -> Result<Array2<f32>, String> {
        ndarray_npy::read_npy(path)
            .map_err(|e| format!("Failed to load {}: {}", path.display(), e))
    }

    /// Helper: Load .npy file as i32 array
    fn load_npy_i32(&self, path: &Path) -> Result<Array2<i32>, String> {
        ndarray_npy::read_npy(path)
            .map_err(|e| format!("Failed to load {}: {}", path.display(), e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_parse() {
        let json = r#"{
            "source": {
                "filename": "test.png",
                "path": "/path/to/test.png",
                "format": "PNG",
                "width": 768,
                "height": 512,
                "channels": 3,
                "file_size_bytes": 123456,
                "checksum_sha256": "abc123"
            },
            "preprocessing": {
                "version": "2.0.0",
                "timestamp": "2025-10-07T00:00:00Z",
                "libraries": {},
                "parameters": {},
                "gpu_used": false
            },
            "analysis_maps": {
                "placement_map": "test_placement.npy",
                "entropy_map": "test_entropy.npy",
                "gradient_map": "test_gradient.npy",
                "texture_map": "test_texture.npy",
                "saliency_map": "test_saliency.npy",
                "distance_map": "test_distance.npy",
                "skeleton": "test_skeleton.npy",
                "segments": "test_segments.npy"
            },
            "statistics": {
                "global_entropy": 0.5,
                "mean_gradient": 0.3,
                "texture_percentage": 25.0,
                "actual_segments": 437
            }
        }"#;

        let metadata: PreprocessingMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(metadata.source.width, 768);
        assert_eq!(metadata.statistics.actual_segments, 437);
    }
}
