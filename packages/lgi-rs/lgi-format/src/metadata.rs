//! LGI metadata (meta chunk)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LGI file metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LgiMetadata {
    /// Encoding parameters used
    pub encoding: Option<EncodingMetadata>,

    /// Quality metrics
    pub quality: Option<QualityMetrics>,

    /// Custom fields
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Encoding parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingMetadata {
    /// Encoder version
    pub encoder_version: String,

    /// Number of optimization iterations
    pub iterations: u32,

    /// Initialization strategy
    pub init_strategy: String,

    /// Whether QA training was used
    pub qa_training: bool,

    /// Encoding time in seconds
    pub encoding_time_secs: f32,
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// PSNR in dB
    pub psnr_db: f32,

    /// SSIM (0-1)
    pub ssim: f32,

    /// Final loss value
    pub final_loss: f32,
}

impl LgiMetadata {
    /// Create new empty metadata
    pub fn new() -> Self {
        Self::default()
    }

    /// Add encoding metadata
    pub fn with_encoding(mut self, encoding: EncodingMetadata) -> Self {
        self.encoding = Some(encoding);
        self
    }

    /// Add quality metrics
    pub fn with_quality(mut self, quality: QualityMetrics) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Add custom field
    pub fn with_custom(mut self, key: String, value: serde_json::Value) -> Self {
        self.custom.insert(key, value);
        self
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_json().expect("Failed to serialize metadata").into_bytes()
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        let json = std::str::from_utf8(bytes)
            .map_err(|e| serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        Self::from_json(json)
    }
}
