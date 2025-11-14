//! SLIC Superpixel Initialization (Strategy E)
//!
//! Loads Gaussian initialization from SLIC preprocessing
//!
//! Workflow:
//! 1. Run: python tools/slic_preprocess.py image.png 500
//! 2. Load: encoder.initialize_slic("slic_init.json")
//! 3. Optimize as normal

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Deserialize, Serialize)]
struct SLICGaussianInit {
    position: [f32; 2],
    scale: [f32; 2],
    rotation: f32,
    color: [f32; 3],

    #[serde(skip_serializing_if = "Option::is_none")]
    segment_id: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pixel_count: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SLICOutput {
    source_image: String,
    image_width: u32,
    image_height: u32,
    n_segments_requested: usize,
    n_gaussians: usize,
    gaussians: Vec<SLICGaussianInit>,
}

/// Load SLIC Gaussian initialization from JSON
///
/// # Arguments
/// * `json_path` - Path to slic_init.json (from Python preprocessing)
///
/// # Returns
/// Vector of Gaussians initialized from superpixel centroids
pub fn load_slic_init<P: AsRef<Path>>(json_path: P) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>, String> {
    let json_str = std::fs::read_to_string(json_path)
        .map_err(|e| format!("Failed to read SLIC JSON: {}", e))?;

    let slic_output: SLICOutput = serde_json::from_str(&json_str)
        .map_err(|e| format!("Failed to parse SLIC JSON: {}", e))?;

    log::info!("📦 Loading SLIC initialization:");
    log::info!("  Source: {}", slic_output.source_image);
    log::info!("  Image: {}×{}", slic_output.image_width, slic_output.image_height);
    log::info!("  Requested segments: {}", slic_output.n_segments_requested);
    log::info!("  Actual Gaussians: {}", slic_output.n_gaussians);

    let gaussians: Vec<_> = slic_output
        .gaussians
        .iter()
        .map(|init| {
            Gaussian2D::new(
                Vector2::new(init.position[0], init.position[1]),
                Euler::new(init.scale[0], init.scale[1], init.rotation),
                Color4::new(init.color[0], init.color[1], init.color[2], 1.0),
                1.0,  // opacity
            )
        })
        .collect();

    log::info!("✅ Loaded {} Gaussians from SLIC", gaussians.len());

    Ok(gaussians)
}

/// Run SLIC preprocessing and load result (subprocess)
///
/// # Arguments
/// * `image_path` - Path to input image
/// * `n_segments` - Target number of superpixels
///
/// # Returns
/// Gaussians initialized from SLIC superpixels
pub fn initialize_slic_subprocess(
    image_path: &str,
    n_segments: usize,
) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>, String> {
    use std::process::Command;

    log::info!("🐍 Running SLIC preprocessing (Python)...");
    log::info!("  Image: {}", image_path);
    log::info!("  Segments: {}", n_segments);

    let output = Command::new("python3")
        .arg("tools/slic_preprocess.py")
        .arg(image_path)
        .arg(n_segments.to_string())
        .arg("slic_init.json")
        .output()
        .map_err(|e| format!("Failed to run SLIC: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("SLIC preprocessing failed: {}", stderr));
    }

    log::info!("✅ SLIC preprocessing complete");

    // Load result
    load_slic_init("slic_init.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slic_json_parse() {
        let json = r#"{
            "source_image": "test.png",
            "image_width": 100,
            "image_height": 100,
            "n_segments_requested": 50,
            "n_gaussians": 48,
            "gaussians": [
                {
                    "position": [0.5, 0.5],
                    "scale": [0.05, 0.05],
                    "rotation": 0.0,
                    "color": [1.0, 0.0, 0.0]
                }
            ]
        }"#;

        let output: SLICOutput = serde_json::from_str(json).unwrap();
        assert_eq!(output.n_gaussians, 48);
        assert_eq!(output.gaussians.len(), 1);
    }
}
