//! Test Results Logging Utility
//!
//! Saves test results to persistent storage for building a corpus of experiments.
//!
//! Structure:
//! ```text
//! test-results/
//!   YYYY-MM-DD/
//!     experiment_name_HH-MM-SS.json      # Metrics and parameters
//!     experiment_name_HH-MM-SS.png       # Rendered result (same timestamp)
//!   rendered-images/
//!     experiment_name_*.png              # Legacy location (deprecated)
//! ```
//!
//! Usage:
//! ```ignore
//! let mut result = TestResult::new("my_experiment", image_path);
//! result.set_param("lr_position", 0.0002);
//! result.set_param("lr_color", 0.02);
//! result.set_param("threshold", 0.1);
//! // ... run optimization ...
//! result.final_psnr = psnr;
//! result.save_with_image(&rendered_image, "lgi-encoder-v2/test-results")?;
//! ```

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use lgi_core::ImageBuffer;

/// Optimizer parameters - stores learning rates, thresholds, etc.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct OptimizerParams {
    /// Learning rate for position parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_position: Option<f32>,
    /// Learning rate for color parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_color: Option<f32>,
    /// Learning rate for scale parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_scale: Option<f32>,
    /// Learning rate for opacity parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_opacity: Option<f32>,
    /// Learning rate for rotation parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_rotation: Option<f32>,
    /// Adam beta1 parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beta1: Option<f32>,
    /// Adam beta2 parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beta2: Option<f32>,
    /// Densification gradient threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub densify_grad_threshold: Option<f32>,
    /// Pruning opacity threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prune_opacity_threshold: Option<f32>,
    /// Densification interval (iterations between events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub densify_interval: Option<usize>,
    /// Starting Gaussian count (for growing approaches)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_gaussians: Option<usize>,
    /// Target Gaussian count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_gaussians: Option<usize>,
    /// Additional custom parameters
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

impl OptimizerParams {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a custom parameter by name
    pub fn set<T: Serialize>(&mut self, key: &str, value: T) {
        if let Ok(v) = serde_json::to_value(value) {
            self.custom.insert(key.to_string(), v);
        }
    }
}

/// Code version information for reproducibility
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct CodeVersion {
    /// Git commit hash (short)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_hash: Option<String>,
    /// Whether there are uncommitted changes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dirty: Option<bool>,
    /// Current branch name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
}

impl CodeVersion {
    /// Capture current git state
    pub fn capture() -> Self {
        let git_hash = std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string());

        let dirty = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .output()
            .ok()
            .map(|o| !o.stdout.is_empty());

        let branch = std::process::Command::new("git")
            .args(["branch", "--show-current"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string());

        Self { git_hash, dirty, branch }
    }
}

/// Test run status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TestStatus {
    /// Test completed successfully
    Completed,
    /// Test was aborted manually
    Aborted,
    /// Test failed due to error
    Failed,
    /// Test is still running
    Running,
}

impl Default for TestStatus {
    fn default() -> Self {
        TestStatus::Running
    }
}

/// Test result metadata
#[derive(Serialize, Deserialize, Debug)]
pub struct TestResult {
    pub experiment_name: String,
    pub timestamp: String,
    /// Test completion status
    #[serde(default)]
    pub status: TestStatus,
    pub image_path: String,
    pub image_dimensions: (u32, u32),
    pub n_gaussians: usize,
    pub iterations: usize,
    pub optimizer: String,
    pub initial_psnr: f32,
    pub final_psnr: f32,
    pub improvement_db: f32,
    pub final_loss: f32,
    pub elapsed_seconds: f32,
    pub notes: String,
    /// Optimizer parameters used for this experiment
    #[serde(default)]
    pub params: OptimizerParams,
    /// Git version info for reproducibility
    #[serde(default)]
    pub code_version: CodeVersion,
    /// Path to the rendered output image (if saved)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered_image_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_metrics: Option<serde_json::Value>,
}

impl TestResult {
    pub fn new(experiment_name: &str, image_path: &str) -> Self {
        let now = chrono_lite_timestamp();
        Self {
            experiment_name: experiment_name.to_string(),
            timestamp: now,
            status: TestStatus::Running,
            image_path: image_path.to_string(),
            image_dimensions: (0, 0),
            n_gaussians: 0,
            iterations: 0,
            optimizer: String::new(),
            initial_psnr: 0.0,
            final_psnr: 0.0,
            improvement_db: 0.0,
            final_loss: 0.0,
            elapsed_seconds: 0.0,
            notes: String::new(),
            params: OptimizerParams::default(),
            code_version: CodeVersion::capture(),
            rendered_image_path: None,
            extra_metrics: None,
        }
    }

    /// Mark test as completed successfully
    pub fn complete(&mut self) {
        self.status = TestStatus::Completed;
    }

    /// Mark test as aborted with reason
    pub fn abort(&mut self, reason: &str, iteration: usize, current_psnr: f32) {
        self.status = TestStatus::Aborted;
        self.notes = format!("ABORTED at iter {}: {} | Previous notes: {}",
            iteration, reason, self.notes);
        self.iterations = iteration;
        self.final_psnr = current_psnr;
        self.improvement_db = current_psnr - self.initial_psnr;
    }

    /// Mark test as failed with error
    pub fn fail(&mut self, error: &str) {
        self.status = TestStatus::Failed;
        self.notes = format!("FAILED: {} | Previous notes: {}", error, self.notes);
    }

    /// Save aborted/failed test results (includes partial data)
    pub fn save_partial(&self, base_dir: &str) -> std::io::Result<String> {
        let date = &self.timestamp[..10];
        let dir = format!("{}/{}", base_dir, date);
        fs::create_dir_all(&dir)?;

        // Use status-based suffix for partial results
        let status_suffix = match self.status {
            TestStatus::Aborted => "aborted",
            TestStatus::Failed => "failed",
            _ => "partial",
        };

        let filename = format!("{}/{}_{}.json",
            dir,
            self.base_filename(),
            status_suffix
        );

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let mut file = File::create(&filename)?;
        file.write_all(json.as_bytes())?;

        Ok(filename)
    }

    /// Set a parameter value (convenience method)
    pub fn set_param<T: Serialize>(&mut self, key: &str, value: T) {
        // Try to set known fields first
        match key {
            "lr_position" => if let Ok(v) = serde_json::from_value(serde_json::to_value(&value).unwrap()) { self.params.lr_position = Some(v); return; }
            "lr_color" => if let Ok(v) = serde_json::from_value(serde_json::to_value(&value).unwrap()) { self.params.lr_color = Some(v); return; }
            "lr_scale" => if let Ok(v) = serde_json::from_value(serde_json::to_value(&value).unwrap()) { self.params.lr_scale = Some(v); return; }
            "lr_opacity" => if let Ok(v) = serde_json::from_value(serde_json::to_value(&value).unwrap()) { self.params.lr_opacity = Some(v); return; }
            "lr_rotation" => if let Ok(v) = serde_json::from_value(serde_json::to_value(&value).unwrap()) { self.params.lr_rotation = Some(v); return; }
            "beta1" => if let Ok(v) = serde_json::from_value(serde_json::to_value(&value).unwrap()) { self.params.beta1 = Some(v); return; }
            "beta2" => if let Ok(v) = serde_json::from_value(serde_json::to_value(&value).unwrap()) { self.params.beta2 = Some(v); return; }
            "densify_grad_threshold" => if let Ok(v) = serde_json::from_value(serde_json::to_value(&value).unwrap()) { self.params.densify_grad_threshold = Some(v); return; }
            "prune_opacity_threshold" => if let Ok(v) = serde_json::from_value(serde_json::to_value(&value).unwrap()) { self.params.prune_opacity_threshold = Some(v); return; }
            "densify_interval" => if let Ok(v) = serde_json::from_value(serde_json::to_value(&value).unwrap()) { self.params.densify_interval = Some(v); return; }
            "start_gaussians" => if let Ok(v) = serde_json::from_value(serde_json::to_value(&value).unwrap()) { self.params.start_gaussians = Some(v); return; }
            "target_gaussians" => if let Ok(v) = serde_json::from_value(serde_json::to_value(&value).unwrap()) { self.params.target_gaussians = Some(v); return; }
            _ => {}
        }
        // Fall back to custom params
        self.params.set(key, value);
    }

    /// Generate the base filename (without extension) for this result
    fn base_filename(&self) -> String {
        format!("{}_{}",
            self.experiment_name.replace(" ", "_").replace("/", "-"),
            &self.timestamp[11..19].replace(":", "-") // HH-MM-SS
        )
    }

    /// Save result to JSON file (without image)
    pub fn save(&self, base_dir: &str) -> std::io::Result<String> {
        let date = &self.timestamp[..10]; // YYYY-MM-DD
        let dir = format!("{}/{}", base_dir, date);
        fs::create_dir_all(&dir)?;

        let filename = format!("{}/{}.json", dir, self.base_filename());

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let mut file = File::create(&filename)?;
        file.write_all(json.as_bytes())?;

        Ok(filename)
    }

    /// Save result with rendered image - both saved to same directory with matching timestamps
    pub fn save_with_image(&mut self, rendered: &ImageBuffer<f32>, base_dir: &str) -> std::io::Result<(String, String)> {
        let date = &self.timestamp[..10]; // YYYY-MM-DD
        let dir = format!("{}/{}", base_dir, date);
        fs::create_dir_all(&dir)?;

        let base_name = self.base_filename();

        // Save rendered image
        let image_filename = format!("{}/{}.png", dir, base_name);
        save_image_to_path(rendered, &image_filename)?;
        self.rendered_image_path = Some(image_filename.clone());

        // Save JSON (now includes rendered_image_path)
        let json_filename = format!("{}/{}.json", dir, base_name);
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut file = File::create(&json_filename)?;
        file.write_all(json.as_bytes())?;

        Ok((json_filename, image_filename))
    }
}

/// Save image to a specific path
fn save_image_to_path(image: &ImageBuffer<f32>, path: &str) -> std::io::Result<()> {
    let width = image.width as u32;
    let height = image.height as u32;
    let mut rgb_data: Vec<u8> = Vec::with_capacity((width * height * 3) as usize);

    for pixel in &image.data {
        rgb_data.push((pixel.r.clamp(0.0, 1.0) * 255.0) as u8);
        rgb_data.push((pixel.g.clamp(0.0, 1.0) * 255.0) as u8);
        rgb_data.push((pixel.b.clamp(0.0, 1.0) * 255.0) as u8);
    }

    image::save_buffer(path, &rgb_data, width, height, image::ColorType::Rgb8)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

/// Save rendered image for visual comparison
pub fn save_rendered_image(
    image: &ImageBuffer<f32>,
    base_dir: &str,
    experiment_name: &str,
    suffix: &str,
) -> std::io::Result<String> {
    let dir = format!("{}/rendered-images", base_dir);
    fs::create_dir_all(&dir)?;

    let timestamp = chrono_lite_timestamp();
    let filename = format!("{}/{}_{}_{}_{}.png",
        dir,
        experiment_name.replace(" ", "_").replace("/", "-"),
        suffix,
        &timestamp[..10],
        &timestamp[11..19].replace(":", "-")
    );

    // Convert f32 image to u8 for saving
    let width = image.width as u32;
    let height = image.height as u32;
    let mut rgb_data: Vec<u8> = Vec::with_capacity((width * height * 3) as usize);

    for pixel in &image.data {
        rgb_data.push((pixel.r.clamp(0.0, 1.0) * 255.0) as u8);
        rgb_data.push((pixel.g.clamp(0.0, 1.0) * 255.0) as u8);
        rgb_data.push((pixel.b.clamp(0.0, 1.0) * 255.0) as u8);
    }

    // Use image crate to save
    image::save_buffer(&filename, &rgb_data, width, height, image::ColorType::Rgb8)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    Ok(filename)
}

/// Simple timestamp without chrono dependency
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = duration.as_secs();

    // Convert to date/time (simplified, assumes UTC)
    let days = secs / 86400;
    let remaining = secs % 86400;
    let hours = remaining / 3600;
    let minutes = (remaining % 3600) / 60;
    let seconds = remaining % 60;

    // Simplified date calculation (good enough for our purposes)
    // This is approximate but works for 2020-2030
    let years_since_1970 = days / 365;
    let year = 1970 + years_since_1970;
    let day_of_year = days % 365;

    // Very rough month/day (ignoring leap years for simplicity)
    let month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1;
    let mut day = day_of_year;
    for &days_in_month in &month_days {
        if day < days_in_month as u64 {
            break;
        }
        day -= days_in_month as u64;
        month += 1;
    }

    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        year, month, day + 1, hours, minutes, seconds)
}

/// Load all test results from a date
pub fn load_results(base_dir: &str, date: &str) -> Vec<TestResult> {
    let dir = format!("{}/{}", base_dir, date);
    let mut results = Vec::new();

    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            if entry.path().extension().map_or(false, |e| e == "json") {
                if let Ok(content) = fs::read_to_string(entry.path()) {
                    if let Ok(result) = serde_json::from_str::<TestResult>(&content) {
                        results.push(result);
                    }
                }
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp() {
        let ts = chrono_lite_timestamp();
        assert!(ts.len() == 19); // YYYY-MM-DDTHH:MM:SS
        assert!(ts.contains("T"));
    }
}
