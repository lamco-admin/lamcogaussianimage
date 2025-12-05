//! Test Results Logging Utility
//!
//! Saves test results to persistent storage for building a corpus of experiments.
//!
//! Structure:
//! ```text
//! test-results/
//!   YYYY-MM-DD/
//!     experiment_name.json      # Metrics and parameters
//!   rendered-images/
//!     experiment_name_*.png     # Visual results
//! ```

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use serde::{Serialize, Deserialize};
use lgi_core::ImageBuffer;

/// Test result metadata
#[derive(Serialize, Deserialize, Debug)]
pub struct TestResult {
    pub experiment_name: String,
    pub timestamp: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_metrics: Option<serde_json::Value>,
}

impl TestResult {
    pub fn new(experiment_name: &str, image_path: &str) -> Self {
        let now = chrono_lite_timestamp();
        Self {
            experiment_name: experiment_name.to_string(),
            timestamp: now,
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
            extra_metrics: None,
        }
    }

    /// Save result to JSON file
    pub fn save(&self, base_dir: &str) -> std::io::Result<String> {
        let date = &self.timestamp[..10]; // YYYY-MM-DD
        let dir = format!("{}/{}", base_dir, date);
        fs::create_dir_all(&dir)?;

        let filename = format!("{}/{}_{}.json",
            dir,
            self.experiment_name.replace(" ", "_").replace("/", "-"),
            &self.timestamp[11..19].replace(":", "-") // HH-MM-SS
        );

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let mut file = File::create(&filename)?;
        file.write_all(json.as_bytes())?;

        Ok(filename)
    }
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
