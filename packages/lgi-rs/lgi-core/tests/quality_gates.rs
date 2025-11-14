//! Quality Gate Tests - Validate Each Technique's Improvement
//!
//! Each module must pass quality gates before we proceed to next.
//! This prevents the v1 mistake of implementing everything before testing.

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_math::{log_cholesky::LogCholesky, color::Color4, vec::Vector2, gaussian::Gaussian2D, parameterization::Euler};

/// Standard test image set
pub struct TestImage {
    pub name: String,
    pub image: ImageBuffer<f32>,
    pub baseline_psnr: f32,
    pub content_type: ContentType,
}

#[derive(Debug, Clone, Copy)]
pub enum ContentType {
    SolidColor,
    Gradient,
    Text,
    Photo,
}

impl TestImage {
    /// Create solid color test image
    pub fn solid_color(name: &str, size: u32, r: f32, g: f32, b: f32) -> Self {
        let mut image = ImageBuffer::new(size, size);
        for pixel in &mut image.data {
            *pixel = Color4::new(r, g, b, 1.0);
        }

        Self {
            name: name.to_string(),
            image,
            baseline_psnr: 17.0,  // From v1 testing
            content_type: ContentType::SolidColor,
        }
    }

    /// Create linear gradient
    pub fn gradient_linear(name: &str, size: u32) -> Self {
        let mut image = ImageBuffer::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let t = x as f32 / size as f32;  // 0 to 1
                // Blue to red
                let r = t;
                let g = 0.0;
                let b = 1.0 - t;
                image.set_pixel(x, y, Color4::new(r, g, b, 1.0));
            }
        }

        Self {
            name: name.to_string(),
            image,
            baseline_psnr: 16.0,
            content_type: ContentType::Gradient,
        }
    }

    /// Create text image (white on black)
    pub fn text_simple(name: &str, size: u32) -> Self {
        // Simple "I" shape for testing
        let mut image = ImageBuffer::new(size, size);

        // Black background
        for pixel in &mut image.data {
            *pixel = Color4::new(0.0, 0.0, 0.0, 1.0);
        }

        // White vertical bar (letter "I")
        let bar_width = size / 8;
        let bar_start = size / 2 - bar_width / 2;
        let bar_end = bar_start + bar_width;

        for y in size / 4..3 * size / 4 {
            for x in bar_start..bar_end {
                image.set_pixel(x, y, Color4::new(1.0, 1.0, 1.0, 1.0));
            }
        }

        Self {
            name: name.to_string(),
            image,
            baseline_psnr: 15.0,  // With geodesic EDT, before was 3 dB
            content_type: ContentType::Text,
        }
    }
}

/// Quality gate result
pub struct GateResult {
    pub technique: String,
    pub passed: bool,
    pub avg_improvement: f32,
    pub results: Vec<TestResult>,
}

pub struct TestResult {
    pub test_name: String,
    pub psnr: f32,
    pub baseline: f32,
    pub improvement: f32,
    pub passed: bool,
}

impl GateResult {
    pub fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════╗");
        println!("║ QUALITY GATE: {}",  self.technique);
        println!("╠═══════════════════════════════════════════════╣");

        for result in &self.results {
            let status = if result.passed { "✅" } else { "❌" };
            println!("║ {} {}: {:.2} dB (baseline: {:.2}, {:+.2} dB)",
                status, result.test_name, result.psnr, result.baseline, result.improvement);
        }

        println!("╠═══════════════════════════════════════════════╣");
        println!("║ Average Improvement: {:+.2} dB", self.avg_improvement);
        println!("║ Status: {}",
            if self.passed { "✅ PASSED - PROCEED" } else { "❌ FAILED - INVESTIGATE" });
        println!("╚═══════════════════════════════════════════════╝\n");
    }
}

/// PSNR computation
pub fn compute_psnr(original: &ImageBuffer<f32>, reconstructed: &ImageBuffer<f32>) -> f32 {
    assert_eq!(original.width, reconstructed.width);
    assert_eq!(original.height, reconstructed.height);

    let mut mse = 0.0;
    let count = (original.width * original.height * 3) as f32;

    for (p1, p2) in original.data.iter().zip(reconstructed.data.iter()) {
        mse += (p1.r - p2.r).powi(2);
        mse += (p1.g - p2.g).powi(2);
        mse += (p1.b - p2.b).powi(2);
    }

    mse /= count;

    if mse < 1e-10 {
        100.0
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psnr_identical_images() {
        let img = TestImage::solid_color("red", 64, 1.0, 0.0, 0.0);
        let psnr = compute_psnr(&img.image, &img.image);

        assert!(psnr > 90.0, "Identical images should have very high PSNR");
    }

    #[test]
    fn test_create_test_images() {
        let solid = TestImage::solid_color("red", 64, 1.0, 0.0, 0.0);
        assert_eq!(solid.image.width, 64);
        assert_eq!(solid.content_type as u8, ContentType::SolidColor as u8);

        let gradient = TestImage::gradient_linear("test", 64);
        assert_eq!(gradient.content_type as u8, ContentType::Gradient as u8);

        let text = TestImage::text_simple("I", 64);
        assert_eq!(text.content_type as u8, ContentType::Text as u8);
    }
}
