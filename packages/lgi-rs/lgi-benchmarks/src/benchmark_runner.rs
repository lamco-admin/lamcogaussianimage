//! Automated benchmark runner with results export

use lgi_core::{ImageBuffer, Renderer};
use lgi_encoder::{Encoder, EncoderConfig};
use crate::metrics::compute_all_metrics;
use std::time::Instant;
use std::path::Path;
use serde::{Serialize, Deserialize};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Image sizes to test
    pub sizes: Vec<u32>,
    /// Gaussian counts to test
    pub gaussian_counts: Vec<usize>,
    /// Quality presets to test
    pub quality_presets: Vec<String>,
    /// Number of runs per configuration
    pub num_runs: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            sizes: vec![128, 256, 512],
            gaussian_counts: vec![100, 200, 500, 1000, 2000],
            quality_presets: vec!["fast".to_string()],
            num_runs: 3,
        }
    }
}

/// Single benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub image_size: u32,
    pub num_gaussians: usize,
    pub quality_preset: String,

    // Timing
    pub encode_time_ms: f32,
    pub decode_time_ms: f32,
    pub total_time_ms: f32,

    // Performance
    pub encode_fps: f32,
    pub decode_fps: f32,
    pub gaussians_per_second: f32,

    // Quality
    pub psnr: f32,
    pub ssim: f32,
    pub mse: f32,
    pub mae: f32,

    // Storage
    pub original_size_kb: u32,
    pub gaussian_size_kb: u32,
    pub compression_ratio: f32,

    // Optimization
    pub iterations_used: usize,
    pub final_loss: f32,
    pub avg_opacity: f32,
}

/// Benchmark runner
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
}

impl BenchmarkRunner {
    /// Create new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Run comprehensive benchmark suite
    pub fn run_suite(&mut self, test_image: &ImageBuffer<f32>) {
        println!("=== LGI Codec Benchmark Suite ===\n");
        println!("Configuration:");
        println!("  Sizes: {:?}", self.config.sizes);
        println!("  Gaussian counts: {:?}", self.config.gaussian_counts);
        println!("  Quality presets: {:?}", self.config.quality_presets);
        println!("  Runs per config: {}\n", self.config.num_runs);

        let total_tests = self.config.sizes.len() *
                         self.config.gaussian_counts.len() *
                         self.config.quality_presets.len() *
                         self.config.num_runs;

        println!("Total tests to run: {}\n", total_tests);

        let mut test_num = 0;

        for &size in &self.config.sizes {
            for &num_gaussians in &self.config.gaussian_counts {
                for quality in &self.config.quality_presets {
                    // Resize test image if needed
                    let test_img = if test_image.width != size || test_image.height != size {
                        resize_image(test_image, size, size)
                    } else {
                        test_image.clone()
                    };

                    for run in 0..self.config.num_runs {
                        test_num += 1;
                        println!("Test {}/{}: size={}, gaussians={}, quality={}, run={}",
                            test_num, total_tests, size, num_gaussians, quality, run + 1);

                        if let Ok(result) = self.run_single_benchmark(&test_img, size, num_gaussians, quality) {
                            self.results.push(result.clone());
                            println!("  ✓ Encode: {:.2}s, Decode: {:.3}s, PSNR: {:.2} dB, SSIM: {:.4}\n",
                                result.encode_time_ms / 1000.0,
                                result.decode_time_ms / 1000.0,
                                result.psnr,
                                result.ssim);
                        } else {
                            println!("  ✗ Failed\n");
                        }
                    }
                }
            }
        }

        println!("=== Benchmark Suite Complete ===");
        println!("Total successful runs: {}", self.results.len());
    }

    /// Run single benchmark
    fn run_single_benchmark(
        &self,
        test_image: &ImageBuffer<f32>,
        size: u32,
        num_gaussians: usize,
        quality_preset: &str,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        // Configure encoder
        let config = match quality_preset {
            "fast" => EncoderConfig::fast(),
            "balanced" => EncoderConfig::balanced(),
            "high" => EncoderConfig::high_quality(),
            "ultra" => EncoderConfig::ultra(),
            _ => EncoderConfig::default(),
        };

        let encoder = Encoder::with_config(config);

        // Encode
        let encode_start = Instant::now();
        let mut iterations = 0;
        let mut final_loss = 0.0;

        let gaussians = encoder.encode_with_progress(test_image, num_gaussians, |iter, loss| {
            iterations = iter;
            final_loss = loss;
        })?;

        let encode_time = encode_start.elapsed();

        // Decode
        let decode_start = Instant::now();
        let renderer = Renderer::new();
        let rendered = renderer.render(&gaussians, test_image.width, test_image.height)?;
        let decode_time = decode_start.elapsed();

        // Compute metrics
        let metrics = compute_all_metrics(test_image, &rendered);
        let avg_opacity = gaussians.iter().map(|g| g.opacity).sum::<f32>() / gaussians.len() as f32;

        // Storage estimate
        let original_size_kb = (test_image.width * test_image.height * 4) / 1024;
        let gaussian_size_kb = (gaussians.len() * 48) / 1024;
        let compression_ratio = gaussian_size_kb as f32 / original_size_kb as f32;

        Ok(BenchmarkResult {
            image_size: size,
            num_gaussians,
            quality_preset: quality_preset.to_string(),

            encode_time_ms: encode_time.as_secs_f32() * 1000.0,
            decode_time_ms: decode_time.as_secs_f32() * 1000.0,
            total_time_ms: (encode_time + decode_time).as_secs_f32() * 1000.0,

            encode_fps: 1.0 / encode_time.as_secs_f32(),
            decode_fps: 1.0 / decode_time.as_secs_f32(),
            gaussians_per_second: gaussians.len() as f32 / encode_time.as_secs_f32(),

            psnr: metrics.psnr,
            ssim: metrics.ssim,
            mse: metrics.mse,
            mae: metrics.mae,

            original_size_kb,
            gaussian_size_kb: gaussian_size_kb as u32,
            compression_ratio,

            iterations_used: iterations,
            final_loss,
            avg_opacity,
        })
    }

    /// Export results to CSV
    pub fn export_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let mut wtr = csv::Writer::from_path(path)?;

        // Write headers
        wtr.write_record(&[
            "image_size", "num_gaussians", "quality_preset",
            "encode_time_ms", "decode_time_ms", "total_time_ms",
            "encode_fps", "decode_fps", "gaussians_per_second",
            "psnr", "ssim", "mse", "mae",
            "original_size_kb", "gaussian_size_kb", "compression_ratio",
            "iterations_used", "final_loss", "avg_opacity",
        ])?;

        // Write data
        for result in &self.results {
            wtr.write_record(&[
                result.image_size.to_string(),
                result.num_gaussians.to_string(),
                result.quality_preset.clone(),
                result.encode_time_ms.to_string(),
                result.decode_time_ms.to_string(),
                result.total_time_ms.to_string(),
                result.encode_fps.to_string(),
                result.decode_fps.to_string(),
                result.gaussians_per_second.to_string(),
                result.psnr.to_string(),
                result.ssim.to_string(),
                result.mse.to_string(),
                result.mae.to_string(),
                result.original_size_kb.to_string(),
                result.gaussian_size_kb.to_string(),
                result.compression_ratio.to_string(),
                result.iterations_used.to_string(),
                result.final_loss.to_string(),
                result.avg_opacity.to_string(),
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }

    /// Export results to JSON
    pub fn export_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.results)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Get results
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Print summary
    pub fn print_summary(&self) {
        if self.results.is_empty() {
            println!("No results to summarize");
            return;
        }

        println!("\n=== Benchmark Summary ===\n");

        // Average metrics
        let avg_psnr: f32 = self.results.iter().map(|r| r.psnr).sum::<f32>() / self.results.len() as f32;
        let avg_ssim: f32 = self.results.iter().map(|r| r.ssim).sum::<f32>() / self.results.len() as f32;
        let avg_encode_time: f32 = self.results.iter().map(|r| r.encode_time_ms).sum::<f32>() / self.results.len() as f32;
        let avg_decode_fps: f32 = self.results.iter().map(|r| r.decode_fps).sum::<f32>() / self.results.len() as f32;

        println!("Average Quality:");
        println!("  PSNR: {:.2} dB", avg_psnr);
        println!("  SSIM: {:.4}", avg_ssim);

        println!("\nAverage Performance:");
        println!("  Encode time: {:.2}s", avg_encode_time / 1000.0);
        println!("  Decode FPS: {:.1}", avg_decode_fps);

        // Find best/worst
        let best_psnr = self.results.iter().max_by(|a, b| a.psnr.partial_cmp(&b.psnr).unwrap()).unwrap();
        let worst_psnr = self.results.iter().min_by(|a, b| a.psnr.partial_cmp(&b.psnr).unwrap()).unwrap();

        println!("\nBest PSNR: {:.2} dB (size={}, gaussians={})",
            best_psnr.psnr, best_psnr.image_size, best_psnr.num_gaussians);
        println!("Worst PSNR: {:.2} dB (size={}, gaussians={})",
            worst_psnr.psnr, worst_psnr.image_size, worst_psnr.num_gaussians);
    }
}

/// Simple image resizing (nearest neighbor for now)
fn resize_image(img: &ImageBuffer<f32>, new_width: u32, new_height: u32) -> ImageBuffer<f32> {
    let mut result = ImageBuffer::new(new_width, new_height);

    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = (x as f32 / new_width as f32 * img.width as f32) as u32;
            let src_y = (y as f32 / new_height as f32 * img.height as f32) as u32;

            if let Some(pixel) = img.get_pixel(src_x.min(img.width - 1), src_y.min(img.height - 1)) {
                result.set_pixel(x, y, pixel);
            }
        }
    }

    result
}
