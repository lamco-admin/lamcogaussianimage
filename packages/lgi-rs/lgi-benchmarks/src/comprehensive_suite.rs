//! Comprehensive Benchmark Suite
//!
//! Tests all compression modes, rendering modes, and edge cases

use lgi_core::{ImageBuffer, Renderer, RenderMode};
use lgi_encoder::{EncoderConfig, OptimizerV2};
use lgi_format::{LgiFile, LgiWriter, LgiReader, CompressionConfig, QuantizationProfile};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use std::io::Cursor;

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub gaussians: usize,
    pub encode_time_ms: f32,
    pub decode_time_ms: f32,
    pub render_time_ms: f32,
    pub psnr_db: f32,
    pub file_size_bytes: usize,
    pub compression_ratio: f32,
    pub quality_profile: String,
}

/// Comprehensive benchmark runner
pub struct ComprehensiveBenchmark;

impl ComprehensiveBenchmark {
    /// Run full benchmark suite
    pub fn run_all() -> Vec<BenchmarkResult> {
        let mut results = Vec::new();

        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  LGI Comprehensive Benchmark Suite                           ║");
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // Test patterns
        let patterns = vec![
            ("Solid Color", Self::create_solid_color(256, 256)),
            ("Gradient", Self::create_gradient(256, 256)),
            ("Checkerboard", Self::create_checkerboard(256, 256)),
        ];

        // Gaussian counts
        let gaussian_counts = vec![100, 500, 1000, 2000];

        // Compression modes
        let compression_modes = vec![
            ("Uncompressed", CompressionConfig::uncompressed()),
            ("LGIQ-B", Self::config_lgiq_b()),
            ("LGIQ-S", Self::config_lgiq_s()),
            ("LGIQ-H", Self::config_lgiq_h()),
            ("LGIQ-X", Self::config_lgiq_x()),
        ];

        // Rendering modes
        let render_modes = vec![
            RenderMode::AlphaComposite,
            RenderMode::AccumulatedSum,
        ];

        println!("Running benchmarks:");
        println!("  - {} test patterns", patterns.len());
        println!("  - {} Gaussian counts", gaussian_counts.len());
        println!("  - {} compression modes", compression_modes.len());
        println!("  - {} rendering modes", render_modes.len());
        println!("  Total: {} combinations\n",
            patterns.len() * gaussian_counts.len() * compression_modes.len() * render_modes.len());

        let mut test_count = 0;
        let total_tests = patterns.len() * gaussian_counts.len() * compression_modes.len() * render_modes.len();

        for (pattern_name, target) in &patterns {
            for &num_gaussians in &gaussian_counts {
                for (compression_name, compression_config) in &compression_modes {
                    for &render_mode in &render_modes {
                        test_count += 1;
                        print!("\r[{}/{}] Testing: {}, {}G, {}, {:?}...",
                            test_count, total_tests, pattern_name, num_gaussians, compression_name, render_mode);

                        if let Ok(result) = Self::run_single_benchmark(
                            &format!("{}-{}G-{}-{:?}", pattern_name, num_gaussians, compression_name, render_mode),
                            target,
                            num_gaussians,
                            compression_config.clone(),
                            render_mode,
                        ) {
                            results.push(result);
                        }
                    }
                }
            }
        }

        println!("\n\n✅ Benchmark suite complete! {} results collected", results.len());

        results
    }

    /// Run single benchmark
    fn run_single_benchmark(
        name: &str,
        target: &ImageBuffer<f32>,
        num_gaussians: usize,
        compression: CompressionConfig,
        render_mode: RenderMode,
    ) -> anyhow::Result<BenchmarkResult> {
        // Encode
        let encode_start = Instant::now();
        let config = EncoderConfig::fast(); // Fast for benchmarking
        let initializer = lgi_core::Initializer::new(config.init_strategy);
        let mut gaussians = initializer.initialize(target, num_gaussians)?;
        let optimizer = OptimizerV2::new(config);
        let _metrics = optimizer.optimize_with_metrics(&mut gaussians, target)?;
        let encode_time_ms = encode_start.elapsed().as_secs_f32() * 1000.0;

        // Compress and save
        let file = LgiFile::with_compression(
            gaussians.clone(),
            target.width,
            target.height,
            compression.clone(),
        );

        let mut buffer = Vec::new();
        LgiWriter::write(&mut buffer, &file)?;
        let file_size_bytes = buffer.len();

        // Decode
        let decode_start = Instant::now();
        let mut cursor = Cursor::new(buffer);
        let loaded = LgiReader::read(&mut cursor)?;
        let reconstructed_gaussians = loaded.gaussians();
        let decode_time_ms = decode_start.elapsed().as_secs_f32() * 1000.0;

        // Render
        let render_start = Instant::now();
        let mut render_config = lgi_core::RenderConfig::default();
        render_config.render_mode = render_mode;
        let renderer = Renderer::with_config(render_config);
        let rendered = renderer.render(&reconstructed_gaussians, target.width, target.height)?;
        let render_time_ms = render_start.elapsed().as_secs_f32() * 1000.0;

        // Measure quality
        let psnr = Self::compute_psnr(&rendered, target);

        // Compression ratio
        let uncompressed_size = (target.width * target.height * 4 * 3) as usize;
        let compression_ratio = uncompressed_size as f32 / file_size_bytes as f32;

        Ok(BenchmarkResult {
            name: name.to_string(),
            gaussians: num_gaussians,
            encode_time_ms,
            decode_time_ms,
            render_time_ms,
            psnr_db: psnr,
            file_size_bytes,
            compression_ratio,
            quality_profile: format!("{:?}", compression.quantization),
        })
    }

    /// Create solid color test image
    fn create_solid_color(width: u32, height: u32) -> ImageBuffer<f32> {
        ImageBuffer::with_background(width, height, Color4::rgb(0.5, 0.5, 0.5))
    }

    /// Create gradient test image
    fn create_gradient(width: u32, height: u32) -> ImageBuffer<f32> {
        let mut buffer = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let v = x as f32 / width as f32;
                buffer.set_pixel(x, y, Color4::rgb(v, 0.5, 0.5));
            }
        }
        buffer
    }

    /// Create checkerboard test image
    fn create_checkerboard(width: u32, height: u32) -> ImageBuffer<f32> {
        let mut buffer = ImageBuffer::new(width, height);
        let square_size = 16;
        for y in 0..height {
            for x in 0..width {
                let checker = ((x / square_size) + (y / square_size)) % 2 == 0;
                let v = if checker { 1.0 } else { 0.0 };
                buffer.set_pixel(x, y, Color4::rgb(v, v, v));
            }
        }
        buffer
    }

    /// Compute PSNR
    fn compute_psnr(img1: &ImageBuffer<f32>, img2: &ImageBuffer<f32>) -> f32 {
        let mut mse = 0.0;
        let count = (img1.width * img1.height * 3) as f32;

        for (p1, p2) in img1.data.iter().zip(img2.data.iter()) {
            mse += (p1.r - p2.r) * (p1.r - p2.r);
            mse += (p1.g - p2.g) * (p1.g - p2.g);
            mse += (p1.b - p2.b) * (p1.b - p2.b);
        }

        mse /= count;

        if mse < 1e-10 {
            100.0
        } else {
            20.0 * (1.0f32 / mse.sqrt()).log10()
        }
    }

    // Helper configs
    fn config_lgiq_b() -> CompressionConfig {
        CompressionConfig::custom(
            QuantizationProfile::LGIQ_B,
            false, 0, true, 9
        )
    }

    fn config_lgiq_s() -> CompressionConfig {
        CompressionConfig::custom(
            QuantizationProfile::LGIQ_S,
            false, 0, true, 9
        )
    }

    fn config_lgiq_h() -> CompressionConfig {
        CompressionConfig::custom(
            QuantizationProfile::LGIQ_H,
            false, 0, true, 9
        )
    }

    fn config_lgiq_x() -> CompressionConfig {
        CompressionConfig::lossless()
    }
}

/// Export results to CSV
pub fn export_results_csv(results: &[BenchmarkResult], path: &str) -> anyhow::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;

    writeln!(file, "Name,Gaussians,EncodeMS,DecodeMS,RenderMS,PSNR,FileSizeKB,CompressionRatio,Profile")?;

    for r in results {
        writeln!(file, "{},{},{:.2},{:.2},{:.2},{:.2},{},{:.2},{}",
            r.name,
            r.gaussians,
            r.encode_time_ms,
            r.decode_time_ms,
            r.render_time_ms,
            r.psnr_db,
            r.file_size_bytes / 1024,
            r.compression_ratio,
            r.quality_profile
        )?;
    }

    Ok(())
}
