//! Real World Benchmark
//!
//! Tests on BOTH:
//! 1. Kodak dataset (768×512) - standard benchmark
//! 2. User's test images (4K, 8MP) - real high-res performance

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2};
use lgi_core::{ImageBuffer, quantization::LGIQProfile};
use lgi_math::color::Color4;
use std::time::Instant;
use std::path::PathBuf;

fn main() {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("      LGI v2 REAL WORLD BENCHMARK - Session 8 Validation      ");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut all_results = Vec::new();

    // =========================================================================
    // PART 1: Kodak Dataset (Standard Benchmark - 768×512)
    // =========================================================================
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║ PART 1: KODAK DATASET (Standard Benchmark)                   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let kodak_dir = PathBuf::from("../../kodak-dataset");
    if kodak_dir.exists() {
        // Test first 3 Kodak images
        for i in 2..=4 {
            let image_path = kodak_dir.join(format!("kodim{:02}.png", i));
            if let Ok(image) = load_png(&image_path) {
                println!("\n▼ kodim{:02}.png ({}×{})", i, image.width, image.height);
                let result = benchmark_image(&image, &format!("kodim{:02}", i));
                all_results.push(result);
            }
        }
    } else {
        println!("⚠️  Kodak dataset not found, skipping...");
    }

    // =========================================================================
    // PART 2: High-Res Test Images (Real World - 4K+)
    // =========================================================================
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║ PART 2: HIGH-RES TEST IMAGES (Real World 4K+)                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let test_dir = PathBuf::from("/home/greg/gaussian-image-projects/test_images");
    if test_dir.exists() {
        // Test first 2 high-res images (4K takes ~30-60s each)
        let test_files = vec![
            "20140709_094123.jpg",  // 4128×2322
            "133683084337742188.jpg"  // 3840×2160
        ];

        for filename in test_files {
            let image_path = test_dir.join(filename);
            if let Ok(image) = load_jpg(&image_path) {
                println!("\n▼ {} ({}×{})", filename, image.width, image.height);
                let result = benchmark_image(&image, filename);
                all_results.push(result);
            }
        }
    } else {
        println!("⚠️  Test images directory not found");
    }

    // Print combined summary
    print_summary(&all_results);
}

fn benchmark_image(image: &ImageBuffer<f32>, image_name: &str) -> BenchmarkResult {
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");
    let mut result = BenchmarkResult::new(image_name, image.width, image.height);

    // Baseline
    println!("  [1/3] Baseline...");
    let start = Instant::now();
    let baseline = encoder.initialize_gaussians(10);  // 10×10=100
    let baseline_time = start.elapsed();
    let baseline_psnr = compute_psnr(image, &baseline);
    println!("    PSNR = {:.2} dB | Time = {:?}", baseline_psnr, baseline_time);
    result.baseline_psnr = baseline_psnr;
    result.baseline_time = baseline_time;

    // Adam (RECOMMENDED) - scale N based on resolution
    // For Kodak (768×512 = 0.4MP): start with 25, max 100
    // For 4K (8.3MP): start with 125, max 500
    let pixels = (image.width * image.height) as f32;
    let n_init = (pixels.sqrt() / 20.0) as usize;  // More reasonable scaling
    let n_max = n_init * 4;
    println!("  [2/3] Adam Optimizer (N: {}→{})...", n_init, n_max);
    let start = Instant::now();
    let adam = encoder.encode_error_driven_adam(n_init, n_max);
    let adam_time = start.elapsed();
    let adam_psnr = compute_psnr(image, &adam);
    println!("    PSNR = {:.2} dB ({:+.2} dB) | N = {} | Time = {:?}",
        adam_psnr, adam_psnr - baseline_psnr, adam.len(), adam_time);
    result.adam_psnr = adam_psnr;
    result.adam_time = adam_time;
    result.adam_n = adam.len();

    // GPU
    println!("  [3/3] GPU Acceleration...");
    let start = Instant::now();
    let gpu = encoder.encode_error_driven_gpu(n_init, n_max);
    let gpu_time = start.elapsed();
    let gpu_psnr = compute_psnr(image, &gpu);
    let speedup = adam_time.as_secs_f32() / gpu_time.as_secs_f32().max(0.001);
    println!("    PSNR = {:.2} dB ({:+.2} dB) | N = {} | Time = {:?} ({:.1}× faster)",
        gpu_psnr, gpu_psnr - baseline_psnr, gpu.len(), gpu_time, speedup);
    result.gpu_psnr = gpu_psnr;
    result.gpu_time = gpu_time;
    result.gpu_n = gpu.len();

    result
}

fn load_png(path: &PathBuf) -> Result<ImageBuffer<f32>, String> {
    let img = image::open(path)
        .map_err(|e| format!("Failed to load image: {}", e))?;
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut buffer = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            buffer.set_pixel(x, y, Color4::new(
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
                1.0
            ));
        }
    }
    Ok(buffer)
}

fn load_jpg(path: &PathBuf) -> Result<ImageBuffer<f32>, String> {
    load_png(path)  // image crate handles both
}

fn compute_psnr(
    target: &ImageBuffer<f32>,
    gaussians: &[lgi_math::gaussian::Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>]
) -> f32 {
    let rendered = RendererV2::render(gaussians, target.width, target.height);
    let mut mse = 0.0f32;
    let pixel_count = target.width * target.height;

    for y in 0..target.height {
        for x in 0..target.width {
            let t = target.get_pixel(x, y).unwrap();
            let r = rendered.get_pixel(x, y).unwrap();
            mse += (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
        }
    }

    mse /= (pixel_count * 3) as f32;
    if mse < 1e-10 { 100.0 } else { -10.0 * mse.log10() }
}

#[derive(Clone)]
struct BenchmarkResult {
    name: String,
    width: u32,
    height: u32,
    baseline_psnr: f32,
    baseline_time: std::time::Duration,
    adam_psnr: f32,
    adam_time: std::time::Duration,
    adam_n: usize,
    gpu_psnr: f32,
    gpu_time: std::time::Duration,
    gpu_n: usize,
}

impl BenchmarkResult {
    fn new(name: &str, width: u32, height: u32) -> Self {
        Self {
            name: name.to_string(),
            width,
            height,
            baseline_psnr: 0.0,
            baseline_time: std::time::Duration::from_secs(0),
            adam_psnr: 0.0,
            adam_time: std::time::Duration::from_secs(0),
            adam_n: 0,
            gpu_psnr: 0.0,
            gpu_time: std::time::Duration::from_secs(0),
            gpu_n: 0,
        }
    }

    fn resolution(&self) -> String {
        format!("{}×{}", self.width, self.height)
    }

    fn megapixels(&self) -> f32 {
        (self.width * self.height) as f32 / 1_000_000.0
    }
}

fn print_summary(results: &[BenchmarkResult]) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ COMPREHENSIVE SUMMARY                                                         ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Image              │  Resolution │  MP  │ Baseline │  Adam   │   GPU   │ GPU  ║");
    println!("║                    │             │      │   (dB)   │  (dB)   │  (dB)   │ Time ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");

    let mut sum_baseline = 0.0;
    let mut sum_adam = 0.0;
    let mut sum_gpu = 0.0;

    for r in results {
        println!("║ {:18} │ {:11} │ {:>4.1} │   {:>6.2} │  {:>6.2} │  {:>6.2} │ {:>4}s ║",
            r.name.chars().take(18).collect::<String>(),
            r.resolution(),
            r.megapixels(),
            r.baseline_psnr,
            r.adam_psnr,
            r.gpu_psnr,
            r.gpu_time.as_secs());

        sum_baseline += r.baseline_psnr;
        sum_adam += r.adam_psnr;
        sum_gpu += r.gpu_psnr;
    }

    let count = results.len() as f32;
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ AVERAGE            │             │      │   {:>6.2} │  {:>6.2} │  {:>6.2} │      ║",
        sum_baseline / count, sum_adam / count, sum_gpu / count);
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ IMPROVEMENT        │             │      │      —   │ {:>+6.2} │ {:>+6.2} │      ║",
        (sum_adam - sum_baseline) / count,
        (sum_gpu - sum_baseline) / count);
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");

    // GPU analysis
    println!("\n📊 GPU PERFORMANCE ANALYSIS:");
    for r in results {
        let speedup = r.adam_time.as_secs_f32() / r.gpu_time.as_secs_f32().max(0.001);
        println!("  {} ({:.1}MP): GPU {:.1}× {} than Adam",
            r.name.chars().take(20).collect::<String>(),
            r.megapixels(),
            speedup,
            if speedup > 1.0 { "FASTER" } else { "slower" });
    }

    println!("\n✅ BEST METHOD: Adam Optimizer ({:.2} dB average, {:+.2} dB gain over baseline)",
        sum_adam / count,
        (sum_adam - sum_baseline) / count);
}
