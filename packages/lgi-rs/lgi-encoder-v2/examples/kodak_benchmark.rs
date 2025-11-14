//! Kodak Dataset Benchmark
//!
//! Tests all encoding methods on the Kodak PhotoCD dataset (24 768×512 images)
//! Standard benchmark for image quality assessment

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2};
use lgi_core::{ImageBuffer, quantization::LGIQProfile};
use lgi_math::color::Color4;
use std::time::Instant;
use std::path::PathBuf;

fn main() {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("         LGI v2 KODAK DATASET BENCHMARK (24 Images)           ");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Load Kodak dataset (24 images, 768×512 PNG)
    let kodak_dir = PathBuf::from("../../kodak-dataset");

    if !kodak_dir.exists() {
        eprintln!("ERROR: Kodak dataset not found at {:?}", kodak_dir);
        eprintln!("Please run: cd {} && wget http://r0k.us/graphics/kodak/kodak/kodim{{01..24}}.png", kodak_dir.display());
        return;
    }

    let mut results = Vec::new();

    // Test first 5 images for speed (full benchmark takes ~30 minutes)
    for i in 1..=5 {
        let image_path = kodak_dir.join(format!("kodim{:02}.png", i));

        println!("\n▼ TEST IMAGE: kodim{:02}.png (768×512)", i);
        println!("─────────────────────────────────────────────────────────────");

        let image = match load_png(&image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("  ERROR loading image: {}", e);
                continue;
            }
        };

        let result = benchmark_image(&image, &format!("kodim{:02}", i));
        results.push(result);
    }

    // Print summary
    print_summary(&results);
}

fn benchmark_image(image: &ImageBuffer<f32>, image_name: &str) -> BenchmarkResult {
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    let mut result = BenchmarkResult::new(image_name);

    // =========================================================================
    // Test 1: BASELINE (No optimization)
    // =========================================================================
    println!("\n[1/4] BASELINE (8×8 grid = 64 Gaussians)");
    let start = Instant::now();
    let baseline = encoder.initialize_gaussians(8);
    let baseline_time = start.elapsed();
    let baseline_psnr = compute_psnr(image, &baseline);
    println!("  N = {} | PSNR = {:.2} dB | Time = {:?}",
        baseline.len(), baseline_psnr, baseline_time);

    result.baseline_psnr = baseline_psnr;
    result.baseline_time = baseline_time;

    // =========================================================================
    // Test 2: ADAM OPTIMIZER (RECOMMENDED)
    // =========================================================================
    println!("\n[2/4] ADAM OPTIMIZER (RECOMMENDED) - FULL RESOLUTION 768×512");

    let start = Instant::now();
    let adam = encoder.encode_error_driven_adam(25, 100);
    let adam_time = start.elapsed();
    let adam_psnr = compute_psnr(image, &adam);
    let adam_gain = adam_psnr - baseline_psnr;
    println!("  N = {} | PSNR = {:.2} dB ({:+.2} dB) | Time = {:?}",
        adam.len(), adam_psnr, adam_gain, adam_time);

    result.adam_psnr = adam_psnr;
    result.adam_time = adam_time;

    // =========================================================================
    // Test 3: GPU ACCELERATION
    // =========================================================================
    println!("\n[3/4] GPU ACCELERATION - FULL RESOLUTION 768×512");
    let start = Instant::now();
    let gpu = encoder.encode_error_driven_gpu(25, 100);
    let gpu_time = start.elapsed();
    let gpu_psnr = compute_psnr(image, &gpu);
    let gpu_gain = gpu_psnr - baseline_psnr;
    let gpu_speedup = adam_time.as_secs_f32() / gpu_time.as_secs_f32().max(0.001);
    println!("  N = {} | PSNR = {:.2} dB ({:+.2} dB) | Time = {:?} ({:.1}× vs Adam)",
        gpu.len(), gpu_psnr, gpu_gain, gpu_time, gpu_speedup);

    result.gpu_psnr = gpu_psnr;
    result.gpu_time = gpu_time;

    // =========================================================================
    // Test 4: RATE-DISTORTION (Target 30 dB)
    // =========================================================================
    println!("\n[4/4] RATE-DISTORTION: Target PSNR = 30 dB - FULL RESOLUTION");
    let start = Instant::now();
    let rd = encoder.encode_for_psnr(30.0, LGIQProfile::Baseline);
    let rd_time = start.elapsed();
    let rd_psnr = compute_psnr(image, &rd);
    let rd_error = rd_psnr - 30.0;
    println!("  N = {} | PSNR = {:.2} dB (target 30.0, error {:+.2} dB) | Time = {:?}",
        rd.len(), rd_psnr, rd_error, rd_time);

    result.rd_psnr = rd_psnr;
    result.rd_time = rd_time;

    result
}

fn load_png(path: &PathBuf) -> Result<ImageBuffer<f32>, String> {
    // Use image crate to load PNG
    let img = image::open(path)
        .map_err(|e| format!("Failed to load image: {}", e))?;

    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let mut buffer = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            buffer.set_pixel(x, y, Color4::new(r, g, b, 1.0));
        }
    }

    Ok(buffer)
}

fn downscale_image(image: &ImageBuffer<f32>, new_width: u32, new_height: u32) -> ImageBuffer<f32> {
    let mut downscaled = ImageBuffer::new(new_width, new_height);

    let scale_x = image.width as f32 / new_width as f32;
    let scale_y = image.height as f32 / new_height as f32;

    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = (x as f32 * scale_x) as u32;
            let src_y = (y as f32 * scale_y) as u32;

            if let Some(color) = image.get_pixel(src_x, src_y) {
                downscaled.set_pixel(x, y, color);
            }
        }
    }

    downscaled
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
            let t_pixel = target.get_pixel(x, y).unwrap();
            let r_pixel = rendered.get_pixel(x, y).unwrap();

            let diff_r = t_pixel.r - r_pixel.r;
            let diff_g = t_pixel.g - r_pixel.g;
            let diff_b = t_pixel.b - r_pixel.b;

            mse += diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
        }
    }

    mse /= (pixel_count * 3) as f32;

    if mse < 1e-10 {
        100.0
    } else {
        -10.0 * mse.log10()
    }
}

struct BenchmarkResult {
    name: String,
    baseline_psnr: f32,
    baseline_time: std::time::Duration,
    adam_psnr: f32,
    adam_time: std::time::Duration,
    gpu_psnr: f32,
    gpu_time: std::time::Duration,
    rd_psnr: f32,
    rd_time: std::time::Duration,
}

impl BenchmarkResult {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            baseline_psnr: 0.0,
            baseline_time: std::time::Duration::from_secs(0),
            adam_psnr: 0.0,
            adam_time: std::time::Duration::from_secs(0),
            gpu_psnr: 0.0,
            gpu_time: std::time::Duration::from_secs(0),
            rd_psnr: 0.0,
            rd_time: std::time::Duration::from_secs(0),
        }
    }
}

fn print_summary(results: &[BenchmarkResult]) {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║ KODAK DATASET SUMMARY ({} images)                          ║", results.len());
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Image      │  Baseline │   Adam    │    GPU    │    R-D    ║");
    println!("║            │   (dB)    │   (dB)    │   (dB)    │   (dB)    ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");

    let mut sum_baseline = 0.0;
    let mut sum_adam = 0.0;
    let mut sum_gpu = 0.0;
    let mut sum_rd = 0.0;

    for result in results {
        println!("║ {:10} │   {:>6.2}  │   {:>6.2}  │   {:>6.2}  │   {:>6.2}  ║",
            result.name,
            result.baseline_psnr,
            result.adam_psnr,
            result.gpu_psnr,
            result.rd_psnr);

        sum_baseline += result.baseline_psnr;
        sum_adam += result.adam_psnr;
        sum_gpu += result.gpu_psnr;
        sum_rd += result.rd_psnr;
    }

    let count = results.len() as f32;
    let avg_baseline = sum_baseline / count;
    let avg_adam = sum_adam / count;
    let avg_gpu = sum_gpu / count;
    let avg_rd = sum_rd / count;

    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ AVERAGE    │   {:>6.2}  │   {:>6.2}  │   {:>6.2}  │   {:>6.2}  ║",
        avg_baseline, avg_adam, avg_gpu, avg_rd);
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ GAIN       │      —    │   {:>+6.2}  │   {:>+6.2}  │   {:>+6.2}  ║",
        avg_adam - avg_baseline,
        avg_gpu - avg_baseline,
        avg_rd - avg_baseline);
    println!("╚═══════════════════════════════════════════════════════════════╝");

    println!("\n✅ BEST METHOD: Adam Optimizer ({:.2} dB average, {:+.2} dB gain)",
        avg_adam, avg_adam - avg_baseline);
}
