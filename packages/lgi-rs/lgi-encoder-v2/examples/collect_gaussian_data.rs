//! Collect Gaussian configuration data for quantum research
//!
//! Encodes a single image with data logging enabled.
//! Usage: cargo run --example collect_gaussian_data -- <input.png> <output.csv>

use lgi_encoder_v2::{EncoderV2, gaussian_logger::{CsvGaussianLogger, GaussianLogger}};
use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use std::env;
use std::path::PathBuf;

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <input_image.png> <output_data.csv>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  cargo run --release --example collect_gaussian_data -- \\");
        eprintln!("    ../../test-data/kodak-dataset/kodim01.png \\");
        eprintln!("    ../../quantum_research/kodak_gaussian_data/kodim01.csv");
        std::process::exit(1);
    }

    let input_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);

    println!("{}", "=".repeat(80));
    println!("GAUSSIAN DATA COLLECTION FOR QUANTUM RESEARCH");
    println!("{}", "=".repeat(80));
    println!();
    println!("Input:  {}", input_path.display());
    println!("Output: {}", output_path.display());
    println!();

    // Create output directory if needed
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create output directory");
    }

    // Load image
    println!("Loading image...");
    let image = load_png(&input_path).expect("Failed to load image");
    println!("  ✓ Loaded {}×{} image", image.width, image.height);
    println!();

    // Create encoder
    println!("Creating encoder...");
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");
    println!("  ✓ Encoder initialized");
    println!("  ✓ Structure tensor computed");
    println!("  ✓ Geodesic EDT computed");
    println!();

    // Create logger
    println!("Initializing data logger...");
    let mut logger = CsvGaussianLogger::new(&output_path).expect("Failed to create logger");

    // Set structure tensor for context extraction
    logger.set_structure_tensor(encoder.get_structure_tensor().clone());
    logger.set_target(image.clone());

    let image_id = input_path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    println!("  ✓ Logger created: {}", output_path.display());
    println!("  ✓ Image ID: {}", image_id);
    println!();

    // Encode with logging
    println!("{}", "=".repeat(80));
    println!("ENCODING WITH DATA COLLECTION");
    println!("{}", "=".repeat(80));
    println!();
    println!("Method: encode_error_driven_adam");
    println!("Initial Gaussians: 25 (5×5 grid)");
    println!("Max Gaussians: 500");
    println!("Logging interval: Every 10th iteration");
    println!();

    let gaussians = encoder.encode_error_driven_adam_with_logger(
        25,
        500,
        &image_id,
        Some(&mut logger)
    );

    println!();
    println!("{}", "=".repeat(80));
    println!("ENCODING COMPLETE");
    println!("{}", "=".repeat(80));
    println!();
    println!("Final Gaussians: {}", gaussians.len());

    // Compute final quality
    use lgi_encoder_v2::renderer_v2::RendererV2;
    let rendered = RendererV2::render(&gaussians, image.width, image.height);
    let final_psnr = compute_psnr(&image, &rendered);
    println!("Final PSNR: {:.2} dB", final_psnr);
    println!();

    // Count snapshots
    println!("Data collected: {}", output_path.display());
    if let Ok(metadata) = std::fs::metadata(&output_path) {
        println!("File size: {:.2} MB", metadata.len() as f64 / 1024.0 / 1024.0);
    }

    println!();
    println!("✓ Complete!");
    println!();
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
                1.0,
            ));
        }
    }
    Ok(buffer)
}

fn compute_psnr(target: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let pixel_count = (target.width * target.height) as f32;

    for y in 0..target.height {
        for x in 0..target.width {
            let t = target.get_pixel(x, y).unwrap();
            let r = rendered.get_pixel(x, y).unwrap();
            mse += (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
        }
    }

    mse /= (pixel_count * 3.0) as f32;
    if mse < 1e-10 { 100.0 } else { -10.0 * mse.log10() }
}
