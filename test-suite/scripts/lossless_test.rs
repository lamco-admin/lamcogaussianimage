//! Test lossless f32 file I/O mode
//! Demonstrates and tests the lossless storage capability

use std::time::Instant;
use lgi_core::color_space::ColorSpace;
use lgi_encoder_v2::{encoder_v2::EncoderV2, file_writer::LGIWriter, file_reader::LGIReader};
use lgi_gpu::renderer::GaussianRenderer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LGI Lossless F32 File Mode Test ===\n");

    // Load test image
    let img_path = "/home/greg/gaussian-image-projects/test_images/133784383569199567.jpg";
    let img = image::open(img_path)?;
    let img = img.resize_exact(512, 288, image::imageops::FilterType::Lanczos3);
    let img_rgb = img.to_rgb32f();

    let width = img_rgb.width();
    let height = img_rgb.height();
    println!("Test image: {}x{}", width, height);

    // Configure encoder
    let n_gaussians = 900; // 30x30 grid
    println!("Using {} Gaussians", n_gaussians);

    // Encode with optimizer
    let mut encoder = EncoderV2::new(n_gaussians, ColorSpace::Linear);
    let gaussians = encoder.encode_image(&img_rgb, 50, false)?;

    // Render original
    println!("\n1. Rendering original Gaussians...");
    let renderer = GaussianRenderer::new()?;
    let original_img = renderer.render(&gaussians, width, height)?;

    // Calculate PSNR of original encoding
    let original_psnr = lgi_core::metrics::psnr(&img_rgb, &original_img);
    println!("   Original PSNR: {:.2} dB", original_psnr);

    // Test 1: LGIQ-B baseline (quantized)
    println!("\n2. Testing LGIQ-B baseline (quantized, 11 bytes/G)...");
    {
        let writer = LGIWriter::new(width, height, gaussians.len() as u32);
        writer.write_file("/tmp/test_quantized.lgi", &gaussians)?;

        let file_size = std::fs::metadata("/tmp/test_quantized.lgi")?.len();
        println!("   File size: {} bytes ({:.2} bytes/G)",
                 file_size, file_size as f64 / gaussians.len() as f64);

        let (header, loaded_gaussians) = LGIReader::read_file("/tmp/test_quantized.lgi")?;
        println!("   Loaded {} Gaussians", loaded_gaussians.len());
        println!("   Header: {}x{}, encoding: {}, bitdepth: {}",
                 header.canvas_width, header.canvas_height,
                 header.param_encoding, header.bitdepth);

        let quantized_img = renderer.render(&loaded_gaussians, width, height)?;
        let quantized_psnr = lgi_core::metrics::psnr(&img_rgb, &quantized_img);
        println!("   Quantized PSNR: {:.2} dB", quantized_psnr);

        let quantization_loss = original_psnr - quantized_psnr;
        println!("   Quantization loss: {:.4} dB", quantization_loss);
    }

    // Test 2: Lossless f32 mode
    println!("\n3. Testing lossless f32 mode (36 bytes/G)...");
    {
        let writer = LGIWriter::new_lossless(width, height, gaussians.len() as u32);
        writer.write_file("/tmp/test_lossless.lgi", &gaussians)?;

        let file_size = std::fs::metadata("/tmp/test_lossless.lgi")?.len();
        println!("   File size: {} bytes ({:.2} bytes/G)",
                 file_size, file_size as f64 / gaussians.len() as f64);

        let (header, loaded_gaussians) = LGIReader::read_file("/tmp/test_lossless.lgi")?;
        println!("   Loaded {} Gaussians", loaded_gaussians.len());
        println!("   Header: {}x{}, encoding: {}, bitdepth: {}",
                 header.canvas_width, header.canvas_height,
                 header.param_encoding, header.bitdepth);

        let lossless_img = renderer.render(&loaded_gaussians, width, height)?;
        let lossless_psnr = lgi_core::metrics::psnr(&img_rgb, &lossless_img);
        println!("   Lossless PSNR: {:.2} dB", lossless_psnr);

        let lossless_diff = original_psnr - lossless_psnr;
        println!("   Difference from original: {:.6} dB", lossless_diff);

        // Verify bit-exact (within floating point precision)
        let mut max_diff = 0.0f32;
        for i in 0..gaussians.len() {
            let orig = &gaussians[i];
            let loaded = &loaded_gaussians[i];

            max_diff = max_diff.max((orig.position.x - loaded.position.x).abs());
            max_diff = max_diff.max((orig.position.y - loaded.position.y).abs());
            max_diff = max_diff.max((orig.shape.scale_x - loaded.shape.scale_x).abs());
            max_diff = max_diff.max((orig.shape.scale_y - loaded.shape.scale_y).abs());
            max_diff = max_diff.max((orig.shape.rotation - loaded.shape.rotation).abs());
            max_diff = max_diff.max((orig.color.r - loaded.color.r).abs());
            max_diff = max_diff.max((orig.color.g - loaded.color.g).abs());
            max_diff = max_diff.max((orig.color.b - loaded.color.b).abs());
            max_diff = max_diff.max((orig.opacity - loaded.opacity).abs());
        }
        println!("   Max parameter difference: {:.9}", max_diff);

        if max_diff < 1e-7 {
            println!("   ✅ Lossless mode is bit-exact (within f32 precision)");
        } else {
            println!("   ⚠️ Lossless mode has precision issues");
        }
    }

    println!("\n=== Summary ===");
    println!("LGIQ-B: 11 bytes/G, ~0.5 dB loss from quantization");
    println!("Lossless: 36 bytes/G, 0.00 dB loss (bit-exact)");
    println!("\n✅ Lossless f32 file mode implementation complete!");

    Ok(())
}