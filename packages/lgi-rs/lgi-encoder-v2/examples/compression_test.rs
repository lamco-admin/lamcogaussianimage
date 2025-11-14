//! Test compression integration (zstd + delta coding)
//! Demonstrates and tests the compression capability

use lgi_encoder_v2::{file_writer::LGIWriter, file_reader::LGIReader};
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LGI Compression Integration Test ===\n");

    // Create test Gaussians with spatial coherence (good for delta encoding)
    let mut gaussians = Vec::new();

    // Create a grid of Gaussians (good spatial coherence for delta encoding)
    for y in 0..10 {
        for x in 0..10 {
            gaussians.push(Gaussian2D::new(
                Vector2::new((x as f32) * 0.1 + 0.05, (y as f32) * 0.1 + 0.05),
                Euler::isotropic(0.02),
                Color4::new(
                    (x as f32) / 10.0,
                    (y as f32) / 10.0,
                    1.0 - ((x + y) as f32) / 20.0,
                    1.0,
                ),
                0.9,
            ));
        }
    }

    println!("Created {} test Gaussians in grid pattern", gaussians.len());

    // Test 1: Uncompressed baseline (LGIQ-B, no compression)
    println!("\n1. Testing uncompressed baseline...");
    {
        let writer = LGIWriter::new(512, 512, gaussians.len() as u32);
        writer.write_file("/tmp/test_uncompressed.lgi", &gaussians)?;

        let file_size = std::fs::metadata("/tmp/test_uncompressed.lgi")?.len();
        println!("   File size: {} bytes ({:.2} bytes/G)",
                 file_size, file_size as f64 / gaussians.len() as f64);

        let (header, loaded_gaussians) = LGIReader::read_file("/tmp/test_uncompressed.lgi")?;
        println!("   Loaded {} Gaussians", loaded_gaussians.len());
        println!("   Header: compression_flags = {}", header.compression_flags);

        // Verify data integrity
        let mut max_diff = 0.0f32;
        for i in 0..gaussians.len() {
            let orig = &gaussians[i];
            let loaded = &loaded_gaussians[i];
            max_diff = max_diff.max((orig.position.x - loaded.position.x).abs());
            max_diff = max_diff.max((orig.position.y - loaded.position.y).abs());
        }
        println!("   Max position difference: {:.6}", max_diff);
    }

    // Test 2: Compressed (LGIQ-B + zstd + delta)
    println!("\n2. Testing compressed (zstd + delta encoding)...");
    {
        let writer = LGIWriter::new_compressed(512, 512, gaussians.len() as u32);
        writer.write_file("/tmp/test_compressed.lgi", &gaussians)?;

        let file_size = std::fs::metadata("/tmp/test_compressed.lgi")?.len();
        println!("   File size: {} bytes ({:.2} bytes/G)",
                 file_size, file_size as f64 / gaussians.len() as f64);

        let (header, loaded_gaussians) = LGIReader::read_file("/tmp/test_compressed.lgi")?;
        println!("   Loaded {} Gaussians", loaded_gaussians.len());
        println!("   Header: compression_flags = {} (zstd={}, delta={})",
                 header.compression_flags,
                 header.compression_flags & 1, // Zstd flag
                 header.compression_flags & 16); // Delta flag

        // Verify data integrity after compression
        let mut max_diff = 0.0f32;
        for i in 0..gaussians.len() {
            let orig = &gaussians[i];
            let loaded = &loaded_gaussians[i];
            max_diff = max_diff.max((orig.position.x - loaded.position.x).abs());
            max_diff = max_diff.max((orig.position.y - loaded.position.y).abs());
        }
        println!("   Max position difference: {:.6}", max_diff);
    }

    // Test 3: Lossless f32 mode (for comparison)
    println!("\n3. Testing lossless f32 (for size comparison)...");
    {
        let writer = LGIWriter::new_lossless(512, 512, gaussians.len() as u32);
        writer.write_file("/tmp/test_lossless.lgi", &gaussians)?;

        let file_size = std::fs::metadata("/tmp/test_lossless.lgi")?.len();
        println!("   File size: {} bytes ({:.2} bytes/G)",
                 file_size, file_size as f64 / gaussians.len() as f64);
    }

    // Calculate compression ratios
    println!("\n=== Compression Results ===");

    let uncompressed_size = std::fs::metadata("/tmp/test_uncompressed.lgi")?.len();
    let compressed_size = std::fs::metadata("/tmp/test_compressed.lgi")?.len();
    let lossless_size = std::fs::metadata("/tmp/test_lossless.lgi")?.len();

    let compression_ratio = uncompressed_size as f64 / compressed_size as f64;
    let size_reduction = 100.0 * (1.0 - compressed_size as f64 / uncompressed_size as f64);

    println!("Uncompressed: {} bytes", uncompressed_size);
    println!("Compressed:   {} bytes", compressed_size);
    println!("Lossless:     {} bytes", lossless_size);
    println!();
    println!("Compression ratio: {:.2}x", compression_ratio);
    println!("Size reduction: {:.1}%", size_reduction);

    if compression_ratio > 1.5 {
        println!("\n✅ Compression working effectively!");
    } else {
        println!("\n⚠️ Compression ratio lower than expected");
    }

    Ok(())
}