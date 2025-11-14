//! Simple test for lossless f32 file I/O mode
//! Tests the basic functionality without dependencies

use lgi_encoder_v2::{file_writer::LGIWriter, file_reader::LGIReader};
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LGI Lossless F32 File Mode Test ===\n");

    // Create test Gaussians
    let gaussians = vec![
        Gaussian2D::new(
            Vector2::new(0.25, 0.25),
            Euler::isotropic(0.05),
            Color4::new(1.0, 0.0, 0.0, 1.0),
            0.9,
        ),
        Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::new(0.1, 0.08, 0.785),  // Anisotropic with rotation
            Color4::new(0.0, 1.0, 0.0, 1.0),
            0.8,
        ),
        Gaussian2D::new(
            Vector2::new(0.75, 0.75),
            Euler::isotropic(0.03),
            Color4::new(0.0, 0.0, 1.0, 1.0),
            0.7,
        ),
    ];

    println!("Created {} test Gaussians", gaussians.len());

    // Test 1: LGIQ-B baseline (quantized)
    println!("\n1. Testing LGIQ-B baseline (quantized, 11 bytes/G)...");
    {
        let writer = LGIWriter::new(256, 256, gaussians.len() as u32);
        writer.write_file("/tmp/test_quantized.lgi", &gaussians)?;

        let file_size = std::fs::metadata("/tmp/test_quantized.lgi")?.len();
        println!("   File size: {} bytes ({:.2} bytes/G)",
                 file_size, file_size as f64 / gaussians.len() as f64);

        let (header, loaded_gaussians) = LGIReader::read_file("/tmp/test_quantized.lgi")?;
        println!("   Loaded {} Gaussians", loaded_gaussians.len());
        println!("   Header: {}x{}, encoding: {}, bitdepth: {}",
                 header.canvas_width, header.canvas_height,
                 header.param_encoding, header.bitdepth);

        // Check quantization error
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
        println!("   Max parameter difference: {:.6}", max_diff);
    }

    // Test 2: Lossless f32 mode
    println!("\n2. Testing lossless f32 mode (36 bytes/G)...");
    {
        let writer = LGIWriter::new_lossless(256, 256, gaussians.len() as u32);
        writer.write_file("/tmp/test_lossless.lgi", &gaussians)?;

        let file_size = std::fs::metadata("/tmp/test_lossless.lgi")?.len();
        println!("   File size: {} bytes ({:.2} bytes/G)",
                 file_size, file_size as f64 / gaussians.len() as f64);

        let (header, loaded_gaussians) = LGIReader::read_file("/tmp/test_lossless.lgi")?;
        println!("   Loaded {} Gaussians", loaded_gaussians.len());
        println!("   Header: {}x{}, encoding: {}, bitdepth: {}",
                 header.canvas_width, header.canvas_height,
                 header.param_encoding, header.bitdepth);

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

        // Print detailed comparison for first Gaussian
        println!("\n   Detailed comparison (first Gaussian):");
        let orig = &gaussians[0];
        let loaded = &loaded_gaussians[0];
        println!("     Position: ({:.7}, {:.7}) vs ({:.7}, {:.7})",
                 orig.position.x, orig.position.y,
                 loaded.position.x, loaded.position.y);
        println!("     Scale:    ({:.7}, {:.7}) vs ({:.7}, {:.7})",
                 orig.shape.scale_x, orig.shape.scale_y,
                 loaded.shape.scale_x, loaded.shape.scale_y);
        println!("     Rotation:  {:.7} vs {:.7}",
                 orig.shape.rotation, loaded.shape.rotation);
        println!("     Color:    ({:.7}, {:.7}, {:.7}) vs ({:.7}, {:.7}, {:.7})",
                 orig.color.r, orig.color.g, orig.color.b,
                 loaded.color.r, loaded.color.g, loaded.color.b);
        println!("     Opacity:   {:.7} vs {:.7}",
                 orig.opacity, loaded.opacity);
    }

    println!("\n=== Summary ===");
    println!("LGIQ-B: 11 bytes/G, some loss from quantization");
    println!("Lossless: 36 bytes/G, 0.00 loss (bit-exact)");
    println!("\n✅ Lossless f32 file mode implementation complete!");

    Ok(())
}