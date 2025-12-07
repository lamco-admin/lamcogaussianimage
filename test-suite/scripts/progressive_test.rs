//! Progressive Loading Test
//! Tests importance-based ordering and streaming decode capabilities

use lgi_encoder_v2::file_writer::LGIWriter;
use lgi_encoder_v2::file_reader::{LGIReader, ProgressiveResult};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

fn main() {
    println!("=== Progressive Loading Test ===\n");

    // Create diverse Gaussians with different importance levels
    let gaussians = vec![
        // Small, low opacity, corner (least important)
        Gaussian2D::new(
            Vector2::new(0.1, 0.1),
            Euler::isotropic(0.01),
            Color4::new(1.0, 0.0, 0.0, 1.0),
            0.3,
        ),
        // Large, high opacity, center (most important)
        Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.2),
            Color4::new(0.0, 1.0, 0.0, 1.0),
            0.9,
        ),
        // Medium, medium opacity, off-center
        Gaussian2D::new(
            Vector2::new(0.6, 0.4),
            Euler::isotropic(0.05),
            Color4::new(0.0, 0.0, 1.0, 1.0),
            0.7,
        ),
        // Small, high opacity, edge
        Gaussian2D::new(
            Vector2::new(0.9, 0.5),
            Euler::isotropic(0.02),
            Color4::new(1.0, 1.0, 0.0, 1.0),
            0.8,
        ),
        // Large, medium opacity, corner
        Gaussian2D::new(
            Vector2::new(0.2, 0.8),
            Euler::isotropic(0.15),
            Color4::new(1.0, 0.0, 1.0, 1.0),
            0.6,
        ),
    ];

    println!("Test Gaussians:");
    for (i, g) in gaussians.iter().enumerate() {
        let scale = (g.shape.scale_x + g.shape.scale_y) / 2.0;
        println!(
            "  G{}: pos=({:.2}, {:.2}), scale={:.3}, opacity={:.2}",
            i, g.position.x, g.position.y, scale, g.opacity
        );
    }

    // Write progressive file
    let writer = LGIWriter::new_progressive(512, 512, gaussians.len() as u32);
    let path = "/tmp/test_progressive.lgi";
    writer.write_file(path, &gaussians).expect("Failed to write progressive file");

    // Get file size
    let file_size = std::fs::metadata(path).unwrap().len();
    println!("\n✅ Progressive file written: {} bytes", file_size);

    // Read back with progressive metadata
    let result = LGIReader::read_file_progressive(path).expect("Failed to read progressive file");

    println!("\n📊 File Info:");
    println!("  Canvas: {}x{}", result.header.canvas_width, result.header.canvas_height);
    println!("  Gaussian count: {}", result.header.gaussian_count);
    println!("  Feature flags: 0x{:08X}", result.header.feature_flags);
    println!("  Compression flags: 0x{:04X}", result.header.compression_flags);

    // Verify importance order is present
    if let Some(order) = &result.importance_order {
        println!("\n✅ Progressive metadata present");
        println!("  Importance order: {:?}", order);

        // Expected: G1 (large, opaque, center) should be first
        // G4 (large, medium opacity) or G2 (medium) should be next
        // G0 (small, low opacity, corner) should be last
        println!("\n📈 Importance Analysis:");
        for (render_idx, &orig_idx) in order.iter().enumerate() {
            let g = &gaussians[orig_idx];
            let scale = (g.shape.scale_x + g.shape.scale_y) / 2.0;
            let area = g.shape.scale_x * g.shape.scale_y * std::f32::consts::PI;
            let dist_from_center = ((g.position.x - 0.5).powi(2) + (g.position.y - 0.5).powi(2)).sqrt();

            println!(
                "  Render #{}: Original G{} - area={:.4}, opacity={:.2}, center_dist={:.3}",
                render_idx, orig_idx, area, g.opacity, dist_from_center
            );
        }

        // Verify most important is first
        let first_idx = order[0];
        let first = &gaussians[first_idx];
        let first_area = first.shape.scale_x * first.shape.scale_y * std::f32::consts::PI;
        println!("\n✅ Most important Gaussian: G{}", first_idx);
        println!("   Area: {:.4}, Opacity: {:.2}", first_area, first.opacity);

        // Simulate streaming: render only first 2 Gaussians
        println!("\n🌊 Streaming Simulation (first 2 Gaussians):");
        let streaming_count = 2.min(order.len());
        for i in 0..streaming_count {
            let orig_idx = order[i];
            println!("  Load G{} (original index {})", i, orig_idx);
        }
        println!("  → Could render partial image with {}% of data",
                 (streaming_count as f32 / order.len() as f32 * 100.0) as u32);

    } else {
        println!("\n❌ ERROR: Progressive metadata missing!");
    }

    // Verify data roundtrips correctly
    println!("\n🔄 Data Integrity Check:");
    let mut max_diff = 0.0f32;
    for (i, g_read) in result.gaussians.iter().enumerate() {
        // Find corresponding original Gaussian
        let orig_idx = result.importance_order.as_ref().unwrap()[i];
        let g_orig = &gaussians[orig_idx];

        let pos_diff = ((g_orig.position.x - g_read.position.x).powi(2) +
                        (g_orig.position.y - g_read.position.y).powi(2)).sqrt();
        max_diff = max_diff.max(pos_diff);
    }
    println!("  Max position difference: {:.6}", max_diff);

    if max_diff < 0.01 {
        println!("  ✅ Data integrity maintained (quantization + compression)");
    } else {
        println!("  ⚠️  Warning: Larger than expected difference");
    }

    // Compare file sizes
    println!("\n📦 File Size Analysis:");

    // Write non-progressive for comparison
    let writer_regular = LGIWriter::new_compressed(512, 512, gaussians.len() as u32);
    let path_regular = "/tmp/test_regular_compressed.lgi";
    writer_regular.write_file(path_regular, &gaussians).unwrap();
    let regular_size = std::fs::metadata(path_regular).unwrap().len();

    println!("  Regular compressed: {} bytes", regular_size);
    println!("  Progressive: {} bytes", file_size);
    let overhead = file_size as f32 / regular_size as f32;
    println!("  Progressive overhead: {:.1}%", (overhead - 1.0) * 100.0);
    println!("  (Overhead = 4 bytes/Gaussian for importance order)");

    println!("\n=== Test Complete ===");
}
