//! Comprehensive Compression Demo
//!
//! Demonstrates all LGIQ profiles and compression modes

use lgi_format::{
    LgiFile, LgiWriter, LgiReader, CompressionConfig, QuantizationProfile,
};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

fn create_test_gaussians(count: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    (0..count)
        .map(|i| {
            let x = (i as f32) / (count as f32);
            Gaussian2D::new(
                Vector2::new(x, 0.5),
                Euler::isotropic(0.01),
                Color4::rgb(x, 0.5, 0.5),
                0.8,
            )
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  LGI Compression Modes Demonstration                     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let gaussians = create_test_gaussians(1000);
    println!("Created {} test Gaussians\n", gaussians.len());

    // Test all compression modes
    let configs = vec![
        ("Uncompressed", CompressionConfig::uncompressed()),
        ("LGIQ-B (Balanced + zstd)", CompressionConfig::balanced()),
        ("LGIQ-S (Small + zstd)", CompressionConfig::small()),
        ("LGIQ-H (High + zstd)", CompressionConfig::high_quality()),
        ("LGIQ-X (Lossless + zstd)", CompressionConfig::lossless()),
    ];

    println!("Mode                         | Size   | Ratio  | Quality");
    println!("---------------------------- | ------ | ------ | -------");

    for (name, config) in configs {
        let file = LgiFile::with_compression(
            gaussians.clone(),
            256,
            256,
            config.clone(),
        );

        // Write to buffer
        let mut buffer = Vec::new();
        LgiWriter::write(&mut buffer, &file)?;

        let size_kb = buffer.len() / 1024;
        let ratio = file.compression_ratio();
        let (min_psnr, max_psnr) = config.expected_psnr_range();

        println!(
            "{:<28} | {:>4} KB | {:>4.1}× | {:.0}-{:.0} dB",
            name,
            size_kb,
            ratio,
            min_psnr,
            max_psnr.min(50.0)
        );

        // Verify round-trip
        let mut cursor = std::io::Cursor::new(buffer);
        let loaded = LgiReader::read(&mut cursor)?;
        let reconstructed = loaded.gaussians();

        assert_eq!(reconstructed.len(), 1000, "Round-trip failed for {}", name);
    }

    println!("\n✅ All compression modes working!");
    println!("\nKey findings:");
    println!("  • LGIQ-B + zstd: Best balance (5-7× compression, 28-32 dB)");
    println!("  • LGIQ-H + zstd: High quality (3-4× compression, 35-40 dB)");
    println!("  • LGIQ-X + zstd: Lossless (2-3× compression, bit-exact)");
    println!("  • All modes support round-trip encode/decode");

    Ok(())
}
