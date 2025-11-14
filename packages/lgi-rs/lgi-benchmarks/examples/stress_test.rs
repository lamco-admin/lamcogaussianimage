//! Stress test to identify breaking points and bottlenecks

use lgi_benchmarks::test_images::{TestImageGenerator, TestPattern};
use lgi_encoder::{Encoder, EncoderConfig};
use lgi_core::Renderer;
use lgi_benchmarks::compute_psnr;
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║   LGI Codec Stress Test                       ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // Test 1: Maximum Gaussian count
    println!("Test 1: Maximum Gaussian Count");
    println!("==================================================");

    for count in [1000, 2000, 5000, 10000].iter() {
        println!("\nTrying {} Gaussians...", count);

        let gen = TestImageGenerator::new(256);
        let test_image = gen.generate(TestPattern::NaturalScene);

        let encoder = Encoder::with_config(EncoderConfig::fast());
        let start = Instant::now();

        match encoder.encode(&test_image, *count) {
            Ok(gaussians) => {
                let encode_time = start.elapsed();
                println!("  ✓ Success: {:.2}s", encode_time.as_secs_f32());
                println!("    Gaussians/sec: {:.0}", *count as f32 / encode_time.as_secs_f32());

                // Try rendering
                let render_start = Instant::now();
                let renderer = Renderer::new();
                match renderer.render(&gaussians, 256, 256) {
                    Ok(rendered) => {
                        let render_time = render_start.elapsed();
                        println!("    Render: {:.3}s ({:.1} FPS)", render_time.as_secs_f32(), 1.0 / render_time.as_secs_f32());

                        let psnr = compute_psnr(&test_image, &rendered);
                        println!("    PSNR: {:.2} dB", psnr);
                    }
                    Err(e) => println!("  ✗ Render failed: {}", e),
                }
            }
            Err(e) => println!("  ✗ Failed: {}", e),
        }
    }

    // Test 2: Maximum resolution
    println!("\n\nTest 2: Maximum Resolution");
    println!("==================================================");

    for size in [256, 512, 1024].iter() {
        println!("\nTrying {}×{} resolution...", size, size);

        let gen = TestImageGenerator::new(*size);
        let test_image = gen.generate(TestPattern::LinearGradient);

        let encoder = Encoder::with_config(EncoderConfig::fast());
        let start = Instant::now();

        match encoder.encode(&test_image, 500) {
            Ok(gaussians) => {
                let encode_time = start.elapsed();
                println!("  ✓ Encode: {:.2}s", encode_time.as_secs_f32());

                let renderer = Renderer::new();
                let render_start = Instant::now();

                match renderer.render(&gaussians, *size, *size) {
                    Ok(rendered) => {
                        let render_time = render_start.elapsed();
                        let megapixels = (size * size) as f32 / 1_000_000.0;
                        println!("    Render: {:.3}s ({:.1} FPS, {:.2} Mpix/s)",
                            render_time.as_secs_f32(),
                            1.0 / render_time.as_secs_f32(),
                            megapixels / render_time.as_secs_f32());

                        let psnr = compute_psnr(&test_image, &rendered);
                        println!("    PSNR: {:.2} dB", psnr);
                    }
                    Err(e) => println!("  ✗ Render failed: {}", e),
                }
            }
            Err(e) => println!("  ✗ Encode failed: {}", e),
        }
    }

    // Test 3: Difficult patterns
    println!("\n\nTest 3: Difficult Patterns");
    println!("==================================================");

    let difficult_patterns = vec![
        ("Random Noise", TestPattern::RandomNoise),
        ("Checkerboard", TestPattern::Checkerboard),
        ("Frequency Sweep", TestPattern::FrequencySweep),
        ("Text Pattern", TestPattern::TextPattern),
    ];

    for (name, pattern) in difficult_patterns {
        println!("\nPattern: {}", name);

        let gen = TestImageGenerator::new(256);
        let test_image = gen.generate(pattern);

        let encoder = Encoder::with_config(EncoderConfig::fast());

        match encoder.encode(&test_image, 500) {
            Ok(gaussians) => {
                let renderer = Renderer::new();

                if let Ok(rendered) = renderer.render(&gaussians, 256, 256) {
                    let psnr = compute_psnr(&test_image, &rendered);
                    println!("  PSNR: {:.2} dB", psnr);

                    if psnr < 15.0 {
                        println!("  ⚠️  Warning: Low PSNR! Pattern may be difficult for Gaussian representation.");
                    }
                } else {
                    println!("  ✗ Render failed");
                }
            }
            Err(e) => println!("  ✗ Encode failed: {}", e),
        }
    }

    // Test 4: Resolution independence
    println!("\n\nTest 4: Resolution Independence");
    println!("==================================================");

    let gen = TestImageGenerator::new(256);
    let test_image_256 = gen.generate(TestPattern::NaturalScene);

    let encoder = Encoder::with_config(EncoderConfig::balanced());
    println!("\nEncoding at 256×256 with 1000 Gaussians...");

    match encoder.encode(&test_image_256, 1000) {
        Ok(gaussians) => {
            println!("  ✓ Encoded successfully");

            let renderer = Renderer::new();

            // Render at various resolutions
            for size in [128, 256, 512, 1024].iter() {
                println!("\n  Rendering at {}×{}...", size, size);

                match renderer.render(&gaussians, *size, *size) {
                    Ok(rendered) => {
                        let start = Instant::now();
                        let _ = renderer.render(&gaussians, *size, *size).unwrap();
                        let time = start.elapsed();

                        println!("    Time: {:.3}s ({:.1} FPS)", time.as_secs_f32(), 1.0 / time.as_secs_f32());
                        println!("    Mpix/s: {:.2}", (size * size) as f32 / 1_000_000.0 / time.as_secs_f32());
                    }
                    Err(e) => println!("    ✗ Failed: {}", e),
                }
            }
        }
        Err(e) => println!("  ✗ Encode failed: {}", e),
    }

    println!("\n\n╔════════════════════════════════════════════════╗");
    println!("║   Stress Test Complete                        ║");
    println!("╚════════════════════════════════════════════════╝");
}
