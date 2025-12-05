//! Test isotropic vs anisotropic edge representation
//!
//! Validates quantum discovery: Small isotropic Gaussians work better than
//! elongated anisotropic ones for edge representation.
//!
//! Quantum channels 3, 4, 7 (high quality) all show σ_x ≈ σ_y despite
//! high edge coherence (>0.96).

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2};
use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    env_logger::init();

    println!("{}", "=".repeat(80));
    println!("ISOTROPIC EDGE VALIDATION");
    println!("Testing Quantum Discovery: σ_x = σ_y for edges");
    println!("{}", "=".repeat(80));
    println!();

    // Test images
    let test_images = vec![
        ("kodim03", "Architecture with sharp edges"),
        ("kodim05", "Building with many edges"),
        ("kodim08", "Portrait (soft + sharp edges)"),
        ("kodim15", "Outdoor scene"),
        ("kodim23", "Garden with texture"),
    ];

    let kodak_dir = PathBuf::from("../../test-data/kodak-dataset");

    let mut results = Vec::new();

    for (idx, (image_id, description)) in test_images.iter().enumerate() {
        println!();
        println!("{}", "=".repeat(80));
        println!("[{}/{}] {}.png - {}", idx + 1, test_images.len(), image_id, description);
        println!("{}", "=".repeat(80));
        println!();

        let image_path = kodak_dir.join(format!("{}.png", image_id));

        let image = match load_png(&image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("  ERROR loading image: {}", e);
                continue;
            }
        };

        println!("Image: {}×{}", image.width, image.height);
        println!();

        // Test 1: Anisotropic (current method)
        println!("  [1/2] Anisotropic (current) - elongated edges...");
        let encoder_aniso = EncoderV2::new(image.clone()).expect("Failed to create encoder");

        let start = Instant::now();
        let gaussians_aniso = encoder_aniso.encode_error_driven_adam(25, 500);
        let time_aniso = start.elapsed();

        let rendered_aniso = RendererV2::render(&gaussians_aniso, image.width, image.height);
        let psnr_aniso = compute_psnr(&image, &rendered_aniso);

        println!("    ✓ PSNR: {:.2} dB | N: {} | Time: {:.1}s",
            psnr_aniso, gaussians_aniso.len(), time_aniso.as_secs_f32());

        // Test 2: Isotropic (quantum-guided)
        println!("  [2/2] Isotropic (quantum-guided) - small round Gaussians...");
        let encoder_iso = EncoderV2::new(image.clone()).expect("Failed to create encoder");

        let start = Instant::now();
        let gaussians_iso = encoder_iso.encode_error_driven_adam_isotropic(25, 500);
        let time_iso = start.elapsed();

        let rendered_iso = RendererV2::render(&gaussians_iso, image.width, image.height);
        let psnr_iso = compute_psnr(&image, &rendered_iso);

        println!("    ✓ PSNR: {:.2} dB | N: {} | Time: {:.1}s",
            psnr_iso, gaussians_iso.len(), time_iso.as_secs_f32());

        // Comparison
        let improvement = psnr_iso - psnr_aniso;
        let improvement_pct = 100.0 * improvement / psnr_aniso;

        println!();
        println!("  COMPARISON:");
        println!("    Anisotropic: {:.2} dB", psnr_aniso);
        println!("    Isotropic:   {:.2} dB", psnr_iso);
        if improvement > 0.0 {
            println!("    Improvement: +{:.2} dB ({:+.1}%) ⭐", improvement, improvement_pct);
        } else {
            println!("    Change:      {:.2} dB ({:+.1}%)", improvement, improvement_pct);
        }

        results.push((image_id.to_string(), psnr_aniso, psnr_iso, improvement));
    }

    // Summary
    println!();
    println!("{}", "=".repeat(80));
    println!("SUMMARY - ISOTROPIC vs ANISOTROPIC EDGES");
    println!("{}", "=".repeat(80));
    println!();

    println!("{:12} | {:>12} | {:>12} | {:>12} | {:>10}",
        "Image", "Anisotropic", "Isotropic", "Improvement", "Status");
    println!("{}", "-".repeat(80));

    let mut total_aniso = 0.0;
    let mut total_iso = 0.0;
    let mut wins = 0;
    let mut losses = 0;

    for (image_id, psnr_aniso, psnr_iso, improvement) in &results {
        let status = if *improvement > 1.0 {
            wins += 1;
            "WIN ⭐"
        } else if *improvement > 0.1 {
            wins += 1;
            "win"
        } else if *improvement < -0.1 {
            losses += 1;
            "loss"
        } else {
            "tie"
        };

        println!("{:12} | {:>10.2} dB | {:>10.2} dB | {:>10.2} dB | {:>10}",
            image_id, psnr_aniso, psnr_iso, improvement, status);

        total_aniso += psnr_aniso;
        total_iso += psnr_iso;
    }

    println!("{}", "-".repeat(80));

    let n = results.len() as f32;
    let avg_aniso = total_aniso / n;
    let avg_iso = total_iso / n;
    let avg_improvement = avg_iso - avg_aniso;

    println!("{:12} | {:>10.2} dB | {:>10.2} dB | {:>10.2} dB |",
        "AVERAGE", avg_aniso, avg_iso, avg_improvement);

    println!();
    println!("Win rate: {}/{} ({:.0}%)", wins, results.len(), 100.0 * wins as f32 / results.len() as f32);
    println!();

    // Interpretation
    println!("{}", "=".repeat(80));
    println!("INTERPRETATION");
    println!("{}", "=".repeat(80));
    println!();

    if avg_improvement > 3.0 {
        println!("🎯 QUANTUM DISCOVERY VALIDATED! ⭐⭐⭐");
        println!();
        println!("  Isotropic edges achieve {:.1} dB better PSNR than anisotropic.", avg_improvement);
        println!("  This confirms quantum channels 3, 4, 7 found the correct approach.");
        println!();
        println!("  RECOMMENDATION: Adopt isotropic edges as new standard immediately.");
        println!("  Edge representation problem is SOLVED using quantum guidance.");
    } else if avg_improvement > 1.0 {
        println!("✓ Quantum discovery shows promise");
        println!();
        println!("  Isotropic edges achieve {:.1} dB better PSNR.", avg_improvement);
        println!("  Modest but consistent improvement.");
        println!();
        println!("  RECOMMENDATION: Further investigation warranted.");
        println!("  Test on larger dataset and analyze per-edge-type.");
    } else if avg_improvement > -0.5 {
        println!("~ Results inconclusive");
        println!();
        println!("  Isotropic and anisotropic perform similarly ({:.2} dB difference).", avg_improvement);
        println!();
        println!("  RECOMMENDATION: Re-analyze quantum channels.");
        println!("  Shape might not be the key factor - investigate scale.");
    } else {
        println!("✗ Anisotropic still better");
        println!();
        println!("  Anisotropic achieves {:.1} dB better PSNR.", -avg_improvement);
        println!();
        println!("  RECOMMENDATION: Quantum channels need reinterpretation.");
        println!("  Isotropic in FINAL state doesn't mean isotropic INITIALIZATION works.");
    }

    println!();
    println!("{}", "=".repeat(80));
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
