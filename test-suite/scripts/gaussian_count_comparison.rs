//! Gaussian Count Determination Comparison
//!
//! Compares three strategies for determining initial Gaussian count:
//! 1. Arbitrary formula (current): sqrt(pixels) / 20
//! 2. Entropy-driven (existing): auto_gaussian_count()
//! 3. Entropy+Gradient hybrid (new): PPM-based
//!
//! Tests on Kodak dataset to see which gives best quality

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2};
use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use std::time::Instant;
use std::path::PathBuf;

fn main() {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("     GAUSSIAN COUNT STRATEGY COMPARISON - Session 8           ");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Testing three strategies for determining initial N:");
    println!("  1. ARBITRARY: sqrt(pixels) / 20");
    println!("  2. ENTROPY:   Based on image complexity (variance)");
    println!("  3. HYBRID:    Entropy (60%) + Gradient (40%)");
    println!();

    let kodak_dir = PathBuf::from("../../kodak-dataset");
    if !kodak_dir.exists() {
        eprintln!("ERROR: Kodak dataset not found at {:?}", kodak_dir);
        return;
    }

    let mut results = Vec::new();

    // Test on first 5 Kodak images
    for i in 1..=5 {
        let image_path = kodak_dir.join(format!("kodim{:02}.png", i));

        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║ kodim{:02}.png (768×512 = 393,216 pixels)                 ║", i);
        println!("╚═══════════════════════════════════════════════════════════╝\n");

        let image = match load_png(&image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("  ERROR loading image: {}", e);
                continue;
            }
        };

        let result = compare_strategies(&image, &format!("kodim{:02}", i));
        results.push(result);
    }

    // Print summary
    print_summary(&results);
}

struct ComparisonResult {
    name: String,
    arbitrary_n: usize,
    arbitrary_psnr: f32,
    arbitrary_time: std::time::Duration,

    entropy_n: usize,
    entropy_psnr: f32,
    entropy_time: std::time::Duration,

    hybrid_n: usize,
    hybrid_psnr: f32,
    hybrid_time: std::time::Duration,
}

fn compare_strategies(image: &ImageBuffer<f32>, image_name: &str) -> ComparisonResult {
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    let pixels = (image.width * image.height) as f32;

    // Strategy 1: Arbitrary formula
    println!("[1/3] ARBITRARY FORMULA...");
    let arbitrary_n = (pixels.sqrt() / 20.0) as usize;
    println!("  N = {} (formula: sqrt(pixels) / 20)", arbitrary_n);

    let start = Instant::now();
    let grid_size = (arbitrary_n as f32).sqrt().ceil() as u32;
    let arbitrary_gaussians = encoder.initialize_gaussians(grid_size);
    let arbitrary_psnr = compute_psnr(image, &arbitrary_gaussians);
    let arbitrary_time = start.elapsed();
    println!("  PSNR = {:.2} dB | Time = {:?}", arbitrary_psnr, arbitrary_time);

    // Strategy 2: Entropy-driven (existing)
    println!("\n[2/3] ENTROPY-DRIVEN (adaptive_gaussian_count)...");
    let entropy_n = encoder.auto_gaussian_count();
    println!("  N = {} (entropy-based)", entropy_n);

    let start = Instant::now();
    let grid_size = (entropy_n as f32).sqrt().ceil() as u32;
    let entropy_gaussians = encoder.initialize_gaussians(grid_size);
    let entropy_psnr = compute_psnr(image, &entropy_gaussians);
    let entropy_time = start.elapsed();
    println!("  PSNR = {:.2} dB ({:+.2} dB vs arbitrary) | Time = {:?}",
        entropy_psnr, entropy_psnr - arbitrary_psnr, entropy_time);

    // Strategy 3: Hybrid (Entropy + Gradient)
    println!("\n[3/3] HYBRID (60% entropy + 40% gradient)...");
    let hybrid_n = encoder.hybrid_gaussian_count();
    println!("  N = {} (hybrid)", hybrid_n);

    let start = Instant::now();
    let grid_size = (hybrid_n as f32).sqrt().ceil() as u32;
    let hybrid_gaussians = encoder.initialize_gaussians(grid_size);
    let hybrid_psnr = compute_psnr(image, &hybrid_gaussians);
    let hybrid_time = start.elapsed();
    println!("  PSNR = {:.2} dB ({:+.2} dB vs arbitrary) | Time = {:?}",
        hybrid_psnr, hybrid_psnr - arbitrary_psnr, hybrid_time);

    ComparisonResult {
        name: image_name.to_string(),
        arbitrary_n,
        arbitrary_psnr,
        arbitrary_time,
        entropy_n,
        entropy_psnr,
        entropy_time,
        hybrid_n,
        hybrid_psnr,
        hybrid_time,
    }
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

fn print_summary(results: &[ComparisonResult]) {
    println!("\n╔═════════════════════════════════════════════════════════════════════════════╗");
    println!("║ SUMMARY: Which strategy determines N best?                                 ║");
    println!("╠═════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Image     │ Arbitrary       │ Entropy         │ Hybrid          │ Winner    ║");
    println!("║           │ N    PSNR (dB)  │ N    PSNR (dB)  │ N    PSNR (dB)  │           ║");
    println!("╠═════════════════════════════════════════════════════════════════════════════╣");

    let mut arbitrary_wins = 0;
    let mut entropy_wins = 0;
    let mut hybrid_wins = 0;

    let mut sum_arbitrary_psnr = 0.0;
    let mut sum_entropy_psnr = 0.0;
    let mut sum_hybrid_psnr = 0.0;

    for r in results {
        // Determine winner
        let best_psnr = r.arbitrary_psnr.max(r.entropy_psnr).max(r.hybrid_psnr);
        let winner = if r.arbitrary_psnr == best_psnr {
            arbitrary_wins += 1;
            "Arbitrary"
        } else if r.entropy_psnr == best_psnr {
            entropy_wins += 1;
            "Entropy  "
        } else {
            hybrid_wins += 1;
            "Hybrid   "
        };

        println!("║ {:9} │ {:4} {:6.2}    │ {:4} {:6.2}    │ {:4} {:6.2}    │ {} ║",
            r.name,
            r.arbitrary_n, r.arbitrary_psnr,
            r.entropy_n, r.entropy_psnr,
            r.hybrid_n, r.hybrid_psnr,
            winner);

        sum_arbitrary_psnr += r.arbitrary_psnr;
        sum_entropy_psnr += r.entropy_psnr;
        sum_hybrid_psnr += r.hybrid_psnr;
    }

    let count = results.len() as f32;
    println!("╠═════════════════════════════════════════════════════════════════════════════╣");
    println!("║ AVERAGE   │      {:6.2}    │      {:6.2}    │      {:6.2}    │           ║",
        sum_arbitrary_psnr / count,
        sum_entropy_psnr / count,
        sum_hybrid_psnr / count);
    println!("╠═════════════════════════════════════════════════════════════════════════════╣");
    println!("║ WINS      │ {} images       │ {} images       │ {} images       │           ║",
        arbitrary_wins, entropy_wins, hybrid_wins);
    println!("╚═════════════════════════════════════════════════════════════════════════════╝");

    println!("\n📊 ANALYSIS:");

    // Best strategy
    if entropy_wins > arbitrary_wins && entropy_wins > hybrid_wins {
        println!("  ✅ ENTROPY-DRIVEN is best ({} wins, {:.2} dB average)",
            entropy_wins, sum_entropy_psnr / count);
        println!("     → Use encoder.auto_gaussian_count() instead of arbitrary formula!");
    } else if hybrid_wins > arbitrary_wins && hybrid_wins > entropy_wins {
        println!("  ✅ HYBRID is best ({} wins, {:.2} dB average)",
            hybrid_wins, sum_hybrid_psnr / count);
        println!("     → Implement entropy+gradient PPM for Gaussian placement!");
    } else {
        println!("  ⚠️  ARBITRARY formula performs surprisingly well ({} wins)",
            arbitrary_wins);
        println!("     → May indicate initialization matters less than optimization");
    }

    // Quality gains
    let entropy_gain = (sum_entropy_psnr - sum_arbitrary_psnr) / count;
    let hybrid_gain = (sum_hybrid_psnr - sum_arbitrary_psnr) / count;

    println!("\n  Average gain over arbitrary:");
    println!("    Entropy: {:+.2} dB", entropy_gain);
    println!("    Hybrid:  {:+.2} dB", hybrid_gain);

    // N comparison
    println!("\n  Gaussian count comparison:");
    for r in results {
        println!("    {}: Arbitrary={}, Entropy={} ({:+.0}%), Hybrid={} ({:+.0}%)",
            r.name,
            r.arbitrary_n,
            r.entropy_n,
            ((r.entropy_n as f32 / r.arbitrary_n as f32) - 1.0) * 100.0,
            r.hybrid_n,
            ((r.hybrid_n as f32 / r.arbitrary_n as f32) - 1.0) * 100.0);
    }

    println!("\n✅ Recommendation: {}",
        if entropy_gain > 0.5 || hybrid_gain > 0.5 {
            "ADOPT data-driven N determination immediately!"
        } else if entropy_gain > 0.0 || hybrid_gain > 0.0 {
            "Data-driven helps but gains are small. Still worth adopting."
        } else {
            "Surprising! Arbitrary formula competitive. Investigate further."
        });
}
