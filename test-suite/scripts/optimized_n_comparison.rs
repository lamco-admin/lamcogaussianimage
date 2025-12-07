//! Optimized Gaussian Count Comparison
//!
//! Critical test: Does entropy-based N advantage (+2.9 dB initialization) persist after optimization?
//!
//! Tests:
//! 1. Arbitrary N=31  + Adam 100 iters
//! 2. Entropy  N~2600 + Adam 100 iters  (WARNING: 80× slower!)
//! 3. Hybrid   N~1700 + Adam 100 iters  (50× slower)
//!
//! This answers: "Is better initialization worth the massive compute cost?"

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2};
use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use std::time::Instant;
use std::path::PathBuf;

fn main() {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("   OPTIMIZED N COMPARISON - Does Better Init Help After Adam?  ");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("WARNING: This benchmark is SLOW!");
    println!("  Arbitrary (N=31):  ~30s per image");
    println!("  Entropy (N~2600):  ~45 min per image (80× slower!)");
    println!("  Hybrid (N~1700):   ~30 min per image (50× slower!)");
    println!();
    println!("Testing on 2 Kodak images only...");
    println!();

    let kodak_dir = PathBuf::from("../../kodak-dataset");
    if !kodak_dir.exists() {
        eprintln!("ERROR: Kodak dataset not found at {:?}", kodak_dir);
        return;
    }

    let mut results = Vec::new();

    // Test on just 2 images (not 5) to keep runtime reasonable
    for i in [2, 3] {  // kodim02, kodim03
        let image_path = kodak_dir.join(format!("kodim{:02}.png", i));

        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║ kodim{:02}.png (768×512)                                   ║", i);
        println!("╚═══════════════════════════════════════════════════════════╝\n");

        let image = match load_png(&image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("  ERROR loading image: {}", e);
                continue;
            }
        };

        let result = compare_optimized(&image, &format!("kodim{:02}", i));
        results.push(result);
    }

    // Print summary
    print_summary(&results);
}

struct OptimizedResult {
    name: String,

    arb_init_psnr: f32,
    arb_final_psnr: f32,
    arb_n: usize,
    arb_time: std::time::Duration,

    ent_init_psnr: f32,
    ent_final_psnr: f32,
    ent_n: usize,
    ent_time: std::time::Duration,

    hyb_init_psnr: f32,
    hyb_final_psnr: f32,
    hyb_n: usize,
    hyb_time: std::time::Duration,
}

fn compare_optimized(image: &ImageBuffer<f32>, image_name: &str) -> OptimizedResult {
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");
    let pixels = (image.width * image.height) as f32;

    // Strategy 1: Arbitrary (N=31)
    println!("[1/3] ARBITRARY (N=31) + Adam 100 iters...");
    let arb_n = (pixels.sqrt() / 20.0) as usize;
    let grid_size = (arb_n as f32).sqrt().ceil() as u32;

    let arb_init = encoder.initialize_gaussians(grid_size);
    let arb_init_psnr = compute_psnr(image, &arb_init);
    println!("  Init: N={}, PSNR={:.2} dB", arb_init.len(), arb_init_psnr);

    let start = Instant::now();
    let arb_final = encoder.encode_error_driven_adam(arb_n, arb_n * 4);
    let arb_time = start.elapsed();
    let arb_final_psnr = compute_psnr(image, &arb_final);
    println!("  Final: N={}, PSNR={:.2} dB ({:+.2} dB gain) | Time={:?}",
        arb_final.len(), arb_final_psnr, arb_final_psnr - arb_init_psnr, arb_time);

    // Strategy 2: Entropy (N~2600)
    println!("\n[2/3] ENTROPY (N~2600) + Adam 100 iters... (WARNING: SLOW!)");
    let ent_n = encoder.auto_gaussian_count();
    let grid_size = (ent_n as f32).sqrt().ceil() as u32;

    let ent_init = encoder.initialize_gaussians(grid_size);
    let ent_init_psnr = compute_psnr(image, &ent_init);
    println!("  Init: N={}, PSNR={:.2} dB", ent_init.len(), ent_init_psnr);

    println!("  Optimizing (this will take ~30-45 minutes)...");
    let start = Instant::now();
    let ent_final = encoder.encode_error_driven_adam(ent_n, ent_n * 4);
    let ent_time = start.elapsed();
    let ent_final_psnr = compute_psnr(image, &ent_final);
    println!("  Final: N={}, PSNR={:.2} dB ({:+.2} dB gain) | Time={:?}",
        ent_final.len(), ent_final_psnr, ent_final_psnr - ent_init_psnr, ent_time);

    // Strategy 3: Hybrid (N~1700)
    println!("\n[3/3] HYBRID (N~1700) + Adam 100 iters... (WARNING: SLOW!)");
    let hyb_n = encoder.hybrid_gaussian_count();
    let grid_size = (hyb_n as f32).sqrt().ceil() as u32;

    let hyb_init = encoder.initialize_gaussians(grid_size);
    let hyb_init_psnr = compute_psnr(image, &hyb_init);
    println!("  Init: N={}, PSNR={:.2} dB", hyb_init.len(), hyb_init_psnr);

    println!("  Optimizing (this will take ~20-30 minutes)...");
    let start = Instant::now();
    let hyb_final = encoder.encode_error_driven_adam(hyb_n, hyb_n * 4);
    let hyb_time = start.elapsed();
    let hyb_final_psnr = compute_psnr(image, &hyb_final);
    println!("  Final: N={}, PSNR={:.2} dB ({:+.2} dB gain) | Time={:?}",
        hyb_final.len(), hyb_final_psnr, hyb_final_psnr - hyb_init_psnr, hyb_time);

    OptimizedResult {
        name: image_name.to_string(),

        arb_init_psnr,
        arb_final_psnr,
        arb_n: arb_final.len(),
        arb_time,

        ent_init_psnr,
        ent_final_psnr,
        ent_n: ent_final.len(),
        ent_time,

        hyb_init_psnr,
        hyb_final_psnr,
        hyb_n: hyb_final.len(),
        hyb_time,
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

fn print_summary(results: &[OptimizedResult]) {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ OPTIMIZED COMPARISON: Does Better N Help After Optimization?                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Image    │ Arbitrary (N~31)  │ Entropy (N~2600)   │ Hybrid (N~1700)    │ Winner║");
    println!("║          │ Init→Final (Gain) │ Init→Final (Gain)  │ Init→Final (Gain)  │       ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");

    let mut arb_wins = 0;
    let mut ent_wins = 0;
    let mut hyb_wins = 0;

    let mut sum_arb_final = 0.0;
    let mut sum_ent_final = 0.0;
    let mut sum_hyb_final = 0.0;

    for r in results {
        let best_psnr = r.arb_final_psnr.max(r.ent_final_psnr).max(r.hyb_final_psnr);

        let winner = if r.arb_final_psnr == best_psnr {
            arb_wins += 1;
            "Arb  "
        } else if r.ent_final_psnr == best_psnr {
            ent_wins += 1;
            "Ent  "
        } else {
            hyb_wins += 1;
            "Hyb  "
        };

        let arb_gain = r.arb_final_psnr - r.arb_init_psnr;
        let ent_gain = r.ent_final_psnr - r.ent_init_psnr;
        let hyb_gain = r.hyb_final_psnr - r.hyb_init_psnr;

        println!("║ {:8} │ {:4.1}→{:5.2} ({:+.1}) │ {:4.1}→{:5.2} ({:+.1})  │ {:4.1}→{:5.2} ({:+.1})  │ {} ║",
            r.name,
            r.arb_init_psnr, r.arb_final_psnr, arb_gain,
            r.ent_init_psnr, r.ent_final_psnr, ent_gain,
            r.hyb_init_psnr, r.hyb_final_psnr, hyb_gain,
            winner);

        sum_arb_final += r.arb_final_psnr;
        sum_ent_final += r.ent_final_psnr;
        sum_hyb_final += r.hyb_final_psnr;
    }

    let count = results.len() as f32;
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ AVERAGE  │       {:5.2}       │       {:5.2}        │       {:5.2}        │       ║",
        sum_arb_final / count,
        sum_ent_final / count,
        sum_hyb_final / count);
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ WINS     │ {} images          │ {} images           │ {} images           │       ║",
        arb_wins, ent_wins, hyb_wins);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    println!("\n📊 CRITICAL ANALYSIS:\n");

    // Quality comparison
    let ent_advantage = (sum_ent_final - sum_arb_final) / count;
    let hyb_advantage = (sum_hyb_final - sum_arb_final) / count;

    println!("  Quality after optimization:");
    println!("    Entropy advantage: {:+.2} dB", ent_advantage);
    println!("    Hybrid advantage:  {:+.2} dB", hyb_advantage);

    // Time comparison
    let total_arb_time: f32 = results.iter().map(|r| r.arb_time.as_secs_f32()).sum();
    let total_ent_time: f32 = results.iter().map(|r| r.ent_time.as_secs_f32()).sum();
    let total_hyb_time: f32 = results.iter().map(|r| r.hyb_time.as_secs_f32()).sum();

    println!("\n  Time cost:");
    println!("    Arbitrary: {:.1}s total", total_arb_time);
    println!("    Entropy:   {:.1}s total ({:.0}× slower)", total_ent_time, total_ent_time / total_arb_time);
    println!("    Hybrid:    {:.1}s total ({:.0}× slower)", total_hyb_time, total_hyb_time / total_arb_time);

    // Quality per second
    let arb_qps = (sum_arb_final / count) / (total_arb_time / count);
    let ent_qps = (sum_ent_final / count) / (total_ent_time / count);
    let hyb_qps = (sum_hyb_final / count) / (total_hyb_time / count);

    println!("\n  Quality per second (efficiency):");
    println!("    Arbitrary: {:.3} dB/s", arb_qps);
    println!("    Entropy:   {:.3} dB/s", ent_qps);
    println!("    Hybrid:    {:.3} dB/s", hyb_qps);

    // Verdict
    println!("\n✅ VERDICT:");

    if ent_advantage > 2.0 && ent_qps > arb_qps {
        println!("  Entropy is WORTH IT: +{:.1} dB gain AND more efficient!", ent_advantage);
        println!("  → Use auto_gaussian_count() everywhere");
    } else if ent_advantage > 2.0 && ent_qps < arb_qps {
        println!("  Entropy gives quality (+{:.1} dB) but is INEFFICIENT", ent_advantage);
        println!("  → Use for high-quality mode only");
        println!("  → Use Arbitrary for fast mode");
    } else if hyb_advantage > 1.0 && hyb_qps > arb_qps {
        println!("  Hybrid is BEST: Good quality (+{:.1} dB) AND efficient", hyb_advantage);
        println!("  → Use hybrid_gaussian_count() as default");
    } else {
        println!("  Arbitrary is COMPETITIVE: Small gains not worth compute cost");
        println!("  → Fix optimizer first, then re-test");
        println!("  → Better initialization won't help if optimizer is broken");
    }

    println!("\n⚠️  Note: If optimizer is still diverging (negative gains), results are INVALID!");
    println!("    Must fix optimizer convergence before trusting these numbers.");
}
