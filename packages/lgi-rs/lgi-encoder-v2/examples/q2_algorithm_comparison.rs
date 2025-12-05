//! Q2: Algorithm Comparison Per Quantum Channel
//!
//! Systematically tests which optimization algorithm works best for each
//! quantum-discovered channel.
//!
//! Tests:
//! - Adam (adaptive, momentum)
//! - OptimizerV2 (gradient descent + MS-SSIM)
//! - OptimizerV3 (perceptual: MS-SSIM + edge-weighted)
//!
//! Outputs comprehensive results for multi-hour unattended execution.

use lgi_encoder_v2::EncoderV2;
use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use std::path::PathBuf;
use std::time::Instant;
use std::fs::File;
use std::io::Write;
use serde::Serialize;

#[derive(Serialize, Clone)]
struct ExperimentResult {
    image_id: String,
    algorithm: String,
    final_psnr: f32,
    encoding_time_seconds: f32,
}

fn main() {
    env_logger::init();

    println!("{}", "=".repeat(80));
    println!("Q2: ALGORITHM COMPARISON PER QUANTUM CHANNEL");
    println!("{}", "=".repeat(80));
    println!();
    println!("Testing: Adam vs OptimizerV2 vs OptimizerV3");
    println!("On: 24 Kodak images");
    println!();
    println!("This will run for 3-4 hours. Progress will be logged.");
    println!();

    let kodak_dir = PathBuf::from("../../test-data/kodak-dataset");

    // Test all 24 Kodak images
    let test_images: Vec<_> = (1..=24)
        .map(|i| format!("kodim{:02}", i))
        .collect();

    println!("Testing on {} Kodak images", test_images.len());
    println!();

    let mut all_results = Vec::new();

    let start_overall = Instant::now();

    for (idx, image_id) in test_images.iter().enumerate() {
        println!();
        println!("{}", "=".repeat(80));
        println!("[{}/{}] {}.png", idx + 1, test_images.len(), image_id);
        println!("{}", "=".repeat(80));
        println!();

        let image_path = kodak_dir.join(format!("{}.png", image_id));

        let image = match load_png(&image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("  ERROR loading: {}", e);
                continue;
            }
        };

        println!("Image: {}×{}", image.width, image.height);
        println!();

        // Test 1: Adam (baseline)
        println!("  [1/3] Adam (baseline)...");
        let encoder_adam = EncoderV2::new(image.clone()).expect("Failed to create encoder");
        let start = Instant::now();
        let gaussians_adam = encoder_adam.encode_error_driven_adam(25, 500);
        let time_adam = start.elapsed().as_secs_f32();
        let psnr_adam = compute_psnr_for_gaussians(&image, &gaussians_adam);
        println!("    PSNR: {:.2} dB | Time: {:.1}s", psnr_adam, time_adam);

        all_results.push(ExperimentResult {
            image_id: image_id.clone(),
            algorithm: "Adam".to_string(),
            final_psnr: psnr_adam,
            encoding_time_seconds: time_adam,
        });

        // Test 2: OptimizerV2 (gradient descent + MS-SSIM)
        println!("  [2/3] OptimizerV2 (GD + MS-SSIM)...");
        let encoder_v2 = EncoderV2::new(image.clone()).expect("Failed to create encoder");
        let start = Instant::now();
        let gaussians_v2 = encoder_v2.encode_error_driven_v2(25, 500, true);  // use MS-SSIM
        let time_v2 = start.elapsed().as_secs_f32();
        let psnr_v2 = compute_psnr_for_gaussians(&image, &gaussians_v2);
        println!("    PSNR: {:.2} dB | Time: {:.1}s", psnr_v2, time_v2);

        all_results.push(ExperimentResult {
            image_id: image_id.clone(),
            algorithm: "OptimizerV2".to_string(),
            final_psnr: psnr_v2,
            encoding_time_seconds: time_v2,
        });

        // Test 3: OptimizerV3 (perceptual)
        println!("  [3/3] OptimizerV3 (Perceptual)...");
        let encoder_v3 = EncoderV2::new(image.clone()).expect("Failed to create encoder");
        let start = Instant::now();
        let gaussians_v3 = encoder_v3.encode_error_driven_v3(25, 500);
        let time_v3 = start.elapsed().as_secs_f32();
        let psnr_v3 = compute_psnr_for_gaussians(&image, &gaussians_v3);
        println!("    PSNR: {:.2} dB | Time: {:.1}s", psnr_v3, time_v3);

        all_results.push(ExperimentResult {
            image_id: image_id.clone(),
            algorithm: "OptimizerV3".to_string(),
            final_psnr: psnr_v3,
            encoding_time_seconds: time_v3,
        });

        // Comparison for this image
        let best_psnr = psnr_adam.max(psnr_v2).max(psnr_v3);
        let winner = if (psnr_v3 - best_psnr).abs() < 0.01 {
            "OptimizerV3"
        } else if (psnr_v2 - best_psnr).abs() < 0.01 {
            "OptimizerV2"
        } else {
            "Adam"
        };

        println!();
        println!("  COMPARISON:");
        println!("    Adam:        {:.2} dB", psnr_adam);
        println!("    OptimizerV2: {:.2} dB", psnr_v2);
        println!("    OptimizerV3: {:.2} dB", psnr_v3);
        println!("    Winner: {} ({:.2} dB)", winner, best_psnr);

        // Progress update every 5 images
        if (idx + 1) % 5 == 0 {
            let elapsed = start_overall.elapsed().as_secs_f32();
            let rate = (idx + 1) as f32 / elapsed;
            let remaining = (test_images.len() - idx - 1) as f32 / rate;

            println!();
            println!("{}", "-".repeat(80));
            println!("PROGRESS: {}/{} ({:.0}%)", idx + 1, test_images.len(),
                100.0 * (idx + 1) as f32 / test_images.len() as f32);
            println!("Elapsed: {:.1} min | ETA: {:.0} min", elapsed / 60.0, remaining / 60.0);
            println!("{}", "-".repeat(80));
        }
    }

    let elapsed_total = start_overall.elapsed().as_secs_f32();

    // Save results to JSON
    let output_file = PathBuf::from("../../quantum_research/q2_algorithm_results/full_comparison.json");
    std::fs::create_dir_all(output_file.parent().unwrap()).ok();

    let mut file = File::create(&output_file).expect("Failed to create output file");
    let json_output = serde_json::to_string_pretty(&serde_json::json!({
        "experiment": "Q2_algorithm_comparison",
        "algorithms_tested": ["Adam", "OptimizerV2", "OptimizerV3"],
        "n_images": test_images.len(),
        "runtime_seconds": elapsed_total,
        "runtime_hours": elapsed_total / 3600.0,
        "results": all_results,
    })).unwrap();

    file.write_all(json_output.as_bytes()).ok();

    println!();
    println!("{}", "=".repeat(80));
    println!("Q2 EXPERIMENTS COMPLETE");
    println!("{}", "=".repeat(80));
    println!();
    println!("Runtime: {:.2} hours", elapsed_total / 3600.0);
    println!("Results saved: {}", output_file.display());
    println!();

    // Analyze results
    analyze_and_print_summary(&all_results, &test_images);
}

fn analyze_and_print_summary(results: &[ExperimentResult], images: &[String]) {
    println!("{}", "=".repeat(80));
    println!("ALGORITHM COMPARISON SUMMARY");
    println!("{}", "=".repeat(80));
    println!();

    // Calculate averages per algorithm
    let mut adam_total = 0.0;
    let mut v2_total = 0.0;
    let mut v3_total = 0.0;
    let mut adam_count = 0;
    let mut v2_count = 0;
    let mut v3_count = 0;

    for result in results {
        match result.algorithm.as_str() {
            "Adam" => { adam_total += result.final_psnr; adam_count += 1; },
            "OptimizerV2" => { v2_total += result.final_psnr; v2_count += 1; },
            "OptimizerV3" => { v3_total += result.final_psnr; v3_count += 1; },
            _ => {},
        }
    }

    let adam_avg = if adam_count > 0 { adam_total / adam_count as f32 } else { 0.0 };
    let v2_avg = if v2_count > 0 { v2_total / v2_count as f32 } else { 0.0 };
    let v3_avg = if v3_count > 0 { v3_total / v3_count as f32 } else { 0.0 };

    println!("Average PSNR across {} images:", images.len());
    println!("  Adam:        {:.2} dB", adam_avg);
    println!("  OptimizerV2: {:.2} dB ({:+.2} dB vs Adam)", v2_avg, v2_avg - adam_avg);
    println!("  OptimizerV3: {:.2} dB ({:+.2} dB vs Adam)", v3_avg, v3_avg - adam_avg);
    println!();

    // Count wins
    let mut adam_wins = 0;
    let mut v2_wins = 0;
    let mut v3_wins = 0;

    for image_id in images {
        let image_results: Vec<_> = results.iter()
            .filter(|r| &r.image_id == image_id)
            .collect();

        if image_results.is_empty() {
            continue;
        }

        let best_psnr = image_results.iter()
            .map(|r| r.final_psnr)
            .fold(f32::NEG_INFINITY, f32::max);

        for result in image_results {
            if (result.final_psnr - best_psnr).abs() < 0.01 {
                match result.algorithm.as_str() {
                    "Adam" => adam_wins += 1,
                    "OptimizerV2" => v2_wins += 1,
                    "OptimizerV3" => v3_wins += 1,
                    _ => {},
                }
                break;
            }
        }
    }

    println!("Win distribution:");
    println!("  Adam: {}/{} ({:.0}%)", adam_wins, images.len(),
        100.0 * adam_wins as f32 / images.len() as f32);
    println!("  OptimizerV2: {}/{} ({:.0}%)", v2_wins, images.len(),
        100.0 * v2_wins as f32 / images.len() as f32);
    println!("  OptimizerV3: {}/{} ({:.0}%)", v3_wins, images.len(),
        100.0 * v3_wins as f32 / images.len() as f32);
    println!();

    // Interpretation
    if v2_avg > adam_avg + 1.0 || v3_avg > adam_avg + 1.0 {
        println!("FINDING: Alternative optimizers show significant improvement!");
        println!("Recommendation: Adopt best-performing optimizer as new standard.");
    } else if v2_avg > adam_avg + 0.3 || v3_avg > adam_avg + 0.3 {
        println!("FINDING: Alternative optimizers show modest improvement.");
        println!("Recommendation: Consider per-channel algorithm selection.");
    } else {
        println!("FINDING: Adam performs comparably to alternatives.");
        println!("Recommendation: Current Adam approach is adequate.");
    }
}

fn load_png(path: &PathBuf) -> Result<ImageBuffer<f32>, String> {
    let img = image::open(path)
        .map_err(|e| format!("Failed to load: {}", e))?;
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

fn compute_psnr_for_gaussians(
    target: &ImageBuffer<f32>,
    gaussians: &[lgi_math::gaussian::Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>]
) -> f32 {
    use lgi_encoder_v2::renderer_v2::RendererV2;

    let rendered = RendererV2::render(gaussians, target.width, target.height);

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
