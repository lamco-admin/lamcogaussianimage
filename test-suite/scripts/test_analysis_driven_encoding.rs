//! Test Analysis-Driven Encoding
//!
//! Uses the unified ImageAnalysis pipeline to:
//! 1. Analyze the image
//! 2. Get suggested encoding strategy
//! 3. Generate feature-aware Gaussians
//! 4. Optimize with Adam
//! 5. Compare against baseline
//!
//! This is the fully integrated approach using all image interpretation tools.

use image::ImageBuffer as ImgBuffer;
use image::Rgb;
use lgi_core::{
    ImageBuffer,
    image_analysis::{ImageAnalysis, AnalysisConfig},
    artifact_detector::ArtifactDetector,
    Initializer, InitStrategy,
};
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_math::{
    gaussian::Gaussian2D,
    parameterization::Euler,
    color::Color4,
};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("    ANALYSIS-DRIVEN ENCODING TEST - Full Image Interpretation Pipeline");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Test multiple images to see how analysis adapts
    let test_images = [
        ("kodim03", "test-data/kodak-dataset/kodim03.png", "Caps, text, wood grain, sky"),
        ("kodim01", "test-data/kodak-dataset/kodim01.png", "House, foliage, gradients"),
        ("kodim23", "test-data/kodak-dataset/kodim23.png", "Lighthouse, building, sky"),
    ];

    let output_dir = format!("{}/analysis_driven_results", RESULTS_DIR);
    std::fs::create_dir_all(&output_dir).ok();

    let mut all_results = Vec::new();

    for (name, path, description) in &test_images {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Image: {} - {}", name, description);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let full_path = format!("{}/../../../../{}", env!("CARGO_MANIFEST_DIR"), path);

        // Load image
        let img = match image::open(&full_path) {
            Ok(img) => img.to_rgb32f(),
            Err(e) => {
                println!("  ❌ Failed to load: {}\n", e);
                continue;
            }
        };

        let width = img.width();
        let height = img.height();
        let mut target = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                target.set_pixel(x, y, Color4::new(pixel[0], pixel[1], pixel[2], 1.0));
            }
        }

        // Step 1: Analyze image
        println!("Step 1: Image Analysis");
        println!("─────────────────────────────────────────────────────────────────────\n");

        let start = Instant::now();
        let analysis = match ImageAnalysis::compute(&target, AnalysisConfig::default()) {
            Ok(a) => a,
            Err(e) => {
                println!("  ❌ Analysis failed: {:?}\n", e);
                continue;
            }
        };
        let analysis_time = start.elapsed();

        analysis.print_summary();
        println!("\n  Analysis time: {:.1}ms\n", analysis_time.as_secs_f64() * 1000.0);

        // Step 2: Get suggested strategy
        println!("Step 2: Suggested Encoding Strategy");
        println!("─────────────────────────────────────────────────────────────────────\n");

        let strategy = analysis.suggest_strategy();
        strategy.print();
        println!();

        // Step 3: Compare baseline vs analysis-driven
        println!("Step 3: Encoding Comparison");
        println!("─────────────────────────────────────────────────────────────────────\n");

        let iterations = 200;
        let n_gaussians = strategy.gaussian_count;

        // Baseline: Grid initialization
        println!("  3a. Baseline (grid init)...");
        let baseline_result = run_encoding(
            name,
            &target,
            || {
                let initializer = Initializer::new(InitStrategy::Grid).with_seed(42);
                initializer.initialize(&target, n_gaussians).expect("Init failed")
            },
            iterations,
            path,
        );

        // Analysis-driven: Feature-aware initialization
        println!("  3b. Analysis-driven (feature-aware)...");
        let analysis_result = run_encoding(
            name,
            &target,
            || analysis.generate_gaussians(n_gaussians, &target, 42),
            iterations,
            path,
        );

        let improvement = analysis_result.1 - baseline_result.1;
        println!("\n  Comparison:");
        println!("    Baseline:         {:.2}dB → {:.2}dB ({:+.2}dB improvement)",
                 baseline_result.0, baseline_result.1, baseline_result.1 - baseline_result.0);
        println!("    Analysis-driven:  {:.2}dB → {:.2}dB ({:+.2}dB improvement)",
                 analysis_result.0, analysis_result.1, analysis_result.1 - analysis_result.0);
        println!("    Analysis vs baseline: {:+.2}dB", improvement);

        // Step 4: Artifact analysis on final result
        println!("\nStep 4: Artifact Analysis");
        println!("─────────────────────────────────────────────────────────────────────\n");

        let detector = ArtifactDetector::default();

        // Get Gaussians for artifact analysis
        let gaussians = analysis.generate_gaussians(n_gaussians, &target, 42);
        let mut opt_gaussians = gaussians;
        let mut optimizer = AdamOptimizer::new();
        optimizer.max_iterations = iterations;
        for _ in 0..iterations {
            optimizer.optimize(&mut opt_gaussians, &target);
        }
        let final_render = RendererV2::render(&opt_gaussians, width, height);

        if let Ok(report) = detector.analyze(&final_render, &target) {
            report.print_summary();

            // Save artifact visualization
            let vis = report.visualize();
            let vis_path = format!("{}/artifacts_{}.png", output_dir, name);
            save_image(&vis, &vis_path);
            println!("\n  💾 Saved artifact map: {}", vis_path);
        }

        all_results.push((name.to_string(), baseline_result.1, analysis_result.1, improvement));
        println!();
    }

    // Final summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                           FINAL SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:10} {:>12} {:>12} {:>12}", "Image", "Baseline", "Analysis", "Difference");
    println!("  {:->10} {:->12} {:->12} {:->12}", "", "", "", "");

    let mut total_baseline = 0.0;
    let mut total_analysis = 0.0;

    for (name, baseline, analysis, diff) in &all_results {
        println!("  {:10} {:>10.2}dB {:>10.2}dB {:>+10.2}dB", name, baseline, analysis, diff);
        total_baseline += baseline;
        total_analysis += analysis;
    }

    let n = all_results.len() as f32;
    println!("  {:->10} {:->12} {:->12} {:->12}", "", "", "", "");
    println!("  {:10} {:>10.2}dB {:>10.2}dB {:>+10.2}dB",
             "Average",
             total_baseline / n,
             total_analysis / n,
             (total_analysis - total_baseline) / n);

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn run_encoding<F>(
    _name: &str,
    target: &ImageBuffer<f32>,
    init_fn: F,
    iterations: usize,
    _image_path: &str,
) -> (f32, f32)
where
    F: FnOnce() -> Vec<Gaussian2D<f32, Euler<f32>>>,
{
    let width = target.width;
    let height = target.height;

    // Initialize
    let mut gaussians = init_fn();
    let n_gaussians = gaussians.len();

    // Initial PSNR
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);

    // Optimize
    let mut optimizer = AdamOptimizer::new();
    optimizer.max_iterations = iterations;
    let mut last_psnr = init_psnr;

    for iter in 0..iterations {
        optimizer.optimize(&mut gaussians, target);

        if iter % 50 == 0 || iter == iterations - 1 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };
            println!("      [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                     iter + 1, iterations, psnr, dir, psnr - last_psnr);
            last_psnr = psnr;
        }
    }

    let final_psnr = last_psnr;
    println!("    → {} Gaussians: {:.2}dB → {:.2}dB", n_gaussians, init_psnr, final_psnr);

    (init_psnr, final_psnr)
}

fn save_image(image: &ImageBuffer<f32>, path: &str) {
    let mut output = ImgBuffer::<Rgb<u8>, _>::new(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            let pixel = image.get_pixel(x, y).unwrap();
            output.put_pixel(x, y, Rgb([
                (pixel.r * 255.0).clamp(0.0, 255.0) as u8,
                (pixel.g * 255.0).clamp(0.0, 255.0) as u8,
                (pixel.b * 255.0).clamp(0.0, 255.0) as u8,
            ]));
        }
    }
    output.save(path).ok();
}
