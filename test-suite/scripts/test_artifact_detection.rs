//! Test Artifact Detection on Encoding Results
//!
//! Encodes an image, then analyzes artifacts at different iteration stages:
//! - Initial (poor fit)
//! - Mid-optimization (improving)
//! - Final (converged)
//!
//! Identifies what artifact types dominate at each stage.

use image::ImageBuffer as ImgBuffer;
use image::Rgb;
use lgi_core::{
    ImageBuffer,
    artifact_detector::{ArtifactDetector, ArtifactReport},
    Initializer, InitStrategy,
};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_math::{
    gaussian::Gaussian2D,
    parameterization::Euler,
    color::Color4,
};

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("        ARTIFACT DETECTION TEST - Encoding Quality Diagnosis");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let image_path = "test-data/kodak-dataset/kodim03.png";
    let full_path = format!("{}/../../../../{}", env!("CARGO_MANIFEST_DIR"), image_path);

    // Load image
    let img = image::open(&full_path).expect("Failed to load image").to_rgb32f();
    let width = img.width();
    let height = img.height();

    let mut target = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            target.set_pixel(x, y, Color4::new(pixel[0], pixel[1], pixel[2], 1.0));
        }
    }

    println!("Image: kodim03 ({}x{})\n", width, height);

    // Initialize Gaussians using correct API
    let n_gaussians = 256;
    let initializer = Initializer::new(InitStrategy::Gradient).with_seed(42);
    let mut gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = initializer
        .initialize(&target, n_gaussians)
        .expect("Initialization failed");

    let detector = ArtifactDetector::default();

    // Analysis stages
    let stages = vec![
        (0, "Initial"),
        (20, "Early"),
        (50, "Mid"),
        (100, "Late"),
        (200, "Final"),
    ];

    // Create optimizer with correct API
    let mut optimizer = AdamOptimizer::new();
    optimizer.max_iterations = 200;

    let output_dir = format!("{}/artifact_detection_results", RESULTS_DIR);
    std::fs::create_dir_all(&output_dir).ok();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Artifact Analysis at Different Optimization Stages");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut results: Vec<(String, f32, ArtifactReport)> = Vec::new();
    let mut current_iter = 0;

    for (target_iter, stage_name) in &stages {
        // Optimize to target iteration
        while current_iter < *target_iter {
            optimizer.optimize(&mut gaussians, &target);
            current_iter += 1;
        }

        // Render and analyze
        let rendered = RendererV2::render(&gaussians, width, height);
        let psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);

        let report = detector.analyze(&rendered, &target).expect("Analysis failed");

        println!("─────────────────────────────────────────────────────────────────────");
        println!("Stage: {} (iter {})", stage_name, target_iter);
        println!("─────────────────────────────────────────────────────────────────────");
        println!("  PSNR: {:.2}dB\n", psnr);
        report.print_summary();
        println!();

        // Save artifact visualization
        let vis = report.visualize();
        let vis_path = format!("{}/artifacts_{}_{}.png", output_dir, stage_name.to_lowercase(), target_iter);
        save_image(&vis, &vis_path);
        println!("  💾 Saved: {}", vis_path);

        // Top error regions
        let top_regions = report.top_error_regions(5);
        if !top_regions.is_empty() {
            println!("  Top error regions:");
            for (i, (x, y, err)) in top_regions.iter().enumerate() {
                println!("    {}: ({}, {}) error={:.3}", i + 1, x, y, err);
            }
        }
        println!();

        results.push((stage_name.to_string(), psnr, report));
    }

    // Summary comparison
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    ARTIFACT PROGRESSION SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:10} {:>8} {:>10} {:>8} {:>8} {:>8} {:>8}",
             "Stage", "PSNR", "Overall", "Bleed", "Blur", "Ring", "Band");
    println!("  {:->10} {:->8} {:->10} {:->8} {:->8} {:->8} {:->8}",
             "", "", "", "", "", "", "");

    for (name, psnr, report) in &results {
        println!("  {:10} {:>6.2}dB {:>9.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
                 name,
                 psnr,
                 report.overall_score * 100.0,
                 report.bleeding_score * 100.0,
                 report.blur_score * 100.0,
                 report.ringing_score * 100.0,
                 report.banding_score * 100.0);
    }

    // Key findings
    println!("\n─────────────────────────────────────────────────────────────────────");
    println!("Key Findings:");
    println!("─────────────────────────────────────────────────────────────────────");

    let initial = &results[0].2;
    let final_rep = &results[results.len() - 1].2;

    if initial.blur_score > final_rep.blur_score + 0.1 {
        println!("  ✓ Blur reduced significantly ({:.1}% → {:.1}%)",
                 initial.blur_score * 100.0, final_rep.blur_score * 100.0);
    }

    if final_rep.bleeding_score > 0.1 {
        println!("  ⚠ Bleeding still significant ({:.1}%) - consider geodesic clamping",
                 final_rep.bleeding_score * 100.0);
    }

    if final_rep.ringing_score > 0.1 {
        println!("  ⚠ Ringing detected ({:.1}%) - consider adjusting overlap",
                 final_rep.ringing_score * 100.0);
    }

    if final_rep.banding_score > 0.1 {
        println!("  ⚠ Banding detected ({:.1}%) - consider adding gradient Gaussians",
                 final_rep.banding_score * 100.0);
    }

    // Dominant artifact
    let dominant = if final_rep.bleeding_score >= final_rep.blur_score &&
                     final_rep.bleeding_score >= final_rep.ringing_score {
        "Bleeding"
    } else if final_rep.blur_score >= final_rep.ringing_score {
        "Blur"
    } else {
        "Ringing"
    };

    println!("\n  Dominant artifact at final stage: {}", dominant);

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn save_image(image: &ImageBuffer<f32>, path: &str) {
    let mut output = ImgBuffer::<Rgb<u8>, _>::new(image.width, image.height);

    for y in 0..image.height {
        for x in 0..image.width {
            let pixel = image.get_pixel(x, y).unwrap();
            output.put_pixel(
                x, y,
                Rgb([
                    (pixel.r * 255.0).clamp(0.0, 255.0) as u8,
                    (pixel.g * 255.0).clamp(0.0, 255.0) as u8,
                    (pixel.b * 255.0).clamp(0.0, 255.0) as u8,
                ])
            );
        }
    }

    output.save(path).ok();
}
