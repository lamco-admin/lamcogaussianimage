//! Test Feature Classifier on Multiple Kodak Images
//!
//! Verifies that the FeatureClassifier correctly identifies different image regions:
//! - kodim03: Caps, text ("BAHAMAS"), wood grain, sky
//! - kodim01: House, foliage, sky gradients
//! - kodim23: Lighthouse, detailed building, sky
//!
//! Outputs:
//! - Feature map visualizations
//! - Classification statistics
//! - Budget allocation recommendations

use image::{ImageBuffer as ImgBuffer, Rgb};
use lgi_core::feature_classifier::{FeatureClassifier, FeatureType, print_classification_summary};
use lgi_core::ImageBuffer;
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("        FEATURE CLASSIFIER TEST - Per-Pixel Image Analysis");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Test images with diverse content
    let test_images = [
        ("kodim03", "test-data/kodak-dataset/kodim03.png", "Caps, text, wood grain, sky"),
        ("kodim01", "test-data/kodak-dataset/kodim01.png", "House, foliage, gradients"),
        ("kodim23", "test-data/kodak-dataset/kodim23.png", "Lighthouse, building, sky"),
        ("kodim08", "test-data/kodak-dataset/kodim08.png", "Portrait, hair, skin"),
    ];

    // Ensure results directory exists - use simple date format
    let output_dir = format!("{}/feature_classifier_results", RESULTS_DIR);
    std::fs::create_dir_all(&output_dir).ok();

    let classifier = FeatureClassifier::default();

    for (name, path, description) in &test_images {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Testing: {} - {}", name, description);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // Load image
        let full_path = format!("{}/../../../../{}", env!("CARGO_MANIFEST_DIR"), path);
        let img = match image::open(&full_path) {
            Ok(img) => img.to_rgb32f(),
            Err(e) => {
                println!("  ❌ Failed to load {}: {}\n", path, e);
                continue;
            }
        };

        // Convert to ImageBuffer
        let width = img.width();
        let height = img.height();
        let mut target = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                target.set_pixel(
                    x, y,
                    lgi_math::color::Color4::new(pixel[0], pixel[1], pixel[2], 1.0)
                );
            }
        }

        // Classify
        let start = Instant::now();
        let result = match classifier.classify(&target) {
            Ok(r) => r,
            Err(e) => {
                println!("  ❌ Classification failed: {:?}\n", e);
                continue;
            }
        };
        let elapsed = start.elapsed();

        println!("  ⏱️  Classification time: {:.1}ms\n", elapsed.as_secs_f64() * 1000.0);

        // Print summary
        print_classification_summary(&result);
        println!();

        // Budget allocation for 256 Gaussians
        let budget = result.allocate_budget(256);
        println!("  Gaussian Budget Allocation (256 total):");
        for ft in [
            FeatureType::Smooth,
            FeatureType::SmoothGradient,
            FeatureType::Edge,
            FeatureType::Corner,
            FeatureType::TextStroke,
            FeatureType::Texture,
            FeatureType::Specular,
        ] {
            let count = budget.get(&ft).unwrap_or(&0);
            if *count > 0 {
                println!("    {:14}: {:3}", ft.name(), count);
            }
        }
        println!();

        // Save visualization
        let vis = result.visualize();
        let vis_path = format!("{}/feature_map_{}.png", output_dir, name);
        save_visualization(&vis, &vis_path);
        println!("  💾 Saved: {}\n", vis_path);

        // Sample positions for edges (verify distribution)
        let edge_samples = result.sample_positions(FeatureType::Edge, 50, 42);
        println!("  Edge sample positions (first 5 of {}):", edge_samples.len());
        for (i, (x, y)) in edge_samples.iter().take(5).enumerate() {
            println!("    {}: ({:.1}, {:.1})", i + 1, x, y);
        }
        println!();
    }

    // Save legend
    save_legend(&output_dir);

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("Feature Classifier Test Complete");
    println!("═══════════════════════════════════════════════════════════════════════");
}

fn save_visualization(image: &ImageBuffer<f32>, path: &str) {
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

fn save_legend(output_dir: &str) {
    // Create a legend image showing feature type colors
    let legend_width = 200;
    let legend_height = 180;
    let mut legend = ImgBuffer::<Rgb<u8>, _>::new(legend_width, legend_height);

    // Background
    for pixel in legend.pixels_mut() {
        *pixel = Rgb([30, 30, 30]);
    }

    let colors = [
        ("Smooth", [0, 0, 102]),
        ("SmoothGrad", [102, 153, 255]),
        ("Edge", [0, 204, 0]),
        ("Corner", [255, 0, 0]),
        ("TextStroke", [255, 255, 0]),
        ("Texture", [255, 153, 0]),
        ("Specular", [255, 255, 255]),
    ];

    for (i, (_, color)) in colors.iter().enumerate() {
        let y_start = (i * 25 + 5) as u32;
        // Draw color swatch
        for y in y_start..(y_start + 20) {
            for x in 5..45 {
                legend.put_pixel(x, y, Rgb(*color));
            }
        }
    }

    let legend_path = format!("{}/feature_map_legend.png", output_dir);
    legend.save(&legend_path).ok();
    println!("  💾 Saved legend: {}\n", legend_path);
}
