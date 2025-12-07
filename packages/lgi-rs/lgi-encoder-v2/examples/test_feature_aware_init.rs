//! Test Feature-Aware Gaussian Initialization
//!
//! Uses the FeatureClassifier to drive different initialization strategies
//! for different regions of the image, then optimizes with Adam.
//!
//! Compares:
//! - Baseline: Uniform jittered grid init
//! - Feature-aware: Different scales/anisotropy per feature type

use image::ImageBuffer as ImgBuffer;
use lgi_core::{
    ImageBuffer,
    feature_classifier::{FeatureClassifier, FeatureType, ClassificationResult},
};
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_math::{
    gaussian::Gaussian2D,
    parameterization::Euler,
    vec::Vector2,
    color::Color4,
};
use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("   FEATURE-AWARE INITIALIZATION TEST - Adaptive Gaussian Strategies");
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

    // Classify features
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 1: Feature Classification");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let classifier = FeatureClassifier::default();
    let classification = classifier.classify(&target).expect("Classification failed");

    lgi_core::feature_classifier::print_classification_summary(&classification);
    println!();

    let n_gaussians = 256;

    // Test 1: Baseline (uniform jittered grid)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: BASELINE - Uniform Jittered Grid");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let baseline_result = run_test(
        "baseline_uniform",
        &target,
        || init_uniform_jittered(width, height, n_gaussians, &target),
        image_path,
    );

    // Test 2: Feature-aware initialization
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: FEATURE-AWARE - Adaptive Per-Region");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let feature_result = run_test(
        "feature_aware",
        &target,
        || init_feature_aware(&classification, n_gaussians, &target),
        image_path,
    );

    // Test 3: Feature-aware with orientation
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: FEATURE-AWARE + ORIENTATION - Structure Tensor Alignment");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let oriented_result = run_test(
        "feature_oriented",
        &target,
        || init_feature_aware_oriented(&classification, n_gaussians, &target),
        image_path,
    );

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                         COMPARISON SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:25} {:>12} {:>12} {:>12}", "Method", "Initial", "Final", "Improvement");
    println!("  {:-<25} {:->12} {:->12} {:->12}", "", "", "", "");

    for (name, (init_psnr, final_psnr)) in [
        ("Baseline (uniform)", baseline_result),
        ("Feature-aware", feature_result),
        ("Feature + orientation", oriented_result),
    ] {
        let improvement = final_psnr - init_psnr;
        println!("  {:25} {:>10.2}dB {:>10.2}dB {:>+10.2}dB", name, init_psnr, final_psnr, improvement);
    }

    let best_vs_baseline = oriented_result.1 - baseline_result.1;
    println!("\n  Feature-aware vs baseline: {:+.2}dB", best_vs_baseline);

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn run_test<F>(
    name: &str,
    target: &ImageBuffer<f32>,
    init_fn: F,
    image_path: &str,
) -> (f32, f32)
where
    F: FnOnce() -> Vec<Gaussian2D<f32, Euler<f32>>>,
{
    let width = target.width;
    let height = target.height;
    let iterations = 200;

    let mut result = TestResult::new(&format!("feature_aware_init_{}", name), image_path);
    result.optimizer = format!("Feature-aware init: {}", name);
    result.image_dimensions = (width, height);

    // Initialize Gaussians
    let start = Instant::now();
    let mut gaussians = init_fn();
    result.n_gaussians = gaussians.len();

    // Compute initial PSNR
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
    result.initial_psnr = init_psnr;

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", gaussians.len(), init_psnr);

    // Optimize with Adam
    let mut optimizer = AdamOptimizer::new();
    optimizer.max_iterations = iterations;

    result.set_param("lr_position", 0.0002);
    result.set_param("lr_color", 0.02);
    result.set_param("lr_scale", 0.005);
    result.set_param("lr_opacity", 0.05);
    result.iterations = iterations;

    let mut last_psnr = init_psnr;
    let mut final_psnr = init_psnr;

    for iter in 0..iterations {
        optimizer.optimize(&mut gaussians, target);

        if iter % 20 == 0 || iter == iterations - 1 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
            let direction = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                     iter + 1, iterations, psnr, direction, psnr - last_psnr);
            last_psnr = psnr;
            final_psnr = psnr;
        }
    }

    let elapsed = start.elapsed();

    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f64() as f32;
    result.complete();

    // Save result with image
    let final_render = RendererV2::render(&gaussians, width, height);
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    println!("  Final: PSNR={:.2}dB (improvement: {:+.2}dB)", final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.1}s", elapsed.as_secs_f64());

    (init_psnr, final_psnr)
}

fn init_uniform_jittered(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    let cols = (count as f32).sqrt().ceil() as u32;
    let rows = (count as f32 / cols as f32).ceil() as u32;
    let cell_w = width as f32 / cols as f32;
    let cell_h = height as f32 / rows as f32;

    for i in 0..count {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;

        // Jittered position within cell
        let jitter_x = rng.gen_range(-0.3..0.3);
        let jitter_y = rng.gen_range(-0.3..0.3);
        let x = (col as f32 + 0.5 + jitter_x) * cell_w;
        let y = (row as f32 + 0.5 + jitter_y) * cell_h;

        // Sample color from target
        let px = (x as u32).min(width - 1);
        let py = (y as u32).min(height - 1);
        let color = target.get_pixel(px, py).unwrap();

        // Uniform scale
        let scale = 0.04 * width.max(height) as f32;

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(scale, scale, 0.0),
            color.clone(),
            1.0,
        ));
    }

    gaussians
}

fn init_feature_aware(
    classification: &ClassificationResult,
    total_count: usize,
    target: &ImageBuffer<f32>,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(total_count);

    let width = classification.width;
    let height = classification.height;
    let scale_unit = (width.max(height) as f32) / 100.0;  // 1% of image

    // Allocate budget
    let budget = classification.allocate_budget(total_count);

    println!("  Budget allocation:");
    for (ft, count) in &budget {
        if *count > 0 {
            println!("    {:14}: {}", ft.name(), count);
        }
    }
    println!();

    // Generate Gaussians per feature type
    for (feature_type, count) in budget {
        if count == 0 {
            continue;
        }

        let (scale_min, scale_max) = feature_type.scale_range();
        let (aniso_min, aniso_max) = feature_type.anisotropy_range();

        // Sample positions from this feature type
        let positions = classification.sample_positions(feature_type, count, 42);

        for (x, y) in positions {
            let px = (x as u32).min(width - 1);
            let py = (y as u32).min(height - 1);
            let color = target.get_pixel(px, py).unwrap();

            // Random scale within range
            let scale_factor = rng.gen_range(scale_min..scale_max);
            let base_scale = scale_factor * (width.max(height) as f32);

            // Random anisotropy within range
            let anisotropy = rng.gen_range(aniso_min..aniso_max);
            let sigma_major = base_scale;
            let sigma_minor = base_scale / anisotropy;

            // Random orientation
            let angle = rng.gen_range(0.0..std::f32::consts::PI);

            gaussians.push(Gaussian2D::new(
                Vector2::new(x, y),
                Euler::new(sigma_major, sigma_minor, angle),
                color.clone(),
                1.0,
            ));
        }
    }

    gaussians
}

fn init_feature_aware_oriented(
    classification: &ClassificationResult,
    total_count: usize,
    target: &ImageBuffer<f32>,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(total_count);

    let width = classification.width;
    let height = classification.height;

    // Allocate budget
    let budget = classification.allocate_budget(total_count);

    println!("  Budget allocation (with orientation):");
    for (ft, count) in &budget {
        if *count > 0 {
            println!("    {:14}: {}", ft.name(), count);
        }
    }
    println!();

    // Generate Gaussians per feature type
    for (feature_type, count) in budget {
        if count == 0 {
            continue;
        }

        let (scale_min, scale_max) = feature_type.scale_range();
        let (aniso_min, aniso_max) = feature_type.anisotropy_range();
        let use_orientation = feature_type.use_orientation();

        // Sample positions from this feature type
        let positions = classification.sample_positions(feature_type, count, 42);

        for (x, y) in positions {
            let px = (x as u32).min(width - 1);
            let py = (y as u32).min(height - 1);
            let color = target.get_pixel(px, py).unwrap();

            // Random scale within range
            let scale_factor = rng.gen_range(scale_min..scale_max);
            let base_scale = scale_factor * (width.max(height) as f32);

            // Get orientation from structure tensor if applicable
            let (sigma_major, sigma_minor, angle) = if use_orientation {
                let tensor = classification.tensor_field.get(px, py);

                // Anisotropy from coherence: more coherent = more elongated
                let coherence_aniso = 1.0 + tensor.coherence * (aniso_max - 1.0);
                let anisotropy = coherence_aniso.clamp(aniso_min, aniso_max);

                // Orientation from eigenvector (edge tangent = minor eigenvector)
                // For Gaussians: elongate along edge tangent
                let edge_tangent_angle = tensor.eigenvector_minor.y.atan2(tensor.eigenvector_minor.x);

                let sigma_major = base_scale;
                let sigma_minor = base_scale / anisotropy;

                (sigma_major, sigma_minor, edge_tangent_angle)
            } else {
                // Random for non-oriented features
                let anisotropy = rng.gen_range(aniso_min..aniso_max);
                let sigma_major = base_scale;
                let sigma_minor = base_scale / anisotropy;
                let angle = rng.gen_range(0.0..std::f32::consts::PI);
                (sigma_major, sigma_minor, angle)
            };

            gaussians.push(Gaussian2D::new(
                Vector2::new(x, y),
                Euler::new(sigma_major, sigma_minor, angle),
                color.clone(),
                1.0,
            ));
        }
    }

    gaussians
}
