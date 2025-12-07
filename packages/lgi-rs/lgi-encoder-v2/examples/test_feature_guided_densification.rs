//! Test Feature-Guided Densification
//!
//! Hypothesis: Combining FeatureClassifier with error-driven placement
//! improves encoding by setting adaptive sigma based on feature type.
//!
//! The key insight is that error-driven hotspot placement works well,
//! but the fixed sigma (0.03) is suboptimal:
//! - Edge hotspots need smaller, more anisotropic Gaussians
//! - Smooth region hotspots need larger Gaussians
//! - Texture hotspots need medium, potentially anisotropic Gaussians
//!
//! This test compares:
//! 1. Baseline: Error-driven with fixed sigma
//! 2. Feature-guided: Error-driven with feature-adaptive sigma

use lgi_core::{
    ImageBuffer,
    feature_classifier::{FeatureClassifier, FeatureType},
    Initializer, InitStrategy,
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

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("   FEATURE-GUIDED DENSIFICATION TEST - Adaptive Sigma for Hotspots");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let image_path = "test-data/kodak-dataset/kodim03.png";
    let full_path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), image_path);

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

    // Classify features once
    println!("Step 1: Feature Classification");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let classifier = FeatureClassifier::default();
    let classification = classifier.classify(&target).expect("Classification failed");
    lgi_core::feature_classifier::print_classification_summary(&classification);
    println!();

    // Test parameters
    let initial_gaussians = 64;
    let iterations_per_pass = 50;
    let max_passes = 6;
    let max_gaussians = 350;

    // Test 1: Baseline (fixed sigma)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: BASELINE - Error-driven with fixed sigma");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let baseline_result = run_error_driven(
        &target,
        initial_gaussians,
        max_gaussians,
        iterations_per_pass,
        max_passes,
        image_path,
        None,  // No feature classification
    );

    // Test 2: Feature-guided
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: FEATURE-GUIDED - Adaptive sigma based on feature type");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let feature_result = run_error_driven(
        &target,
        initial_gaussians,
        max_gaussians,
        iterations_per_pass,
        max_passes,
        image_path,
        Some(&classification),
    );

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                         COMPARISON SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let (b_init, b_final, b_count) = baseline_result;
    let (f_init, f_final, f_count) = feature_result;

    println!("  {:25} {:>12} {:>12} {:>12} {:>8}", "Method", "Initial", "Final", "Improvement", "N");
    println!("  {:->25} {:->12} {:->12} {:->12} {:->8}", "", "", "", "", "");
    println!("  {:25} {:>10.2}dB {:>10.2}dB {:>+10.2}dB {:>8}",
             "Baseline (fixed σ)", b_init, b_final, b_final - b_init, b_count);
    println!("  {:25} {:>10.2}dB {:>10.2}dB {:>+10.2}dB {:>8}",
             "Feature-guided (σ(f))", f_init, f_final, f_final - f_init, f_count);

    let advantage = f_final - b_final;
    println!("\n  Feature-guided advantage: {:+.2}dB", advantage);

    if advantage > 0.1 {
        println!("  ✓ Feature-guided densification improves quality!");
    } else if advantage < -0.1 {
        println!("  ⚠ Fixed sigma performs better - review adaptive logic");
    } else {
        println!("  ~ Results comparable - sigma adaptation has minimal impact");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn run_error_driven(
    target: &ImageBuffer<f32>,
    initial_gaussians: usize,
    max_gaussians: usize,
    iterations_per_pass: usize,
    max_passes: usize,
    image_path: &str,
    classification: Option<&lgi_core::feature_classifier::ClassificationResult>,
) -> (f32, f32, usize) {
    let width = target.width;
    let height = target.height;
    let w_f = width as f32;
    let h_f = height as f32;

    let method_name = if classification.is_some() { "feature_guided" } else { "fixed_sigma" };
    let mut result = TestResult::new(&format!("densification_{}", method_name), image_path);
    result.optimizer = format!("Error-driven densification ({})", method_name);
    result.image_dimensions = (width, height);

    let start = Instant::now();

    // Initialize with grid
    let initializer = Initializer::new(InitStrategy::Grid).with_seed(42);
    let mut gaussians = initializer.initialize(target, initial_gaussians).expect("Init failed");

    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
    result.initial_psnr = init_psnr;

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", gaussians.len(), init_psnr);

    let mut last_psnr = init_psnr;

    for pass in 0..max_passes {
        if gaussians.len() >= max_gaussians {
            println!("  ✓ Reached max Gaussians");
            break;
        }

        // Optimize current Gaussians
        let mut optimizer = AdamOptimizer::new();
        optimizer.max_iterations = iterations_per_pass;

        for _ in 0..iterations_per_pass {
            optimizer.optimize(&mut gaussians, target);
        }

        let rendered = RendererV2::render(&gaussians, width, height);
        let psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
        let direction = if psnr > last_psnr { "↑" } else { "↓" };

        println!("  Pass {}: N={:4}, PSNR={:.2}dB {} ({:+.2}dB)",
                 pass, gaussians.len(), psnr, direction, psnr - last_psnr);

        // Find high-error hotspots
        let hotspots = find_hotspots(target, &rendered, &gaussians, 30);

        if hotspots.is_empty() {
            println!("    → No valid hotspots remaining");
            break;
        }

        // Add Gaussians at hotspots with sigma based on feature type
        let added = add_gaussians_at_hotspots(
            &mut gaussians,
            target,
            &hotspots,
            max_gaussians,
            classification,
        );

        println!("    → Added {} Gaussians", added);

        last_psnr = psnr;
    }

    // Final polish
    println!("  Final polish (30 iterations)...");
    let mut optimizer = AdamOptimizer::new();
    optimizer.max_iterations = 30;
    for _ in 0..30 {
        optimizer.optimize(&mut gaussians, target);
    }

    let elapsed = start.elapsed();
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = lgi_core::error_metrics::compute_psnr(target, &final_render);

    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f64() as f32;
    result.iterations = max_passes * iterations_per_pass + 30;
    result.n_gaussians = gaussians.len();
    result.set_param("max_passes", max_passes as f64);
    result.set_param("iterations_per_pass", iterations_per_pass as f64);
    result.set_param("initial_gaussians", initial_gaussians as f64);
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    println!("  Final: {} Gaussians, {:.2}dB ({:+.2}dB improvement)",
             gaussians.len(), final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.1}s", elapsed.as_secs_f64());

    (init_psnr, final_psnr, gaussians.len())
}

struct Hotspot {
    x: u32,
    y: u32,
    error: f32,
}

fn find_hotspots(
    target: &ImageBuffer<f32>,
    rendered: &ImageBuffer<f32>,
    existing: &[Gaussian2D<f32, Euler<f32>>],
    max_count: usize,
) -> Vec<Hotspot> {
    let width = target.width;
    let height = target.height;
    let w_f = width as f32;
    let h_f = height as f32;

    // Compute error map
    let mut error_map: Vec<f32> = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let t = target.get_pixel(x, y).unwrap();
            let r = rendered.get_pixel(x, y).unwrap();
            let err = (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
            error_map.push(err);
        }
    }

    // Find threshold (top 5%)
    let mut sorted = error_map.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold_idx = (sorted.len() as f32 * 0.05) as usize;
    let threshold = sorted.get(threshold_idx).copied().unwrap_or(0.0);

    // Collect candidates
    let mut candidates: Vec<Hotspot> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if error_map[idx] > threshold {
                candidates.push(Hotspot { x, y, error: error_map[idx] });
            }
        }
    }

    // Sort by error
    candidates.sort_by(|a, b| b.error.partial_cmp(&a.error).unwrap());

    // Filter with spatial clustering
    let min_dist_sq = 0.02_f32.powi(2);
    let mut selected = Vec::new();

    for candidate in candidates {
        let cx = candidate.x as f32 / w_f;
        let cy = candidate.y as f32 / h_f;

        // Check existing Gaussians
        let too_close_existing = existing.iter().any(|g| {
            let dx = g.position.x - cx;
            let dy = g.position.y - cy;
            dx * dx + dy * dy < min_dist_sq
        });
        if too_close_existing { continue; }

        // Check already selected
        let too_close_selected = selected.iter().any(|h: &Hotspot| {
            let hx = h.x as f32 / w_f;
            let hy = h.y as f32 / h_f;
            (hx - cx).powi(2) + (hy - cy).powi(2) < min_dist_sq
        });
        if too_close_selected { continue; }

        selected.push(candidate);
        if selected.len() >= max_count { break; }
    }

    selected
}

fn add_gaussians_at_hotspots(
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    hotspots: &[Hotspot],
    max_gaussians: usize,
    classification: Option<&lgi_core::feature_classifier::ClassificationResult>,
) -> usize {
    let width = target.width as f32;
    let height = target.height as f32;
    let mut added = 0;

    for hotspot in hotspots {
        if gaussians.len() >= max_gaussians { break; }

        let position = Vector2::new(hotspot.x as f32 / width, hotspot.y as f32 / height);
        let color = target.get_pixel(hotspot.x, hotspot.y)
            .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        // Determine sigma based on feature type (if available)
        let param = if let Some(class) = classification {
            // Access feature_map directly (row-major order)
            let idx = (hotspot.y * class.width + hotspot.x) as usize;
            let feature_type = class.feature_map[idx];
            feature_type_to_euler(feature_type, class, hotspot.x, hotspot.y)
        } else {
            // Default fixed sigma
            Euler::isotropic(0.03)
        };

        gaussians.push(Gaussian2D::new(position, param, color, 1.0));
        added += 1;
    }

    added
}

fn feature_type_to_euler(
    feature_type: FeatureType,
    classification: &lgi_core::feature_classifier::ClassificationResult,
    x: u32,
    y: u32,
) -> Euler<f32> {
    // Get structure tensor info for orientation
    let tensor = classification.tensor_field.get(x, y);
    let orientation = tensor.eigenvector_minor.y.atan2(tensor.eigenvector_minor.x);
    let coherence = tensor.coherence;

    // Scale ranges based on feature type (normalized, relative to image)
    let (sigma_base, anisotropy) = match feature_type {
        FeatureType::Smooth => (0.06, 1.0),           // Large, isotropic
        FeatureType::SmoothGradient => (0.045, 1.2),  // Medium, slightly elongated
        FeatureType::Edge => {
            // Edges: smaller, anisotropic along edge tangent
            let aniso = 1.5 + coherence * 3.0;  // 1.5-4.5 based on coherence
            (0.025, aniso)
        },
        FeatureType::Corner => (0.02, 1.0),           // Small, isotropic
        FeatureType::TextStroke => {
            // Text: very small, highly anisotropic
            let aniso = 2.0 + coherence * 5.0;  // 2-7
            (0.015, aniso)
        },
        FeatureType::Texture => (0.035, 1.3),         // Medium, moderately anisotropic
        FeatureType::Specular => (0.02, 1.0),         // Small, isotropic
    };

    // Create parameterization
    let sigma_major = sigma_base;
    let sigma_minor = sigma_base / anisotropy;

    // Use orientation for anisotropic features
    let final_angle = if matches!(feature_type, FeatureType::Edge | FeatureType::TextStroke) {
        orientation
    } else {
        0.0
    };

    Euler::new(sigma_major, sigma_minor, final_angle)
}
