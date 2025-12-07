//! Test Local Contrast-Aware Gaussian Scaling
//!
//! Hypothesis: Gaussians in high-contrast regions should be smaller
//! while those in uniform regions can be larger without quality loss.
//!
//! This uses local variance (already computed by FeatureClassifier) as
//! a proxy for local contrast to set initial sigma values.
//!
//! The key insight is that a single global sigma doesn't adapt to image content.
//! High-detail regions need fine Gaussians; smooth regions need coarse ones.

use lgi_core::{
    ImageBuffer,
    feature_classifier::FeatureClassifier,
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
    println!("   LOCAL CONTRAST-AWARE SCALING TEST - Adaptive Sigma from Variance");
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

    // Get local variance from classifier
    println!("Step 1: Local Variance Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let classifier = FeatureClassifier::default();
    let classification = classifier.classify(&target).expect("Classification failed");
    let local_variance = &classification.local_variance;

    // Analyze variance distribution
    let max_var = local_variance.iter().cloned().fold(0.0_f32, f32::max);
    let min_var = local_variance.iter().cloned().fold(f32::MAX, f32::min);
    let mean_var: f32 = local_variance.iter().sum::<f32>() / local_variance.len() as f32;

    println!("  Variance range: [{:.4}, {:.4}]", min_var, max_var);
    println!("  Mean variance: {:.4}", mean_var);

    // Count high/low variance regions
    let high_var_threshold = mean_var * 2.0;
    let low_var_threshold = mean_var * 0.5;
    let high_var_count = local_variance.iter().filter(|&&v| v > high_var_threshold).count();
    let low_var_count = local_variance.iter().filter(|&&v| v < low_var_threshold).count();

    println!("  High contrast (>2x mean): {:.1}%", 100.0 * high_var_count as f32 / local_variance.len() as f32);
    println!("  Low contrast (<0.5x mean): {:.1}%\n", 100.0 * low_var_count as f32 / local_variance.len() as f32);

    let n_gaussians = 256;
    let iterations = 200;

    // Test 1: Baseline (fixed sigma)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: BASELINE - Fixed Sigma");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let baseline_result = run_encoding(
        "baseline_fixed",
        &target,
        || init_fixed_sigma(width, height, n_gaussians, &target, 0.04),
        iterations,
        image_path,
    );

    // Test 2: Contrast-aware (smaller in high-contrast)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: CONTRAST-AWARE - Variance-Adaptive Sigma");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let contrast_result = run_encoding(
        "contrast_aware",
        &target,
        || init_contrast_aware(width, height, n_gaussians, &target, local_variance, max_var),
        iterations,
        image_path,
    );

    // Test 3: Inverse contrast (larger in high-contrast - sanity check)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: INVERSE (Sanity Check) - Larger Sigma in High Contrast");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let inverse_result = run_encoding(
        "inverse_contrast",
        &target,
        || init_inverse_contrast(width, height, n_gaussians, &target, local_variance, max_var),
        iterations,
        image_path,
    );

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    CONTRAST SCALING SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:25} {:>12} {:>12} {:>12}", "Method", "Initial", "Final", "Improvement");
    println!("  {:-<25} {:->12} {:->12} {:->12}", "", "", "", "");

    for (name, (init_psnr, final_psnr)) in [
        ("Baseline (fixed σ)", baseline_result),
        ("Contrast-aware", contrast_result),
        ("Inverse (sanity)", inverse_result),
    ] {
        let improvement = final_psnr - init_psnr;
        println!("  {:25} {:>10.2}dB {:>10.2}dB {:>+10.2}dB", name, init_psnr, final_psnr, improvement);
    }

    let best_improvement = contrast_result.1 - baseline_result.1;
    let inverse_improvement = inverse_result.1 - baseline_result.1;

    println!("\n  Contrast-aware vs baseline: {:+.2}dB", best_improvement);
    println!("  Inverse vs baseline: {:+.2}dB (should be negative)", inverse_improvement);

    if best_improvement > 0.05 && best_improvement > inverse_improvement {
        println!("\n  ✓ Contrast-aware scaling improves quality!");
    } else if inverse_improvement > best_improvement {
        println!("\n  ⚠ Inverse performed better - unexpected! Review logic");
    } else {
        println!("\n  ~ Results similar - optimization may dominate initialization");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn run_encoding<F>(
    name: &str,
    target: &ImageBuffer<f32>,
    init_fn: F,
    iterations: usize,
    image_path: &str,
) -> (f32, f32)
where
    F: FnOnce() -> Vec<Gaussian2D<f32, Euler<f32>>>,
{
    let width = target.width;
    let height = target.height;

    let mut result = TestResult::new(&format!("contrast_scaling_{}", name), image_path);
    result.optimizer = format!("Contrast-aware: {}", name);
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
    result.iterations = iterations;

    let mut last_psnr = init_psnr;
    let mut final_psnr = init_psnr;

    for iter in 0..iterations {
        optimizer.optimize(&mut gaussians, target);

        if iter % 40 == 0 || iter == iterations - 1 {
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

fn init_fixed_sigma(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    sigma: f32,
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

        // Jittered position
        let jitter_x = rng.gen_range(-0.3..0.3);
        let jitter_y = rng.gen_range(-0.3..0.3);
        let px = ((col as f32 + 0.5 + jitter_x) * cell_w) as u32;
        let py = ((row as f32 + 0.5 + jitter_y) * cell_h) as u32;
        let px = px.min(width - 1);
        let py = py.min(height - 1);

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        // Random orientation
        let angle = rng.gen_range(0.0..std::f32::consts::PI);

        gaussians.push(Gaussian2D::new(
            position,
            Euler::isotropic(sigma),
            color.clone(),
            1.0,
        ));
    }

    gaussians
}

fn init_contrast_aware(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    local_variance: &[f32],
    max_variance: f32,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    let cols = (count as f32).sqrt().ceil() as u32;
    let rows = (count as f32 / cols as f32).ceil() as u32;
    let cell_w = width as f32 / cols as f32;
    let cell_h = height as f32 / rows as f32;

    // Sigma range: high contrast gets small sigma, low contrast gets large
    let sigma_min = 0.02;
    let sigma_max = 0.07;

    for i in 0..count {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;

        // Jittered position
        let jitter_x = rng.gen_range(-0.3..0.3);
        let jitter_y = rng.gen_range(-0.3..0.3);
        let px = ((col as f32 + 0.5 + jitter_x) * cell_w) as u32;
        let py = ((row as f32 + 0.5 + jitter_y) * cell_h) as u32;
        let px = px.min(width - 1);
        let py = py.min(height - 1);

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        // Get local variance at this position
        let idx = (py * width + px) as usize;
        let variance = local_variance.get(idx).copied().unwrap_or(0.0);

        // Normalize variance and invert: high variance → small sigma
        let norm_var = (variance / max_variance.max(0.001)).clamp(0.0, 1.0);
        let sigma = sigma_max - norm_var * (sigma_max - sigma_min);

        gaussians.push(Gaussian2D::new(
            position,
            Euler::isotropic(sigma),
            color.clone(),
            1.0,
        ));
    }

    gaussians
}

fn init_inverse_contrast(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    local_variance: &[f32],
    max_variance: f32,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    let cols = (count as f32).sqrt().ceil() as u32;
    let rows = (count as f32 / cols as f32).ceil() as u32;
    let cell_w = width as f32 / cols as f32;
    let cell_h = height as f32 / rows as f32;

    // Inverse: high contrast gets LARGE sigma (sanity check - should be worse)
    let sigma_min = 0.02;
    let sigma_max = 0.07;

    for i in 0..count {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;

        // Jittered position
        let jitter_x = rng.gen_range(-0.3..0.3);
        let jitter_y = rng.gen_range(-0.3..0.3);
        let px = ((col as f32 + 0.5 + jitter_x) * cell_w) as u32;
        let py = ((row as f32 + 0.5 + jitter_y) * cell_h) as u32;
        let px = px.min(width - 1);
        let py = py.min(height - 1);

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        // Get local variance at this position
        let idx = (py * width + px) as usize;
        let variance = local_variance.get(idx).copied().unwrap_or(0.0);

        // INVERSE: high variance → large sigma (opposite of correct)
        let norm_var = (variance / max_variance.max(0.001)).clamp(0.0, 1.0);
        let sigma = sigma_min + norm_var * (sigma_max - sigma_min);

        gaussians.push(Gaussian2D::new(
            position,
            Euler::isotropic(sigma),
            color.clone(),
            1.0,
        ));
    }

    gaussians
}
