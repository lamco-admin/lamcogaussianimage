//! Test Adaptive Densification (3DGS Split/Clone/Prune)
//!
//! Tests the 3DGS-style adaptive densification that dynamically:
//! - SPLIT large Gaussians in high-error regions
//! - CLONE small Gaussians in high-error regions
//! - PRUNE low-opacity Gaussians
//!
//! Run: cargo run --release --example test_adaptive_densification -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::adaptive_densification::{AdaptiveDensifier, apply_densification};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   ADAPTIVE DENSIFICATION TEST                                ║");
    println!("║   Testing 3DGS-style Split/Clone/Prune                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);
    println!();

    // Test with fewer initial Gaussians to see if densification helps
    let initial_grid_size = 12;  // 144 Gaussians initially
    let target_gaussians = 256;  // Target count after densification

    // Test 1: Baseline (no densification)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Baseline (No Densification)");
    println!("═══════════════════════════════════════════════════════════════");

    let baseline_result = test_baseline(&target, initial_grid_size, image_path);

    // Test 2: With densification during optimization
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: With Adaptive Densification");
    println!("═══════════════════════════════════════════════════════════════");

    let densified_result = test_with_densification(&target, initial_grid_size, target_gaussians, image_path);

    // Test 3: Start with more Gaussians (fair comparison)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 3: Baseline with Same Final Count (256 Gaussians)");
    println!("═══════════════════════════════════════════════════════════════");

    let fair_baseline = test_baseline(&target, 16, image_path);  // 16×16 = 256

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<35} {:>8} {:>8} {:>10} {:>8}",
             "Method", "N_init", "N_final", "PSNR", "Time");
    println!("{:-<35} {:-<8} {:-<8} {:-<10} {:-<8}", "", "", "", "", "");
    println!("{:<35} {:>8} {:>8} {:>10.2} {:>7.1}s",
             "Baseline (144 Gaussians)",
             baseline_result.0, baseline_result.0,
             baseline_result.1, baseline_result.2);
    println!("{:<35} {:>8} {:>8} {:>10.2} {:>7.1}s",
             "With Densification",
             initial_grid_size * initial_grid_size, densified_result.0,
             densified_result.1, densified_result.2);
    println!("{:<35} {:>8} {:>8} {:>10.2} {:>7.1}s",
             "Baseline (256 Gaussians)",
             fair_baseline.0, fair_baseline.0,
             fair_baseline.1, fair_baseline.2);
    println!("{:-<35} {:-<8} {:-<8} {:-<10} {:-<8}", "", "", "", "", "");

    // Determine result
    let densification_benefit = densified_result.1 - fair_baseline.1;
    if densification_benefit > 0.1 {
        println!("\nRESULT: Densification improves quality by {:+.2} dB!", densification_benefit);
    } else if densification_benefit > -0.1 {
        println!("\nRESULT: Densification provides similar quality ({:+.2} dB)", densification_benefit);
    } else {
        println!("\nRESULT: Densification hurts quality ({:+.2} dB)", densification_benefit);
    }
    println!("═══════════════════════════════════════════════════════════════");
}

/// Test baseline without densification
/// Returns: (n_gaussians, final_psnr, elapsed_seconds)
fn test_baseline(target: &ImageBuffer<f32>, grid_size: usize, image_path: &str) -> (usize, f32, f32) {
    let (width, height) = (target.width, target.height);
    let mut gaussians = create_gaussians(target, grid_size);
    let n_gaussians = gaussians.len();

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n_gaussians, init_psnr);

    let start = Instant::now();
    let mut adam = AdamOptimizer::new();
    adam.max_iterations = 100;
    adam.optimize(&mut gaussians, target);
    let elapsed = start.elapsed().as_secs_f32();

    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);

    println!("  Final:   {} Gaussians, PSNR={:.2}dB ({:+.2}dB) in {:.1}s",
             n_gaussians, final_psnr, final_psnr - init_psnr, elapsed);

    // Save result
    let mut result = TestResult::new("adaptive_densification_baseline", image_path);
    result.optimizer = format!("Adam (no densification, {} Gaussians)", n_gaussians);
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = 100;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;
    let _ = result.save(RESULTS_DIR);

    (n_gaussians, final_psnr, elapsed)
}

/// Test with adaptive densification
/// Returns: (final_n_gaussians, final_psnr, elapsed_seconds)
fn test_with_densification(
    target: &ImageBuffer<f32>,
    initial_grid_size: usize,
    target_count: usize,
    image_path: &str,
) -> (usize, f32, f32) {
    let (width, height) = (target.width, target.height);
    let mut gaussians = create_gaussians(target, initial_grid_size);
    let n_initial = gaussians.len();

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n_initial, init_psnr);
    println!("  Target:  {} Gaussians", target_count);

    let start = Instant::now();

    // Configure densification
    let mut densifier = AdaptiveDensifier::default();
    // Lower threshold to encourage more densification
    densifier.densify_grad_threshold = 0.0001;
    densifier.densify_interval = 20;  // Check every 20 iterations
    densifier.prune_interval = 50;    // Prune every 50 iterations

    // Run optimization with periodic densification
    let mut adam = AdamOptimizer::new();

    let total_iterations = 100;
    let densify_iterations = 50;  // Densify during first 50 iterations

    for phase_iter in 0..5 {
        // Optimize for a block
        adam.max_iterations = 20;
        adam.optimize(&mut gaussians, target);

        // Densify if still in densification phase and under target count
        if phase_iter < 3 && gaussians.len() < target_count {
            let grad_mags = densifier.compute_gaussian_gradients(&gaussians, target);
            let (to_add, to_remove) = densifier.densify(&gaussians, &grad_mags);

            let n_added = to_add.len();
            let n_removed = to_remove.len();

            if n_added > 0 || n_removed > 0 {
                apply_densification(&mut gaussians, to_add, to_remove);
                println!("    Phase {}: +{} added, -{} removed, now {} Gaussians",
                        phase_iter + 1, n_added, n_removed, gaussians.len());
            }
        }

        // Prune if appropriate
        if (phase_iter + 1) * 20 >= 50 {
            let to_prune = densifier.prune(&gaussians);
            if !to_prune.is_empty() {
                apply_densification(&mut gaussians, vec![], to_prune.clone());
                println!("    Pruned {} low-opacity Gaussians, now {}",
                        to_prune.len(), gaussians.len());
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);

    println!("  Final:   {} Gaussians, PSNR={:.2}dB ({:+.2}dB) in {:.1}s",
             gaussians.len(), final_psnr, final_psnr - init_psnr, elapsed);

    // Save result
    let mut result = TestResult::new("adaptive_densification_enabled", image_path);
    result.optimizer = "Adam + Adaptive Densification".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.iterations = total_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;
    result.extra_metrics = Some(json!({
        "initial_gaussians": n_initial,
        "final_gaussians": gaussians.len(),
        "target_gaussians": target_count,
        "densify_interval": 20,
        "prune_interval": 50,
        "densify_threshold": 0.0001,
    }));
    let _ = result.save(RESULTS_DIR);

    (gaussians.len(), final_psnr, elapsed)
}

fn create_gaussians(target: &ImageBuffer<f32>, grid_size: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(grid_size * grid_size);

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            // Add small jitter for better coverage
            let jitter_x = ((gx * 7 + gy * 13) % 10) as f32 / 100.0 - 0.05;
            let jitter_y = ((gx * 11 + gy * 17) % 10) as f32 / 100.0 - 0.05;

            let x = ((gx as f32 + 0.5) / grid_size as f32 + jitter_x).clamp(0.01, 0.99);
            let y = ((gy as f32 + 0.5) / grid_size as f32 + jitter_y).clamp(0.01, 0.99);

            let px = ((x * width as f32) as u32).min(width - 1);
            let py = ((y * height as f32) as u32).min(height - 1);
            let color = target.get_pixel(px, py).unwrap();

            let scale = 1.0 / grid_size as f32;
            gaussians.push(Gaussian2D::new(
                Vector2::new(x, y),
                Euler::new(scale, scale, 0.0),
                color,
                1.0,
            ));
        }
    }

    gaussians
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height * 3) as f32;
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= count;
    if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}
