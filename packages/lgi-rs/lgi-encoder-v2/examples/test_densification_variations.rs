//! Test adaptive densification variations
//!
//! The OPTIMIZER_INVENTORY.md noted that densification was tested with one configuration
//! (starting from 144 → 288 Gaussians). This test explores:
//! - Starting from fewer Gaussians (64, 100) to let densification grow more
//! - Different gradient thresholds (lower = more aggressive densification)
//! - More iterations after densification to allow refinement
//!
//! Run: cargo run --release --example test_densification_variations -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::adaptive_densification::{AdaptiveDensifier, apply_densification};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   ADAPTIVE DENSIFICATION VARIATIONS TEST                     ║");
    println!("║   Testing sparse starts + densification growth               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);

    // Test configurations: (name, initial_count, grad_threshold, densify_interval, total_iterations)
    let configs = vec![
        // Baselines (no densification)
        ("baseline_64", 64, 999.0, 1000, 200),   // High threshold = no densification
        ("baseline_100", 100, 999.0, 1000, 200),
        ("baseline_256", 256, 999.0, 1000, 200),

        // Sparse start + aggressive densification
        ("sparse_64_aggressive", 64, 0.00005, 15, 200),  // Very low threshold
        ("sparse_64_moderate", 64, 0.0001, 20, 200),
        ("sparse_64_conservative", 64, 0.0002, 25, 200),

        // Medium start + densification
        ("medium_100_aggressive", 100, 0.00005, 15, 200),
        ("medium_100_moderate", 100, 0.0001, 20, 200),

        // Different densify intervals
        ("sparse_64_fast_densify", 64, 0.0001, 10, 200),  // Every 10 iterations
        ("sparse_64_slow_densify", 64, 0.0001, 30, 200),  // Every 30 iterations
    ];

    let mut results = Vec::new();

    for (name, initial_count, grad_threshold, densify_interval, total_iterations) in &configs {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Configuration: {}", name);
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Initial Gaussians: {}", initial_count);
        println!("  Gradient threshold: {}", grad_threshold);
        println!("  Densify interval: {} iterations", densify_interval);

        let start = Instant::now();

        // Create initial Gaussians (jittered grid)
        let mut gaussians = create_jittered_grid(&target, *initial_count);

        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);
        println!("  Initial PSNR: {:.2}dB (N={})", init_psnr, gaussians.len());

        // Setup densifier
        let densifier = AdaptiveDensifier {
            densify_grad_threshold: *grad_threshold,
            split_scale_threshold: 0.04,  // Slightly lower than default for more splits
            prune_opacity_threshold: 0.003,
            densify_interval: *densify_interval,
            prune_interval: 100,  // Prune every 100 iterations
        };

        // Adam optimizer
        let mut adam = AdamOptimizer::new();
        adam.max_iterations = 1;

        let mut last_psnr = init_psnr;
        let mut densification_events = 0;
        let mut prune_events = 0;

        for iter in 0..*total_iterations {
            // Optimization step
            adam.optimize(&mut gaussians, &target);

            // Densification check
            if iter > 0 && iter % densify_interval == 0 && *grad_threshold < 1.0 {
                let grad_mags = densifier.compute_gaussian_gradients(&gaussians, &target);
                let (to_add, to_remove) = densifier.densify(&gaussians, &grad_mags);

                if !to_add.is_empty() || !to_remove.is_empty() {
                    let before_count = gaussians.len();
                    apply_densification(&mut gaussians, to_add, to_remove);
                    let after_count = gaussians.len();

                    if after_count != before_count {
                        densification_events += 1;
                        println!("    [iter {:3}] Densified: {} → {} Gaussians",
                                 iter, before_count, after_count);
                    }
                }
            }

            // Pruning check
            if iter > 0 && iter % densifier.prune_interval == 0 {
                let to_prune = densifier.prune(&gaussians);
                if !to_prune.is_empty() {
                    let before_count = gaussians.len();
                    // Remove in reverse order
                    let mut sorted = to_prune;
                    sorted.sort_unstable_by(|a, b| b.cmp(a));
                    for idx in sorted {
                        if idx < gaussians.len() {
                            gaussians.swap_remove(idx);
                        }
                    }
                    prune_events += 1;
                    println!("    [iter {:3}] Pruned: {} → {} Gaussians",
                             iter, before_count, gaussians.len());
                }
            }

            // Progress report
            if iter % 40 == 0 {
                let rendered = RendererV2::render(&gaussians, width, height);
                let psnr = compute_psnr(&rendered, &target);
                let direction = if psnr > last_psnr { "↑" } else { "↓" };
                println!("    [{:3}/{:3}] PSNR={:.2}dB {} N={}",
                         iter, total_iterations, psnr, direction, gaussians.len());
                last_psnr = psnr;
            }
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        let improvement = final_psnr - init_psnr;

        println!("  Final PSNR: {:.2}dB ({:+.2}dB)", final_psnr, improvement);
        println!("  Final Gaussians: {} (from {})", gaussians.len(), initial_count);
        println!("  Densification events: {}, Prune events: {}", densification_events, prune_events);
        println!();

        results.push((
            name.to_string(),
            *initial_count,
            gaussians.len(),
            init_psnr,
            final_psnr,
            elapsed.as_secs_f32(),
        ));

        // Save result
        let exp_name = format!("densification_{}", name);
        let mut result = TestResult::new(&exp_name, image_path);
        result.optimizer = format!("Adam + Densification ({})", name);
        result.image_dimensions = (width, height);
        result.n_gaussians = gaussians.len();
        result.iterations = *total_iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = improvement;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({
            "initial_count": initial_count,
            "final_count": gaussians.len(),
            "grad_threshold": grad_threshold,
            "densify_interval": densify_interval,
            "densification_events": densification_events,
            "prune_events": prune_events,
        }));
        let _ = result.save(RESULTS_DIR);
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<28} {:>6} {:>6} {:>10} {:>10} {:>8}",
             "Configuration", "Init", "Final", "Init PSNR", "Final PSNR", "Time");
    println!("{:-<28} {:-<6} {:-<6} {:-<10} {:-<10} {:-<8}", "", "", "", "", "", "");

    for (name, init_n, final_n, init_psnr, final_psnr, time) in &results {
        println!("{:<28} {:>6} {:>6} {:>9.2}dB {:>9.2}dB {:>7.1}s",
                 name, init_n, final_n, init_psnr, final_psnr, time);
    }

    println!("\nHYPOTHESIS: Starting sparse and growing via densification should");
    println!("be competitive with starting at the target count, potentially with");
    println!("better placement of Gaussians in high-error regions.");

    // Efficiency comparison
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                   EFFICIENCY COMPARISON                       ");
    println!("═══════════════════════════════════════════════════════════════");

    // Find baseline_256 result
    if let Some(baseline) = results.iter().find(|r| r.0 == "baseline_256") {
        println!("Baseline (256 Gaussians, no densification): {:.2}dB", baseline.4);
        println!("\nConfigurations reaching similar quality with fewer initial Gaussians:");

        for (name, init_n, final_n, _, final_psnr, _) in &results {
            if *init_n < 256 && *final_psnr >= baseline.4 - 0.5 {
                println!("  {} → {} Gaussians: {:.2}dB (within 0.5dB of baseline)",
                         name, final_n, final_psnr);
            }
        }
    }
}

fn create_jittered_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n);
    let cell_size = 1.0 / grid_size as f32;

    for i in 0..n {
        let gx = i % grid_size;
        let gy = i / grid_size;

        let jx = ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5;
        let jy = ((i * 11 + 7) % 100) as f32 / 100.0 - 0.5;

        let x = ((gx as f32 + 0.5 + jx * 0.4) / grid_size as f32).clamp(0.01, 0.99);
        let y = ((gy as f32 + 0.5 + jy * 0.4) / grid_size as f32).clamp(0.01, 0.99);

        let px = ((x * width as f32) as u32).min(width - 1);
        let py = ((y * height as f32) as u32).min(height - 1);
        let color = target.get_pixel(px, py).unwrap();

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(cell_size, cell_size, 0.0),
            color,
            1.0,
        ));
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
