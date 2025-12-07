//! Overnight Densification Test - Properly tests sparse starts with densification
//!
//! The previous test only ran baselines before being interrupted.
//! This test focuses on the actual sparse→dense growth configurations.
//!
//! Run: cargo run --release --example test_overnight_densification -p lgi-encoder-v2

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
    println!("║   OVERNIGHT DENSIFICATION TEST                               ║");
    println!("║   Sparse starts → Dense via split/clone/prune                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Test on multiple images
    let test_images = vec![
        ("kodim03", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png"),
        ("kodim05", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim05.png"),
        ("kodim08", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim08.png"),
    ];

    // Densification configurations to test
    // Format: (name, initial_count, grad_threshold, densify_interval, max_gaussians, iterations)
    let configs = vec![
        // Sparse starts with aggressive densification
        ("sparse_32_aggressive", 32, 0.00005, 10, 300, 300),
        ("sparse_32_moderate", 32, 0.0001, 15, 300, 300),
        ("sparse_64_aggressive", 64, 0.00005, 10, 300, 300),
        ("sparse_64_moderate", 64, 0.0001, 15, 300, 300),
        ("sparse_64_conservative", 64, 0.0002, 20, 300, 300),

        // Medium starts
        ("medium_100_aggressive", 100, 0.00005, 10, 350, 300),
        ("medium_100_moderate", 100, 0.0001, 15, 350, 300),

        // Very sparse with long training
        ("very_sparse_16", 16, 0.00003, 8, 300, 400),

        // Baseline: 256 direct (no densification)
        ("baseline_256", 256, 999.0, 1000, 256, 200),
    ];

    for (image_name, image_path) in &test_images {
        println!("\n{}", "=".repeat(70));
        println!("Processing image: {}", image_name);
        println!("{}\n", "=".repeat(70));

        let target = match ImageBuffer::load(image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Failed to load {}: {:?}", image_path, e);
                continue;
            }
        };
        let (width, height) = (target.width, target.height);

        for (name, initial_count, grad_threshold, densify_interval, max_gaussians, iterations) in &configs {
            println!("───────────────────────────────────────────────────────────────");
            println!("Config: {} on {}", name, image_name);
            println!("───────────────────────────────────────────────────────────────");
            println!("  Initial: {} Gaussians, max: {}", initial_count, max_gaussians);
            println!("  Threshold: {:.5}, Interval: {}", grad_threshold, densify_interval);

            let start = Instant::now();

            // Create initial Gaussians
            let mut gaussians = create_jittered_grid(&target, *initial_count);
            let init_render = RendererV2::render(&gaussians, width, height);
            let init_psnr = compute_psnr(&init_render, &target);
            println!("  Initial PSNR: {:.2}dB (N={})", init_psnr, gaussians.len());

            // Setup densifier
            let densifier = AdaptiveDensifier {
                densify_grad_threshold: *grad_threshold,
                split_scale_threshold: 0.03,
                prune_opacity_threshold: 0.003,
                densify_interval: *densify_interval,
                prune_interval: 50,
            };

            // Adam optimizer (one step at a time)
            let mut adam = AdamOptimizer::new();
            adam.max_iterations = 1;

            let mut last_psnr = init_psnr;
            let mut densification_events = 0;
            let mut prune_events = 0;
            let mut total_added = 0;
            let mut total_removed = 0;

            for iter in 0..*iterations {
                // Optimization step
                adam.optimize(&mut gaussians, &target);

                // Densification check
                if iter > 10 && iter % *densify_interval == 0 && *grad_threshold < 1.0 {
                    if gaussians.len() < *max_gaussians {
                        let grad_mags = densifier.compute_gaussian_gradients(&gaussians, &target);
                        let (to_add, to_remove) = densifier.densify(&gaussians, &grad_mags);

                        if !to_add.is_empty() {
                            let before = gaussians.len();
                            let to_add_limited: Vec<_> = to_add.into_iter()
                                .take(*max_gaussians - before)
                                .collect();
                            apply_densification(&mut gaussians, to_add_limited, vec![]);
                            let added = gaussians.len() - before;
                            if added > 0 {
                                densification_events += 1;
                                total_added += added;
                            }
                        }
                    }
                }

                // Pruning check
                if iter > 20 && iter % densifier.prune_interval == 0 {
                    let to_prune = densifier.prune(&gaussians);
                    if !to_prune.is_empty() {
                        let before = gaussians.len();
                        let mut sorted = to_prune;
                        sorted.sort_unstable_by(|a, b| b.cmp(a));
                        for idx in sorted {
                            if idx < gaussians.len() {
                                gaussians.swap_remove(idx);
                            }
                        }
                        let removed = before - gaussians.len();
                        if removed > 0 {
                            prune_events += 1;
                            total_removed += removed;
                        }
                    }
                }

                // Progress report
                if iter % 30 == 0 || iter == *iterations - 1 {
                    let rendered = RendererV2::render(&gaussians, width, height);
                    let psnr = compute_psnr(&rendered, &target);
                    let direction = if psnr > last_psnr { "↑" } else { "↓" };
                    println!("    [{:3}/{:3}] PSNR={:.2}dB {} N={} (+{} -{} events)",
                             iter, iterations, psnr, direction, gaussians.len(),
                             densification_events, prune_events);
                    last_psnr = psnr;
                }
            }

            let elapsed = start.elapsed();
            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = compute_psnr(&final_render, &target);
            let improvement = final_psnr - init_psnr;

            println!("  Final: {:.2}dB ({:+.2}dB) in {:.1}s", final_psnr, improvement, elapsed.as_secs_f32());
            println!("  Gaussians: {} → {} (added: {}, removed: {})",
                     initial_count, gaussians.len(), total_added, total_removed);

            // Save result
            let exp_name = format!("overnight_densify_{}_{}", name, image_name);
            let mut result = TestResult::new(&exp_name, image_path);
            result.optimizer = format!("Adam + Densification ({})", name);
            result.image_dimensions = (width, height);
            result.n_gaussians = gaussians.len();
            result.iterations = *iterations;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = improvement;
            result.elapsed_seconds = elapsed.as_secs_f32();
            result.extra_metrics = Some(json!({
                "initial_count": initial_count,
                "final_count": gaussians.len(),
                "max_gaussians": max_gaussians,
                "grad_threshold": grad_threshold,
                "densify_interval": densify_interval,
                "densification_events": densification_events,
                "prune_events": prune_events,
                "total_added": total_added,
                "total_removed": total_removed,
            }));
            let _ = result.save(RESULTS_DIR);
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("OVERNIGHT DENSIFICATION TEST COMPLETE");
    println!("{}", "=".repeat(70));
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
            Euler::new(cell_size * 1.2, cell_size * 1.2, 0.0),  // Slightly larger for coverage
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
