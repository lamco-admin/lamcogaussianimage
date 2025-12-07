//! Systematic test of ALL initialization strategies
//!
//! Tests every InitStrategy in lgi-core with the Adam optimizer.
//! This is micro-level exploration - testing the building blocks.
//!
//! Run: cargo run --release --example test_all_init_strategies -p lgi-encoder-v2

use lgi_core::{ImageBuffer, initializer::{Initializer, InitStrategy}};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   SYSTEMATIC INIT STRATEGY EXPLORATION                       ║");
    println!("║   Testing ALL initialization methods with Adam optimizer     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);
    println!("");

    // All initialization strategies to test
    let strategies = vec![
        ("Random", InitStrategy::Random),
        ("Grid", InitStrategy::Grid),
        ("Gradient", InitStrategy::Gradient),
        ("Importance", InitStrategy::Importance),
        ("SLIC", InitStrategy::SLIC),
    ];

    // Test with different Gaussian counts
    let gaussian_counts = vec![64, 128, 256];

    // Test with different initial scales
    let scales = vec![0.01, 0.02, 0.03];

    let mut all_results = Vec::new();

    for n in &gaussian_counts {
        for scale in &scales {
            println!("\n══════════════════════════════════════════════════════════════");
            println!("Testing N={} Gaussians, scale={:.2}", n, scale);
            println!("══════════════════════════════════════════════════════════════");

            for (name, strategy) in &strategies {
                println!("\n--- {} ---", name);

                // Initialize
                let init_start = Instant::now();
                let initializer = Initializer::new(*strategy)
                    .with_scale(*scale)
                    .with_seed(42);

                let gaussians_result = initializer.initialize(&target, *n);

                let gaussians = match gaussians_result {
                    Ok(g) => g,
                    Err(e) => {
                        println!("  FAILED: {:?}", e);
                        continue;
                    }
                };

                let init_time = init_start.elapsed().as_secs_f32();
                let actual_n = gaussians.len();

                // Initial render and PSNR
                let init_render = RendererV2::render(&gaussians, width, height);
                let init_psnr = compute_psnr(&init_render, &target);
                println!("  Init: {:.2}dB (N={}, {:.3}s)", init_psnr, actual_n, init_time);

                // Optimize with Adam
                let opt_start = Instant::now();
                let mut gaussians = gaussians;
                let mut adam = AdamOptimizer::new();
                adam.max_iterations = 100;
                adam.optimize(&mut gaussians, &target);
                let opt_time = opt_start.elapsed().as_secs_f32();

                // Final render and PSNR
                let final_render = RendererV2::render(&gaussians, width, height);
                let final_psnr = compute_psnr(&final_render, &target);
                let improvement = final_psnr - init_psnr;

                println!("  Final: {:.2}dB (+{:.2}dB, {:.1}s)", final_psnr, improvement, opt_time);

                // Check for anisotropy in SLIC results
                if *name == "SLIC" {
                    let anisotropy_sum: f32 = gaussians.iter()
                        .map(|g| {
                            let ratio = g.shape.scale_x / g.shape.scale_y.max(0.001);
                            (ratio - 1.0).abs()
                        })
                        .sum();
                    let avg_anisotropy = anisotropy_sum / gaussians.len() as f32;
                    println!("  Avg anisotropy: {:.3}", avg_anisotropy);
                }

                // Save result
                let mut result = TestResult::new(
                    &format!("init_{}_{}_s{:.0}", name.to_lowercase(), n, scale * 100.0),
                    image_path,
                );
                result.optimizer = format!("Adam + {} init (scale={})", name, scale);
                result.n_gaussians = actual_n;
                result.iterations = 100;
                result.initial_psnr = init_psnr;
                result.final_psnr = final_psnr;
                result.improvement_db = improvement;
                result.elapsed_seconds = init_time + opt_time;
                result.image_dimensions = (width, height);
                result.set_param("init_strategy", name.to_string());
                result.set_param("init_scale", *scale);
                result.set_param("target_n", *n);
                result.set_param("actual_n", actual_n);
                result.set_param("init_time_s", init_time);
                result.set_param("opt_time_s", opt_time);
                result.complete();

                let _ = result.save_with_image(&final_render, RESULTS_DIR);

                all_results.push((name.to_string(), *n, *scale, init_psnr, final_psnr, improvement));
            }
        }
    }

    // Summary table
    println!("\n\n");
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                          RESULTS SUMMARY                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!("");
    println!("{:<12} {:>6} {:>6} {:>10} {:>10} {:>10}",
             "Strategy", "N", "Scale", "Init dB", "Final dB", "Improve");
    println!("{}", "─".repeat(74));

    for (name, n, scale, init_psnr, final_psnr, improvement) in &all_results {
        println!("{:<12} {:>6} {:>6.2} {:>10.2} {:>10.2} {:>+10.2}",
                 name, n, scale, init_psnr, final_psnr, improvement);
    }

    // Best results per N
    println!("\n");
    println!("Best final PSNR per Gaussian count:");
    for n in &gaussian_counts {
        let best = all_results.iter()
            .filter(|(_, count, _, _, _, _)| *count == *n)
            .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap());
        if let Some((name, _, scale, _, final_psnr, _)) = best {
            println!("  N={}: {} (scale={:.2}) → {:.2} dB", n, name, scale, final_psnr);
        }
    }

    println!("\n✓ All init strategies tested!");
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mse: f32 = rendered
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(r, t)| {
            let dr = r.r - t.r;
            let dg = r.g - t.g;
            let db = r.b - t.b;
            (dr * dr + dg * dg + db * db) / 3.0
        })
        .sum::<f32>()
        / rendered.data.len() as f32;

    if mse < 1e-10 {
        100.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}
