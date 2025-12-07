//! Test Iterative Refinement with Artifact Feedback
//!
//! Full closed-loop encoding:
//! 1. Analyze image
//! 2. Initialize Gaussians based on features
//! 3. Optimize (phase 1)
//! 4. Analyze artifacts
//! 5. Apply refinement actions
//! 6. Optimize (phase 2)
//! 7. Repeat until converged

use lgi_core::{
    ImageBuffer,
    image_analysis::{ImageAnalysis, AnalysisConfig},
    iterative_refinement::{IterativeRefiner, RefinementConfig, RefinementAction, apply_action, print_actions},
};
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_math::color::Color4;
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("      ITERATIVE REFINEMENT TEST - Artifact-Guided Optimization");
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

    // Step 1: Initial Analysis
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 1: Initial Image Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let analysis = ImageAnalysis::compute(&target, AnalysisConfig::default())
        .expect("Analysis failed");
    analysis.print_summary();
    println!();

    let strategy = analysis.suggest_strategy();
    strategy.print();
    println!();

    let refiner = IterativeRefiner::new(RefinementConfig {
        error_threshold: 0.015,
        max_add_per_iter: 24,
        max_prune_per_iter: 8,
        bleeding_threshold: 0.15,
        blur_threshold: 0.25,
        ..Default::default()
    });

    // Initialize test result
    let mut result = TestResult::new("iterative_refinement", image_path);
    result.optimizer = "Iterative Refinement with Artifact Feedback".to_string();
    result.image_dimensions = (width, height);

    // Step 2: Initialize Gaussians
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 2: Feature-Aware Initialization");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let initial_count = strategy.gaussian_count;
    let mut gaussians = analysis.generate_gaussians(initial_count, &target, 42);
    println!("  Initialized: {} Gaussians", gaussians.len());

    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
    result.initial_psnr = init_psnr;
    result.n_gaussians = gaussians.len();
    println!("  Initial PSNR: {:.2}dB\n", init_psnr);

    // Refinement loop parameters
    let max_refinement_rounds = 3;
    let iterations_per_round = 100;
    let mut optimizer = AdamOptimizer::new();
    optimizer.max_iterations = iterations_per_round;

    let start = Instant::now();
    let mut last_psnr = init_psnr;
    let mut round_psnrs = vec![init_psnr];

    for round in 0..max_refinement_rounds {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("ROUND {} - {} Gaussians", round + 1, gaussians.len());
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // Phase A: Optimization
        println!("  Phase A: Optimization ({} iterations)", iterations_per_round);

        for iter in 0..iterations_per_round {
            optimizer.optimize(&mut gaussians, &target);

            if iter % 25 == 0 || iter == iterations_per_round - 1 {
                let rendered = RendererV2::render(&gaussians, width, height);
                let psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
                let dir = if psnr > last_psnr { "↑" } else { "↓" };
                println!("    [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                         iter + 1, iterations_per_round, psnr, dir, psnr - last_psnr);
                last_psnr = psnr;
            }
        }

        let rendered = RendererV2::render(&gaussians, width, height);
        let post_opt_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
        println!("\n  Post-optimization: {:.2}dB\n", post_opt_psnr);

        // Phase B: Analyze and Refine
        println!("  Phase B: Artifact Analysis & Refinement");

        let actions = refiner.analyze_and_suggest(&gaussians, &rendered, &target, &analysis)
            .expect("Analysis failed");

        print_actions(&actions);

        // Check if any meaningful actions
        let has_actions = actions.iter().any(|a| !matches!(a, RefinementAction::NoAction { .. }));

        if !has_actions {
            println!("\n  ✓ Converged - no further refinement needed");
            break;
        }

        // Apply actions
        let pre_apply_count = gaussians.len();
        for action in &actions {
            apply_action(&mut gaussians, action, &target);
        }
        let post_apply_count = gaussians.len();

        println!("\n  Applied: {} → {} Gaussians", pre_apply_count, post_apply_count);

        // Reset optimizer state for new Gaussians
        optimizer = AdamOptimizer::new();
        optimizer.max_iterations = iterations_per_round;

        round_psnrs.push(post_opt_psnr);
        println!();
    }

    let elapsed = start.elapsed();

    // Final render
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

    // Complete result
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f64() as f32;
    result.iterations = max_refinement_rounds * iterations_per_round;
    result.n_gaussians = gaussians.len();
    result.complete();

    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    ITERATIVE REFINEMENT SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  Rounds: {}", round_psnrs.len());
    println!("  PSNR progression:");
    for (i, psnr) in round_psnrs.iter().enumerate() {
        let label = if i == 0 { "Initial" } else { &format!("Round {}", i) };
        println!("    {:10}: {:.2}dB", label, psnr);
    }
    println!("    {:10}: {:.2}dB", "Final", final_psnr);
    println!();
    println!("  Initial Gaussians: {}", initial_count);
    println!("  Final Gaussians: {}", gaussians.len());
    println!("  Total improvement: {:+.2}dB", final_psnr - init_psnr);
    println!("  Time: {:.1}s", elapsed.as_secs_f64());

    println!("\n═══════════════════════════════════════════════════════════════════════");
}
