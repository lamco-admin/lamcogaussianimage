//! Experiment C: Investigate ErrorDrivenAdam Plateau
//!
//! Previous tests showed ErrorDrivenAdam plateauing at 424/512 Gaussians.
//! This experiment investigates WHY by testing:
//! 1. Reduced min_gaussian_spacing (0.02 → 0.01 → 0.005)
//! 2. More aggressive split_percentile (0.05 → 0.10)
//! 3. Longer optimization per pass (50 → 100)
//! 4. Diagnostic logging of why placements fail
//!
//! Run: cargo run --release --example exp_plateau_investigation -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::{AdamOptimizer, LearningRates};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

#[derive(Clone)]
struct PlateauConfig {
    name: &'static str,
    min_spacing: f32,
    split_percentile: f32,
    iterations_per_pass: usize,
    max_passes: usize,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   EXPERIMENT C: ErrorDrivenAdam Plateau Investigation        ║");
    println!("║   Why does optimization stall before max Gaussians?          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";

    let target = match ImageBuffer::load(image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            return;
        }
    };

    println!("Image: kodim03.png ({}×{})\n", target.width, target.height);

    // Test configurations
    let configs = vec![
        PlateauConfig {
            name: "baseline",
            min_spacing: 0.02,
            split_percentile: 0.05,
            iterations_per_pass: 50,
            max_passes: 8,
        },
        PlateauConfig {
            name: "tight_spacing",
            min_spacing: 0.01,  // Halved spacing
            split_percentile: 0.05,
            iterations_per_pass: 50,
            max_passes: 8,
        },
        PlateauConfig {
            name: "very_tight",
            min_spacing: 0.005, // Very tight spacing
            split_percentile: 0.05,
            iterations_per_pass: 50,
            max_passes: 10,
        },
        PlateauConfig {
            name: "aggressive_split",
            min_spacing: 0.01,
            split_percentile: 0.10, // Top 10% instead of 5%
            iterations_per_pass: 50,
            max_passes: 10,
        },
        PlateauConfig {
            name: "longer_passes",
            min_spacing: 0.01,
            split_percentile: 0.05,
            iterations_per_pass: 100, // Double iterations
            max_passes: 8,
        },
    ];

    let mut results = Vec::new();

    for config in &configs {
        println!("\n{}", "═".repeat(70));
        println!("Config: {} (spacing={}, percentile={}, iters/pass={})",
                 config.name, config.min_spacing, config.split_percentile, config.iterations_per_pass);
        println!("{}", "═".repeat(70));

        let result = run_with_diagnostics(&target, image_path, config);
        results.push((config.name, result));
    }

    // Summary
    println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                     PLATEAU INVESTIGATION RESULTS                         ║");
    println!("╠═════════════════╤═══════════╤═══════════╤═════════════╤═══════════════════╣");
    println!("║  Config         │ N_final   │ Final PSNR│ Improvement │ Plateau Reason    ║");
    println!("╠═════════════════╪═══════════╪═══════════╪═════════════╪═══════════════════╣");

    for (name, (n_final, psnr, improvement, reason)) in &results {
        println!("║  {:14} │   {:4}    │  {:.2} dB │   {:+.2} dB  │ {:17} ║",
                 name, n_final, psnr, improvement, reason);
    }
    println!("╚═════════════════╧═══════════╧═══════════╧═════════════╧═══════════════════╝");

    // Analysis
    let max_n = results.iter().map(|(_, (n, _, _, _))| *n).max().unwrap_or(0);
    let best_psnr = results.iter().map(|(_, (_, p, _, _))| *p).fold(f32::NEG_INFINITY, f32::max);

    println!("\n┌───────────────────────────────────────────────────────────────┐");
    println!("│ KEY OBSERVATIONS:                                             │");
    println!("├───────────────────────────────────────────────────────────────┤");
    println!("│ Max Gaussians reached: {:4}                                   │", max_n);
    println!("│ Best PSNR achieved:    {:.2} dB                               │", best_psnr);

    // Check if tighter spacing helps
    let baseline = results.iter().find(|(n, _)| *n == "baseline");
    let tight = results.iter().find(|(n, _)| *n == "tight_spacing");
    if let (Some((_, (n_b, _, _, _))), Some((_, (n_t, _, _, _)))) = (baseline, tight) {
        if n_t > n_b {
            println!("│ → Tighter spacing HELPS: {} → {} Gaussians             │", n_b, n_t);
        } else {
            println!("│ → Tighter spacing didn't help (spacing not the issue)     │");
        }
    }
    println!("└───────────────────────────────────────────────────────────────┘");
}

fn run_with_diagnostics(
    target: &ImageBuffer<f32>,
    image_path: &str,
    config: &PlateauConfig,
) -> (usize, f32, f32, &'static str) {
    let (width, height) = (target.width, target.height);
    let initial_gaussians = 64;
    let max_gaussians = 512;
    let min_improvement_db = 0.05;

    let start = Instant::now();

    // Initialize
    let grid_size = (initial_gaussians as f32).sqrt().ceil() as u32;
    let mut gaussians = initialize_uniform_grid(target, grid_size);

    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(target, &rendered);

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", gaussians.len(), init_psnr);

    let mut last_psnr = init_psnr;
    let mut stall_count = 0;
    let mut plateau_reason: &'static str = "completed";

    // Diagnostic counters
    let mut total_candidates = 0usize;
    let mut rejected_existing = 0usize;
    let mut rejected_selected = 0usize;

    for pass in 0..config.max_passes {
        // Run Adam optimization
        let mut adam = AdamOptimizer::new();
        adam.learning_rates = LearningRates::default();
        adam.max_iterations = config.iterations_per_pass;
        adam.optimize(&mut gaussians, target);

        let rendered = RendererV2::render(&gaussians, width, height);
        let psnr = compute_psnr(target, &rendered);
        let improvement = psnr - last_psnr;
        let direction = if improvement > 0.0 { "↑" } else { "↓" };

        println!("  Pass {}: N={:4}, PSNR={:.2}dB {} ({:+.2}dB)",
                 pass, gaussians.len(), psnr, direction, improvement);

        // Check limits
        if gaussians.len() >= max_gaussians {
            plateau_reason = "max_gaussians";
            println!("  ✓ Reached max Gaussians limit");
            break;
        }

        if pass > 0 && improvement < min_improvement_db {
            stall_count += 1;
            if stall_count >= 2 {
                plateau_reason = "stalled";
                println!("  ✓ Improvement stalled");
                break;
            }
        } else {
            stall_count = 0;
        }

        last_psnr = psnr;

        // Find hotspots with detailed diagnostics
        let error_map = compute_error_map(target, &rendered);
        let (hotspots, diag) = find_hotspots_with_diagnostics(
            &error_map, target, &gaussians,
            config.min_spacing, config.split_percentile
        );

        total_candidates += diag.0;
        rejected_existing += diag.1;
        rejected_selected += diag.2;

        if hotspots.is_empty() {
            plateau_reason = "no_hotspots";
            println!("  ✓ No suitable hotspots (candidates={}, rejected_existing={}, rejected_selected={})",
                     diag.0, diag.1, diag.2);
            break;
        }

        let added = add_gaussians_at_hotspots(&mut gaussians, target, &hotspots, max_gaussians);
        println!("    → Added {} Gaussians (candidates={}, rejected={})",
                 added, diag.0, diag.1 + diag.2);
    }

    // Final polish
    let mut adam = AdamOptimizer::new();
    adam.max_iterations = 20;
    adam.optimize(&mut gaussians, target);

    let elapsed = start.elapsed().as_secs_f32();
    let final_rendered = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(target, &final_rendered);
    let improvement = final_psnr - init_psnr;

    println!("\n  Final: {} Gaussians, PSNR={:.2}dB ({:+.2}dB), {:.1}s",
             gaussians.len(), final_psnr, improvement, elapsed);
    println!("  Diagnostics: {} candidates, {} rejected (existing), {} rejected (spacing)",
             total_candidates, rejected_existing, rejected_selected);

    // Save result
    let mut result = TestResult::new(&format!("plateau_{}", config.name), image_path);
    result.optimizer = format!("ErrorDrivenAdam ({})", config.name);
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.iterations = config.max_passes * config.iterations_per_pass;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.elapsed_seconds = elapsed;
    result.set_param("min_spacing", config.min_spacing);
    result.set_param("split_percentile", config.split_percentile);
    result.set_param("iterations_per_pass", config.iterations_per_pass as f32);
    result.set_param("total_candidates", total_candidates as f32);
    result.set_param("rejected_existing", rejected_existing as f32);
    result.set_param("rejected_selected", rejected_selected as f32);
    result.notes = format!("plateau_reason: {}", plateau_reason);
    result.complete();
    let _ = result.save_with_image(&final_rendered, RESULTS_DIR);

    (gaussians.len(), final_psnr, improvement, plateau_reason)
}

// ==================== HELPER FUNCTIONS ====================

fn initialize_uniform_grid(target: &ImageBuffer<f32>, grid_size: u32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut gaussians = Vec::new();
    let num_gaussians = grid_size * grid_size;
    let gamma = 0.8;
    let width = target.width as f32;
    let height = target.height as f32;
    let sigma_base_px = gamma * ((width * height) / num_gaussians as f32).sqrt();
    let sigma_norm = (sigma_base_px / width).clamp(0.01, 0.25);

    let step_x = target.width / grid_size;
    let step_y = target.height / grid_size;

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx * step_x + step_x / 2).min(target.width - 1);
            let y = (gy * step_y + step_y / 2).min(target.height - 1);
            let position = Vector2::new(x as f32 / width, y as f32 / height);
            let color = target.get_pixel(x, y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            gaussians.push(Gaussian2D::new(
                position,
                Euler::isotropic(sigma_norm),
                color,
                1.0,
            ));
        }
    }
    gaussians
}

fn compute_error_map(target: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> Vec<f32> {
    target.data.iter().zip(rendered.data.iter())
        .map(|(t, r)| (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2))
        .collect()
}

struct Hotspot { x: u32, y: u32, error: f32 }

fn find_hotspots_with_diagnostics(
    error_map: &[f32],
    target: &ImageBuffer<f32>,
    existing: &[Gaussian2D<f32, Euler<f32>>],
    min_spacing: f32,
    percentile: f32,
) -> (Vec<Hotspot>, (usize, usize, usize)) {  // (hotspots, (candidates, rejected_existing, rejected_selected))
    let (width, height) = (target.width, target.height);
    let w_f = width as f32;
    let h_f = height as f32;

    let mut sorted = error_map.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold_idx = (sorted.len() as f32 * percentile) as usize;
    let threshold = sorted.get(threshold_idx).copied().unwrap_or(0.0);

    let mut candidates: Vec<Hotspot> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if error_map[idx] > threshold {
                candidates.push(Hotspot { x, y, error: error_map[idx] });
            }
        }
    }
    candidates.sort_by(|a, b| b.error.partial_cmp(&a.error).unwrap());

    let total_candidates = candidates.len();
    let mut rejected_existing = 0usize;
    let mut rejected_selected = 0usize;

    let min_dist_sq = min_spacing.powi(2);
    let mut selected: Vec<Hotspot> = Vec::new();

    for candidate in candidates {
        let cx = candidate.x as f32 / w_f;
        let cy = candidate.y as f32 / h_f;

        let too_close_existing = existing.iter().any(|g| {
            let dx = g.position.x - cx;
            let dy = g.position.y - cy;
            dx * dx + dy * dy < min_dist_sq
        });

        if too_close_existing {
            rejected_existing += 1;
            continue;
        }

        let too_close_selected = selected.iter().any(|h| {
            let hx = h.x as f32 / w_f;
            let hy = h.y as f32 / h_f;
            let dx = hx - cx;
            let dy = hy - cy;
            dx * dx + dy * dy < min_dist_sq
        });

        if too_close_selected {
            rejected_selected += 1;
            continue;
        }

        selected.push(candidate);
        if selected.len() >= 30 { break; }
    }

    (selected, (total_candidates, rejected_existing, rejected_selected))
}

fn add_gaussians_at_hotspots(
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    hotspots: &[Hotspot],
    max_gaussians: usize,
) -> usize {
    let width = target.width as f32;
    let height = target.height as f32;
    let mut added = 0;

    for hotspot in hotspots {
        if gaussians.len() >= max_gaussians { break; }
        let position = Vector2::new(hotspot.x as f32 / width, hotspot.y as f32 / height);
        let color = target.get_pixel(hotspot.x, hotspot.y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
        gaussians.push(Gaussian2D::new(position, Euler::isotropic(0.03), color, 1.0));
        added += 1;
    }
    added
}

fn compute_psnr(target: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (t, r) in target.data.iter().zip(rendered.data.iter()) {
        mse += (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
    }
    mse /= (target.width * target.height * 3) as f32;
    if mse < 1e-10 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}
