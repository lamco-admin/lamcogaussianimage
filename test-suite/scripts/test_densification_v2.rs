//! Densification V2 Test - Fixed issues from overnight run
//!
//! Key changes from original:
//! 1. DELAY densification until iter 50+ (let initial Gaussians optimize first)
//! 2. RESET opacity after split (children get opacity 1.0, not 0.5)
//! 3. SLOWER growth (higher threshold, longer interval)
//! 4. MORE iterations after growth (to let new Gaussians optimize)
//!
//! Run: cargo run --release --example test_densification_v2 -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   DENSIFICATION V2 TEST - Fixed Opacity & Timing Issues      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}\n", width, height);

    // V2 configurations with delayed densification
    // Format: (name, initial, delay_iters, threshold, interval, max, total_iters)
    let configs = vec![
        // Baseline: no densification
        ("baseline_256", 256, 999, 999.0, 999, 256, 200),

        // V2: Delayed start, slow growth
        ("v2_64_delayed", 64, 50, 0.0003, 30, 256, 300),
        ("v2_64_very_delayed", 64, 80, 0.0002, 25, 256, 350),

        // V2: Start from 100, grow to 256
        ("v2_100_delayed", 100, 50, 0.0003, 30, 256, 300),

        // V2: Ultra conservative - only split worst performers
        ("v2_64_conservative", 64, 100, 0.0005, 40, 256, 400),

        // V2: Match 256 exactly
        ("v2_128_to_256", 128, 60, 0.0003, 30, 256, 300),
    ];

    let mut results = Vec::new();

    for (name, initial, delay, threshold, interval, max, total) in &configs {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Config: {}", name);
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Initial: {}, Delay: {}, Threshold: {:.4}", initial, delay, threshold);
        println!("  Interval: {}, Max: {}, Total iters: {}", interval, max, total);

        let start = Instant::now();

        // Create initial Gaussians
        let mut gaussians = create_jittered_grid(&target, *initial);

        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);
        println!("  Initial PSNR: {:.2}dB (N={})", init_psnr, gaussians.len());

        // Adam optimizer
        let mut adam = AdamOptimizer::new();
        adam.max_iterations = 1;

        let mut last_psnr = init_psnr;
        let mut densification_events = 0;
        let mut total_added = 0;

        for iter in 0..*total {
            // Optimization step
            adam.optimize(&mut gaussians, &target);

            // Densification: only after delay, only if under max
            if iter >= *delay && iter % *interval == 0 && *threshold < 1.0 {
                if gaussians.len() < *max {
                    let added = densify_v2(&mut gaussians, &target, *threshold, *max);
                    if added > 0 {
                        densification_events += 1;
                        total_added += added;
                        println!("    [iter {:3}] Densified: +{} → {} total",
                                 iter, added, gaussians.len());
                    }
                }
            }

            // Progress
            if iter % 40 == 0 || iter == *total - 1 {
                let rendered = RendererV2::render(&gaussians, width, height);
                let psnr = compute_psnr(&rendered, &target);
                let direction = if psnr > last_psnr { "↑" } else { "↓" };
                println!("    [{:3}/{:3}] PSNR={:.2}dB {} N={}",
                         iter, total, psnr, direction, gaussians.len());
                last_psnr = psnr;
            }
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        let improvement = final_psnr - init_psnr;

        println!("  Final: {:.2}dB ({:+.2}dB) in {:.1}s", final_psnr, improvement, elapsed.as_secs_f32());
        println!("  Gaussians: {} → {} (events: {})", initial, gaussians.len(), densification_events);

        results.push((name.to_string(), *initial, gaussians.len(), init_psnr, final_psnr, elapsed.as_secs_f32()));

        // Save result
        let exp_name = format!("densification_v2_{}", name);
        let mut result = TestResult::new(&exp_name, image_path);
        result.optimizer = format!("Adam + DensificationV2 (delay={})", delay);
        result.image_dimensions = (width, height);
        result.n_gaussians = gaussians.len();
        result.iterations = *total;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = improvement;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({
            "initial_count": initial,
            "final_count": gaussians.len(),
            "delay_iters": delay,
            "threshold": threshold,
            "interval": interval,
            "max_gaussians": max,
            "densification_events": densification_events,
            "total_added": total_added,
        }));
        let _ = result.save(RESULTS_DIR);
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<25} {:>6} {:>6} {:>10} {:>10}", "Config", "Init", "Final", "PSNR", "vs 256");
    println!("{:-<25} {:-<6} {:-<6} {:-<10} {:-<10}", "", "", "", "", "");

    let baseline_psnr = results.iter().find(|r| r.0 == "baseline_256").map(|r| r.4).unwrap_or(0.0);

    for (name, init_n, final_n, _, final_psnr, _) in &results {
        let vs_baseline = final_psnr - baseline_psnr;
        println!("{:<25} {:>6} {:>6} {:>9.2}dB {:>+9.2}dB",
                 name, init_n, final_n, final_psnr, vs_baseline);
    }
}

/// V2 Densification: Split worst-performing Gaussians with FULL opacity reset
fn densify_v2(
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    threshold: f32,
    max_count: usize,
) -> usize {
    let width = target.width;
    let height = target.height;
    let rendered = RendererV2::render(gaussians, width, height);

    // Compute per-Gaussian error contribution
    let mut errors: Vec<(usize, f32)> = Vec::new();

    for (i, gaussian) in gaussians.iter().enumerate() {
        let px_x = (gaussian.position.x * width as f32) as u32;
        let px_y = (gaussian.position.y * height as f32) as u32;

        // Measure error in larger region (9x9)
        let mut error_sum = 0.0;
        let mut count = 0;

        for dy in -4i32..=4 {
            for dx in -4i32..=4 {
                let x = (px_x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                let y = (px_y as i32 + dy).clamp(0, height as i32 - 1) as u32;

                if let (Some(t), Some(r)) = (target.get_pixel(x, y), rendered.get_pixel(x, y)) {
                    let err = (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
                    error_sum += err;
                    count += 1;
                }
            }
        }

        let avg_error = if count > 0 { error_sum / count as f32 } else { 0.0 };
        if avg_error > threshold {
            errors.push((i, avg_error));
        }
    }

    // Sort by error (highest first)
    errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // How many can we add?
    let room = max_count.saturating_sub(gaussians.len());
    let to_split = errors.len().min(room);

    if to_split == 0 {
        return 0;
    }

    // Split the worst-performing Gaussians
    let mut new_gaussians = Vec::new();

    for &(idx, _) in errors.iter().take(to_split) {
        let parent = &gaussians[idx];

        // Split along major axis
        let angle = parent.shape.rotation;
        let offset_x = angle.cos() * parent.shape.scale_x * 0.5;
        let offset_y = angle.sin() * parent.shape.scale_x * 0.5;

        // Child 1: Offset in +direction, FULL OPACITY, smaller scale
        new_gaussians.push(Gaussian2D::new(
            Vector2::new(
                (parent.position.x + offset_x).clamp(0.01, 0.99),
                (parent.position.y + offset_y).clamp(0.01, 0.99),
            ),
            Euler::new(
                parent.shape.scale_x * 0.7,  // Reduce scale
                parent.shape.scale_y * 0.7,
                parent.shape.rotation,
            ),
            parent.color,
            1.0,  // FULL OPACITY - key fix!
        ));

        // Also reduce parent's scale and re-optimize its position slightly
        // (We keep the parent but shrink it)
    }

    // Add new Gaussians (keep all parents)
    gaussians.extend(new_gaussians.clone());

    new_gaussians.len()
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
