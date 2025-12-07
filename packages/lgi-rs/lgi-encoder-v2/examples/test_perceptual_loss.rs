//! Test perceptual loss functions for Gaussian optimization
//!
//! Hypothesis: Hybrid L2 + MS-SSIM loss improves perceptual quality while maintaining PSNR.
//!
//! Tests:
//! 1. Pure L2 loss (baseline)
//! 2. Pure MS-SSIM loss
//! 3. Hybrid: α·L2 + (1-α)·(1-MS-SSIM)
//!
//! Run: cargo run --release --example test_perceptual_loss -p lgi-encoder-v2

use lgi_core::{ImageBuffer, ms_ssim::MSSSIM};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   PERCEPTUAL LOSS EXPERIMENTS                                ║");
    println!("║   Testing L2, MS-SSIM, and hybrid loss functions             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load");
    let (width, height) = (target.width, target.height);

    let grid_size = 16;  // 256 Gaussians
    let n_gaussians = grid_size * grid_size;
    let iterations = 100;

    println!("Image: {}x{}, Gaussians: {}, Iterations: {}", width, height, n_gaussians, iterations);
    println!();

    // Create baseline Gaussians (same for all tests)
    let base_gaussians = create_jittered_gaussians(&target, grid_size);

    let init_render = RendererV2::render(&base_gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, &target);
    let init_msssim = compute_ms_ssim(&init_render, &target);

    println!("Initial: PSNR={:.2}dB, MS-SSIM={:.4}\n", init_psnr, init_msssim);

    // Test 1: Pure L2 loss (Adam baseline)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Pure L2 Loss (Adam baseline)");
    println!("═══════════════════════════════════════════════════════════════");

    let (psnr_l2, msssim_l2, time_l2) = run_adam_l2(&base_gaussians, &target, iterations);
    let improve_l2 = psnr_l2 - init_psnr;

    println!("  Final:  PSNR={:.2}dB, MS-SSIM={:.4}", psnr_l2, msssim_l2);
    println!("  Result: {:+.2}dB improvement in {:.1}s\n", improve_l2, time_l2);

    save_result("perceptual_pure_l2", image_path, (width, height), n_gaussians, iterations,
                init_psnr, psnr_l2, improve_l2, time_l2, "L2", init_msssim, msssim_l2, 1.0);

    // Test 2: Hybrid 80% L2 + 20% MS-SSIM
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 2: Hybrid Loss (80% L2 + 20% MS-SSIM)");
    println!("═══════════════════════════════════════════════════════════════");

    let (psnr_h80, msssim_h80, time_h80) = run_hybrid(&base_gaussians, &target, iterations, 0.8);
    let improve_h80 = psnr_h80 - init_psnr;

    println!("  Final:  PSNR={:.2}dB, MS-SSIM={:.4}", psnr_h80, msssim_h80);
    println!("  Result: {:+.2}dB improvement in {:.1}s\n", improve_h80, time_h80);

    save_result("perceptual_hybrid_80_20", image_path, (width, height), n_gaussians, iterations,
                init_psnr, psnr_h80, improve_h80, time_h80, "Hybrid 80/20", init_msssim, msssim_h80, 0.8);

    // Test 3: Hybrid 50% L2 + 50% MS-SSIM
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 3: Hybrid Loss (50% L2 + 50% MS-SSIM)");
    println!("═══════════════════════════════════════════════════════════════");

    let (psnr_h50, msssim_h50, time_h50) = run_hybrid(&base_gaussians, &target, iterations, 0.5);
    let improve_h50 = psnr_h50 - init_psnr;

    println!("  Final:  PSNR={:.2}dB, MS-SSIM={:.4}", psnr_h50, msssim_h50);
    println!("  Result: {:+.2}dB improvement in {:.1}s\n", improve_h50, time_h50);

    save_result("perceptual_hybrid_50_50", image_path, (width, height), n_gaussians, iterations,
                init_psnr, psnr_h50, improve_h50, time_h50, "Hybrid 50/50", init_msssim, msssim_h50, 0.5);

    // Test 4: Hybrid 20% L2 + 80% MS-SSIM
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 4: Hybrid Loss (20% L2 + 80% MS-SSIM)");
    println!("═══════════════════════════════════════════════════════════════");

    let (psnr_h20, msssim_h20, time_h20) = run_hybrid(&base_gaussians, &target, iterations, 0.2);
    let improve_h20 = psnr_h20 - init_psnr;

    println!("  Final:  PSNR={:.2}dB, MS-SSIM={:.4}", psnr_h20, msssim_h20);
    println!("  Result: {:+.2}dB improvement in {:.1}s\n", improve_h20, time_h20);

    save_result("perceptual_hybrid_20_80", image_path, (width, height), n_gaussians, iterations,
                init_psnr, psnr_h20, improve_h20, time_h20, "Hybrid 20/80", init_msssim, msssim_h20, 0.2);

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<20} {:>10} {:>10} {:>10} {:>10}", "Loss Function", "PSNR", "Improve", "MS-SSIM", "Time");
    println!("{:-<20} {:-<10} {:-<10} {:-<10} {:-<10}", "", "", "", "", "");
    println!("{:<20} {:>10.2} {:>+10.2} {:>10.4} {:>9.1}s", "L2 (baseline)", psnr_l2, improve_l2, msssim_l2, time_l2);
    println!("{:<20} {:>10.2} {:>+10.2} {:>10.4} {:>9.1}s", "Hybrid 80/20", psnr_h80, improve_h80, msssim_h80, time_h80);
    println!("{:<20} {:>10.2} {:>+10.2} {:>10.4} {:>9.1}s", "Hybrid 50/50", psnr_h50, improve_h50, msssim_h50, time_h50);
    println!("{:<20} {:>10.2} {:>+10.2} {:>10.4} {:>9.1}s", "Hybrid 20/80", psnr_h20, improve_h20, msssim_h20, time_h20);
    println!("{:-<20} {:-<10} {:-<10} {:-<10} {:-<10}", "", "", "", "", "");

    // Analysis
    println!("\nAnalysis:");
    let best_psnr = psnr_l2.max(psnr_h80).max(psnr_h50).max(psnr_h20);
    let best_msssim = msssim_l2.max(msssim_h80).max(msssim_h50).max(msssim_h20);

    if psnr_l2 == best_psnr && msssim_l2 >= best_msssim * 0.99 {
        println!("  HYPOTHESIS REJECTED: Pure L2 achieves best or comparable quality on both metrics.");
    } else if psnr_h50 == best_psnr || psnr_h80 == best_psnr {
        println!("  HYPOTHESIS CONFIRMED: Hybrid loss improves over pure L2!");
        println!("  Best config achieves {:+.2}dB PSNR gain with {:.4} MS-SSIM.", best_psnr - psnr_l2, best_msssim);
    } else {
        println!("  MIXED RESULTS: Trade-offs between PSNR and MS-SSIM observed.");
    }
    println!("═══════════════════════════════════════════════════════════════");
}

/// Run Adam optimizer with pure L2 loss (baseline)
fn run_adam_l2(
    base_gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    iterations: usize,
) -> (f32, f32, f32) {
    let mut gaussians = base_gaussians.to_vec();
    let (width, height) = (target.width, target.height);

    let start = Instant::now();
    let mut adam = AdamOptimizer::new();
    adam.max_iterations = iterations;
    adam.optimize(&mut gaussians, target);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(&gaussians, width, height);
    let psnr = compute_psnr(&final_render, target);
    let msssim = compute_ms_ssim(&final_render, target);

    (psnr, msssim, elapsed.as_secs_f32())
}

/// Run custom optimization with hybrid L2 + MS-SSIM loss
fn run_hybrid(
    base_gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    iterations: usize,
    l2_weight: f32,
) -> (f32, f32, f32) {
    let mut gaussians = base_gaussians.to_vec();
    let (width, height) = (target.width, target.height);
    let msssim_weight = 1.0 - l2_weight;

    let start = Instant::now();

    // Adam-like optimizer with hybrid loss
    let mut m_pos: Vec<(f32, f32)> = vec![(0.0, 0.0); gaussians.len()];
    let mut v_pos: Vec<(f32, f32)> = vec![(0.0, 0.0); gaussians.len()];
    let mut m_color: Vec<(f32, f32, f32)> = vec![(0.0, 0.0, 0.0); gaussians.len()];
    let mut v_color: Vec<(f32, f32, f32)> = vec![(0.0, 0.0, 0.0); gaussians.len()];
    let mut m_scale: Vec<(f32, f32)> = vec![(0.0, 0.0); gaussians.len()];
    let mut v_scale: Vec<(f32, f32)> = vec![(0.0, 0.0); gaussians.len()];

    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-8f32;

    let lr_pos_init = 0.0002f32;
    let lr_color = 0.02f32;
    let lr_scale = 0.005f32;

    let mut best_loss = f32::INFINITY;
    let mut best_gaussians = gaussians.clone();
    let mut patience = 0;
    let mut last_psnr = 0.0f32;

    for iter in 0..iterations {
        let decay = 0.1_f32.powf(iter as f32 / iterations as f32);
        let lr_pos = lr_pos_init * decay;

        let rendered = RendererV2::render(&gaussians, width, height);

        // Compute hybrid loss
        let l2_loss = compute_l2_loss(&rendered, target);
        let msssim_score = compute_ms_ssim(&rendered, target);
        let msssim_loss = 1.0 - msssim_score;
        let hybrid_loss = l2_weight * l2_loss + msssim_weight * msssim_loss;

        // Progress display
        if iter % 10 == 0 {
            let psnr = compute_psnr(&rendered, target);
            let direction = if psnr > last_psnr { "↑" } else if psnr < last_psnr { "↓" } else { "→" };
            let delta = psnr - last_psnr;
            println!("    [{:3}/{:3}] loss={:.6} (L2={:.4}, MS-SSIM={:.4}) PSNR={:.2}dB {} ({:+.3}dB)",
                     iter, iterations, hybrid_loss, l2_loss, msssim_score, psnr, direction, delta);
            last_psnr = psnr;
        }

        // Track best
        if hybrid_loss < best_loss {
            best_loss = hybrid_loss;
            best_gaussians.clone_from_slice(&gaussians);
            patience = 0;
        } else {
            patience += 1;
            if patience >= 30 {
                println!("    [EARLY STOP] No improvement for 30 iterations at iter {}", iter);
                break;
            }
        }

        // Compute gradients (using L2 gradients - MS-SSIM gradients are complex)
        // This is a simplification: we use L2 gradients but weight the loss with MS-SSIM
        let grads = compute_gradients(&gaussians, &rendered, target);

        // Adam update
        let t = (iter + 1) as f32;
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);

        for (i, (gaussian, grad)) in gaussians.iter_mut().zip(grads.iter()).enumerate() {
            // Position
            m_pos[i].0 = beta1 * m_pos[i].0 + (1.0 - beta1) * grad.position_x;
            m_pos[i].1 = beta1 * m_pos[i].1 + (1.0 - beta1) * grad.position_y;
            v_pos[i].0 = beta2 * v_pos[i].0 + (1.0 - beta2) * grad.position_x.powi(2);
            v_pos[i].1 = beta2 * v_pos[i].1 + (1.0 - beta2) * grad.position_y.powi(2);

            let m_hat_x = m_pos[i].0 / bc1;
            let m_hat_y = m_pos[i].1 / bc1;
            let v_hat_x = v_pos[i].0 / bc2;
            let v_hat_y = v_pos[i].1 / bc2;

            gaussian.position.x -= lr_pos * m_hat_x / (v_hat_x.sqrt() + eps);
            gaussian.position.y -= lr_pos * m_hat_y / (v_hat_y.sqrt() + eps);
            gaussian.position.x = gaussian.position.x.clamp(0.0, 1.0);
            gaussian.position.y = gaussian.position.y.clamp(0.0, 1.0);

            // Color
            m_color[i].0 = beta1 * m_color[i].0 + (1.0 - beta1) * grad.color_r;
            m_color[i].1 = beta1 * m_color[i].1 + (1.0 - beta1) * grad.color_g;
            m_color[i].2 = beta1 * m_color[i].2 + (1.0 - beta1) * grad.color_b;
            v_color[i].0 = beta2 * v_color[i].0 + (1.0 - beta2) * grad.color_r.powi(2);
            v_color[i].1 = beta2 * v_color[i].1 + (1.0 - beta2) * grad.color_g.powi(2);
            v_color[i].2 = beta2 * v_color[i].2 + (1.0 - beta2) * grad.color_b.powi(2);

            gaussian.color.r -= lr_color * (m_color[i].0 / bc1) / ((v_color[i].0 / bc2).sqrt() + eps);
            gaussian.color.g -= lr_color * (m_color[i].1 / bc1) / ((v_color[i].1 / bc2).sqrt() + eps);
            gaussian.color.b -= lr_color * (m_color[i].2 / bc1) / ((v_color[i].2 / bc2).sqrt() + eps);
            gaussian.color.r = gaussian.color.r.clamp(0.0, 1.0);
            gaussian.color.g = gaussian.color.g.clamp(0.0, 1.0);
            gaussian.color.b = gaussian.color.b.clamp(0.0, 1.0);

            // Scale
            m_scale[i].0 = beta1 * m_scale[i].0 + (1.0 - beta1) * grad.scale_x;
            m_scale[i].1 = beta1 * m_scale[i].1 + (1.0 - beta1) * grad.scale_y;
            v_scale[i].0 = beta2 * v_scale[i].0 + (1.0 - beta2) * grad.scale_x.powi(2);
            v_scale[i].1 = beta2 * v_scale[i].1 + (1.0 - beta2) * grad.scale_y.powi(2);

            gaussian.shape.scale_x -= lr_scale * (m_scale[i].0 / bc1) / ((v_scale[i].0 / bc2).sqrt() + eps);
            gaussian.shape.scale_y -= lr_scale * (m_scale[i].1 / bc1) / ((v_scale[i].1 / bc2).sqrt() + eps);
            gaussian.shape.scale_x = gaussian.shape.scale_x.clamp(0.01, 0.25);
            gaussian.shape.scale_y = gaussian.shape.scale_y.clamp(0.01, 0.25);
        }
    }

    gaussians.copy_from_slice(&best_gaussians);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(&gaussians, width, height);
    let psnr = compute_psnr(&final_render, target);
    let msssim = compute_ms_ssim(&final_render, target);

    (psnr, msssim, elapsed.as_secs_f32())
}

#[derive(Clone)]
struct Gradient {
    position_x: f32,
    position_y: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    scale_x: f32,
    scale_y: f32,
}

fn compute_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
) -> Vec<Gradient> {
    let width = target.width;
    let height = target.height;
    let mut gradients: Vec<Gradient> = gaussians.iter().map(|_| Gradient {
        position_x: 0.0, position_y: 0.0,
        color_r: 0.0, color_g: 0.0, color_b: 0.0,
        scale_x: 0.0, scale_y: 0.0,
    }).collect();

    for y in 0..height {
        for x in 0..width {
            let px = x as f32 / width as f32;
            let py = y as f32 / height as f32;

            let rendered_color = rendered.get_pixel(x, y).unwrap();
            let target_color = target.get_pixel(x, y).unwrap();

            let error_r = 2.0 * (rendered_color.r - target_color.r);
            let error_g = 2.0 * (rendered_color.g - target_color.g);
            let error_b = 2.0 * (rendered_color.b - target_color.b);

            for (i, gaussian) in gaussians.iter().enumerate() {
                let dx = px - gaussian.position.x;
                let dy = py - gaussian.position.y;

                let sx = gaussian.shape.scale_x.max(1e-6);
                let sy = gaussian.shape.scale_y.max(1e-6);

                let dist_sq = (dx / sx).powi(2) + (dy / sy).powi(2);
                if dist_sq > 16.0 { continue; }

                let weight = (-0.5 * dist_sq).exp();
                if weight < 1e-6 { continue; }

                // Color gradients
                gradients[i].color_r += error_r * weight;
                gradients[i].color_g += error_g * weight;
                gradients[i].color_b += error_b * weight;

                // Position gradients
                let error_weighted = error_r * gaussian.color.r
                                   + error_g * gaussian.color.g
                                   + error_b * gaussian.color.b;

                gradients[i].position_x += error_weighted * weight * dx / (sx * sx);
                gradients[i].position_y += error_weighted * weight * dy / (sy * sy);

                // Scale gradients
                gradients[i].scale_x += error_weighted * weight * (dx * dx) / (sx * sx * sx);
                gradients[i].scale_y += error_weighted * weight * (dy * dy) / (sy * sy * sy);
            }
        }
    }

    let pixel_count = (width * height) as f32;
    for grad in &mut gradients {
        grad.position_x /= pixel_count;
        grad.position_y /= pixel_count;
        grad.color_r /= pixel_count;
        grad.color_g /= pixel_count;
        grad.color_b /= pixel_count;
        grad.scale_x /= pixel_count;
        grad.scale_y /= pixel_count;
    }

    gradients
}

fn create_jittered_gaussians(target: &ImageBuffer<f32>, grid_size: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(grid_size * grid_size);

    for i in 0..(grid_size * grid_size) {
        let gx = i % grid_size;
        let gy = i / grid_size;

        // Jitter within cell
        let jx = ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5;
        let jy = ((i * 11 + 7) % 100) as f32 / 100.0 - 0.5;

        let x = ((gx as f32 + 0.5 + jx * 0.4) / grid_size as f32).clamp(0.01, 0.99);
        let y = ((gy as f32 + 0.5 + jy * 0.4) / grid_size as f32).clamp(0.01, 0.99);

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

fn compute_l2_loss(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height) as f32;
    let mut loss = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        loss += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    loss / count
}

fn compute_ms_ssim(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let msssim = MSSSIM::default();
    msssim.compute(target, rendered)
}

fn save_result(exp_name: &str, image_path: &str, dims: (u32, u32), n_gaussians: usize,
               iterations: usize, init_psnr: f32, final_psnr: f32, improvement: f32,
               elapsed: f32, loss_type: &str, init_msssim: f32, final_msssim: f32, l2_weight: f32) {
    let mut result = TestResult::new(exp_name, image_path);
    result.optimizer = format!("Adam + {} loss", loss_type);
    result.image_dimensions = dims;
    result.n_gaussians = n_gaussians;
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.final_loss = 0.0;  // Not stored
    result.elapsed_seconds = elapsed;
    result.extra_metrics = Some(json!({
        "loss_type": loss_type,
        "l2_weight": l2_weight,
        "msssim_weight": 1.0 - l2_weight,
        "initial_msssim": init_msssim,
        "final_msssim": final_msssim,
        "msssim_improvement": final_msssim - init_msssim,
        "placement": "jittered_grid",
    }));
    let _ = result.save(RESULTS_DIR);
}
