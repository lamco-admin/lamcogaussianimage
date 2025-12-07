//! Experiment A: ErrorDrivenAdam + MS-SSIM Hybrid Loss
//!
//! Combines two winning approaches:
//! 1. ErrorDrivenAdam - adaptive Gaussian placement via error maps
//! 2. Hybrid Loss (α=0.84) - 84% MSE + 16% MS-SSIM
//!
//! Hypothesis: Combining adaptive placement with perceptual loss should
//! improve both PSNR and MS-SSIM metrics simultaneously.
//!
//! Run: cargo run --release --example exp_error_driven_hybrid -p lgi-encoder-v2

use lgi_core::{ImageBuffer, ms_ssim::MSSSIM, ms_ssim_gradients::MsssimGradients};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";
const ALPHA: f32 = 0.84; // 3DGS optimal: 84% MSE + 16% MS-SSIM

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   EXPERIMENT A: ErrorDrivenAdam + Hybrid Loss (α=0.84)       ║");
    println!("║   Combining adaptive placement with perceptual loss          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";

    let target = match ImageBuffer::load(image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            return;
        }
    };

    let (width, height) = (target.width, target.height);
    println!("Image: kodim03.png ({}×{})\n", width, height);

    // Configuration
    let initial_gaussians = 64;
    let max_gaussians = 256;
    let iterations_per_pass = 50;
    let max_passes = 8;
    let min_improvement_db = 0.05;
    let min_gaussian_spacing = 0.02;
    let split_percentile = 0.05;

    println!("Config:");
    println!("  Initial Gaussians: {}", initial_gaussians);
    println!("  Max Gaussians: {}", max_gaussians);
    println!("  Iterations per pass: {}", iterations_per_pass);
    println!("  Hybrid loss α: {} ({}% MSE + {}% MS-SSIM)",
             ALPHA, (ALPHA * 100.0) as i32, ((1.0 - ALPHA) * 100.0) as i32);
    println!();

    let start = Instant::now();

    // Initialize with uniform grid
    let grid_size = (initial_gaussians as f32).sqrt().ceil() as u32;
    let mut gaussians = initialize_uniform_grid(&target, grid_size);

    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&target, &rendered);
    let ms_ssim = MSSSIM::default();
    let init_ssim = ms_ssim.compute(&rendered, &target);

    println!("Initial: {} Gaussians, PSNR={:.2}dB, MS-SSIM={:.4}",
             gaussians.len(), init_psnr, init_ssim);

    let mut last_psnr = init_psnr;
    let mut stall_count = 0;

    for pass in 0..max_passes {
        // Run hybrid Adam optimization for this pass
        run_hybrid_adam_pass(&mut gaussians, &target, iterations_per_pass, ALPHA);

        let rendered = RendererV2::render(&gaussians, width, height);
        let psnr = compute_psnr(&target, &rendered);
        let ssim = ms_ssim.compute(&rendered, &target);
        let improvement = psnr - last_psnr;
        let direction = if improvement > 0.0 { "↑" } else { "↓" };

        println!("  Pass {}: N={:4}, PSNR={:.2}dB {} ({:+.2}dB), MS-SSIM={:.4}",
                 pass, gaussians.len(), psnr, direction, improvement, ssim);

        // Check convergence
        if gaussians.len() >= max_gaussians {
            println!("  ✓ Reached max Gaussians limit");
            break;
        }

        if pass > 0 && improvement < min_improvement_db {
            stall_count += 1;
            if stall_count >= 2 {
                println!("  ✓ Improvement stalled ({:.3}dB < {:.3}dB threshold)",
                         improvement, min_improvement_db);
                break;
            }
        } else {
            stall_count = 0;
        }

        last_psnr = psnr;

        // Find high-error regions and add Gaussians
        let error_map = compute_error_map(&target, &rendered);
        let hotspots = find_clustered_hotspots(&error_map, &target, &gaussians,
                                                min_gaussian_spacing, split_percentile);

        if hotspots.is_empty() {
            println!("  ✓ No suitable hotspots remaining");
            break;
        }

        let added = add_gaussians_at_hotspots(&mut gaussians, &target, &hotspots, max_gaussians);
        println!("    → Added {} Gaussians at high-error locations", added);
    }

    // Final polish
    println!("\n  Final polish (20 iterations)...");
    run_hybrid_adam_pass(&mut gaussians, &target, 20, ALPHA);

    let elapsed = start.elapsed().as_secs_f32();
    let final_rendered = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&target, &final_rendered);
    let final_ssim = ms_ssim.compute(&final_rendered, &target);
    let improvement = final_psnr - init_psnr;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     FINAL RESULTS                            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Gaussians: {:4} (started with {})                          ║", gaussians.len(), initial_gaussians);
    println!("║  PSNR:      {:.2}dB ({:+.2}dB improvement)                   ║", final_psnr, improvement);
    println!("║  MS-SSIM:   {:.4} ({:+.4} from {:.4})                        ║", final_ssim, final_ssim - init_ssim, init_ssim);
    println!("║  Time:      {:.1}s                                           ║", elapsed);
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Save result
    let mut result = TestResult::new("error_driven_hybrid_adam", image_path);
    result.optimizer = format!("ErrorDrivenAdam + Hybrid Loss (α={})", ALPHA);
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.iterations = max_passes * iterations_per_pass;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.elapsed_seconds = elapsed;
    result.set_param("alpha", ALPHA);
    result.set_param("initial_gaussians", initial_gaussians as f32);
    result.set_param("max_gaussians", max_gaussians as f32);
    result.set_param("iterations_per_pass", iterations_per_pass as f32);
    result.set_param("init_msssim", init_ssim);
    result.set_param("final_msssim", final_ssim);
    result.complete();
    let _ = result.save_with_image(&final_rendered, RESULTS_DIR);

    println!("\n✓ Results saved to {}", RESULTS_DIR);
}

/// Run hybrid Adam optimization pass
fn run_hybrid_adam_pass(
    gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    iterations: usize,
    alpha: f32,
) {
    let (width, height) = (target.width, target.height);
    let n = gaussians.len();

    // Adam state
    let n_params = n * 7;
    let mut m = vec![0.0_f32; n_params];
    let mut v = vec![0.0_f32; n_params];

    // Per-parameter learning rates
    let lr_position = 0.0002;
    let lr_color = 0.02;
    let lr_scale = 0.005;

    let beta1: f32 = 0.9;
    let beta2: f32 = 0.999;
    let epsilon: f32 = 1e-8;

    let ssim_grads_calc = MsssimGradients::default();

    for iter in 0..iterations {
        let rendered = RendererV2::render(gaussians, width, height);

        // Compute MSE gradients
        let mse_grads = compute_mse_gradients(gaussians, target, &rendered);

        // Compute MS-SSIM gradients
        let ssim_pixel_grads = if alpha < 1.0 {
            ssim_grads_calc.compute_pixel_gradients(target, &rendered)
        } else {
            ImageBuffer::new(width, height)
        };

        let ssim_grads = if alpha < 1.0 {
            backprop_ssim_gradients(gaussians, &ssim_pixel_grads)
        } else {
            vec![[0.0; 7]; n]
        };

        // Combine gradients
        let t = (iter + 1) as f32;
        let lr_correction = (1.0 - beta2.powf(t)).sqrt() / (1.0 - beta1.powf(t));

        for (i, gaussian) in gaussians.iter_mut().enumerate() {
            let base = i * 7;
            let mse_grad = &mse_grads[i];
            let ssim_grad = &ssim_grads[i];

            for j in 0..7 {
                let idx = base + j;
                let g = alpha * mse_grad[j] + (1.0 - alpha) * ssim_grad[j];

                m[idx] = beta1 * m[idx] + (1.0 - beta1) * g;
                v[idx] = beta2 * v[idx] + (1.0 - beta2) * g * g;

                let lr = match j {
                    0 | 1 => lr_position,
                    2 | 3 | 4 => lr_color,
                    5 | 6 => lr_scale,
                    _ => lr_color,
                };

                let update = lr * lr_correction * m[idx] / (v[idx].sqrt() + epsilon);

                match j {
                    0 => gaussian.position.x -= update,
                    1 => gaussian.position.y -= update,
                    2 => gaussian.color.r -= update,
                    3 => gaussian.color.g -= update,
                    4 => gaussian.color.b -= update,
                    5 => gaussian.shape.scale_x -= update,
                    6 => gaussian.shape.scale_y -= update,
                    _ => {}
                }
            }

            // Clamp parameters
            gaussian.position.x = gaussian.position.x.clamp(0.0, 1.0);
            gaussian.position.y = gaussian.position.y.clamp(0.0, 1.0);
            gaussian.color.r = gaussian.color.r.clamp(0.0, 1.0);
            gaussian.color.g = gaussian.color.g.clamp(0.0, 1.0);
            gaussian.color.b = gaussian.color.b.clamp(0.0, 1.0);
            gaussian.shape.scale_x = gaussian.shape.scale_x.clamp(0.005, 0.5);
            gaussian.shape.scale_y = gaussian.shape.scale_y.clamp(0.005, 0.5);
        }
    }
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

fn find_clustered_hotspots(
    error_map: &[f32],
    target: &ImageBuffer<f32>,
    existing: &[Gaussian2D<f32, Euler<f32>>],
    min_spacing: f32,
    percentile: f32,
) -> Vec<Hotspot> {
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

        if too_close_existing { continue; }

        let too_close_selected = selected.iter().any(|h| {
            let hx = h.x as f32 / w_f;
            let hy = h.y as f32 / h_f;
            let dx = hx - cx;
            let dy = hy - cy;
            dx * dx + dy * dy < min_dist_sq
        });

        if too_close_selected { continue; }

        selected.push(candidate);
        if selected.len() >= 30 { break; }
    }
    selected
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

fn compute_mse_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    rendered: &ImageBuffer<f32>,
) -> Vec<[f32; 7]> {
    let (width, height) = (target.width, target.height);
    let n = gaussians.len();
    let mut grads: Vec<[f32; 7]> = vec![[0.0; 7]; n];

    for y in 0..height {
        for px in 0..width {
            let norm_x = (px as f32 + 0.5) / width as f32;
            let norm_y = (y as f32 + 0.5) / height as f32;

            let rendered_color = rendered.get_pixel(px, y).unwrap();
            let target_color = target.get_pixel(px, y).unwrap();

            let error_r = 2.0 * (rendered_color.r - target_color.r);
            let error_g = 2.0 * (rendered_color.g - target_color.g);
            let error_b = 2.0 * (rendered_color.b - target_color.b);

            for (i, gaussian) in gaussians.iter().enumerate() {
                let dx = norm_x - gaussian.position.x;
                let dy = norm_y - gaussian.position.y;
                let sx = gaussian.shape.scale_x.max(1e-6);
                let sy = gaussian.shape.scale_y.max(1e-6);
                let dist_sq = (dx / sx).powi(2) + (dy / sy).powi(2);
                if dist_sq > 16.0 { continue; }
                let weight = (-0.5 * dist_sq).exp();
                if weight < 1e-6 { continue; }

                grads[i][2] += error_r * weight;
                grads[i][3] += error_g * weight;
                grads[i][4] += error_b * weight;

                let error_weighted = error_r * gaussian.color.r + error_g * gaussian.color.g + error_b * gaussian.color.b;
                grads[i][0] += error_weighted * weight * dx / (sx * sx);
                grads[i][1] += error_weighted * weight * dy / (sy * sy);
                grads[i][5] += error_weighted * weight * (dx * dx) / (sx * sx * sx);
                grads[i][6] += error_weighted * weight * (dy * dy) / (sy * sy * sy);
            }
        }
    }

    let pixel_count = (width * height) as f32;
    for grad in &mut grads { for g in grad.iter_mut() { *g /= pixel_count; } }
    grads
}

fn backprop_ssim_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    pixel_grads: &ImageBuffer<f32>,
) -> Vec<[f32; 7]> {
    let (width, height) = (pixel_grads.width, pixel_grads.height);
    let n = gaussians.len();
    let mut grads: Vec<[f32; 7]> = vec![[0.0; 7]; n];

    for y in 0..height {
        for px in 0..width {
            let norm_x = (px as f32 + 0.5) / width as f32;
            let norm_y = (y as f32 + 0.5) / height as f32;
            let pixel_grad = pixel_grads.get_pixel(px, y).unwrap();

            if pixel_grad.r.abs() < 1e-8 && pixel_grad.g.abs() < 1e-8 && pixel_grad.b.abs() < 1e-8 { continue; }

            for (i, gaussian) in gaussians.iter().enumerate() {
                let dx = norm_x - gaussian.position.x;
                let dy = norm_y - gaussian.position.y;
                let sx = gaussian.shape.scale_x.max(1e-6);
                let sy = gaussian.shape.scale_y.max(1e-6);
                let dist_sq = (dx / sx).powi(2) + (dy / sy).powi(2);
                if dist_sq > 16.0 { continue; }
                let weight = (-0.5 * dist_sq).exp();
                if weight < 1e-6 { continue; }

                grads[i][2] += pixel_grad.r * weight;
                grads[i][3] += pixel_grad.g * weight;
                grads[i][4] += pixel_grad.b * weight;

                let grad_weighted = pixel_grad.r * gaussian.color.r + pixel_grad.g * gaussian.color.g + pixel_grad.b * gaussian.color.b;
                grads[i][0] += grad_weighted * weight * dx / (sx * sx);
                grads[i][1] += grad_weighted * weight * dy / (sy * sy);
                grads[i][5] += grad_weighted * weight * (dx * dx) / (sx * sx * sx);
                grads[i][6] += grad_weighted * weight * (dy * dy) / (sy * sy * sy);
            }
        }
    }

    let pixel_count = (width * height) as f32;
    for grad in &mut grads { for g in grad.iter_mut() { *g /= pixel_count; } }
    grads
}

fn compute_psnr(target: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (t, r) in target.data.iter().zip(rendered.data.iter()) {
        mse += (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
    }
    mse /= (target.width * target.height * 3) as f32;
    if mse < 1e-10 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}
