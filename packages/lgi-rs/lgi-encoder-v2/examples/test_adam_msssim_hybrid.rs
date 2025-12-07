//! Adam + MS-SSIM Hybrid Loss Experiment
//!
//! Tests hybrid loss (α·MSE + (1-α)·MS-SSIM) with Adam optimizer.
//! Compares different alpha values to find optimal perceptual/PSNR balance.
//!
//! Run: cargo run --release --example test_adam_msssim_hybrid -p lgi-encoder-v2

use lgi_core::{ImageBuffer, ms_ssim::MSSSIM, ms_ssim_gradients::MsssimGradients};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       ADAM + MS-SSIM HYBRID LOSS EXPERIMENT                  ║");
    println!("║       Finding optimal MSE/MS-SSIM balance                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";

    let target = match ImageBuffer::load(image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            return;
        }
    };

    println!("Image: {} ({}x{})\n",
        image_path.split('/').last().unwrap_or("unknown"),
        target.width, target.height);

    // Test different alpha values
    // α = 1.0: Pure MSE (baseline)
    // α = 0.84: 3DGS paper recommendation
    // α = 0.5: Equal blend
    // α = 0.2: MS-SSIM dominant

    let alphas = [1.0, 0.84, 0.5, 0.2];
    let mut results = Vec::new();

    for &alpha in &alphas {
        println!("\n{}", "═".repeat(60));
        println!("Testing α = {} ({}% MSE + {}% MS-SSIM)",
            alpha, (alpha * 100.0) as i32, ((1.0 - alpha) * 100.0) as i32);
        println!("{}", "═".repeat(60));

        let result = run_hybrid_optimization(&target, image_path, alpha, 200);
        results.push((alpha, result));
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     SUMMARY                                  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Alpha │ Final PSNR │ Improvement │ Final MS-SSIM │ Time    ║");
    println!("╠════════╪════════════╪═════════════╪═══════════════╪═════════╣");

    for (alpha, (psnr, improvement, ssim, time)) in &results {
        println!("║  {:.2}  │  {:.2} dB  │   {:+.2} dB   │    {:.4}     │ {:.1}s   ║",
            alpha, psnr, improvement, ssim, time);
    }
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Find best for each metric
    let best_psnr = results.iter().max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap());
    let best_ssim = results.iter().max_by(|a, b| a.1.2.partial_cmp(&b.1.2).unwrap());

    println!("\nBest PSNR:    α={:.2} with {:.2}dB", best_psnr.unwrap().0, best_psnr.unwrap().1.0);
    println!("Best MS-SSIM: α={:.2} with {:.4}", best_ssim.unwrap().0, best_ssim.unwrap().1.2);
}

/// Run hybrid loss optimization with Adam
fn run_hybrid_optimization(
    target: &ImageBuffer<f32>,
    image_path: &str,
    alpha: f32,  // 1.0 = pure MSE, 0.0 = pure MS-SSIM
    max_iterations: usize,
) -> (f32, f32, f32, f32) {  // (final_psnr, improvement, final_ssim, time)
    let (width, height) = (target.width, target.height);
    let n_gaussians = 256;

    // Initialize Gaussians
    let mut gaussians = create_jittered_grid(target, n_gaussians);

    // Compute initial metrics
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&rendered, target);
    let ms_ssim = MSSSIM::default();
    let init_ssim = ms_ssim.compute(&rendered, target);

    println!("  Initial: PSNR={:.2}dB, MS-SSIM={:.4}", init_psnr, init_ssim);

    // Adam state for each Gaussian parameter
    // Per-gaussian: x, y, r, g, b, sx, sy (7 params)
    let n_params = n_gaussians * 7;
    let mut m = vec![0.0_f32; n_params];  // First moment
    let mut v = vec![0.0_f32; n_params];  // Second moment

    // Per-parameter learning rates (3DGS-calibrated)
    let lr_position = 0.0002;
    let lr_color = 0.02;
    let lr_scale = 0.005;

    let beta1: f32 = 0.9;
    let beta2: f32 = 0.999;
    let epsilon: f32 = 1e-8;

    let ssim_grads = MsssimGradients::default();
    let start = Instant::now();
    let mut last_psnr = init_psnr;

    for iter in 0..max_iterations {
        let rendered = RendererV2::render(&gaussians, width, height);

        // Compute MSE gradients
        let mse_grads = compute_mse_gradients(&gaussians, target);

        // Compute MS-SSIM gradients (pixel-level, then backprop to Gaussians)
        let ssim_pixel_grads = if alpha < 1.0 {
            ssim_grads.compute_pixel_gradients(target, &rendered)
        } else {
            // Skip MS-SSIM computation if alpha = 1.0
            ImageBuffer::new(width, height)
        };

        let ssim_gaussian_grads = if alpha < 1.0 {
            backprop_ssim_gradients(&gaussians, &ssim_pixel_grads)
        } else {
            vec![[0.0; 7]; n_gaussians]
        };

        // Combine gradients: α * MSE_grad + (1-α) * SSIM_grad
        let combined_grads: Vec<[f32; 7]> = mse_grads.iter()
            .zip(ssim_gaussian_grads.iter())
            .map(|(mse, ssim)| {
                let mut combined = [0.0; 7];
                for i in 0..7 {
                    combined[i] = alpha * mse[i] + (1.0 - alpha) * ssim[i];
                }
                combined
            })
            .collect();

        // Adam update
        let t = (iter + 1) as f32;
        let lr_correction = (1.0 - beta2.powf(t)).sqrt() / (1.0 - beta1.powf(t));

        for (i, gaussian) in gaussians.iter_mut().enumerate() {
            let base = i * 7;
            let grad = &combined_grads[i];

            // Update each parameter with Adam
            for j in 0..7 {
                let idx = base + j;
                let g = grad[j];

                // Adam moments
                m[idx] = beta1 * m[idx] + (1.0 - beta1) * g;
                v[idx] = beta2 * v[idx] + (1.0 - beta2) * g * g;

                let lr = match j {
                    0 | 1 => lr_position,  // x, y
                    2 | 3 | 4 => lr_color, // r, g, b
                    5 | 6 => lr_scale,     // sx, sy
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

        // Progress logging
        if iter % 20 == 0 || iter == max_iterations - 1 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = compute_psnr(&rendered, target);
            let ssim = ms_ssim.compute(&rendered, target);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };

            println!("    [{:3}/{:3}] PSNR={:.2}dB {} MS-SSIM={:.4}",
                iter, max_iterations, psnr, dir, ssim);
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();

    // Final metrics
    let final_rendered = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_rendered, target);
    let final_ssim = ms_ssim.compute(&final_rendered, target);
    let improvement = final_psnr - init_psnr;

    println!("  Final: PSNR={:.2}dB ({:+.2}dB), MS-SSIM={:.4} ({:+.4})",
        final_psnr, improvement, final_ssim, final_ssim - init_ssim);

    // Save result
    let mut result = TestResult::new(&format!("adam_msssim_alpha_{}", (alpha * 100.0) as i32), image_path);
    result.optimizer = format!("Adam + Hybrid Loss (α={})", alpha);
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = max_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.elapsed_seconds = elapsed;
    result.set_param("alpha", alpha);
    result.set_param("lr_position", lr_position);
    result.set_param("lr_color", lr_color);
    result.set_param("lr_scale", lr_scale);
    result.set_param("init_msssim", init_ssim);
    result.set_param("final_msssim", final_ssim);
    result.complete();
    let _ = result.save_with_image(&final_rendered, RESULTS_DIR);

    (final_psnr, improvement, final_ssim, elapsed)
}

// ==================== HELPER FUNCTIONS ====================

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
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(cell_size, cell_size, 0.0),
            color,
            1.0,
        ));
    }
    gaussians
}

/// Compute MSE gradients for all Gaussian parameters
fn compute_mse_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
) -> Vec<[f32; 7]> {
    let width = target.width;
    let height = target.height;
    let n = gaussians.len();

    let mut grads: Vec<[f32; 7]> = vec![[0.0; 7]; n];
    let rendered = RendererV2::render(gaussians, width, height);

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

                // Color gradients
                grads[i][2] += error_r * weight;
                grads[i][3] += error_g * weight;
                grads[i][4] += error_b * weight;

                let error_weighted = error_r * gaussian.color.r
                                   + error_g * gaussian.color.g
                                   + error_b * gaussian.color.b;

                // Position gradients
                grads[i][0] += error_weighted * weight * dx / (sx * sx);
                grads[i][1] += error_weighted * weight * dy / (sy * sy);

                // Scale gradients
                grads[i][5] += error_weighted * weight * (dx * dx) / (sx * sx * sx);
                grads[i][6] += error_weighted * weight * (dy * dy) / (sy * sy * sy);
            }
        }
    }

    let pixel_count = (width * height) as f32;
    for grad in &mut grads {
        for g in grad.iter_mut() {
            *g /= pixel_count;
        }
    }
    grads
}

/// Backpropagate MS-SSIM pixel gradients to Gaussian parameters
fn backprop_ssim_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    pixel_grads: &ImageBuffer<f32>,
) -> Vec<[f32; 7]> {
    let width = pixel_grads.width;
    let height = pixel_grads.height;
    let n = gaussians.len();

    let mut grads: Vec<[f32; 7]> = vec![[0.0; 7]; n];

    for y in 0..height {
        for px in 0..width {
            let norm_x = (px as f32 + 0.5) / width as f32;
            let norm_y = (y as f32 + 0.5) / height as f32;

            let pixel_grad = pixel_grads.get_pixel(px, y).unwrap();

            // Skip if gradient is negligible
            if pixel_grad.r.abs() < 1e-8 && pixel_grad.g.abs() < 1e-8 && pixel_grad.b.abs() < 1e-8 {
                continue;
            }

            for (i, gaussian) in gaussians.iter().enumerate() {
                let dx = norm_x - gaussian.position.x;
                let dy = norm_y - gaussian.position.y;
                let sx = gaussian.shape.scale_x.max(1e-6);
                let sy = gaussian.shape.scale_y.max(1e-6);

                let dist_sq = (dx / sx).powi(2) + (dy / sy).powi(2);
                if dist_sq > 16.0 { continue; }

                let weight = (-0.5 * dist_sq).exp();
                if weight < 1e-6 { continue; }

                // Color gradients from MS-SSIM
                grads[i][2] += pixel_grad.r * weight;
                grads[i][3] += pixel_grad.g * weight;
                grads[i][4] += pixel_grad.b * weight;

                let grad_weighted = pixel_grad.r * gaussian.color.r
                                  + pixel_grad.g * gaussian.color.g
                                  + pixel_grad.b * gaussian.color.b;

                // Position gradients
                grads[i][0] += grad_weighted * weight * dx / (sx * sx);
                grads[i][1] += grad_weighted * weight * dy / (sy * sy);

                // Scale gradients
                grads[i][5] += grad_weighted * weight * (dx * dx) / (sx * sx * sx);
                grads[i][6] += grad_weighted * weight * (dy * dy) / (sy * sy * sy);
            }
        }
    }

    let pixel_count = (width * height) as f32;
    for grad in &mut grads {
        for g in grad.iter_mut() {
            *g /= pixel_count;
        }
    }
    grads
}

fn compute_mse(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse / (rendered.width * rendered.height * 3) as f32
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mse = compute_mse(rendered, target);
    if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}
