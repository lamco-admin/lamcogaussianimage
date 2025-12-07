//! Experiment B: Finer α sweep around optimal (0.84)
//!
//! Maps the loss landscape more precisely around the 3DGS-recommended α=0.84.
//! Tests: 0.90, 0.86, 0.84, 0.82, 0.78
//!
//! Goal: Determine if 0.84 is truly optimal or if there's a better nearby value.
//!
//! Run: cargo run --release --example exp_fine_alpha_sweep -p lgi-encoder-v2

use lgi_core::{ImageBuffer, ms_ssim::MSSSIM, ms_ssim_gradients::MsssimGradients};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   EXPERIMENT B: Fine α Sweep (0.78 - 0.90)                   ║");
    println!("║   Mapping loss landscape around 3DGS optimal                 ║");
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

    // Fine-grained alpha sweep around 0.84
    let alphas = [0.90, 0.86, 0.84, 0.82, 0.78];
    let mut results = Vec::new();

    for &alpha in &alphas {
        println!("\n{}", "═".repeat(60));
        println!("Testing α = {:.2} ({}% MSE + {}% MS-SSIM)",
            alpha, (alpha * 100.0) as i32, ((1.0 - alpha) * 100.0) as i32);
        println!("{}", "═".repeat(60));

        let result = run_hybrid_optimization(&target, image_path, alpha, 200);
        results.push((alpha, result));
    }

    // Summary table
    println!("\n╔════════════════════════════════════════════════════════════════════╗");
    println!("║                        FINE α SWEEP RESULTS                        ║");
    println!("╠════════╤═══════════╤═════════════╤═══════════════╤════════╤════════╣");
    println!("║  Alpha │ Final PSNR│ Improvement │ Final MS-SSIM │ Δ SSIM │  Time  ║");
    println!("╠════════╪═══════════╪═════════════╪═══════════════╪════════╪════════╣");

    for (alpha, (psnr, improvement, ssim, init_ssim, time)) in &results {
        let ssim_delta = ssim - init_ssim;
        println!("║  {:.2}  │  {:.2} dB │   {:+.2} dB   │     {:.4}    │ {:+.4} │ {:.1}s  ║",
            alpha, psnr, improvement, ssim, ssim_delta, time);
    }
    println!("╚════════╧═══════════╧═════════════╧═══════════════╧════════╧════════╝");

    // Find optima
    let best_psnr = results.iter().max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap());
    let best_ssim = results.iter().max_by(|a, b| a.1.2.partial_cmp(&b.1.2).unwrap());
    let best_improvement = results.iter().max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap());

    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│ OBSERVATIONS:                                                │");
    println!("├──────────────────────────────────────────────────────────────┤");
    println!("│ Best PSNR:       α={:.2} with {:.2}dB                         │",
             best_psnr.unwrap().0, best_psnr.unwrap().1.0);
    println!("│ Best MS-SSIM:    α={:.2} with {:.4}                          │",
             best_ssim.unwrap().0, best_ssim.unwrap().1.2);
    println!("│ Best Improvement: α={:.2} with {:+.2}dB                        │",
             best_improvement.unwrap().0, best_improvement.unwrap().1.1);
    println!("└──────────────────────────────────────────────────────────────┘");

    // Analyze curve shape
    println!("\n┌──────────────────────────────────────────────────────────────┐");
    println!("│ CURVE ANALYSIS:                                              │");
    println!("├──────────────────────────────────────────────────────────────┤");

    // Check if 0.84 is truly optimal or if there's a better value
    let psnr_at_84 = results.iter().find(|(a, _)| (*a - 0.84).abs() < 0.01).map(|(_, r)| r.0);
    let max_psnr = results.iter().map(|(_, r)| r.0).fold(f32::NEG_INFINITY, f32::max);

    if let Some(p84) = psnr_at_84 {
        if (p84 - max_psnr).abs() < 0.05 {
            println!("│ α=0.84 is near-optimal (within 0.05dB of best)             │");
        } else {
            println!("│ α=0.84 is NOT optimal - better α exists in this range      │");
        }
    }
    println!("└──────────────────────────────────────────────────────────────┘");
}

fn run_hybrid_optimization(
    target: &ImageBuffer<f32>,
    image_path: &str,
    alpha: f32,
    max_iterations: usize,
) -> (f32, f32, f32, f32, f32) {  // (final_psnr, improvement, final_ssim, init_ssim, time)
    let (width, height) = (target.width, target.height);
    let n_gaussians = 256;

    let mut gaussians = create_jittered_grid(target, n_gaussians);

    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&rendered, target);
    let ms_ssim = MSSSIM::default();
    let init_ssim = ms_ssim.compute(&rendered, target);

    println!("  Initial: PSNR={:.2}dB, MS-SSIM={:.4}", init_psnr, init_ssim);

    // Adam state
    let n_params = n_gaussians * 7;
    let mut m = vec![0.0_f32; n_params];
    let mut v = vec![0.0_f32; n_params];

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
        let mse_grads = compute_mse_gradients(&gaussians, target);

        let ssim_pixel_grads = if alpha < 1.0 {
            ssim_grads.compute_pixel_gradients(target, &rendered)
        } else {
            ImageBuffer::new(width, height)
        };

        let ssim_gaussian_grads = if alpha < 1.0 {
            backprop_ssim_gradients(&gaussians, &ssim_pixel_grads)
        } else {
            vec![[0.0; 7]; n_gaussians]
        };

        let t = (iter + 1) as f32;
        let lr_correction = (1.0 - beta2.powf(t)).sqrt() / (1.0 - beta1.powf(t));

        for (i, gaussian) in gaussians.iter_mut().enumerate() {
            let base = i * 7;
            let mse_grad = &mse_grads[i];
            let ssim_grad = &ssim_gaussian_grads[i];

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

            gaussian.position.x = gaussian.position.x.clamp(0.0, 1.0);
            gaussian.position.y = gaussian.position.y.clamp(0.0, 1.0);
            gaussian.color.r = gaussian.color.r.clamp(0.0, 1.0);
            gaussian.color.g = gaussian.color.g.clamp(0.0, 1.0);
            gaussian.color.b = gaussian.color.b.clamp(0.0, 1.0);
            gaussian.shape.scale_x = gaussian.shape.scale_x.clamp(0.005, 0.5);
            gaussian.shape.scale_y = gaussian.shape.scale_y.clamp(0.005, 0.5);
        }

        if iter % 40 == 0 || iter == max_iterations - 1 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = compute_psnr(&rendered, target);
            let ssim = ms_ssim.compute(&rendered, target);
            let dir = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {} MS-SSIM={:.4}", iter, max_iterations, psnr, dir, ssim);
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let final_rendered = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_rendered, target);
    let final_ssim = ms_ssim.compute(&final_rendered, target);
    let improvement = final_psnr - init_psnr;

    println!("  Final: PSNR={:.2}dB ({:+.2}dB), MS-SSIM={:.4}", final_psnr, improvement, final_ssim);

    // Save result
    let mut result = TestResult::new(&format!("fine_alpha_{}", (alpha * 100.0) as i32), image_path);
    result.optimizer = format!("Adam + Hybrid (α={})", alpha);
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = max_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement;
    result.elapsed_seconds = elapsed;
    result.set_param("alpha", alpha);
    result.set_param("init_msssim", init_ssim);
    result.set_param("final_msssim", final_ssim);
    result.complete();
    let _ = result.save_with_image(&final_rendered, RESULTS_DIR);

    (final_psnr, improvement, final_ssim, init_ssim, elapsed)
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

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= (rendered.width * rendered.height * 3) as f32;
    if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}
