//! Test Different Loss Functions
//!
//! Compares:
//! - L2 (MSE) - Current baseline
//! - L1 (MAE) - Often sharper
//! - D-SSIM - Perceptual
//! - 3DGS Loss (L1 + D-SSIM) - Standard 3DGS approach
//!
//! Run: cargo run --release --example test_loss_functions -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::loss_functions::{compute_l1_loss, compute_l2_loss, compute_ssim, compute_dssim_loss, compute_3dgs_loss};
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   LOSS FUNCTIONS COMPARISON TEST                             ║");
    println!("║   L2 vs L1 vs D-SSIM vs 3DGS (L1 + D-SSIM)                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);
    println!();

    let grid_size = 16;  // 256 Gaussians
    let iterations = 100;

    // Test each loss function
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: L2 Loss (MSE) - Current Baseline");
    println!("═══════════════════════════════════════════════════════════════");
    let l2_result = test_with_loss(&target, grid_size, iterations, LossType::L2, image_path);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: L1 Loss (MAE)");
    println!("═══════════════════════════════════════════════════════════════");
    let l1_result = test_with_loss(&target, grid_size, iterations, LossType::L1, image_path);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 3: D-SSIM Loss");
    println!("═══════════════════════════════════════════════════════════════");
    let dssim_result = test_with_loss(&target, grid_size, iterations, LossType::DSSIM, image_path);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 4: 3DGS Loss (L1 + D-SSIM)");
    println!("═══════════════════════════════════════════════════════════════");
    let gs3d_result = test_with_loss(&target, grid_size, iterations, LossType::GS3D, image_path);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<25} {:>10} {:>10} {:>10} {:>8}",
             "Loss Function", "PSNR", "Improve", "SSIM", "Time");
    println!("{:-<25} {:-<10} {:-<10} {:-<10} {:-<8}", "", "", "", "", "");
    println!("{:<25} {:>10.2} {:>9.2}dB {:>10.4} {:>7.1}s",
             "L2 (MSE)", l2_result.0, l2_result.1, l2_result.2, l2_result.3);
    println!("{:<25} {:>10.2} {:>9.2}dB {:>10.4} {:>7.1}s",
             "L1 (MAE)", l1_result.0, l1_result.1, l1_result.2, l1_result.3);
    println!("{:<25} {:>10.2} {:>9.2}dB {:>10.4} {:>7.1}s",
             "D-SSIM", dssim_result.0, dssim_result.1, dssim_result.2, dssim_result.3);
    println!("{:<25} {:>10.2} {:>9.2}dB {:>10.4} {:>7.1}s",
             "3DGS (L1+D-SSIM)", gs3d_result.0, gs3d_result.1, gs3d_result.2, gs3d_result.3);
    println!("{:-<25} {:-<10} {:-<10} {:-<10} {:-<8}", "", "", "", "", "");

    // Find best
    let results = vec![
        ("L2 (MSE)", l2_result),
        ("L1 (MAE)", l1_result),
        ("D-SSIM", dssim_result),
        ("3DGS", gs3d_result),
    ];

    let best_psnr = results.iter().max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap()).unwrap();
    let best_ssim = results.iter().max_by(|a, b| a.1.2.partial_cmp(&b.1.2).unwrap()).unwrap();

    println!("\nBest PSNR: {} with {:.2} dB", best_psnr.0, best_psnr.1.0);
    println!("Best SSIM: {} with {:.4}", best_ssim.0, best_ssim.1.2);
    println!("═══════════════════════════════════════════════════════════════");
}

#[derive(Clone, Copy)]
enum LossType {
    L2,
    L1,
    DSSIM,
    GS3D,
}

impl LossType {
    fn name(&self) -> &str {
        match self {
            LossType::L2 => "L2",
            LossType::L1 => "L1",
            LossType::DSSIM => "D-SSIM",
            LossType::GS3D => "3DGS",
        }
    }
}

/// Test with a specific loss function
/// Returns: (final_psnr, improvement, ssim, elapsed)
fn test_with_loss(
    target: &ImageBuffer<f32>,
    grid_size: usize,
    iterations: usize,
    loss_type: LossType,
    image_path: &str,
) -> (f32, f32, f32, f32) {
    let (width, height) = (target.width, target.height);
    let mut gaussians = create_gaussians(target, grid_size);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", gaussians.len(), init_psnr);

    let start = Instant::now();

    // Custom optimizer with configurable loss
    let learning_rates = LearningRates {
        position: 0.0002,
        color: 0.02,
        scale: 0.005,
        opacity: 0.05,
        position_decay: 0.995,
    };

    let mut best_psnr = init_psnr;
    let mut best_gaussians = gaussians.clone();

    for iter in 1..=iterations {
        let rendered = RendererV2::render(&gaussians, width, height);

        // Compute loss based on type
        let loss = match loss_type {
            LossType::L2 => compute_l2_loss(&rendered, target),
            LossType::L1 => compute_l1_loss(&rendered, target),
            LossType::DSSIM => compute_dssim_loss(&rendered, target),
            LossType::GS3D => compute_3dgs_loss(&rendered, target),
        };

        // Compute gradients (finite differences for this test)
        let gradients = compute_gradients(&gaussians, target, width, height, loss_type);

        // Update Gaussians
        let pos_lr = learning_rates.position * learning_rates.position_decay.powi(iter as i32);
        for (i, gaussian) in gaussians.iter_mut().enumerate() {
            // Position
            gaussian.position.x -= pos_lr * gradients[i].dx;
            gaussian.position.y -= pos_lr * gradients[i].dy;
            gaussian.position.x = gaussian.position.x.clamp(0.01, 0.99);
            gaussian.position.y = gaussian.position.y.clamp(0.01, 0.99);

            // Color
            gaussian.color.r -= learning_rates.color * gradients[i].dr;
            gaussian.color.g -= learning_rates.color * gradients[i].dg;
            gaussian.color.b -= learning_rates.color * gradients[i].db;
            gaussian.color.r = gaussian.color.r.clamp(0.0, 1.0);
            gaussian.color.g = gaussian.color.g.clamp(0.0, 1.0);
            gaussian.color.b = gaussian.color.b.clamp(0.0, 1.0);

            // Scale
            gaussian.shape.scale_x -= learning_rates.scale * gradients[i].dsx;
            gaussian.shape.scale_y -= learning_rates.scale * gradients[i].dsy;
            gaussian.shape.scale_x = gaussian.shape.scale_x.clamp(0.001, 0.5);
            gaussian.shape.scale_y = gaussian.shape.scale_y.clamp(0.001, 0.5);
        }

        // Track best
        let current_psnr = compute_psnr(&rendered, target);
        if current_psnr > best_psnr {
            best_psnr = current_psnr;
            best_gaussians = gaussians.clone();
        }

        if iter % 20 == 0 {
            println!("  Iteration {}/{}: loss = {:.6}, PSNR = {:.2}dB", iter, iterations, loss, current_psnr);
        }
    }

    // Restore best
    gaussians = best_gaussians;
    let elapsed = start.elapsed().as_secs_f32();

    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);
    let final_ssim = compute_ssim(&final_render, target);

    println!("  Final: PSNR={:.2}dB ({:+.2}dB), SSIM={:.4} in {:.1}s",
             final_psnr, final_psnr - init_psnr, final_ssim, elapsed);

    // Save result
    let mut result = TestResult::new(&format!("loss_function_{}", loss_type.name()), image_path);
    result.optimizer = format!("Custom GD with {} loss", loss_type.name());
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;
    result.extra_metrics = Some(json!({
        "loss_type": loss_type.name(),
        "final_ssim": final_ssim,
        "learning_rates": {
            "position": learning_rates.position,
            "color": learning_rates.color,
            "scale": learning_rates.scale,
        },
    }));
    let _ = result.save(RESULTS_DIR);

    (final_psnr, final_psnr - init_psnr, final_ssim, elapsed)
}

struct LearningRates {
    position: f32,
    color: f32,
    scale: f32,
    opacity: f32,
    position_decay: f32,
}

struct Gradient {
    dx: f32,
    dy: f32,
    dr: f32,
    dg: f32,
    db: f32,
    dsx: f32,
    dsy: f32,
}

fn compute_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    width: u32,
    height: u32,
    loss_type: LossType,
) -> Vec<Gradient> {
    let eps = 0.0001;
    let mut gradients = Vec::with_capacity(gaussians.len());

    for i in 0..gaussians.len() {
        let mut grad = Gradient {
            dx: 0.0, dy: 0.0,
            dr: 0.0, dg: 0.0, db: 0.0,
            dsx: 0.0, dsy: 0.0,
        };

        // Position X gradient
        let mut gs = gaussians.to_vec();
        gs[i].position.x += eps;
        let render_plus = RendererV2::render(&gs, width, height);
        gs[i].position.x -= 2.0 * eps;
        let render_minus = RendererV2::render(&gs, width, height);
        let loss_plus = compute_loss_by_type(&render_plus, target, loss_type);
        let loss_minus = compute_loss_by_type(&render_minus, target, loss_type);
        grad.dx = (loss_plus - loss_minus) / (2.0 * eps);

        // Position Y gradient
        gs = gaussians.to_vec();
        gs[i].position.y += eps;
        let render_plus = RendererV2::render(&gs, width, height);
        gs[i].position.y -= 2.0 * eps;
        let render_minus = RendererV2::render(&gs, width, height);
        let loss_plus = compute_loss_by_type(&render_plus, target, loss_type);
        let loss_minus = compute_loss_by_type(&render_minus, target, loss_type);
        grad.dy = (loss_plus - loss_minus) / (2.0 * eps);

        // Color R gradient
        gs = gaussians.to_vec();
        gs[i].color.r = (gs[i].color.r + eps).min(1.0);
        let render_plus = RendererV2::render(&gs, width, height);
        gs[i].color.r = (gaussians[i].color.r - eps).max(0.0);
        let render_minus = RendererV2::render(&gs, width, height);
        let loss_plus = compute_loss_by_type(&render_plus, target, loss_type);
        let loss_minus = compute_loss_by_type(&render_minus, target, loss_type);
        grad.dr = (loss_plus - loss_minus) / (2.0 * eps);

        // Color G gradient
        gs = gaussians.to_vec();
        gs[i].color.g = (gs[i].color.g + eps).min(1.0);
        let render_plus = RendererV2::render(&gs, width, height);
        gs[i].color.g = (gaussians[i].color.g - eps).max(0.0);
        let render_minus = RendererV2::render(&gs, width, height);
        let loss_plus = compute_loss_by_type(&render_plus, target, loss_type);
        let loss_minus = compute_loss_by_type(&render_minus, target, loss_type);
        grad.dg = (loss_plus - loss_minus) / (2.0 * eps);

        // Color B gradient
        gs = gaussians.to_vec();
        gs[i].color.b = (gs[i].color.b + eps).min(1.0);
        let render_plus = RendererV2::render(&gs, width, height);
        gs[i].color.b = (gaussians[i].color.b - eps).max(0.0);
        let render_minus = RendererV2::render(&gs, width, height);
        let loss_plus = compute_loss_by_type(&render_plus, target, loss_type);
        let loss_minus = compute_loss_by_type(&render_minus, target, loss_type);
        grad.db = (loss_plus - loss_minus) / (2.0 * eps);

        gradients.push(grad);
    }

    gradients
}

fn compute_loss_by_type(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>, loss_type: LossType) -> f32 {
    match loss_type {
        LossType::L2 => compute_l2_loss(rendered, target),
        LossType::L1 => compute_l1_loss(rendered, target),
        LossType::DSSIM => compute_dssim_loss(rendered, target),
        LossType::GS3D => compute_3dgs_loss(rendered, target),
    }
}

fn create_gaussians(target: &ImageBuffer<f32>, grid_size: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(grid_size * grid_size);

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let jitter_x = ((gx * 7 + gy * 13) % 10) as f32 / 100.0 - 0.05;
            let jitter_y = ((gx * 11 + gy * 17) % 10) as f32 / 100.0 - 0.05;

            let x = ((gx as f32 + 0.5) / grid_size as f32 + jitter_x).clamp(0.01, 0.99);
            let y = ((gy as f32 + 0.5) / grid_size as f32 + jitter_y).clamp(0.01, 0.99);

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
    }

    gaussians
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mse = compute_l2_loss(rendered, target);
    if mse <= 0.0 { 100.0 } else { 10.0_f32 * (1.0_f32 / mse).log10() }
}
