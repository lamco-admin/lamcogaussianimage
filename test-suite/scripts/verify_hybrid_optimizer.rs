//! Verify Hybrid Adam→L-BFGS Optimizer
//!
//! Tests the hybrid optimizer with analytical gradients for L-BFGS phase.
//! Hypothesis: L-BFGS refinement after Adam warm-start improves final quality.
//!
//! Run: cargo run --release --example verify_hybrid_optimizer -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::hybrid_optimizer::{HybridOptimizer, HybridConfig};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   HYBRID OPTIMIZER VERIFICATION                              ║");
    println!("║   Testing Adam warm-start + L-BFGS refinement                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load");
    let (width, height) = (target.width, target.height);

    let grid_size = 16;  // 256 Gaussians
    let n_gaussians = grid_size * grid_size;

    println!("Image: {}x{}, Gaussians: {}", width, height, n_gaussians);
    println!();

    // Create identical starting Gaussians for fair comparison
    let base_gaussians = create_gaussians(&target, grid_size);

    let init_render = RendererV2::render(&base_gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, &target);
    let init_loss = compute_loss(&init_render, &target);

    println!("Initial: PSNR={:.2}dB, loss={:.6}\n", init_psnr, init_loss);

    // Test 1: Adam only (50 iterations for comparison)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Adam Only (50 iterations)");
    println!("═══════════════════════════════════════════════════════════════");

    let mut adam_gaussians = base_gaussians.clone();
    let start_adam = Instant::now();

    let mut adam = AdamOptimizer::new();
    adam.max_iterations = 50;
    let adam_loss = adam.optimize(&mut adam_gaussians, &target);

    let adam_elapsed = start_adam.elapsed();
    let adam_render = RendererV2::render(&adam_gaussians, width, height);
    let adam_psnr = compute_psnr(&adam_render, &target);

    println!("  Final:  PSNR={:.2}dB, loss={:.6}", adam_psnr, adam_loss);
    println!("  Result: {:+.2}dB improvement in {:.1}s\n", adam_psnr - init_psnr, adam_elapsed.as_secs_f32());

    // Save Adam result
    let mut adam_result = TestResult::new("verify_hybrid_adam_only", image_path);
    adam_result.optimizer = "Adam (50 iterations)".to_string();
    adam_result.image_dimensions = (width, height);
    adam_result.n_gaussians = n_gaussians;
    adam_result.iterations = 50;
    adam_result.initial_psnr = init_psnr;
    adam_result.final_psnr = adam_psnr;
    adam_result.improvement_db = adam_psnr - init_psnr;
    adam_result.final_loss = adam_loss;
    adam_result.elapsed_seconds = adam_elapsed.as_secs_f32();
    let _ = adam_result.save(RESULTS_DIR);

    // Test 2: Hybrid (30 Adam + 20 L-BFGS)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 2: Hybrid (30 Adam + 20 L-BFGS)");
    println!("═══════════════════════════════════════════════════════════════");

    let mut hybrid_gaussians = base_gaussians.clone();
    let start_hybrid = Instant::now();

    let mut config = HybridConfig::default();
    config.adam_iterations = 30;
    config.lbfgs_iterations = 20;

    let mut hybrid = HybridOptimizer::new(config);
    let hybrid_loss = hybrid.optimize(&mut hybrid_gaussians, &target);

    let hybrid_elapsed = start_hybrid.elapsed();
    let hybrid_render = RendererV2::render(&hybrid_gaussians, width, height);
    let hybrid_psnr = compute_psnr(&hybrid_render, &target);

    println!("  Final:  PSNR={:.2}dB, loss={:.6}", hybrid_psnr, hybrid_loss);
    println!("  Result: {:+.2}dB improvement in {:.1}s\n", hybrid_psnr - init_psnr, hybrid_elapsed.as_secs_f32());

    // Save Hybrid result
    let mut hybrid_result = TestResult::new("verify_hybrid_adam_lbfgs", image_path);
    hybrid_result.optimizer = "Hybrid (30 Adam + 20 L-BFGS)".to_string();
    hybrid_result.image_dimensions = (width, height);
    hybrid_result.n_gaussians = n_gaussians;
    hybrid_result.iterations = 50;  // Total iterations
    hybrid_result.initial_psnr = init_psnr;
    hybrid_result.final_psnr = hybrid_psnr;
    hybrid_result.improvement_db = hybrid_psnr - init_psnr;
    hybrid_result.final_loss = hybrid_loss;
    hybrid_result.elapsed_seconds = hybrid_elapsed.as_secs_f32();
    hybrid_result.extra_metrics = Some(json!({
        "adam_iterations": 30,
        "lbfgs_iterations": 20,
        "uses_analytical_gradients": true,
    }));
    let _ = hybrid_result.save(RESULTS_DIR);

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("                        COMPARISON                             ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<25} {:>10} {:>10} {:>10}", "Method", "PSNR", "Improve", "Time");
    println!("{:-<25} {:-<10} {:-<10} {:-<10}", "", "", "", "");
    println!("{:<25} {:>10.2} {:>+10.2} {:>9.1}s", "Adam (50 iter)", adam_psnr, adam_psnr - init_psnr, adam_elapsed.as_secs_f32());
    println!("{:<25} {:>10.2} {:>+10.2} {:>9.1}s", "Hybrid (30+20)", hybrid_psnr, hybrid_psnr - init_psnr, hybrid_elapsed.as_secs_f32());
    println!("{:-<25} {:-<10} {:-<10} {:-<10}", "", "", "", "");

    let lbfgs_benefit = hybrid_psnr - adam_psnr;
    if lbfgs_benefit > 0.05 {
        println!("\nHYPOTHESIS CONFIRMED: L-BFGS refinement adds {:+.2}dB over Adam alone!", lbfgs_benefit);
    } else if lbfgs_benefit > -0.05 {
        println!("\nRESULT: L-BFGS refinement provides similar quality ({:+.2}dB)", lbfgs_benefit);
    } else {
        println!("\nHYPOTHESIS REJECTED: L-BFGS refinement hurt quality ({:+.2}dB)", lbfgs_benefit);
    }
    println!("═══════════════════════════════════════════════════════════════");
}

fn create_gaussians(target: &ImageBuffer<f32>, grid_size: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(grid_size * grid_size);
    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx as f32 + 0.5) / grid_size as f32;
            let y = (gy as f32 + 0.5) / grid_size as f32;
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
    let count = (rendered.width * rendered.height * 3) as f32;
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= count;
    if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}

fn compute_loss(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height) as f32;
    let mut loss = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        loss += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    loss / count
}
