//! VERIFICATION TEST: Does Adam optimizer actually reduce loss?
//!
//! Purpose: Sanity check that the optimizer is functioning at all.
//! NOT a quality test - just "does loss go down?"
//!
//! Run: cargo run --release --example verify_adam_basic -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::{TestResult, save_rendered_image};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("=== VERIFICATION: Adam Optimizer Basic Function ===\n");

    // Load test image
    let image_path = std::env::args().nth(1)
        .unwrap_or_else(|| "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png".to_string());

    println!("Loading: {}", image_path);
    let img = image::open(&image_path).expect("Failed to load image");
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    println!("Dimensions: {}x{}", width, height);

    // Convert to ImageBuffer
    let mut target = ImageBuffer::new(width, height);
    for (x, y, pixel) in rgb.enumerate_pixels() {
        target.set_pixel(x, y, Color4::new(
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
            1.0,
        ));
    }

    // Create simple grid of Gaussians
    let grid_size = 16; // 16x16 = 256 Gaussians (small for quick test)
    let n_gaussians = grid_size * grid_size;
    println!("Gaussians: {} ({}x{} grid)", n_gaussians, grid_size, grid_size);

    let mut gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = Vec::with_capacity(n_gaussians);

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx as f32 + 0.5) / grid_size as f32;
            let y = (gy as f32 + 0.5) / grid_size as f32;

            // Sample color from target at this position
            let px = (x * width as f32) as u32;
            let py = (y * height as f32) as u32;
            let px = px.min(width - 1);
            let py = py.min(height - 1);
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

    // Render initial state
    let initial_render = RendererV2::render(&gaussians, width, height);
    let initial_loss = compute_mse(&initial_render, &target);
    let initial_psnr = mse_to_psnr(initial_loss);

    println!("\n--- INITIAL STATE ---");
    println!("Loss (MSE): {:.6}", initial_loss);
    println!("PSNR: {:.2} dB", initial_psnr);

    // Run optimizer
    println!("\n--- RUNNING ADAM OPTIMIZER ---");
    let iterations = 50; // Short test

    let mut optimizer = AdamOptimizer::new();
    optimizer.max_iterations = iterations;
    // Using default per-param LRs

    let start = Instant::now();
    let final_loss = optimizer.optimize(&mut gaussians, &target);
    let elapsed = start.elapsed();

    // Render final state
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = mse_to_psnr(final_loss);

    println!("\n--- FINAL STATE ---");
    println!("Loss (MSE): {:.6}", final_loss);
    println!("PSNR: {:.2} dB", final_psnr);
    println!("Time: {:.2}s", elapsed.as_secs_f32());

    // VERIFICATION
    println!("\n=== VERIFICATION RESULTS ===");
    let loss_decreased = final_loss < initial_loss;
    let psnr_improved = final_psnr > initial_psnr;
    let improvement_db = final_psnr - initial_psnr;

    println!("Loss decreased: {} ({:.6} -> {:.6})",
             if loss_decreased { "YES ✓" } else { "NO ✗" },
             initial_loss, final_loss);
    println!("PSNR improved: {} ({:.2} -> {:.2} dB, +{:.2} dB)",
             if psnr_improved { "YES ✓" } else { "NO ✗" },
             initial_psnr, final_psnr, improvement_db);

    // Save result
    let mut result = TestResult::new("verify_adam_basic", &image_path);
    result.optimizer = "Adam (per-param LRs, default config)".to_string();
    result.n_gaussians = n_gaussians;
    result.iterations = iterations;
    result.initial_psnr = initial_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement_db;
    result.final_loss = final_loss;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.notes = format!(
        "VERIFICATION TEST. Loss decreased: {}. Grid: {}x{}. Per-param LRs: pos={}, color={}, scale={}",
        loss_decreased,
        grid_size, grid_size,
        optimizer.learning_rates.position,
        optimizer.learning_rates.color,
        optimizer.learning_rates.scale,
    );

    match result.save(RESULTS_DIR) {
        Ok(path) => println!("\nResult saved: {}", path),
        Err(e) => eprintln!("Failed to save result: {}", e),
    }

    // Save rendered image
    match save_rendered_image(&final_render, RESULTS_DIR, "verify_adam_basic", "final") {
        Ok(path) => println!("Image saved: {}", path),
        Err(e) => eprintln!("Failed to save image: {}", e),
    }

    // Final verdict
    println!("\n=== VERDICT ===");
    if loss_decreased && improvement_db > 0.5 {
        println!("PASS: Adam optimizer appears to function correctly");
        println!("      (Loss decreased, PSNR improved by {:.2} dB)", improvement_db);
    } else if loss_decreased {
        println!("MARGINAL: Loss decreased but improvement small ({:.2} dB)", improvement_db);
        println!("          May indicate suboptimal hyperparameters");
    } else {
        println!("FAIL: Optimizer did not reduce loss!");
        println!("      This indicates a BUG in the optimizer or gradients");
    }
}

fn compute_mse(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut sum = 0.0;
    let count = (rendered.width * rendered.height * 3) as f32;

    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        sum += (r.r - t.r).powi(2);
        sum += (r.g - t.g).powi(2);
        sum += (r.b - t.b).powi(2);
    }

    sum / count
}

fn mse_to_psnr(mse: f32) -> f32 {
    if mse <= 0.0 {
        return 100.0; // Perfect match
    }
    10.0 * (1.0 / mse).log10()
}
