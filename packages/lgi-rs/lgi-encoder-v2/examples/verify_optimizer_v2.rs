//! VERIFICATION TEST: Does OptimizerV2 actually reduce loss?
//!
//! Run: cargo run --release --example verify_optimizer_v2 -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::optimizer_v2::OptimizerV2;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::{TestResult, save_rendered_image};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("=== VERIFICATION: OptimizerV2 Basic Function ===\n");

    let image_path = std::env::args().nth(1)
        .unwrap_or_else(|| "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png".to_string());

    println!("Loading: {}", image_path);
    let img = image::open(&image_path).expect("Failed to load image");
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    println!("Dimensions: {}x{}", width, height);

    let mut target = ImageBuffer::new(width, height);
    for (x, y, pixel) in rgb.enumerate_pixels() {
        target.set_pixel(x, y, Color4::new(
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
            1.0,
        ));
    }

    // Same setup as Adam test
    let grid_size = 16;
    let n_gaussians = grid_size * grid_size;
    println!("Gaussians: {} ({}x{} grid)", n_gaussians, grid_size, grid_size);

    let mut gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = Vec::with_capacity(n_gaussians);

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx as f32 + 0.5) / grid_size as f32;
            let y = (gy as f32 + 0.5) / grid_size as f32;

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

    let initial_render = RendererV2::render(&gaussians, width, height);
    let initial_loss = compute_mse(&initial_render, &target);
    let initial_psnr = mse_to_psnr(initial_loss);

    println!("\n--- INITIAL STATE ---");
    println!("Loss (MSE): {:.6}", initial_loss);
    println!("PSNR: {:.2} dB", initial_psnr);

    println!("\n--- RUNNING OPTIMIZER V2 ---");
    let iterations = 50;

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = iterations;
    // Using default config (no GPU, no MS-SSIM)

    println!("Config: lr_position={}, lr_color={}, lr_scale={}",
             optimizer.learning_rate_position,
             optimizer.learning_rate_color,
             optimizer.learning_rate_scale);

    let start = Instant::now();
    let final_loss = optimizer.optimize(&mut gaussians, &target);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = mse_to_psnr(final_loss);

    println!("\n--- FINAL STATE ---");
    println!("Loss (MSE): {:.6}", final_loss);
    println!("PSNR: {:.2} dB", final_psnr);
    println!("Time: {:.2}s", elapsed.as_secs_f32());

    println!("\n=== VERIFICATION RESULTS ===");
    let loss_decreased = final_loss < initial_loss;
    let psnr_improved = final_psnr > initial_psnr;
    let improvement_db = final_psnr - initial_psnr;

    println!("Loss decreased: {} ({:.6} -> {:.6})",
             if loss_decreased { "YES ✓" } else { "NO ✗" },
             initial_loss, final_loss);
    println!("PSNR improved: {} ({:.2} -> {:.2} dB, {:+.2} dB)",
             if psnr_improved { "YES ✓" } else { "NO ✗" },
             initial_psnr, final_psnr, improvement_db);

    // Save result
    let mut result = TestResult::new("verify_optimizer_v2", &image_path);
    result.optimizer = "OptimizerV2 (default config, L2 loss)".to_string();
    result.n_gaussians = n_gaussians;
    result.iterations = iterations;
    result.initial_psnr = initial_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = improvement_db;
    result.final_loss = final_loss;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.notes = format!(
        "VERIFICATION. Loss decreased: {}. lr_pos={}, lr_color={}, lr_scale={}",
        loss_decreased,
        optimizer.learning_rate_position,
        optimizer.learning_rate_color,
        optimizer.learning_rate_scale,
    );

    match result.save(RESULTS_DIR) {
        Ok(path) => println!("\nResult saved: {}", path),
        Err(e) => eprintln!("Failed to save result: {}", e),
    }

    match save_rendered_image(&final_render, RESULTS_DIR, "verify_optimizer_v2", "final") {
        Ok(path) => println!("Image saved: {}", path),
        Err(e) => eprintln!("Failed to save image: {}", e),
    }

    println!("\n=== VERDICT ===");
    if loss_decreased && improvement_db > 0.5 {
        println!("PASS: OptimizerV2 appears to function correctly");
    } else if loss_decreased {
        println!("MARGINAL: Loss decreased but improvement small ({:.2} dB)", improvement_db);
    } else {
        println!("FAIL: Optimizer did not reduce loss!");
    }
}

fn compute_mse(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut sum = 0.0;
    let count = (rendered.width * rendered.height * 3) as f32;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        sum += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    sum / count
}

fn mse_to_psnr(mse: f32) -> f32 {
    if mse <= 0.0 { return 100.0; }
    10.0 * (1.0 / mse).log10()
}
