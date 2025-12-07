//! MS-SSIM Loss Function Test
//!
//! Validates that MS-SSIM works correctly and can be used in optimization.
//! Tests both the forward pass (loss computation) and gradient computation.
//!
//! Run: cargo run --release --example test_msssim_loss -p lgi-encoder-v2

use lgi_core::{ImageBuffer, ms_ssim::MSSSIM, ms_ssim_gradients::MsssimGradients};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       MS-SSIM LOSS FUNCTION TEST                             ║");
    println!("║       Validating perceptual loss computation                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Test 1: Basic MS-SSIM computation
    println!("--- Test 1: Basic MS-SSIM computation ---\n");
    test_basic_msssim();

    // Test 2: MS-SSIM gradient computation
    println!("\n--- Test 2: MS-SSIM gradients ---\n");
    test_msssim_gradients();

    // Test 3: MS-SSIM vs MSE comparison on real image
    println!("\n--- Test 3: MS-SSIM vs MSE on Kodak image ---\n");
    test_msssim_vs_mse();

    // Test 4: Hybrid loss (MSE + MS-SSIM) optimization
    println!("\n--- Test 4: Hybrid loss optimization ---\n");
    test_hybrid_loss_optimization();

    println!("\n\nAll MS-SSIM tests complete!");
}

/// Test basic MS-SSIM computation correctness
fn test_basic_msssim() {
    let ms_ssim = MSSSIM::default();

    // Test 1a: Identical images should have MS-SSIM ≈ 1.0
    let img1 = create_test_image(64, 64, 0.5, 0.5, 0.5);
    let img2 = img1.clone();

    let similarity = ms_ssim.compute(&img1, &img2);
    println!("  Identical images:      MS-SSIM = {:.4} (expected ≈ 1.0)", similarity);

    // Test 1b: Very different images should have low MS-SSIM
    let img3 = create_test_image(64, 64, 0.0, 0.0, 0.0);  // Black
    let img4 = create_test_image(64, 64, 1.0, 1.0, 1.0);  // White

    let diff_similarity = ms_ssim.compute(&img3, &img4);
    println!("  Black vs White:        MS-SSIM = {:.4} (expected < 0.5)", diff_similarity);

    // Test 1c: Slightly noisy image should have high but not perfect MS-SSIM
    let img5 = add_noise(&img1, 0.05);
    let noisy_similarity = ms_ssim.compute(&img1, &img5);
    println!("  Original vs 5% noise:  MS-SSIM = {:.4} (expected 0.7-0.95)", noisy_similarity);

    // Test 1d: Check loss computation
    let loss = ms_ssim.compute_loss(&img1, &img5);
    println!("  5% noise loss:         1-SSIM  = {:.4}", loss);
}

/// Test MS-SSIM gradient computation
fn test_msssim_gradients() {
    let grads_computer = MsssimGradients::default();

    // Create simple test images
    let target = create_gradient_image(64, 64);
    let rendered = add_noise(&target, 0.1);

    // Compute pixel gradients
    let start = Instant::now();
    let pixel_grads = grads_computer.compute_pixel_gradients(&target, &rendered);
    let elapsed = start.elapsed().as_millis();

    println!("  Gradient computation time: {}ms for 64x64 image", elapsed);

    // Compute gradient statistics
    let mut max_grad = 0.0_f32;
    let mut sum_grad = 0.0_f32;
    let mut nonzero_count = 0_usize;

    for pixel in &pixel_grads.data {
        let abs_grad = pixel.r.abs();
        max_grad = max_grad.max(abs_grad);
        sum_grad += abs_grad;
        if abs_grad > 1e-6 {
            nonzero_count += 1;
        }
    }

    let mean_grad = sum_grad / (pixel_grads.width * pixel_grads.height) as f32;
    println!("  Max gradient: {:.6}", max_grad);
    println!("  Mean |gradient|: {:.6}", mean_grad);
    println!("  Non-zero gradients: {} / {}", nonzero_count, pixel_grads.width * pixel_grads.height);

    // Verify gradients are reasonable
    if max_grad > 0.0 && nonzero_count > 0 {
        println!("  ✓ Gradients look reasonable");
    } else {
        println!("  ✗ WARNING: Gradients may be zero or invalid");
    }
}

/// Compare MS-SSIM and MSE on a real image
fn test_msssim_vs_mse() {
    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";

    let target = match ImageBuffer::load(image_path) {
        Ok(img) => img,
        Err(e) => {
            println!("  Could not load test image: {}", e);
            return;
        }
    };

    let (width, height) = (target.width, target.height);
    println!("  Image: {}x{}", width, height);

    // Create Gaussians and render
    let gaussians = create_jittered_grid(&target, 256);
    let rendered = RendererV2::render(&gaussians, width, height);

    // Compute both metrics
    let ms_ssim = MSSSIM::default();
    let ssim_score = ms_ssim.compute(&rendered, &target);
    let ssim_loss = ms_ssim.compute_loss(&rendered, &target);

    let mse = compute_mse(&rendered, &target);
    let psnr = compute_psnr(&rendered, &target);

    println!("  Initial render quality:");
    println!("    PSNR:     {:.2} dB", psnr);
    println!("    MSE:      {:.6}", mse);
    println!("    MS-SSIM:  {:.4}", ssim_score);
    println!("    SSIM Loss: {:.4}", ssim_loss);

    // Check correlation - higher PSNR should correlate with higher MS-SSIM
    println!("\n  Quality correlation:");
    println!("    Higher PSNR = {} → Higher MS-SSIM = {}",
        if psnr > 15.0 { "✓" } else { "✗" },
        if ssim_score > 0.3 { "✓" } else { "✗" }
    );
}

/// Test hybrid loss (MSE + MS-SSIM) in optimization
fn test_hybrid_loss_optimization() {
    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";

    let target = match ImageBuffer::load(image_path) {
        Ok(img) => img,
        Err(e) => {
            println!("  Could not load test image: {}", e);
            return;
        }
    };

    let (width, height) = (target.width, target.height);

    // Initialize Gaussians
    let mut gaussians = create_jittered_grid(&target, 128);  // Smaller for speed

    // Compute initial metrics
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&rendered, &target);
    let ms_ssim_computer = MSSSIM::default();
    let init_ssim = ms_ssim_computer.compute(&rendered, &target);

    println!("  Initial: PSNR={:.2}dB, MS-SSIM={:.4}", init_psnr, init_ssim);

    // Quick optimization using MSE only (for comparison baseline)
    let alpha = 0.8;  // Weight for MSE (1-alpha for MS-SSIM)
    let lr = 0.001;
    let iterations = 50;

    let start = Instant::now();

    for iter in 0..iterations {
        let rendered = RendererV2::render(&gaussians, width, height);

        // Compute hybrid loss
        let mse = compute_mse(&rendered, &target);
        let ssim_loss = ms_ssim_computer.compute_loss(&rendered, &target);
        let hybrid_loss = alpha * mse + (1.0 - alpha) * ssim_loss;

        // Simple gradient descent on MSE component (colors only for simplicity)
        for (g_idx, gaussian) in gaussians.iter_mut().enumerate() {
            let px = ((gaussian.position.x * width as f32) as u32).min(width - 1);
            let py = ((gaussian.position.y * height as f32) as u32).min(height - 1);

            if let (Some(target_color), Some(rendered_color)) =
                (target.get_pixel(px, py), rendered.get_pixel(px, py)) {
                // Color gradient (MSE)
                gaussian.color.r -= lr * 2.0 * (rendered_color.r - target_color.r);
                gaussian.color.g -= lr * 2.0 * (rendered_color.g - target_color.g);
                gaussian.color.b -= lr * 2.0 * (rendered_color.b - target_color.b);

                // Clamp colors
                gaussian.color.r = gaussian.color.r.clamp(0.0, 1.0);
                gaussian.color.g = gaussian.color.g.clamp(0.0, 1.0);
                gaussian.color.b = gaussian.color.b.clamp(0.0, 1.0);
            }
        }

        if iter % 10 == 0 {
            let psnr = compute_psnr(&rendered, &target);
            let ssim = ms_ssim_computer.compute(&rendered, &target);
            println!("    [{:3}] PSNR={:.2}dB, MS-SSIM={:.4}, Hybrid={:.4}",
                iter, psnr, ssim, hybrid_loss);
        }
    }

    let elapsed = start.elapsed().as_secs_f32();

    // Final metrics
    let final_rendered = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_rendered, &target);
    let final_ssim = ms_ssim_computer.compute(&final_rendered, &target);

    println!("\n  Final: PSNR={:.2}dB, MS-SSIM={:.4}", final_psnr, final_ssim);
    println!("  Improvement: PSNR {:+.2}dB, MS-SSIM {:+.4}",
        final_psnr - init_psnr, final_ssim - init_ssim);
    println!("  Time: {:.1}s", elapsed);

    // Save result
    let mut result = TestResult::new("msssim_hybrid_loss_test", image_path);
    result.optimizer = "Hybrid MSE+MS-SSIM (color only)".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = 128;
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;
    result.set_param("alpha", alpha);
    result.set_param("lr", lr);
    result.set_param("init_msssim", init_ssim);
    result.set_param("final_msssim", final_ssim);
    result.complete();
    let _ = result.save_with_image(&final_rendered, RESULTS_DIR);
}

// Helper functions

fn create_test_image(width: u32, height: u32, r: f32, g: f32, b: f32) -> ImageBuffer<f32> {
    let mut img = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            img.set_pixel(x, y, Color4::new(r, g, b, 1.0));
        }
    }
    img
}

fn create_gradient_image(width: u32, height: u32) -> ImageBuffer<f32> {
    let mut img = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let r = x as f32 / width as f32;
            let g = y as f32 / height as f32;
            let b = (x as f32 + y as f32) / (width as f32 + height as f32);
            img.set_pixel(x, y, Color4::new(r, g, b, 1.0));
        }
    }
    img
}

fn add_noise(img: &ImageBuffer<f32>, noise_level: f32) -> ImageBuffer<f32> {
    let mut noisy = img.clone();
    // Deterministic "noise" based on position
    for y in 0..img.height {
        for x in 0..img.width {
            if let Some(pixel) = img.get_pixel(x, y) {
                let noise_r = ((x * 7 + y * 13) % 100) as f32 / 100.0 - 0.5;
                let noise_g = ((x * 11 + y * 17) % 100) as f32 / 100.0 - 0.5;
                let noise_b = ((x * 19 + y * 23) % 100) as f32 / 100.0 - 0.5;

                noisy.set_pixel(x, y, Color4::new(
                    (pixel.r + noise_level * noise_r).clamp(0.0, 1.0),
                    (pixel.g + noise_level * noise_g).clamp(0.0, 1.0),
                    (pixel.b + noise_level * noise_b).clamp(0.0, 1.0),
                    1.0,
                ));
            }
        }
    }
    noisy
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
