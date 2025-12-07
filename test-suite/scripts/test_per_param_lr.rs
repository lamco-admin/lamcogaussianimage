//! Test per-parameter learning rates for Adam optimizer
//!
//! Compares the new per-parameter LR Adam with the old single-LR Adam
//! on a single Kodak image (kodim01).

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, adam_optimizer::AdamOptimizer};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Per-Parameter Learning Rate Adam Test                       ║");
    println!("║  Comparing single LR (0.01) vs per-parameter LRs            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load Kodak image (absolute path to ensure it works)
    let kodak_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    println!("Loading {}...", kodak_path);

    let target = match ImageBuffer::load(kodak_path) {
        Ok(img) => {
            println!("  Loaded: {}x{}", img.width, img.height);
            img
        }
        Err(e) => {
            eprintln!("ERROR: Could not load image: {}", e);
            eprintln!("Creating synthetic test image instead...");
            create_synthetic_test_image()
        }
    };

    let grid_size = 32;  // 1024 Gaussians
    let num_gaussians = grid_size * grid_size;

    println!("\nUsing {} Gaussians ({}x{} grid)", num_gaussians, grid_size, grid_size);
    println!("Max iterations: 200\n");

    // Test 1: Old single learning rate (0.01)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Single Learning Rate (LR=0.01 for all parameters)");
    println!("═══════════════════════════════════════════════════════════════");

    let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
    let mut gaussians_single_lr = encoder.initialize_gaussians(grid_size);

    let init_rendered = RendererV2::render(&gaussians_single_lr, target.width, target.height);
    let init_psnr = compute_psnr(&target, &init_rendered);
    println!("Initial PSNR: {:.2} dB", init_psnr);

    let mut optimizer_single = AdamOptimizer::with_single_lr(0.01);
    optimizer_single.max_iterations = 200;

    let final_loss_single = optimizer_single.optimize(&mut gaussians_single_lr, &target);

    let final_rendered_single = RendererV2::render(&gaussians_single_lr, target.width, target.height);
    let final_psnr_single = compute_psnr(&target, &final_rendered_single);

    println!("\nFinal: {:.2} dB (loss: {:.6})", final_psnr_single, final_loss_single);
    println!("Improvement: {:+.2} dB", final_psnr_single - init_psnr);

    // Test 2: New per-parameter learning rates
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: Per-Parameter Learning Rates");
    println!("  Position: 0.0002 -> 0.00002 (exponential decay)");
    println!("  Color:    0.02");
    println!("  Scale:    0.005");
    println!("  Opacity:  0.05");
    println!("═══════════════════════════════════════════════════════════════");

    let encoder2 = EncoderV2::new(target.clone()).expect("Encoder failed");
    let mut gaussians_per_param = encoder2.initialize_gaussians(grid_size);

    let mut optimizer_per_param = AdamOptimizer::new();  // Default uses per-param LRs
    optimizer_per_param.max_iterations = 200;

    let final_loss_per_param = optimizer_per_param.optimize(&mut gaussians_per_param, &target);

    let final_rendered_per_param = RendererV2::render(&gaussians_per_param, target.width, target.height);
    let final_psnr_per_param = compute_psnr(&target, &final_rendered_per_param);

    println!("\nFinal: {:.2} dB (loss: {:.6})", final_psnr_per_param, final_loss_per_param);
    println!("Improvement: {:+.2} dB", final_psnr_per_param - init_psnr);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Initial PSNR:              {:.2} dB", init_psnr);
    println!("Single LR (0.01):          {:.2} dB", final_psnr_single);
    println!("Per-Parameter LRs:         {:.2} dB", final_psnr_per_param);
    println!("");
    let diff = final_psnr_per_param - final_psnr_single;
    if diff > 0.0 {
        println!("Per-param LR WINS by:      {:+.2} dB", diff);
    } else if diff < 0.0 {
        println!("Single LR WINS by:         {:+.2} dB", -diff);
    } else {
        println!("TIE (identical results)");
    }
    println!("═══════════════════════════════════════════════════════════════");
}

fn compute_psnr(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (original.width * original.height * 3) as f32;

    for (p1, p2) in original.data.iter().zip(rendered.data.iter()) {
        mse += (p1.r - p2.r).powi(2);
        mse += (p1.g - p2.g).powi(2);
        mse += (p1.b - p2.b).powi(2);
    }

    mse /= count;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}

fn create_synthetic_test_image() -> ImageBuffer<f32> {
    let mut img = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let xf = x as f32 / 255.0;
            let yf = y as f32 / 255.0;

            // Complex pattern with gradients and edges
            let r = (xf * yf).sqrt();
            let g = ((1.0 - xf) * yf).sqrt();
            let b = (xf * (1.0 - yf)).sqrt();

            img.set_pixel(x, y, lgi_math::color::Color4::new(r, g, b, 1.0));
        }
    }
    img
}
