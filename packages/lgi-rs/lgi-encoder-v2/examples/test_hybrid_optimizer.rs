//! Test Hybrid Adam → L-BFGS optimizer
//!
//! Compares: Adam-only vs Adam→L-BFGS hybrid on a Kodak image

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{
    EncoderV2,
    renderer_v2::RendererV2,
    adam_optimizer::AdamOptimizer,
    hybrid_optimizer::{HybridOptimizer, HybridConfig},
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Hybrid Adam → L-BFGS Optimizer Test                         ║");
    println!("║  Comparing: Adam-only (150 iter) vs Hybrid (100+50 iter)     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load Kodak image
    let kodak_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    println!("Loading {}...", kodak_path);

    let target = match ImageBuffer::load(kodak_path) {
        Ok(img) => {
            println!("  Loaded: {}x{}", img.width, img.height);
            img
        }
        Err(e) => {
            eprintln!("ERROR: Could not load image: {}", e);
            eprintln!("Creating synthetic test image...");
            create_synthetic_test_image()
        }
    };

    let grid_size = 24;  // 576 Gaussians (smaller for faster L-BFGS with finite differences)
    let num_gaussians = grid_size * grid_size;

    println!("\nUsing {} Gaussians ({}x{} grid)", num_gaussians, grid_size, grid_size);

    // Test 1: Adam-only (150 iterations with per-param LRs)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 1: Adam-Only (150 iterations, per-parameter LRs)");
    println!("═══════════════════════════════════════════════════════════════");

    let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
    let mut gaussians_adam = encoder.initialize_gaussians(grid_size);

    let init_rendered = RendererV2::render(&gaussians_adam, target.width, target.height);
    let init_psnr = compute_psnr(&target, &init_rendered);
    println!("Initial PSNR: {:.2} dB", init_psnr);

    let mut adam = AdamOptimizer::new();
    adam.max_iterations = 150;

    let adam_loss = adam.optimize(&mut gaussians_adam, &target);

    let adam_rendered = RendererV2::render(&gaussians_adam, target.width, target.height);
    let adam_psnr = compute_psnr(&target, &adam_rendered);

    println!("\nAdam-only Final: {:.2} dB (loss: {:.6})", adam_psnr, adam_loss);
    println!("Improvement: {:+.2} dB", adam_psnr - init_psnr);

    // Test 2: Hybrid Adam (100) → L-BFGS (50)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: Hybrid Adam(100) → L-BFGS(50)");
    println!("═══════════════════════════════════════════════════════════════");

    let encoder2 = EncoderV2::new(target.clone()).expect("Encoder failed");
    let mut gaussians_hybrid = encoder2.initialize_gaussians(grid_size);

    let config = HybridConfig {
        adam_iterations: 100,
        lbfgs_iterations: 50,
        lbfgs_history: 10,
        ..Default::default()
    };

    let mut hybrid = HybridOptimizer::new(config);
    let hybrid_loss = hybrid.optimize(&mut gaussians_hybrid, &target);

    let hybrid_rendered = RendererV2::render(&gaussians_hybrid, target.width, target.height);
    let hybrid_psnr = compute_psnr(&target, &hybrid_rendered);

    println!("\nHybrid Final: {:.2} dB (loss: {:.6})", hybrid_psnr, hybrid_loss);
    println!("Improvement: {:+.2} dB", hybrid_psnr - init_psnr);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Initial PSNR:              {:.2} dB", init_psnr);
    println!("Adam-only (150 iter):      {:.2} dB  (loss: {:.6})", adam_psnr, adam_loss);
    println!("Hybrid (100+50 iter):      {:.2} dB  (loss: {:.6})", hybrid_psnr, hybrid_loss);
    println!("");
    let diff = hybrid_psnr - adam_psnr;
    if diff > 0.1 {
        println!("Hybrid WINS by:            {:+.2} dB", diff);
    } else if diff < -0.1 {
        println!("Adam WINS by:              {:+.2} dB", -diff);
    } else {
        println!("Essentially TIED (diff: {:+.2} dB)", diff);
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

            let r = (xf * yf).sqrt();
            let g = ((1.0 - xf) * yf).sqrt();
            let b = (xf * (1.0 - yf)).sqrt();

            img.set_pixel(x, y, lgi_math::color::Color4::new(r, g, b, 1.0));
        }
    }
    img
}
