//! Test Levenberg-Marquardt optimizer
//!
//! Compares: Adam (warm-start) vs Adam→L-M hybrid on a Kodak image
//!
//! NOTE: L-M with finite differences is VERY slow for large problems.
//! Using small Gaussian count (16x16 = 256) for reasonable test time.

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{
    EncoderV2,
    renderer_v2::RendererV2,
    adam_optimizer::AdamOptimizer,
    lm_optimizer::{LMOptimizer, LMConfig},
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Levenberg-Marquardt Optimizer Test                          ║");
    println!("║  Comparing: Adam-only vs Adam→L-M hybrid                     ║");
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

    // Small grid for L-M (finite differences is expensive!)
    let grid_size = 16;  // 256 Gaussians
    let num_gaussians = grid_size * grid_size;

    println!("\nUsing {} Gaussians ({}x{} grid)", num_gaussians, grid_size, grid_size);
    println!("NOTE: L-M uses finite differences for Jacobian - this will be slow\n");

    // Test 1: Adam-only (100 iterations with per-param LRs)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Adam-Only (100 iterations, per-parameter LRs)");
    println!("═══════════════════════════════════════════════════════════════");

    let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
    let mut gaussians_adam = encoder.initialize_gaussians(grid_size);

    let init_rendered = RendererV2::render(&gaussians_adam, target.width, target.height);
    let init_psnr = compute_psnr(&target, &init_rendered);
    println!("Initial PSNR: {:.2} dB", init_psnr);

    let mut adam = AdamOptimizer::new();
    adam.max_iterations = 100;

    let start_adam = std::time::Instant::now();
    let adam_loss = adam.optimize(&mut gaussians_adam, &target);
    let adam_time = start_adam.elapsed();

    let adam_rendered = RendererV2::render(&gaussians_adam, target.width, target.height);
    let adam_psnr = compute_psnr(&target, &adam_rendered);

    println!("\nAdam-only Final: {:.2} dB (loss: {:.6})", adam_psnr, adam_loss);
    println!("Time: {:.2}s", adam_time.as_secs_f32());
    println!("Improvement: {:+.2} dB", adam_psnr - init_psnr);

    // Test 2: Adam (50) → L-M (20)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: Hybrid Adam(50) → L-M(20)");
    println!("═══════════════════════════════════════════════════════════════");

    let encoder2 = EncoderV2::new(target.clone()).expect("Encoder failed");
    let mut gaussians_hybrid = encoder2.initialize_gaussians(grid_size);

    // Phase 1: Adam warm-start
    println!("\n--- Phase 1: Adam Warm-Start (50 iterations) ---");
    let mut adam2 = AdamOptimizer::new();
    adam2.max_iterations = 50;

    let start_hybrid = std::time::Instant::now();
    let adam_warmup_loss = adam2.optimize(&mut gaussians_hybrid, &target);
    let adam_phase_time = start_hybrid.elapsed();

    let warmup_rendered = RendererV2::render(&gaussians_hybrid, target.width, target.height);
    let warmup_psnr = compute_psnr(&target, &warmup_rendered);
    println!("After Adam: {:.2} dB (loss: {:.6}, time: {:.2}s)",
             warmup_psnr, adam_warmup_loss, adam_phase_time.as_secs_f32());

    // Phase 2: L-M refinement
    println!("\n--- Phase 2: L-M Refinement (20 iterations) ---");
    println!("(This may take several minutes due to finite difference Jacobian)");

    let config = LMConfig {
        max_iterations: 20,
        tolerance: 1e-8,
        ..Default::default()
    };

    let mut lm = LMOptimizer::new(config);
    let lm_loss = lm.optimize(&mut gaussians_hybrid, &target);
    let total_time = start_hybrid.elapsed();

    let hybrid_rendered = RendererV2::render(&gaussians_hybrid, target.width, target.height);
    let hybrid_psnr = compute_psnr(&target, &hybrid_rendered);

    println!("\nHybrid Final: {:.2} dB (loss: {:.6})", hybrid_psnr, lm_loss);
    println!("Total Time: {:.2}s", total_time.as_secs_f32());
    println!("Improvement: {:+.2} dB", hybrid_psnr - init_psnr);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Initial PSNR:              {:.2} dB", init_psnr);
    println!("Adam-only (100 iter):      {:.2} dB  ({:.2}s)", adam_psnr, adam_time.as_secs_f32());
    println!("Hybrid Adam(50)+LM(20):    {:.2} dB  ({:.2}s)", hybrid_psnr, total_time.as_secs_f32());
    println!("");
    let diff = hybrid_psnr - adam_psnr;
    if diff > 0.1 {
        println!("L-M Hybrid WINS by:        {:+.2} dB", diff);
    } else if diff < -0.1 {
        println!("Adam-only WINS by:         {:+.2} dB", -diff);
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
