//! Comprehensive Optimizer Benchmark
//!
//! Runs all optimizers on a single image and saves:
//! - JSON metrics to test-results/YYYY-MM-DD/
//! - Rendered images to test-results/rendered-images/
//!
//! This creates a permanent record for analysis.

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{
    EncoderV2,
    renderer_v2::RendererV2,
    adam_optimizer::AdamOptimizer,
    test_results::{TestResult, save_rendered_image},
};

const RESULTS_DIR: &str = "/home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs/lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Comprehensive Optimizer Benchmark                           ║");
    println!("║  Results saved to test-results/                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load test image
    let kodak_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = match ImageBuffer::load(kodak_path) {
        Ok(img) => {
            println!("Loaded: {} ({}x{})\n", kodak_path, img.width, img.height);
            img
        }
        Err(e) => {
            eprintln!("ERROR: Could not load image: {}", e);
            return;
        }
    };

    let grid_size = 24;  // 576 Gaussians
    let iterations = 150;

    // Test 1: Adam with per-param LRs
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Adam with Per-Parameter Learning Rates");
    println!("═══════════════════════════════════════════════════════════════");

    let mut result = TestResult::new("adam_perparam_kodim03", kodak_path);
    result.optimizer = "Adam (per-param LRs)".to_string();
    result.n_gaussians = (grid_size * grid_size) as usize;
    result.iterations = iterations;
    result.image_dimensions = (target.width, target.height);

    let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
    let mut gaussians = encoder.initialize_gaussians(grid_size);

    let init_rendered = RendererV2::render(&gaussians, target.width, target.height);
    result.initial_psnr = compute_psnr(&target, &init_rendered);
    println!("  Initial PSNR: {:.2} dB", result.initial_psnr);

    let mut adam = AdamOptimizer::new();
    adam.max_iterations = iterations;

    let start = std::time::Instant::now();
    result.final_loss = adam.optimize(&mut gaussians, &target);
    result.elapsed_seconds = start.elapsed().as_secs_f32();

    let final_rendered = RendererV2::render(&gaussians, target.width, target.height);
    result.final_psnr = compute_psnr(&target, &final_rendered);
    result.improvement_db = result.final_psnr - result.initial_psnr;

    println!("  Final PSNR: {:.2} dB", result.final_psnr);
    println!("  Improvement: {:+.2} dB", result.improvement_db);
    println!("  Time: {:.2}s", result.elapsed_seconds);

    // Save results
    match result.save(RESULTS_DIR) {
        Ok(path) => println!("  Saved metrics: {}", path),
        Err(e) => eprintln!("  ERROR saving metrics: {}", e),
    }

    match save_rendered_image(&final_rendered, RESULTS_DIR, "adam_perparam_kodim03", "final") {
        Ok(path) => println!("  Saved image: {}", path),
        Err(e) => eprintln!("  ERROR saving image: {}", e),
    }

    // Test 2: Error-driven Adam (isotropic)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: Error-Driven Adam (Isotropic Edges)");
    println!("═══════════════════════════════════════════════════════════════");

    let mut result2 = TestResult::new("adam_isotropic_kodim03", kodak_path);
    result2.optimizer = "Adam (isotropic edges)".to_string();
    result2.iterations = 0; // Error-driven has variable iterations
    result2.image_dimensions = (target.width, target.height);
    result2.notes = "Quantum-guided: isotropic Gaussians at edges".to_string();

    let encoder2 = EncoderV2::new(target.clone()).expect("Encoder failed");

    let init_grid = 20; // ~400 initial
    let max_n = 800;
    let grid_init = encoder2.initialize_gaussians(init_grid);
    let init_rendered2 = RendererV2::render(&grid_init, target.width, target.height);
    result2.initial_psnr = compute_psnr(&target, &init_rendered2);
    result2.n_gaussians = (init_grid * init_grid) as usize;
    println!("  Initial PSNR: {:.2} dB ({} Gaussians)", result2.initial_psnr, result2.n_gaussians);

    let start2 = std::time::Instant::now();
    let gaussians_iso = encoder2.encode_error_driven_adam_isotropic(
        (init_grid * init_grid) as usize,
        max_n
    );
    result2.elapsed_seconds = start2.elapsed().as_secs_f32();
    result2.n_gaussians = gaussians_iso.len();

    let final_rendered2 = RendererV2::render(&gaussians_iso, target.width, target.height);
    result2.final_psnr = compute_psnr(&target, &final_rendered2);
    result2.improvement_db = result2.final_psnr - result2.initial_psnr;
    result2.final_loss = compute_mse(&target, &final_rendered2);

    println!("  Final PSNR: {:.2} dB ({} Gaussians)", result2.final_psnr, result2.n_gaussians);
    println!("  Improvement: {:+.2} dB", result2.improvement_db);
    println!("  Time: {:.2}s", result2.elapsed_seconds);

    match result2.save(RESULTS_DIR) {
        Ok(path) => println!("  Saved metrics: {}", path),
        Err(e) => eprintln!("  ERROR saving metrics: {}", e),
    }

    match save_rendered_image(&final_rendered2, RESULTS_DIR, "adam_isotropic_kodim03", "final") {
        Ok(path) => println!("  Saved image: {}", path),
        Err(e) => eprintln!("  ERROR saving image: {}", e),
    }

    // Test 3: Standard Error-driven Adam (anisotropic)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 3: Error-Driven Adam (Anisotropic - Standard)");
    println!("═══════════════════════════════════════════════════════════════");

    let mut result3 = TestResult::new("adam_anisotropic_kodim03", kodak_path);
    result3.optimizer = "Adam (anisotropic edges)".to_string();
    result3.image_dimensions = (target.width, target.height);
    result3.notes = "Standard: elongated Gaussians along edges".to_string();

    let encoder3 = EncoderV2::new(target.clone()).expect("Encoder failed");
    result3.initial_psnr = result2.initial_psnr; // Same starting point

    let start3 = std::time::Instant::now();
    let gaussians_aniso = encoder3.encode_error_driven_adam(
        (init_grid * init_grid) as usize,
        max_n
    );
    result3.elapsed_seconds = start3.elapsed().as_secs_f32();
    result3.n_gaussians = gaussians_aniso.len();

    let final_rendered3 = RendererV2::render(&gaussians_aniso, target.width, target.height);
    result3.final_psnr = compute_psnr(&target, &final_rendered3);
    result3.improvement_db = result3.final_psnr - result3.initial_psnr;
    result3.final_loss = compute_mse(&target, &final_rendered3);

    println!("  Final PSNR: {:.2} dB ({} Gaussians)", result3.final_psnr, result3.n_gaussians);
    println!("  Improvement: {:+.2} dB", result3.improvement_db);
    println!("  Time: {:.2}s", result3.elapsed_seconds);

    match result3.save(RESULTS_DIR) {
        Ok(path) => println!("  Saved metrics: {}", path),
        Err(e) => eprintln!("  ERROR saving metrics: {}", e),
    }

    match save_rendered_image(&final_rendered3, RESULTS_DIR, "adam_anisotropic_kodim03", "final") {
        Ok(path) => println!("  Saved image: {}", path),
        Err(e) => eprintln!("  ERROR saving image: {}", e),
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<25} {:>8} {:>8} {:>10}", "Test", "PSNR", "N", "Time");
    println!("{:-<25} {:-<8} {:-<8} {:-<10}", "", "", "", "");
    println!("{:<25} {:>8.2} {:>8} {:>10.2}s",
             "Adam per-param", result.final_psnr, result.n_gaussians, result.elapsed_seconds);
    println!("{:<25} {:>8.2} {:>8} {:>10.2}s",
             "Adam isotropic", result2.final_psnr, result2.n_gaussians, result2.elapsed_seconds);
    println!("{:<25} {:>8.2} {:>8} {:>10.2}s",
             "Adam anisotropic", result3.final_psnr, result3.n_gaussians, result3.elapsed_seconds);
    println!("═══════════════════════════════════════════════════════════════");

    // Comparison
    let iso_vs_aniso = result2.final_psnr - result3.final_psnr;
    println!("\nIsotropic vs Anisotropic: {:+.2} dB", iso_vs_aniso);
    if iso_vs_aniso > 0.1 {
        println!(">>> ISOTROPIC WINS (quantum validated)");
    } else if iso_vs_aniso < -0.1 {
        println!(">>> ANISOTROPIC WINS (classical approach)");
    } else {
        println!(">>> TIED (within 0.1 dB)");
    }

    println!("\nResults saved to: {}", RESULTS_DIR);
}

fn compute_psnr(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mse = compute_mse(original, rendered);
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}

fn compute_mse(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (original.width * original.height * 3) as f32;

    for (p1, p2) in original.data.iter().zip(rendered.data.iter()) {
        mse += (p1.r - p2.r).powi(2);
        mse += (p1.g - p2.g).powi(2);
        mse += (p1.b - p2.b).powi(2);
    }

    mse / count
}
