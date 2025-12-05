//! Test optimizer on multiple Kodak images
//!
//! Runs Adam with per-param LRs on several Kodak images to see
//! how performance varies by image type.

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{
    EncoderV2,
    renderer_v2::RendererV2,
    adam_optimizer::AdamOptimizer,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Multi-Kodak Adam Optimizer Test                             ║");
    println!("║  Testing per-param LRs across different image types          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let kodak_dir = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset";

    // Test on a variety of Kodak images with different characteristics
    let test_images = vec![
        ("kodim01.png", "Houses - geometric, sharp edges"),
        ("kodim03.png", "Hats - natural, gradients"),
        ("kodim05.png", "Flowers - organic, colorful"),
        ("kodim08.png", "Church - architectural, texture"),
        ("kodim13.png", "Peppers - smooth regions, colors"),
    ];

    let grid_size = 24;  // 576 Gaussians
    let max_iterations = 150;

    println!("Grid size: {}x{} = {} Gaussians", grid_size, grid_size, grid_size * grid_size);
    println!("Max iterations: {}\n", max_iterations);

    let mut results = Vec::new();

    for (filename, description) in &test_images {
        let path = format!("{}/{}", kodak_dir, filename);
        println!("═══════════════════════════════════════════════════════════════");
        println!("{}: {}", filename, description);
        println!("═══════════════════════════════════════════════════════════════");

        let target = match ImageBuffer::load(&path) {
            Ok(img) => {
                println!("  Loaded: {}x{}", img.width, img.height);
                img
            }
            Err(e) => {
                eprintln!("  ERROR: Could not load: {}", e);
                continue;
            }
        };

        let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
        let mut gaussians = encoder.initialize_gaussians(grid_size);

        let init_rendered = RendererV2::render(&gaussians, target.width, target.height);
        let init_psnr = compute_psnr(&target, &init_rendered);

        let mut adam = AdamOptimizer::new();
        adam.max_iterations = max_iterations;

        let start = std::time::Instant::now();
        let final_loss = adam.optimize(&mut gaussians, &target);
        let elapsed = start.elapsed();

        let final_rendered = RendererV2::render(&gaussians, target.width, target.height);
        let final_psnr = compute_psnr(&target, &final_rendered);

        let improvement = final_psnr - init_psnr;

        println!("  Initial PSNR: {:.2} dB", init_psnr);
        println!("  Final PSNR:   {:.2} dB (loss: {:.6})", final_psnr, final_loss);
        println!("  Improvement:  {:+.2} dB", improvement);
        println!("  Time:         {:.2}s\n", elapsed.as_secs_f32());

        results.push((filename.to_string(), init_psnr, final_psnr, improvement, elapsed.as_secs_f32()));
    }

    // Summary table
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<12} {:>10} {:>10} {:>10} {:>8}", "Image", "Init", "Final", "Improve", "Time");
    println!("{:-<12} {:-<10} {:-<10} {:-<10} {:-<8}", "", "", "", "", "");

    let mut total_improvement = 0.0;
    for (name, init, final_psnr, improvement, time) in &results {
        println!("{:<12} {:>10.2} {:>10.2} {:>+10.2} {:>8.2}s",
                 name, init, final_psnr, improvement, time);
        total_improvement += improvement;
    }

    let avg_improvement = total_improvement / results.len() as f32;
    println!("{:-<12} {:-<10} {:-<10} {:-<10} {:-<8}", "", "", "", "", "");
    println!("{:<12} {:>10} {:>10} {:>+10.2} {:>8}", "AVERAGE", "", "", avg_improvement, "");
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
