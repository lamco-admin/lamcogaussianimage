//! Test OptimizerV3 with perceptual loss functions
//!
//! OptimizerV3 supports:
//! - MS-SSIM loss (structural similarity)
//! - Edge-weighted loss (higher weight in edge regions)
//! - Standard L2 loss
//!
//! This test compares these options against Adam optimizer baseline.
//!
//! Run: cargo run --release --example test_optimizer_v3_perceptual -p lgi-encoder-v2

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::optimizer_v3_perceptual::OptimizerV3;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   OPTIMIZER V3 PERCEPTUAL LOSS TEST                          ║");
    println!("║   Comparing MS-SSIM, Edge-weighted, and L2 losses            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);
    println!("Computing structure tensor...\n");

    let tensor_field = StructureTensorField::compute(&target, 1.0, 2.0)
        .expect("Failed to compute structure tensor");

    let n_gaussians = 256;
    let iterations = 100;

    let mut results = Vec::new();

    // Test 1: Adam optimizer baseline
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Adam Optimizer Baseline");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let start = Instant::now();
        let mut gaussians = create_jittered_grid(&target, n_gaussians);

        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);
        println!("  Initial PSNR: {:.2}dB", init_psnr);

        let mut adam = AdamOptimizer::new();
        adam.max_iterations = iterations;

        let mut last_psnr = init_psnr;
        for iter in 0..iterations {
            if iter % 20 == 0 {
                let rendered = RendererV2::render(&gaussians, width, height);
                let psnr = compute_psnr(&rendered, &target);
                let direction = if psnr > last_psnr { "↑" } else { "↓" };
                println!("    [{:3}/{:3}] PSNR={:.2}dB {}", iter, iterations, psnr, direction);
                last_psnr = psnr;
            }
            adam.max_iterations = 1;
            adam.optimize(&mut gaussians, &target);
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        println!("  Final PSNR: {:.2}dB ({:+.2}dB)", final_psnr, final_psnr - init_psnr);

        results.push(("Adam Baseline".to_string(), init_psnr, final_psnr, elapsed.as_secs_f32()));

        let mut result = TestResult::new("v3_perceptual_adam_baseline", image_path);
        result.optimizer = "Adam (L2 loss)".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({"test_type": "adam_baseline", "loss_type": "L2"}));
        let _ = result.save(RESULTS_DIR);
    }

    // Test 2: OptimizerV3 with L2 loss
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: OptimizerV3 with L2 Loss");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let start = Instant::now();
        let mut gaussians = create_jittered_grid(&target, n_gaussians);

        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);
        println!("  Initial PSNR: {:.2}dB", init_psnr);

        let mut optimizer = OptimizerV3 {
            use_ms_ssim: false,
            use_edge_weighted: false,
            max_iterations: iterations,
            ..Default::default()
        };

        optimizer.optimize(&mut gaussians, &target, &tensor_field);

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        println!("  Final PSNR: {:.2}dB ({:+.2}dB)", final_psnr, final_psnr - init_psnr);

        results.push(("V3 L2 Loss".to_string(), init_psnr, final_psnr, elapsed.as_secs_f32()));

        let mut result = TestResult::new("v3_perceptual_l2", image_path);
        result.optimizer = "OptimizerV3 (L2 loss)".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({"test_type": "v3_l2", "loss_type": "L2"}));
        let _ = result.save(RESULTS_DIR);
    }

    // Test 3: OptimizerV3 with MS-SSIM loss
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 3: OptimizerV3 with MS-SSIM Loss");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let start = Instant::now();
        let mut gaussians = create_jittered_grid(&target, n_gaussians);

        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);
        println!("  Initial PSNR: {:.2}dB", init_psnr);

        let mut optimizer = OptimizerV3 {
            use_ms_ssim: true,
            use_edge_weighted: false,
            max_iterations: iterations,
            ..Default::default()
        };

        optimizer.optimize(&mut gaussians, &target, &tensor_field);

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        println!("  Final PSNR: {:.2}dB ({:+.2}dB)", final_psnr, final_psnr - init_psnr);

        results.push(("V3 MS-SSIM".to_string(), init_psnr, final_psnr, elapsed.as_secs_f32()));

        let mut result = TestResult::new("v3_perceptual_msssim", image_path);
        result.optimizer = "OptimizerV3 (MS-SSIM loss)".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({"test_type": "v3_msssim", "loss_type": "MS-SSIM"}));
        let _ = result.save(RESULTS_DIR);
    }

    // Test 4: OptimizerV3 with Edge-weighted loss
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 4: OptimizerV3 with Edge-weighted Loss");
    println!("═══════════════════════════════════════════════════════════════");
    {
        let start = Instant::now();
        let mut gaussians = create_jittered_grid(&target, n_gaussians);

        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);
        println!("  Initial PSNR: {:.2}dB", init_psnr);

        let mut optimizer = OptimizerV3 {
            use_ms_ssim: false,
            use_edge_weighted: true,
            max_iterations: iterations,
            ..Default::default()
        };

        optimizer.optimize(&mut gaussians, &target, &tensor_field);

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);
        println!("  Final PSNR: {:.2}dB ({:+.2}dB)", final_psnr, final_psnr - init_psnr);

        results.push(("V3 Edge-weighted".to_string(), init_psnr, final_psnr, elapsed.as_secs_f32()));

        let mut result = TestResult::new("v3_perceptual_edge_weighted", image_path);
        result.optimizer = "OptimizerV3 (Edge-weighted loss)".to_string();
        result.image_dimensions = (width, height);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({"test_type": "v3_edge_weighted", "loss_type": "Edge-weighted"}));
        let _ = result.save(RESULTS_DIR);
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<25} {:>10} {:>10} {:>10} {:>8}",
             "Configuration", "Init PSNR", "Final PSNR", "Improve", "Time");
    println!("{:-<25} {:-<10} {:-<10} {:-<10} {:-<8}", "", "", "", "", "");

    for (name, init, final_psnr, time) in &results {
        println!("{:<25} {:>9.2}dB {:>9.2}dB {:>9.2}dB {:>7.1}s",
                 name, init, final_psnr, final_psnr - init, time);
    }

    println!("\nNOTE: Different loss functions optimize for different perceptual");
    println!("qualities. MS-SSIM may produce visually sharper edges even if PSNR");
    println!("is slightly lower. Visual inspection is recommended.");
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
        let color = target.get_pixel(px, py).unwrap();

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(cell_size, cell_size, 0.0),
            color,
            1.0,
        ));
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
