//! Test V2 with Adam's exact learning rates
//!
//! Key finding: Adam uses position LR=0.0002, V2 uses 0.1 (500× larger!)
//! This test configures V2 to use Adam's LR configuration.
//!
//! Run: cargo run --release --example test_v2_with_adam_lrs -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::optimizer_v2::OptimizerV2;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::{TestResult, save_rendered_image};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   V2 WITH ADAM'S EXACT LEARNING RATES                        ║");
    println!("║   Testing if Adam's LR config closes the gap                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load");
    let (width, height) = (target.width, target.height);

    let grid_size = 16;
    let n_gaussians = grid_size * grid_size;
    let iterations = 100;

    println!("Image: {}x{}, Gaussians: {}, Iterations: {}\n", width, height, n_gaussians, iterations);

    // Test different LR configurations
    let lr_configs = vec![
        ("V2 Default", 0.1, 0.6, 0.1, 0.02),
        ("Adam's LRs", 0.0002, 0.02, 0.005, 0.0),  // position, color, scale, rotation
        ("Adam 10×", 0.002, 0.2, 0.05, 0.0),       // 10× Adam's rates
        ("Adam 100×", 0.02, 2.0, 0.5, 0.0),        // 100× Adam's rates (may be too high)
    ];

    for (name, lr_pos, lr_color, lr_scale, lr_rot) in &lr_configs {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Testing: {} (pos={}, color={}, scale={}, rot={})",
                 name, lr_pos, lr_color, lr_scale, lr_rot);
        println!("═══════════════════════════════════════════════════════════════");

        // Create fresh Gaussians
        let mut gaussians = create_gaussians(&target, grid_size);

        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = iterations;
        optimizer.learning_rate_position = *lr_pos;
        optimizer.learning_rate_color = *lr_color;
        optimizer.learning_rate_scale = *lr_scale;
        optimizer.learning_rate_rotation = *lr_rot;

        let start = Instant::now();
        let _ = optimizer.optimize(&mut gaussians, &target);
        let elapsed = start.elapsed();

        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);

        println!("  Initial PSNR: {:.2} dB", init_psnr);
        println!("  Final PSNR:   {:.2} dB", final_psnr);
        println!("  Improvement:  {:+.2} dB", final_psnr - init_psnr);
        println!("  Time:         {:.1}s\n", elapsed.as_secs_f32());

        // Save result
        let exp_name = format!("v2_adam_lr_{}", name.replace(" ", "_").replace("×", "x"));
        let mut result = TestResult::new(&exp_name, image_path);
        result.optimizer = format!("V2 ({})", name);
        result.image_dimensions = (width, height);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({
            "lr_position": lr_pos,
            "lr_color": lr_color,
            "lr_scale": lr_scale,
            "lr_rotation": lr_rot,
        }));
        let _ = result.save(RESULTS_DIR);
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("SUMMARY: If Adam's LRs help V2, the gap should close");
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
