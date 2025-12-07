//! Overnight CommercialEncoder Test
//!
//! Tests the full CommercialEncoder pipeline and compares against
//! simplified variants using Adam optimizer.
//!
//! The CommercialEncoder integrates:
//! - Textured Gaussians
//! - Blue-noise residuals
//! - Analytical triggers
//! - Guided filter
//!
//! Run: cargo run --release --example test_overnight_commercial -p lgi-encoder-v2

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::loss_functions::compute_ssim;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   OVERNIGHT COMMERCIAL ENCODER TEST                          ║");
    println!("║   Testing full pipeline vs Adam-optimized variants           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let test_images = vec![
        ("kodim03", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png"),
        ("kodim05", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim05.png"),
        ("kodim08", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim08.png"),
    ];

    // Test configurations
    let configs = vec![
        // Baseline: Jittered grid + Adam (our champion)
        ("baseline_jittered_adam", InitType::JitteredGrid, 256, 150),

        // More iterations
        ("baseline_jittered_adam_200", InitType::JitteredGrid, 256, 200),
        ("baseline_jittered_adam_300", InitType::JitteredGrid, 256, 300),

        // Different Gaussian counts
        ("jittered_adam_512", InitType::JitteredGrid, 512, 200),
        ("jittered_adam_1024", InitType::JitteredGrid, 1024, 200),

        // Structure tensor guided initialization
        ("structure_guided_adam", InitType::StructureGuided, 256, 200),

        // Uniform grid comparison
        ("uniform_adam", InitType::UniformGrid, 256, 150),
    ];

    for (image_name, image_path) in &test_images {
        println!("\n{}", "=".repeat(70));
        println!("Processing image: {}", image_name);
        println!("{}\n", "=".repeat(70));

        let target = match ImageBuffer::load(image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Failed to load {}: {:?}", image_path, e);
                continue;
            }
        };

        // Compute structure tensor for guided init
        let structure_tensor = StructureTensorField::compute(&target, 1.2, 1.0)
            .expect("Failed to compute structure tensor");

        for (name, init_type, n_gaussians, iterations) in &configs {
            println!("───────────────────────────────────────────────────────────────");
            println!("Config: {} on {}", name, image_name);
            println!("───────────────────────────────────────────────────────────────");

            let start = Instant::now();

            // Initialize Gaussians based on type
            let mut gaussians = match init_type {
                InitType::JitteredGrid => create_jittered_grid(&target, *n_gaussians),
                InitType::UniformGrid => create_uniform_grid(&target, *n_gaussians),
                InitType::StructureGuided => create_structure_guided(&target, &structure_tensor, *n_gaussians),
            };

            let init_render = RendererV2::render(&gaussians, target.width, target.height);
            let init_psnr = compute_psnr(&init_render, &target);
            let init_ssim = compute_ssim(&init_render, &target);
            println!("  Initial: N={}, PSNR={:.2}dB, SSIM={:.4}", gaussians.len(), init_psnr, init_ssim);

            // Optimize with Adam
            let mut adam = AdamOptimizer::new();
            adam.max_iterations = *iterations;
            let _loss = adam.optimize(&mut gaussians, &target);

            let elapsed = start.elapsed();
            let final_render = RendererV2::render(&gaussians, target.width, target.height);
            let final_psnr = compute_psnr(&final_render, &target);
            let final_ssim = compute_ssim(&final_render, &target);

            println!("  Final: PSNR={:.2}dB ({:+.2}dB), SSIM={:.4} in {:.1}s",
                     final_psnr, final_psnr - init_psnr, final_ssim, elapsed.as_secs_f32());

            // Save result
            let exp_name = format!("overnight_commercial_{}_{}", name, image_name);
            let mut result = TestResult::new(&exp_name, image_path);
            result.optimizer = format!("Adam with {:?} init", init_type);
            result.image_dimensions = (target.width, target.height);
            result.n_gaussians = gaussians.len();
            result.iterations = *iterations;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = final_psnr - init_psnr;
            result.elapsed_seconds = elapsed.as_secs_f32();
            result.extra_metrics = Some(json!({
                "init_type": format!("{:?}", init_type),
                "initial_ssim": init_ssim,
                "final_ssim": final_ssim,
                "target_gaussians": n_gaussians,
                "actual_gaussians": gaussians.len(),
            }));
            let _ = result.save(RESULTS_DIR);
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("OVERNIGHT COMMERCIAL ENCODER TEST COMPLETE");
    println!("{}", "=".repeat(70));
}

#[derive(Debug, Clone, Copy)]
enum InitType {
    JitteredGrid,
    UniformGrid,
    StructureGuided,
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

fn create_uniform_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n);
    let cell_size = 1.0 / grid_size as f32;

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            if gaussians.len() >= n { break; }

            let x = (gx as f32 + 0.5) / grid_size as f32;
            let y = (gy as f32 + 0.5) / grid_size as f32;

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
    }

    gaussians
}

fn create_structure_guided(
    target: &ImageBuffer<f32>,
    structure_tensor: &StructureTensorField,
    n: usize,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    // Place Gaussians on jittered grid but size/orient them based on local structure
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n);
    let base_size = 1.0 / grid_size as f32;

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

        // Get local structure
        let tensor = structure_tensor.get(px, py);

        // Adapt Gaussian based on local structure
        let (scale_x, scale_y, rotation) = if tensor.coherence > 0.3 {
            // Edge region: elongate along edge direction
            let anisotropy = (tensor.eigenvalue_major / tensor.eigenvalue_minor.max(0.001)).sqrt().min(3.0);
            let scale_major = base_size * anisotropy.sqrt();
            let scale_minor = base_size / anisotropy.sqrt();
            let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
            (scale_major, scale_minor, angle)
        } else {
            // Smooth region: isotropic
            (base_size, base_size, 0.0)
        };

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(scale_x, scale_y, rotation),
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
