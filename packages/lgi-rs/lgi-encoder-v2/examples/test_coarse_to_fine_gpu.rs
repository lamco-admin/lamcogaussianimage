//! Coarse-to-Fine Hierarchical Encoding Test (GPU)
//!
//! HYPOTHESIS: Layered approach - large Gaussians first, then smaller refinement -
//! captures detail better than single-pass uniform allocation.
//!
//! APPROACH:
//! - Level 0: 64 large Gaussians → capture broad color regions
//! - Level 1: 128 medium Gaussians → major shapes
//! - Level 2: 320 small Gaussians → details
//! - Each level encodes RESIDUAL error from previous levels
//!
//! Uses GPU Adam for speed.

use image::GenericImageView;
use lgi_core::{ImageBuffer, StructureTensorField, error_metrics};
use lgi_encoder_v2::test_results::TestResult;
use lgi_gpu::{GpuAdamOptimizer, GpuLearningRates, GpuManager};
use lgi_math::{
    color::Color4,
    gaussian::Gaussian2D,
    parameterization::Euler,
    vec::Vector2,
};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  COARSE-TO-FINE HIERARCHICAL ENCODING TEST (GPU)");
    println!("  Focus: Does layered multi-scale approach improve quality?");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Initialize GPU
    let gpu_manager = match GpuManager::new() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to initialize GPU: {:?}", e);
            return;
        }
    };
    println!("GPU: {}", gpu_manager.capabilities().device_name);

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let img = image::open(image_path).expect("Failed to load image");
    let (width, height) = img.dimensions();

    let mut target = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            target.set_pixel(x, y, Color4::new(
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
                1.0,
            ));
        }
    }

    println!("Image: {}x{}\n", width, height);

    // Test configurations - same total Gaussian count for fair comparison
    let total_gaussians = 512;
    let iterations_per_level = 100;

    // Run flat baseline
    println!("═══ BASELINE: Flat 512 Gaussians ═══");
    let baseline = run_flat_baseline(&gpu_manager, &target, total_gaussians, 200, image_path);

    // Run coarse-to-fine
    println!("\n═══ COARSE-TO-FINE: 64 + 128 + 320 Gaussians ═══");
    let ctf = run_coarse_to_fine(&gpu_manager, &target, iterations_per_level, image_path);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  RESULTS SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                          Baseline    Coarse-Fine  Difference");
    println!("───────────────────────────────────────────────────────────────────");
    println!("Final PSNR:              {:>8.2} dB  {:>8.2} dB  {:>+8.2} dB",
             baseline.0, ctf.0, ctf.0 - baseline.0);
    println!("Time:                    {:>8.1} s   {:>8.1} s", baseline.1, ctf.1);
    println!("═══════════════════════════════════════════════════════════════════\n");

    if ctf.0 > baseline.0 {
        println!("RESULT: Coarse-to-fine IMPROVED by {:.2} dB", ctf.0 - baseline.0);
    } else {
        println!("RESULT: Coarse-to-fine did NOT improve ({:.2} dB worse)", baseline.0 - ctf.0);
    }
}

fn run_flat_baseline(
    gpu: &GpuManager,
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    iterations: usize,
    image_path: &str,
) -> (f32, f32) {
    let start = Instant::now();

    let mut result = TestResult::new("coarse_to_fine_baseline", image_path);
    result.optimizer = "GPU Adam - Flat allocation".to_string();
    result.n_gaussians = n_gaussians;
    result.iterations = iterations;
    result.image_dimensions = (target.width, target.height);
    result.set_param("approach", "flat");
    result.set_param("levels", 1);

    // Initialize on jittered grid
    let mut gaussians = init_jittered_grid(target.width, target.height, n_gaussians, 0.02);
    sample_colors(&mut gaussians, target);

    let init_psnr = compute_psnr_cpu(&gaussians, target);
    result.initial_psnr = init_psnr;
    println!("  Initial PSNR: {:.2} dB", init_psnr);

    // GPU optimize
    let mut optimizer = GpuAdamOptimizer::new(gpu.device()).expect("Failed to create optimizer");
    optimizer.max_iterations = iterations;
    optimizer.learning_rates = GpuLearningRates {
        position: 0.0003,
        position_final: 0.00003,
        color: 0.02,
        scale: 0.003,
        opacity: 0.05,
        rotation: 0.01,
    };

    let _loss = optimizer.optimize(gpu.device(), gpu.queue(), &mut gaussians, target)
        .expect("Optimization failed");

    let final_psnr = compute_psnr_cpu(&gaussians, target);
    let elapsed = start.elapsed().as_secs_f32();

    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;
    result.complete();

    let rendered = render_cpu(&gaussians, target.width, target.height);
    let _ = result.save_with_image(&rendered, RESULTS_DIR);

    println!("  Final PSNR: {:.2} dB (+{:.2} dB)", final_psnr, final_psnr - init_psnr);

    (final_psnr, elapsed)
}

fn run_coarse_to_fine(
    gpu: &GpuManager,
    target: &ImageBuffer<f32>,
    iterations_per_level: usize,
    image_path: &str,
) -> (f32, f32) {
    let start = Instant::now();

    let mut result = TestResult::new("coarse_to_fine_hierarchical", image_path);
    result.optimizer = "GPU Adam - Coarse-to-fine (64+128+320)".to_string();
    result.n_gaussians = 512;  // Total
    result.iterations = iterations_per_level * 3;
    result.image_dimensions = (target.width, target.height);
    result.set_param("approach", "coarse_to_fine");
    result.set_param("levels", 3);
    result.set_param("level_0_count", 64);
    result.set_param("level_1_count", 128);
    result.set_param("level_2_count", 320);
    result.set_param("iterations_per_level", iterations_per_level);

    let mut all_gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = Vec::new();

    // LEVEL 0: Coarse - 64 large Gaussians
    println!("  Level 0: 64 large Gaussians (scale=0.08)");
    let mut level0 = init_jittered_grid(target.width, target.height, 64, 0.08);
    sample_colors(&mut level0, target);

    let init_psnr = compute_psnr_cpu(&level0, target);
    result.initial_psnr = init_psnr;
    println!("    Initial: {:.2} dB", init_psnr);

    let mut optimizer = GpuAdamOptimizer::new(gpu.device()).expect("Failed to create optimizer");
    optimizer.max_iterations = iterations_per_level;
    optimizer.learning_rates = GpuLearningRates {
        position: 0.001,  // Larger LR for coarse level
        position_final: 0.0001,
        color: 0.03,
        scale: 0.01,
        opacity: 0.05,
        rotation: 0.02,
    };

    let _ = optimizer.optimize(gpu.device(), gpu.queue(), &mut level0, target);
    let psnr0 = compute_psnr_cpu(&level0, target);
    println!("    After opt: {:.2} dB", psnr0);
    all_gaussians.extend(level0);

    // LEVEL 1: Medium - 128 Gaussians targeting high-error regions
    println!("  Level 1: 128 medium Gaussians (scale=0.03)");
    let residual1 = compute_residual(&all_gaussians, target);
    let mut level1 = init_error_weighted(target.width, target.height, 128, 0.03, &residual1);
    sample_colors(&mut level1, target);  // Sample from original, not residual

    // Combine and optimize ALL Gaussians together
    all_gaussians.extend(level1);
    let psnr1_init = compute_psnr_cpu(&all_gaussians, target);
    println!("    Combined init: {:.2} dB", psnr1_init);

    optimizer.max_iterations = iterations_per_level;
    optimizer.learning_rates = GpuLearningRates {
        position: 0.0003,
        position_final: 0.00003,
        color: 0.02,
        scale: 0.005,
        opacity: 0.05,
        rotation: 0.01,
    };

    let _ = optimizer.optimize(gpu.device(), gpu.queue(), &mut all_gaussians, target);
    let psnr1 = compute_psnr_cpu(&all_gaussians, target);
    println!("    After opt: {:.2} dB", psnr1);

    // LEVEL 2: Fine - 320 small Gaussians targeting remaining error
    println!("  Level 2: 320 small Gaussians (scale=0.015)");
    let residual2 = compute_residual(&all_gaussians, target);
    let mut level2 = init_error_weighted(target.width, target.height, 320, 0.015, &residual2);
    sample_colors(&mut level2, target);

    all_gaussians.extend(level2);
    let psnr2_init = compute_psnr_cpu(&all_gaussians, target);
    println!("    Combined init: {:.2} dB", psnr2_init);

    optimizer.max_iterations = iterations_per_level;
    optimizer.learning_rates = GpuLearningRates {
        position: 0.0002,
        position_final: 0.00002,
        color: 0.015,
        scale: 0.003,
        opacity: 0.05,
        rotation: 0.005,
    };

    let _ = optimizer.optimize(gpu.device(), gpu.queue(), &mut all_gaussians, target);
    let final_psnr = compute_psnr_cpu(&all_gaussians, target);
    let elapsed = start.elapsed().as_secs_f32();

    println!("    Final: {:.2} dB", final_psnr);

    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;
    result.complete();

    let rendered = render_cpu(&all_gaussians, target.width, target.height);
    let _ = result.save_with_image(&rendered, RESULTS_DIR);

    (final_psnr, elapsed)
}

fn init_jittered_grid(width: u32, height: u32, n: usize, scale: f32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let aspect = width as f32 / height as f32;
    let cols = ((n as f32 * aspect).sqrt()).round() as usize;
    let rows = (n as f32 / cols as f32).ceil() as usize;

    let mut gaussians = Vec::with_capacity(n);
    let jitter = 0.3;  // 30% cell size jitter

    for i in 0..rows {
        for j in 0..cols {
            if gaussians.len() >= n { break; }

            let base_x = (j as f32 + 0.5) / cols as f32;
            let base_y = (i as f32 + 0.5) / rows as f32;

            let jx = (rng.gen::<f32>() - 0.5) * jitter / cols as f32;
            let jy = (rng.gen::<f32>() - 0.5) * jitter / rows as f32;

            gaussians.push(Gaussian2D {
                position: Vector2::new((base_x + jx).clamp(0.01, 0.99), (base_y + jy).clamp(0.01, 0.99)),
                shape: Euler { scale_x: scale, scale_y: scale, rotation: 0.0 },
                color: Color4::new(0.5, 0.5, 0.5, 1.0),
                opacity: 1.0,
                weight: None,
            });
        }
    }
    gaussians
}

fn init_error_weighted(
    width: u32,
    height: u32,
    n: usize,
    scale: f32,
    error_map: &ImageBuffer<f32>,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Compute error importance for sampling
    let mut importance: Vec<(f32, f32, f32)> = Vec::new();  // (x, y, error)
    let mut total_error = 0.0f32;

    for y in 0..height {
        for x in 0..width {
            if let Some(p) = error_map.get_pixel(x, y) {
                let error = (p.r.abs() + p.g.abs() + p.b.abs()) / 3.0;
                importance.push((x as f32 / width as f32, y as f32 / height as f32, error));
                total_error += error;
            }
        }
    }

    // Importance sampling
    let mut gaussians = Vec::with_capacity(n);

    for _ in 0..n {
        let target = rng.gen::<f32>() * total_error;
        let mut cumulative = 0.0;
        let mut chosen = (0.5, 0.5);

        for &(x, y, e) in &importance {
            cumulative += e;
            if cumulative >= target {
                chosen = (x, y);
                break;
            }
        }

        // Add jitter
        let jx = (rng.gen::<f32>() - 0.5) * 0.05;
        let jy = (rng.gen::<f32>() - 0.5) * 0.05;

        gaussians.push(Gaussian2D {
            position: Vector2::new((chosen.0 + jx).clamp(0.01, 0.99), (chosen.1 + jy).clamp(0.01, 0.99)),
            shape: Euler { scale_x: scale, scale_y: scale, rotation: 0.0 },
            color: Color4::new(0.5, 0.5, 0.5, 1.0),
            opacity: 1.0,
            weight: None,
        });
    }

    gaussians
}

fn sample_colors(gaussians: &mut [Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) {
    for g in gaussians.iter_mut() {
        let px = (g.position.x * target.width as f32) as u32;
        let py = (g.position.y * target.height as f32) as u32;
        if let Some(c) = target.get_pixel(px.min(target.width - 1), py.min(target.height - 1)) {
            g.color = c;
        }
    }
}

fn compute_residual(gaussians: &[Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) -> ImageBuffer<f32> {
    let rendered = render_cpu(gaussians, target.width, target.height);
    let mut residual = ImageBuffer::new(target.width, target.height);

    for y in 0..target.height {
        for x in 0..target.width {
            if let (Some(r), Some(t)) = (rendered.get_pixel(x, y), target.get_pixel(x, y)) {
                residual.set_pixel(x, y, Color4::new(
                    (t.r - r.r).abs(),
                    (t.g - r.g).abs(),
                    (t.b - r.b).abs(),
                    1.0,
                ));
            }
        }
    }
    residual
}

fn compute_psnr_cpu(gaussians: &[Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) -> f32 {
    let rendered = render_cpu(gaussians, target.width, target.height);
    error_metrics::psnr(&rendered, target)
}

fn render_cpu(gaussians: &[Gaussian2D<f32, Euler<f32>>], width: u32, height: u32) -> ImageBuffer<f32> {
    let mut buffer = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let px = x as f32 / width as f32;
            let py = y as f32 / height as f32;

            let mut weight_sum = 0.0f32;
            let mut color_sum = Color4::new(0.0, 0.0, 0.0, 0.0);

            for g in gaussians {
                let dx = px - g.position.x;
                let dy = py - g.position.y;

                let cos_t = g.shape.rotation.cos();
                let sin_t = g.shape.rotation.sin();
                let dx_rot = dx * cos_t + dy * sin_t;
                let dy_rot = -dx * sin_t + dy * cos_t;

                let sx = g.shape.scale_x.max(0.001);
                let sy = g.shape.scale_y.max(0.001);
                let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);

                if dist_sq > 12.25 { continue; }

                let gaussian_val = (-0.5 * dist_sq).exp();
                let weight = g.opacity * gaussian_val;

                weight_sum += weight;
                color_sum.r += weight * g.color.r;
                color_sum.g += weight * g.color.g;
                color_sum.b += weight * g.color.b;
            }

            let color = if weight_sum > 1e-10 {
                Color4::new(
                    color_sum.r / weight_sum,
                    color_sum.g / weight_sum,
                    color_sum.b / weight_sum,
                    1.0,
                )
            } else {
                Color4::new(0.0, 0.0, 0.0, 1.0)
            };

            buffer.set_pixel(x, y, color);
        }
    }
    buffer
}
