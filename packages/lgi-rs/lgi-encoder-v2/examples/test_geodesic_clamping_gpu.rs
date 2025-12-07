//! Geodesic Sigma Clamping Test (GPU)
//!
//! HYPOTHESIS: Hard constraints preventing Gaussians from crossing edges
//! produce sharper boundaries than soft loss penalties.
//!
//! APPROACH:
//! - Compute structure tensor → detect edges
//! - Compute geodesic EDT → distance to nearest edge
//! - Clamp Gaussian sigma based on geodesic distance
//! - Formula: σ_max = 0.5 + 0.3 × d (can't extend past nearby edges)
//!
//! Uses GPU Adam for speed, CPU for structure analysis.

use image::GenericImageView;
use lgi_core::{ImageBuffer, StructureTensorField, GeodesicEDT, error_metrics};
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
    println!("  GEODESIC SIGMA CLAMPING TEST (GPU)");
    println!("  Focus: Do hard edge constraints prevent color bleeding?");
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

    println!("Image: {}x{}", width, height);

    // Compute structure tensor and geodesic EDT
    println!("Computing structure tensor...");
    let tensor_field = StructureTensorField::compute(&target, 1.2, 2.5)
        .expect("Failed to compute structure tensor");

    println!("Computing geodesic EDT...");
    let gedt = GeodesicEDT::compute(&target, &tensor_field, 0.5, 50.0)
        .expect("Failed to compute geodesic EDT");

    println!("Analysis complete.\n");

    let n_gaussians = 512;
    let iterations = 200;

    // Run baseline (no clamping)
    println!("═══ BASELINE: No sigma clamping ═══");
    let baseline = run_baseline(&gpu_manager, &target, n_gaussians, iterations, image_path);

    // Run with geodesic clamping
    println!("\n═══ GEODESIC CLAMPING: σ ≤ 0.5 + 0.3×d ═══");
    let clamped = run_with_clamping(&gpu_manager, &target, &gedt, n_gaussians, iterations, image_path);

    // Run with aggressive clamping
    println!("\n═══ AGGRESSIVE CLAMPING: σ ≤ 0.3 + 0.2×d ═══");
    let aggressive = run_with_aggressive_clamping(&gpu_manager, &target, &gedt, n_gaussians, iterations, image_path);

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  RESULTS SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                          Baseline    Geodesic    Aggressive");
    println!("───────────────────────────────────────────────────────────────────");
    println!("Final PSNR:              {:>8.2} dB  {:>8.2} dB  {:>8.2} dB",
             baseline.0, clamped.0, aggressive.0);
    println!("vs Baseline:                  -       {:>+8.2} dB  {:>+8.2} dB",
             clamped.0 - baseline.0, aggressive.0 - baseline.0);
    println!("Time:                    {:>8.1} s   {:>8.1} s   {:>8.1} s",
             baseline.1, clamped.1, aggressive.1);
    println!("═══════════════════════════════════════════════════════════════════\n");

    let best = if clamped.0 > baseline.0 && clamped.0 >= aggressive.0 {
        ("Geodesic clamping", clamped.0 - baseline.0)
    } else if aggressive.0 > baseline.0 {
        ("Aggressive clamping", aggressive.0 - baseline.0)
    } else {
        ("Baseline (no clamping)", 0.0)
    };

    println!("RESULT: Best approach is {} ({:+.2} dB vs baseline)", best.0, best.1);
}

fn run_baseline(
    gpu: &GpuManager,
    target: &ImageBuffer<f32>,
    n_gaussians: usize,
    iterations: usize,
    image_path: &str,
) -> (f32, f32) {
    let start = Instant::now();

    let mut result = TestResult::new("geodesic_clamping_baseline", image_path);
    result.optimizer = "GPU Adam - No sigma clamping".to_string();
    result.n_gaussians = n_gaussians;
    result.iterations = iterations;
    result.image_dimensions = (target.width, target.height);
    result.set_param("clamping", "none");

    let mut gaussians = init_jittered_grid(target.width, target.height, n_gaussians, 0.02);
    sample_colors(&mut gaussians, target);

    let init_psnr = compute_psnr_cpu(&gaussians, target);
    result.initial_psnr = init_psnr;
    println!("  Initial PSNR: {:.2} dB", init_psnr);

    let mut optimizer = GpuAdamOptimizer::new(gpu.device()).expect("Failed to create optimizer");
    optimizer.max_iterations = iterations;
    optimizer.learning_rates = GpuLearningRates::default();

    let _ = optimizer.optimize(gpu.device(), gpu.queue(), &mut gaussians, target);

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

fn run_with_clamping(
    gpu: &GpuManager,
    target: &ImageBuffer<f32>,
    gedt: &GeodesicEDT,
    n_gaussians: usize,
    iterations: usize,
    image_path: &str,
) -> (f32, f32) {
    let start = Instant::now();

    let mut result = TestResult::new("geodesic_clamping_standard", image_path);
    result.optimizer = "GPU Adam + Geodesic clamping (σ ≤ 0.5 + 0.3×d)".to_string();
    result.n_gaussians = n_gaussians;
    result.iterations = iterations;
    result.image_dimensions = (target.width, target.height);
    result.set_param("clamping", "geodesic_standard");
    result.set_param("clamp_offset", 0.5);
    result.set_param("clamp_scale", 0.3);

    let mut gaussians = init_jittered_grid(target.width, target.height, n_gaussians, 0.02);
    sample_colors(&mut gaussians, target);

    // Apply geodesic clamping to initial scales
    apply_geodesic_clamping(&mut gaussians, gedt, target.width, target.height, 0.5, 0.3);

    let init_psnr = compute_psnr_cpu(&gaussians, target);
    result.initial_psnr = init_psnr;
    println!("  Initial PSNR (after clamping): {:.2} dB", init_psnr);

    // Run optimizer in chunks, re-applying clamping periodically
    let chunk_size = 50;
    let mut optimizer = GpuAdamOptimizer::new(gpu.device()).expect("Failed to create optimizer");
    optimizer.learning_rates = GpuLearningRates::default();

    for chunk in 0..(iterations / chunk_size) {
        optimizer.max_iterations = chunk_size;
        let _ = optimizer.optimize(gpu.device(), gpu.queue(), &mut gaussians, target);

        // Re-apply clamping after each chunk
        apply_geodesic_clamping(&mut gaussians, gedt, target.width, target.height, 0.5, 0.3);

        let psnr = compute_psnr_cpu(&gaussians, target);
        println!("    Chunk {}: {:.2} dB", chunk + 1, psnr);
    }

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

fn run_with_aggressive_clamping(
    gpu: &GpuManager,
    target: &ImageBuffer<f32>,
    gedt: &GeodesicEDT,
    n_gaussians: usize,
    iterations: usize,
    image_path: &str,
) -> (f32, f32) {
    let start = Instant::now();

    let mut result = TestResult::new("geodesic_clamping_aggressive", image_path);
    result.optimizer = "GPU Adam + Aggressive clamping (σ ≤ 0.3 + 0.2×d)".to_string();
    result.n_gaussians = n_gaussians;
    result.iterations = iterations;
    result.image_dimensions = (target.width, target.height);
    result.set_param("clamping", "geodesic_aggressive");
    result.set_param("clamp_offset", 0.3);
    result.set_param("clamp_scale", 0.2);

    let mut gaussians = init_jittered_grid(target.width, target.height, n_gaussians, 0.015);
    sample_colors(&mut gaussians, target);

    apply_geodesic_clamping(&mut gaussians, gedt, target.width, target.height, 0.3, 0.2);

    let init_psnr = compute_psnr_cpu(&gaussians, target);
    result.initial_psnr = init_psnr;
    println!("  Initial PSNR (after clamping): {:.2} dB", init_psnr);

    let chunk_size = 50;
    let mut optimizer = GpuAdamOptimizer::new(gpu.device()).expect("Failed to create optimizer");
    optimizer.learning_rates = GpuLearningRates::default();

    for chunk in 0..(iterations / chunk_size) {
        optimizer.max_iterations = chunk_size;
        let _ = optimizer.optimize(gpu.device(), gpu.queue(), &mut gaussians, target);

        apply_geodesic_clamping(&mut gaussians, gedt, target.width, target.height, 0.3, 0.2);

        let psnr = compute_psnr_cpu(&gaussians, target);
        println!("    Chunk {}: {:.2} dB", chunk + 1, psnr);
    }

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

fn apply_geodesic_clamping(
    gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
    gedt: &GeodesicEDT,
    width: u32,
    height: u32,
    offset: f32,
    scale_factor: f32,
) {
    let norm = width.max(height) as f32;

    for g in gaussians.iter_mut() {
        let px = (g.position.x * width as f32).clamp(0.0, width as f32 - 1.0) as u32;
        let py = (g.position.y * height as f32).clamp(0.0, height as f32 - 1.0) as u32;

        // Get geodesic distance (in pixels)
        let d = gedt.get_distance(px, py);

        // Convert to normalized coordinates and compute max sigma
        let d_norm = d / norm;
        let sigma_max = offset / norm + scale_factor * d_norm;

        // Clamp both scale_x and scale_y
        g.shape.scale_x = g.shape.scale_x.min(sigma_max).max(0.002);
        g.shape.scale_y = g.shape.scale_y.min(sigma_max).max(0.002);
    }
}

fn init_jittered_grid(width: u32, height: u32, n: usize, scale: f32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let aspect = width as f32 / height as f32;
    let cols = ((n as f32 * aspect).sqrt()).round() as usize;
    let rows = (n as f32 / cols as f32).ceil() as usize;

    let mut gaussians = Vec::with_capacity(n);

    for i in 0..rows {
        for j in 0..cols {
            if gaussians.len() >= n { break; }

            let base_x = (j as f32 + 0.5) / cols as f32;
            let base_y = (i as f32 + 0.5) / rows as f32;

            let jx = (rng.gen::<f32>() - 0.5) * 0.3 / cols as f32;
            let jy = (rng.gen::<f32>() - 0.5) * 0.3 / rows as f32;

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

fn sample_colors(gaussians: &mut [Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) {
    for g in gaussians.iter_mut() {
        let px = (g.position.x * target.width as f32) as u32;
        let py = (g.position.y * target.height as f32) as u32;
        if let Some(c) = target.get_pixel(px.min(target.width - 1), py.min(target.height - 1)) {
            g.color = c;
        }
    }
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
