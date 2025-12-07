//! ISOLATED Saliency Detection Test
//!
//! Tests ONLY the saliency detection module's contribution to encoding quality.
//! This is a toolkit component test - NOT trying to find optimal configuration.
//!
//! Question: Does placing MORE Gaussians in salient (high-gradient) regions help?
//!
//! Comparison:
//! - Control: Uniform grid placement
//! - Treatment: Saliency-weighted placement (more Gaussians in detailed areas)
//!
//! Keeping ALL other parameters identical.

use lgi_core::{ImageBuffer, saliency_detection::SaliencyDetector};
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_gpu::GpuAdamOptimizer;
use lgi_math::{
    gaussian::Gaussian2D,
    parameterization::Euler,
    vec::Vector2,
    color::Color4,
};
use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("   ISOLATED TOOLKIT TEST: Saliency Detection");
    println!("   Question: Does saliency-weighted placement help quality?");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Initialize GPU
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).expect("Failed to find GPU adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("LGI Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        },
    )).expect("Failed to create GPU device");

    println!("GPU: {}\n", adapter.get_info().name);

    // Test on images with varying saliency distribution
    let images = vec![
        ("kodim03.png", "parrots - high saliency contrast"),
        ("kodim01.png", "lighthouse - sky vs detail"),
        ("kodim07.png", "face - focused subject"),
        ("kodim13.png", "plane - simple background"),
    ];

    let n_gaussians = 256;
    let iterations = 200;
    let init_scale = 0.04;

    let mut results: Vec<(String, f32, f32, f32, f32)> = Vec::new();

    for (img_name, description) in &images {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Image: {} ({})", img_name, description);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let image_path = format!("test-data/kodak-dataset/{}", img_name);
        let full_path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), image_path);

        let img = image::open(&full_path).expect("Failed to load image").to_rgb32f();
        let width = img.width();
        let height = img.height();

        let mut target = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                target.set_pixel(x, y, Color4::new(pixel[0], pixel[1], pixel[2], 1.0));
            }
        }

        // Compute saliency map
        println!("  Computing saliency map...");
        let detector = SaliencyDetector::default();
        let saliency_map = detector.compute_saliency_map(&target);

        // Analyze saliency distribution
        let (high_saliency_pct, saliency_variance) = analyze_saliency(&saliency_map);
        println!("  High-saliency pixels: {:.1}%", high_saliency_pct * 100.0);
        println!("  Saliency variance: {:.4}", saliency_variance);

        // ═══════════════════════════════════════════════════════════════
        // CONTROL: Uniform grid placement
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [CONTROL] Uniform grid placement:");
        let control_gaussians = init_uniform(width, height, n_gaussians, &target, init_scale);
        let control_result = run_optimization(
            &device, &queue, control_gaussians, &target, iterations,
            &format!("isolated_saliency_control_{}", img_name.replace(".png", "")),
            &image_path, "Control: Uniform placement"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 control_result.0, control_result.1, control_result.2);

        // ═══════════════════════════════════════════════════════════════
        // TREATMENT: Saliency-weighted placement
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [TREATMENT] Saliency-weighted placement:");
        let treatment_gaussians = init_saliency_weighted(width, height, n_gaussians, &target, init_scale, &saliency_map);
        let treatment_result = run_optimization(
            &device, &queue, treatment_gaussians, &target, iterations,
            &format!("isolated_saliency_treatment_{}", img_name.replace(".png", "")),
            &image_path, "Treatment: Saliency-weighted"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 treatment_result.0, treatment_result.1, treatment_result.2);

        // ═══════════════════════════════════════════════════════════════
        // COMPARISON
        // ═══════════════════════════════════════════════════════════════
        let module_impact = treatment_result.1 - control_result.1;
        println!("\n  ▶ Saliency Detection Impact: {:+.3}dB", module_impact);

        results.push((
            img_name.to_string(),
            control_result.1,
            treatment_result.1,
            module_impact,
            saliency_variance,
        ));
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("        SALIENCY DETECTION MODULE - ISOLATED TEST SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:15} {:>12} {:>12} {:>12} {:>10}",
             "Image", "Control", "Treatment", "Impact", "Sal Var");
    println!("  {:->15} {:->12} {:->12} {:->12} {:->10}", "", "", "", "", "");

    let mut total_impact = 0.0;
    for (name, control, treatment, impact, variance) in &results {
        println!("  {:15} {:>10.2}dB {:>10.2}dB {:>+10.3}dB {:>8.4}",
                 name, control, treatment, impact, variance);
        total_impact += impact;
    }

    let avg_impact = total_impact / results.len() as f32;
    println!("\n  Average Impact: {:+.3}dB", avg_impact);

    // Correlation with saliency variance
    println!("\n  Correlation with Saliency Variance:");
    let (high_var, low_var): (Vec<_>, Vec<_>) = results.iter()
        .partition(|(_, _, _, _, v)| *v > 0.01);

    if !high_var.is_empty() && !low_var.is_empty() {
        let high_var_avg: f32 = high_var.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / high_var.len() as f32;
        let low_var_avg: f32 = low_var.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / low_var.len() as f32;
        println!("    High saliency variance: {:+.3}dB avg impact", high_var_avg);
        println!("    Low saliency variance:  {:+.3}dB avg impact", low_var_avg);
    }

    // Conclusion
    println!("\n  TOOLKIT FINDING:");
    if avg_impact > 0.1 {
        println!("    ✓ Saliency-weighted placement HELPS ({:+.2}dB average)", avg_impact);
        println!("    → Recommended for images with distinct salient regions");
    } else if avg_impact < -0.1 {
        println!("    ✗ Saliency-weighted placement HURTS ({:+.2}dB average)", avg_impact);
        println!("    → Uniform placement may be more robust");
    } else {
        println!("    ○ Saliency weighting has MINIMAL impact ({:+.2}dB average)", avg_impact);
        println!("    → May help perceptually for specific content types");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn analyze_saliency(saliency_map: &[f32]) -> (f32, f32) {
    let total: f32 = saliency_map.iter().sum();
    let mean = total / saliency_map.len() as f32;

    let high_count = saliency_map.iter().filter(|&&s| s > 0.3).count();
    let high_pct = high_count as f32 / saliency_map.len() as f32;

    let variance: f32 = saliency_map.iter()
        .map(|&s| (s - mean).powi(2))
        .sum::<f32>() / saliency_map.len() as f32;

    (high_pct, variance)
}

fn init_uniform(
    width: u32, height: u32, count: usize, target: &ImageBuffer<f32>, scale: f32
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    let cols = (count as f32).sqrt().ceil() as u32;
    let rows = (count as f32 / cols as f32).ceil() as u32;
    let cell_w = width as f32 / cols as f32;
    let cell_h = height as f32 / rows as f32;

    for i in 0..count {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;

        let jitter_x: f32 = rng.gen_range(-0.3..0.3);
        let jitter_y: f32 = rng.gen_range(-0.3..0.3);
        let px = ((col as f32 + 0.5 + jitter_x) * cell_w).clamp(0.0, width as f32 - 1.0) as u32;
        let py = ((row as f32 + 0.5 + jitter_y) * cell_h).clamp(0.0, height as f32 - 1.0) as u32;

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        gaussians.push(Gaussian2D::new(position, Euler::isotropic(scale), color, 1.0));
    }

    gaussians
}

fn init_saliency_weighted(
    width: u32, height: u32, count: usize, target: &ImageBuffer<f32>, scale: f32,
    saliency_map: &[f32]
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    // Build cumulative probability distribution based on saliency
    // Higher saliency = higher probability of placing a Gaussian
    let min_saliency = saliency_map.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_saliency = saliency_map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let saliency_range = (max_saliency - min_saliency).max(0.001);

    // Normalize and add base probability
    let probabilities: Vec<f32> = saliency_map.iter()
        .map(|&s| {
            let normalized = (s - min_saliency) / saliency_range;
            0.3 + 0.7 * normalized  // Base 30% + 70% from saliency
        })
        .collect();

    let total_prob: f32 = probabilities.iter().sum();
    let mut cumulative = Vec::with_capacity(probabilities.len());
    let mut sum = 0.0;
    for p in &probabilities {
        sum += p / total_prob;
        cumulative.push(sum);
    }

    // Sample positions according to saliency distribution
    for _ in 0..count {
        // Binary search for random sample
        let r: f32 = rng.gen();
        let idx = cumulative.partition_point(|&c| c < r);
        let idx = idx.min(cumulative.len() - 1);

        // Convert index to pixel coordinates
        let px = (idx as u32) % width;
        let py = (idx as u32) / width;

        // Add small jitter
        let jitter_x: f32 = rng.gen_range(-2.0..2.0);
        let jitter_y: f32 = rng.gen_range(-2.0..2.0);
        let px = (px as f32 + jitter_x).clamp(0.0, width as f32 - 1.0) as u32;
        let py = (py as f32 + jitter_y).clamp(0.0, height as f32 - 1.0) as u32;

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        gaussians.push(Gaussian2D::new(position, Euler::isotropic(scale), color, 1.0));
    }

    gaussians
}

fn run_optimization(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    iterations: usize,
    test_name: &str,
    image_path: &str,
    description: &str,
) -> (f32, f32, f32) {
    let width = target.width;
    let height = target.height;

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(target, &init_render);

    let start = Instant::now();

    let mut optimizer = GpuAdamOptimizer::new(device).expect("Failed to create Adam");
    optimizer.max_iterations = iterations;

    let _ = optimizer.optimize(device, queue, &mut gaussians, target);

    let elapsed = start.elapsed();
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = lgi_core::error_metrics::compute_psnr(target, &final_render);

    // Save result
    let mut result = TestResult::new(test_name, image_path);
    result.optimizer = description.to_string();
    result.n_gaussians = gaussians.len();
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f64() as f32;
    result.image_dimensions = (width, height);
    result.set_param("module_under_test", "saliency_detection".to_string());
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (init_psnr, final_psnr, final_psnr - init_psnr)
}
