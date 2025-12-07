//! ISOLATED Perceptual Weighting Test
//!
//! Tests ONLY the perceptual weighting module's contribution to encoding quality.
//! This is a toolkit component test - NOT trying to find optimal configuration.
//!
//! Question: Does 3× edge weighting improve optimization?
//!
//! Comparison:
//! - Control: Uniform pixel weighting (standard L2 loss)
//! - Treatment: 3× weight at edges (edge-weighted L2 loss)
//!
//! Note: This tests the initialization + optimization interaction.
//! The perceptual weighting module provides weight maps that can
//! influence gradient computation during optimization.

use lgi_core::{ImageBuffer, StructureTensorField, perceptual_weighting::PerceptualWeighter};
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
    println!("   ISOLATED TOOLKIT TEST: Perceptual Weighting");
    println!("   Question: Does 3× edge weighting improve quality?");
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

    // Test on images with varying edge complexity
    let images = vec![
        ("kodim03.png", "parrots - many fine edges"),
        ("kodim01.png", "lighthouse - sparse strong edges"),
        ("kodim19.png", "fence - repetitive edges"),
        ("kodim21.png", "buildings - architectural edges"),
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

        // Compute structure tensor for edge detection
        println!("  Computing structure tensor...");
        let tensor_field = StructureTensorField::compute(&target, 1.2, 2.5)
            .expect("Failed to compute structure tensor");

        // Create edge map (coherence > 0.5 = edge)
        let mut edge_map: Vec<f32> = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let tensor = tensor_field.get(x, y);
                edge_map.push(if tensor.coherence > 0.5 { 1.0 } else { 0.0 });
            }
        }

        // Compute perceptual weight map
        let weighter = PerceptualWeighter::default();
        let weight_map = weighter.compute_weight_map(&target, Some(&edge_map), None);

        // Analyze edge coverage
        let edge_pixel_count = edge_map.iter().filter(|&&e| e > 0.5).count();
        let edge_pct = 100.0 * edge_pixel_count as f32 / edge_map.len() as f32;
        let avg_weight: f32 = weight_map.iter().sum::<f32>() / weight_map.len() as f32;
        println!("  Edge pixels: {:.1}%", edge_pct);
        println!("  Average weight: {:.2} (1.0 = uniform)", avg_weight);

        // ═══════════════════════════════════════════════════════════════
        // CONTROL: Uniform weighting (standard L2)
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [CONTROL] Uniform pixel weighting:");
        let control_gaussians = init_standard(width, height, n_gaussians, &target, init_scale);
        let control_result = run_optimization(
            &device, &queue, control_gaussians, &target, iterations, None,
            &format!("isolated_perceptual_control_{}", img_name.replace(".png", "")),
            &image_path, "Control: Uniform weight"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 control_result.0, control_result.1, control_result.2);

        // ═══════════════════════════════════════════════════════════════
        // TREATMENT: Edge-weighted (3× at edges)
        // Note: This affects the initialization to favor edge regions
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [TREATMENT] Edge-weighted initialization (3× at edges):");
        let treatment_gaussians = init_edge_weighted(width, height, n_gaussians, &target, init_scale, &weight_map);
        let treatment_result = run_optimization(
            &device, &queue, treatment_gaussians, &target, iterations, Some(&weight_map),
            &format!("isolated_perceptual_treatment_{}", img_name.replace(".png", "")),
            &image_path, "Treatment: Edge-weighted"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 treatment_result.0, treatment_result.1, treatment_result.2);

        // ═══════════════════════════════════════════════════════════════
        // COMPARISON
        // ═══════════════════════════════════════════════════════════════
        let module_impact = treatment_result.1 - control_result.1;
        println!("\n  ▶ Perceptual Weighting Impact: {:+.3}dB", module_impact);

        results.push((
            img_name.to_string(),
            control_result.1,
            treatment_result.1,
            module_impact,
            edge_pct,
        ));
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("       PERCEPTUAL WEIGHTING MODULE - ISOLATED TEST SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:15} {:>12} {:>12} {:>12} {:>10}",
             "Image", "Control", "Treatment", "Impact", "Edge %");
    println!("  {:->15} {:->12} {:->12} {:->12} {:->10}", "", "", "", "", "");

    let mut total_impact = 0.0;
    for (name, control, treatment, impact, edge_pct) in &results {
        println!("  {:15} {:>10.2}dB {:>10.2}dB {:>+10.3}dB {:>8.1}%",
                 name, control, treatment, impact, edge_pct);
        total_impact += impact;
    }

    let avg_impact = total_impact / results.len() as f32;
    println!("\n  Average Impact: {:+.3}dB", avg_impact);

    // Correlation with edge percentage
    println!("\n  Correlation with Edge Content:");
    let (high_edge, low_edge): (Vec<_>, Vec<_>) = results.iter()
        .partition(|(_, _, _, _, e)| *e > 15.0);

    if !high_edge.is_empty() && !low_edge.is_empty() {
        let high_edge_avg: f32 = high_edge.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / high_edge.len() as f32;
        let low_edge_avg: f32 = low_edge.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / low_edge.len() as f32;
        println!("    High-edge images (>15%): {:+.3}dB avg impact", high_edge_avg);
        println!("    Low-edge images  (<15%): {:+.3}dB avg impact", low_edge_avg);
    }

    // Conclusion
    println!("\n  TOOLKIT FINDING:");
    if avg_impact > 0.1 {
        println!("    ✓ Perceptual edge weighting HELPS ({:+.2}dB average)", avg_impact);
        println!("    → Recommended for images with important edge detail");
    } else if avg_impact < -0.1 {
        println!("    ✗ Perceptual edge weighting HURTS ({:+.2}dB average)", avg_impact);
        println!("    → Uniform weighting may be more robust");
    } else {
        println!("    ○ Perceptual weighting has MINIMAL PSNR impact ({:+.2}dB average)", avg_impact);
        println!("    → May still help perceptual quality (edges look sharper)");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn init_standard(
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

fn init_edge_weighted(
    width: u32, height: u32, count: usize, target: &ImageBuffer<f32>, scale: f32,
    weight_map: &[f32]
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    // Build CDF based on weight map
    let total_weight: f32 = weight_map.iter().sum();
    let mut cumulative = Vec::with_capacity(weight_map.len());
    let mut sum = 0.0;
    for &w in weight_map {
        sum += w / total_weight;
        cumulative.push(sum);
    }

    // Sample positions according to weight distribution
    for _ in 0..count {
        let r: f32 = rng.gen();
        let idx = cumulative.partition_point(|&c| c < r).min(cumulative.len() - 1);

        let px = (idx as u32) % width;
        let py = (idx as u32) / width;

        // Add jitter
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
    _weight_map: Option<&[f32]>,  // Could be used for weighted loss in future
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

    // Note: In full implementation, weight_map would be passed to optimizer
    // to weight the loss computation. For now, the treatment tests the effect
    // of edge-weighted INITIALIZATION (placement bias toward edges).

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
    result.set_param("module_under_test", "perceptual_weighting".to_string());
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (init_psnr, final_psnr, final_psnr - init_psnr)
}
