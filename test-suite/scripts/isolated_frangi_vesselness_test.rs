//! ISOLATED Frangi Vesselness Test
//!
//! Tests ONLY the Frangi vesselness filter's contribution to encoding quality.
//! This is a toolkit component test - NOT trying to find optimal configuration.
//!
//! Question: Does identifying tubular structures and using extreme anisotropy help?
//!
//! Comparison:
//! - Control: Uniform isotropic Gaussians
//! - Treatment: Extreme anisotropy (1:8 ratio) in tubular regions
//!
//! Keeping ALL other parameters identical.

use lgi_core::{ImageBuffer, StructureTensorField, frangi_vesselness::FrangiFilter};
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
    println!("   ISOLATED TOOLKIT TEST: Frangi Vesselness Filter");
    println!("   Question: Does extreme anisotropy for tubular structures help?");
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

    // Test on images with varying amounts of tubular structures
    let images = vec![
        ("kodim03.png", "parrots - some thin feathers"),
        ("kodim19.png", "fence - linear structures"),
        ("kodim05.png", "flowers - organic stems"),
        ("kodim08.png", "castle - architectural lines"),
    ];

    let n_gaussians = 256;
    let iterations = 200;
    let base_scale = 0.04;

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

        // Compute structure tensor and Frangi vesselness
        println!("  Computing structure tensor and Frangi vesselness...");
        let tensor_field = StructureTensorField::compute(&target, 1.2, 2.5)
            .expect("Failed to compute structure tensor");
        let frangi = FrangiFilter::default();
        let vesselness = frangi.compute_vesselness(&tensor_field);

        // Analyze vesselness distribution
        let (high_vessel_pct, max_vessel) = analyze_vesselness(&vesselness);
        println!("  High-vesselness pixels (>0.3): {:.1}%", high_vessel_pct * 100.0);
        println!("  Max vesselness: {:.3}", max_vessel);

        // ═══════════════════════════════════════════════════════════════
        // CONTROL: Uniform isotropic initialization
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [CONTROL] Isotropic Gaussians (no vesselness info):");
        let control_gaussians = init_isotropic(width, height, n_gaussians, &target, base_scale);
        let control_result = run_optimization(
            &device, &queue, control_gaussians, &target, iterations,
            &format!("isolated_frangi_control_{}", img_name.replace(".png", "")),
            &image_path, "Control: Isotropic"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 control_result.0, control_result.1, control_result.2);

        // ═══════════════════════════════════════════════════════════════
        // TREATMENT: Extreme anisotropy in tubular regions
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [TREATMENT] Extreme anisotropy (1:8) for tubular regions:");
        let treatment_gaussians = init_vesselness_aware(
            width, height, n_gaussians, &target, base_scale, &vesselness, &tensor_field
        );
        let treatment_result = run_optimization(
            &device, &queue, treatment_gaussians, &target, iterations,
            &format!("isolated_frangi_treatment_{}", img_name.replace(".png", "")),
            &image_path, "Treatment: Vesselness-aware"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 treatment_result.0, treatment_result.1, treatment_result.2);

        // ═══════════════════════════════════════════════════════════════
        // COMPARISON
        // ═══════════════════════════════════════════════════════════════
        let module_impact = treatment_result.1 - control_result.1;
        println!("\n  ▶ Frangi Vesselness Impact: {:+.3}dB", module_impact);

        results.push((
            img_name.to_string(),
            control_result.1,
            treatment_result.1,
            module_impact,
            high_vessel_pct,
        ));
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("        FRANGI VESSELNESS MODULE - ISOLATED TEST SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:15} {:>12} {:>12} {:>12} {:>10}",
             "Image", "Control", "Treatment", "Impact", "Vessel %");
    println!("  {:->15} {:->12} {:->12} {:->12} {:->10}", "", "", "", "", "");

    let mut total_impact = 0.0;
    for (name, control, treatment, impact, vessel_pct) in &results {
        println!("  {:15} {:>10.2}dB {:>10.2}dB {:>+10.3}dB {:>8.1}%",
                 name, control, treatment, impact, vessel_pct * 100.0);
        total_impact += impact;
    }

    let avg_impact = total_impact / results.len() as f32;
    println!("\n  Average Impact: {:+.3}dB", avg_impact);

    // Correlation analysis
    println!("\n  Correlation with Tubular Content:");
    let (high_vessel, low_vessel): (Vec<_>, Vec<_>) = results.iter()
        .partition(|(_, _, _, _, v)| *v > 0.05);

    if !high_vessel.is_empty() && !low_vessel.is_empty() {
        let high_vessel_avg: f32 = high_vessel.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / high_vessel.len() as f32;
        let low_vessel_avg: f32 = low_vessel.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / low_vessel.len() as f32;
        println!("    High tubular content (>5%): {:+.3}dB avg impact", high_vessel_avg);
        println!("    Low tubular content  (<5%): {:+.3}dB avg impact", low_vessel_avg);
    }

    // Conclusion
    println!("\n  TOOLKIT FINDING:");
    if avg_impact > 0.1 {
        println!("    ✓ Frangi vesselness HELPS for tubular regions ({:+.2}dB average)", avg_impact);
        println!("    → Recommended for: hair, fences, text, blood vessels, power lines");
    } else if avg_impact < -0.1 {
        println!("    ✗ Frangi vesselness HURTS quality ({:+.2}dB average)", avg_impact);
        println!("    → Extreme anisotropy may destabilize optimization");
    } else {
        println!("    ○ Frangi vesselness has MINIMAL impact ({:+.2}dB average)", avg_impact);
        println!("    → May help perceptually for specific content types");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn analyze_vesselness(vesselness: &[f32]) -> (f32, f32) {
    let high_count = vesselness.iter().filter(|&&v| v > 0.3).count();
    let high_pct = high_count as f32 / vesselness.len() as f32;
    let max_val = vesselness.iter().cloned().fold(0.0_f32, f32::max);
    (high_pct, max_val)
}

fn init_isotropic(
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

fn init_vesselness_aware(
    width: u32, height: u32, count: usize, target: &ImageBuffer<f32>, base_scale: f32,
    vesselness: &[f32], tensor_field: &StructureTensorField
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);  // Same seed as control!
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

        // Get vesselness at this position
        let idx = (py * width + px) as usize;
        let v = vesselness.get(idx).copied().unwrap_or(0.0);

        // If high vesselness, use extreme anisotropy aligned with structure
        let (scale_x, scale_y, rotation) = if v > 0.3 {
            // Get orientation from structure tensor
            let tensor = tensor_field.get(px, py);
            let tangent = tensor.eigenvector_minor;
            let angle = tangent.y.atan2(tangent.x);

            // Extreme anisotropy: 1:8 ratio (long along tangent, thin across)
            let anisotropy = 8.0;
            (
                base_scale * anisotropy,   // Long along tubular direction
                base_scale / anisotropy.sqrt(),  // Thin perpendicular
                angle
            )
        } else {
            // Isotropic for non-tubular regions
            (base_scale, base_scale, 0.0)
        };

        gaussians.push(Gaussian2D::new(
            position,
            Euler::new(scale_x, scale_y, rotation),
            color,
            1.0
        ));
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
    result.set_param("module_under_test", "frangi_vesselness".to_string());
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (init_psnr, final_psnr, final_psnr - init_psnr)
}
