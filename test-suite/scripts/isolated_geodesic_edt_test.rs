//! ISOLATED Geodesic EDT Test
//!
//! Tests ONLY the geodesic EDT module's contribution to encoding quality.
//! This is a toolkit component test - NOT trying to find optimal configuration.
//!
//! Question: Does sigma clamping prevent color bleeding and improve quality?
//!
//! Comparison:
//! - Control: No sigma clamping (Gaussians can extend freely)
//! - Treatment: Geodesic sigma clamping (σ ≤ 0.5 + 0.3×d)
//!
//! Keeping ALL other parameters identical.

use lgi_core::{ImageBuffer, StructureTensorField, geodesic_edt::GeodesicEDT};
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
    println!("   ISOLATED TOOLKIT TEST: Geodesic EDT");
    println!("   Question: Does sigma clamping prevent bleeding and help quality?");
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

    // Test on images with varying edge density (important for anti-bleeding)
    let images = vec![
        ("kodim03.png", "parrots - complex boundaries"),
        ("kodim08.png", "castle - architectural edges"),
        ("kodim19.png", "fence - fine detail edges"),
        ("kodim23.png", "houses - strong contrasts"),
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

        // Compute structure tensor and geodesic EDT
        println!("  Computing structure tensor and geodesic EDT...");
        let tensor_field = StructureTensorField::compute(&target, 1.2, 2.5)
            .expect("Failed to compute structure tensor");
        let gedt = GeodesicEDT::compute(&target, &tensor_field, 0.5, 50.0)
            .expect("Failed to compute geodesic EDT");

        // Analyze clamping statistics
        let (clamped_count, avg_clamp_ratio) = analyze_clamping(&gedt, width, height, base_scale);
        println!("  Gaussians that would be clamped: {:.1}%",
                 100.0 * clamped_count as f32 / n_gaussians as f32);
        println!("  Average clamp ratio (clamped/desired): {:.2}", avg_clamp_ratio);

        // ═══════════════════════════════════════════════════════════════
        // CONTROL: No sigma clamping
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [CONTROL] No sigma clamping:");
        let control_gaussians = init_unclamped(width, height, n_gaussians, &target, base_scale);
        let control_result = run_optimization(
            &device, &queue, control_gaussians, &target, iterations,
            &format!("isolated_geodesic_edt_control_{}", img_name.replace(".png", "")),
            &image_path, "Control: No clamping"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 control_result.0, control_result.1, control_result.2);

        // ═══════════════════════════════════════════════════════════════
        // TREATMENT: With geodesic sigma clamping
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [TREATMENT] Geodesic sigma clamping (σ ≤ 0.5 + 0.3×d):");
        let treatment_gaussians = init_with_clamping(width, height, n_gaussians, &target, base_scale, &gedt);
        let treatment_result = run_optimization(
            &device, &queue, treatment_gaussians, &target, iterations,
            &format!("isolated_geodesic_edt_treatment_{}", img_name.replace(".png", "")),
            &image_path, "Treatment: Geodesic clamping"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 treatment_result.0, treatment_result.1, treatment_result.2);

        // ═══════════════════════════════════════════════════════════════
        // COMPARISON
        // ═══════════════════════════════════════════════════════════════
        let module_impact = treatment_result.1 - control_result.1;
        println!("\n  ▶ Geodesic EDT Impact: {:+.3}dB", module_impact);

        results.push((
            img_name.to_string(),
            control_result.1,
            treatment_result.1,
            module_impact,
            avg_clamp_ratio,
        ));
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("          GEODESIC EDT MODULE - ISOLATED TEST SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:15} {:>12} {:>12} {:>12} {:>10}",
             "Image", "Control", "Treatment", "Impact", "Clamp %");
    println!("  {:->15} {:->12} {:->12} {:->12} {:->10}", "", "", "", "", "");

    let mut total_impact = 0.0;
    for (name, control, treatment, impact, clamp_ratio) in &results {
        println!("  {:15} {:>10.2}dB {:>10.2}dB {:>+10.3}dB {:>8.1}%",
                 name, control, treatment, impact, (1.0 - clamp_ratio) * 100.0);
        total_impact += impact;
    }

    let avg_impact = total_impact / results.len() as f32;
    println!("\n  Average Impact: {:+.3}dB", avg_impact);

    // Correlation analysis
    println!("\n  Correlation with Clamping Amount:");
    let (high_clamp, low_clamp): (Vec<_>, Vec<_>) = results.iter()
        .partition(|(_, _, _, _, c)| *c < 0.8);  // High clamping = ratio < 0.8

    if !high_clamp.is_empty() && !low_clamp.is_empty() {
        let high_clamp_avg: f32 = high_clamp.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / high_clamp.len() as f32;
        let low_clamp_avg: f32 = low_clamp.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / low_clamp.len() as f32;
        println!("    High clamping images: {:+.3}dB avg impact", high_clamp_avg);
        println!("    Low clamping images:  {:+.3}dB avg impact", low_clamp_avg);
    }

    // Conclusion
    println!("\n  TOOLKIT FINDING:");
    if avg_impact > 0.1 {
        println!("    ✓ Geodesic EDT HELPS prevent bleeding ({:+.2}dB average)", avg_impact);
        println!("    → Recommended for images with strong edges/boundaries");
    } else if avg_impact < -0.1 {
        println!("    ✗ Geodesic EDT HURTS quality ({:+.2}dB average)", avg_impact);
        println!("    → Clamping may be too aggressive");
    } else {
        println!("    ○ Geodesic EDT has MINIMAL impact ({:+.2}dB average)", avg_impact);
        println!("    → May help perceptually even if PSNR unchanged");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn analyze_clamping(gedt: &GeodesicEDT, width: u32, height: u32, desired_scale: f32) -> (usize, f32) {
    let mut clamped_count = 0;
    let mut total_ratio = 0.0;
    let mut count = 0;

    // Sample at grid positions
    let cols = 16;
    let rows = 16;
    for row in 0..rows {
        for col in 0..cols {
            let x = (col * width / cols).min(width - 1);
            let y = (row * height / rows).min(height - 1);

            let clamped = gedt.clamp_sigma(x, y, desired_scale);
            if clamped < desired_scale * 0.99 {
                clamped_count += 1;
            }
            total_ratio += clamped / desired_scale;
            count += 1;
        }
    }

    (clamped_count, total_ratio / count as f32)
}

fn init_unclamped(
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

        // NO clamping - use full scale
        gaussians.push(Gaussian2D::new(position, Euler::isotropic(scale), color, 1.0));
    }

    gaussians
}

fn init_with_clamping(
    width: u32, height: u32, count: usize, target: &ImageBuffer<f32>, base_scale: f32,
    gedt: &GeodesicEDT
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

        // CLAMP sigma by geodesic distance to edge
        // Formula: σ ≤ 0.5 + 0.3 × d (pixels)
        // Convert from normalized to pixel scale for clamping, then back
        let pixel_scale = base_scale * width.max(height) as f32;
        let clamped_pixel_scale = gedt.clamp_sigma(px, py, pixel_scale);
        let clamped_scale = clamped_pixel_scale / width.max(height) as f32;

        gaussians.push(Gaussian2D::new(position, Euler::isotropic(clamped_scale), color, 1.0));
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
    result.set_param("module_under_test", "geodesic_edt".to_string());
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (init_psnr, final_psnr, final_psnr - init_psnr)
}
