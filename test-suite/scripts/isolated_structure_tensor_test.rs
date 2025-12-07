//! ISOLATED Structure Tensor Test
//!
//! Tests ONLY the structure tensor module's contribution to encoding quality.
//! This is a toolkit component test - NOT trying to find optimal configuration.
//!
//! Question: Does edge-aligned anisotropic initialization help?
//!
//! Comparison:
//! - Control: Isotropic initialization (no structure tensor)
//! - Treatment: Anisotropic initialization using structure tensor
//!
//! Keeping ALL other parameters identical.

use lgi_core::{ImageBuffer, StructureTensorField};
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
    println!("   ISOLATED TOOLKIT TEST: Structure Tensor");
    println!("   Question: Does edge-aligned anisotropic init help?");
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

    // Test on multiple images with different edge characteristics
    let images = vec![
        ("kodim03.png", "parrots - complex feathers"),
        ("kodim01.png", "lighthouse - strong edges"),
        ("kodim23.png", "houses - architectural lines"),
        ("kodim05.png", "flowers - organic edges"),
    ];

    // Fixed test parameters - keeping everything constant except the module under test
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

        // Compute structure tensor for this image
        println!("  Computing structure tensor...");
        let tensor_field = StructureTensorField::compute(&target, 1.2, 2.5)
            .expect("Failed to compute structure tensor");

        // Analyze image edge content
        let edge_count = analyze_edge_content(&tensor_field);
        println!("  Edge pixels (coherence > 0.5): {:.1}%",
                 100.0 * edge_count as f32 / (width * height) as f32);

        // ═══════════════════════════════════════════════════════════════
        // CONTROL: Isotropic initialization (NO structure tensor)
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [CONTROL] Isotropic initialization:");
        let control_gaussians = init_isotropic(width, height, n_gaussians, &target, init_scale);
        let control_result = run_optimization(
            &device, &queue, control_gaussians, &target, iterations,
            &format!("isolated_structure_tensor_control_{}", img_name.replace(".png", "")),
            &image_path, "Control: Isotropic init"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 control_result.0, control_result.1, control_result.2);

        // ═══════════════════════════════════════════════════════════════
        // TREATMENT: Anisotropic initialization (WITH structure tensor)
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [TREATMENT] Edge-aligned anisotropic initialization:");
        let treatment_gaussians = init_anisotropic(width, height, n_gaussians, &target, init_scale, &tensor_field);
        let treatment_result = run_optimization(
            &device, &queue, treatment_gaussians, &target, iterations,
            &format!("isolated_structure_tensor_treatment_{}", img_name.replace(".png", "")),
            &image_path, "Treatment: Anisotropic init"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 treatment_result.0, treatment_result.1, treatment_result.2);

        // ═══════════════════════════════════════════════════════════════
        // COMPARISON
        // ═══════════════════════════════════════════════════════════════
        let module_impact = treatment_result.1 - control_result.1;
        println!("\n  ▶ Structure Tensor Impact: {:+.3}dB", module_impact);

        results.push((
            img_name.to_string(),
            control_result.1,
            treatment_result.1,
            module_impact,
            edge_count as f32 / (width * height) as f32,
        ));
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("        STRUCTURE TENSOR MODULE - ISOLATED TEST SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:15} {:>12} {:>12} {:>12} {:>10}",
             "Image", "Control", "Treatment", "Impact", "Edge %");
    println!("  {:->15} {:->12} {:->12} {:->12} {:->10}", "", "", "", "", "");

    let mut total_impact = 0.0;
    for (name, control, treatment, impact, edge_pct) in &results {
        println!("  {:15} {:>10.2}dB {:>10.2}dB {:>+10.3}dB {:>8.1}%",
                 name, control, treatment, impact, edge_pct * 100.0);
        total_impact += impact;
    }

    let avg_impact = total_impact / results.len() as f32;
    println!("\n  Average Impact: {:+.3}dB", avg_impact);

    // Correlation analysis
    println!("\n  Correlation with Edge Content:");
    let (high_edge, low_edge): (Vec<_>, Vec<_>) = results.iter()
        .partition(|(_, _, _, _, e)| *e > 0.15);

    if !high_edge.is_empty() && !low_edge.is_empty() {
        let high_edge_avg: f32 = high_edge.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / high_edge.len() as f32;
        let low_edge_avg: f32 = low_edge.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / low_edge.len() as f32;
        println!("    High-edge images (>15%): {:+.3}dB avg impact", high_edge_avg);
        println!("    Low-edge images  (<15%): {:+.3}dB avg impact", low_edge_avg);
    }

    // Conclusion
    println!("\n  TOOLKIT FINDING:");
    if avg_impact > 0.1 {
        println!("    ✓ Structure tensor HELPS for anisotropic init ({:+.2}dB average)", avg_impact);
        println!("    → Recommended for images with strong edges");
    } else if avg_impact < -0.1 {
        println!("    ✗ Structure tensor HURTS quality ({:+.2}dB average)", avg_impact);
        println!("    → Not recommended for general use");
    } else {
        println!("    ○ Structure tensor has MINIMAL impact ({:+.2}dB average)", avg_impact);
        println!("    → Optional - may help on specific image types");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn analyze_edge_content(tensor_field: &StructureTensorField) -> usize {
    let mut edge_count = 0;
    for y in 0..tensor_field.height {
        for x in 0..tensor_field.width {
            if tensor_field.get(x, y).coherence > 0.5 {
                edge_count += 1;
            }
        }
    }
    edge_count
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

        // ISOTROPIC: Same scale in all directions, no rotation
        gaussians.push(Gaussian2D::new(position, Euler::isotropic(scale), color, 1.0));
    }

    gaussians
}

fn init_anisotropic(
    width: u32, height: u32, count: usize, target: &ImageBuffer<f32>, base_scale: f32,
    tensor_field: &StructureTensorField
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42); // Same seed as control!
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

        // Get structure tensor at this position
        let tensor = tensor_field.get(px, py);

        // ANISOTROPIC: Use structure tensor for orientation and shape
        let (scale_x, scale_y, rotation) = if tensor.is_edge(0.5) {
            // At edge: make Gaussian long along edge tangent, thin across
            let anisotropy = 1.0 + tensor.coherence;  // 1.5-2x elongation at strong edges
            (
                base_scale * anisotropy,           // Long along tangent
                base_scale / anisotropy,           // Thin across edge
                tensor.eigenvector_minor.y.atan2(tensor.eigenvector_minor.x)  // Align with tangent
            )
        } else {
            // Flat region: isotropic
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
    result.set_param("module_under_test", "structure_tensor".to_string());
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (init_psnr, final_psnr, final_psnr - init_psnr)
}
