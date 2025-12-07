//! ISOLATED Text Stroke Detection Test
//!
//! Tests ONLY the text stroke detection module's contribution to encoding quality.
//! This is a toolkit component test - NOT trying to find optimal configuration.
//!
//! Question: Do the text stroke presets (σ⊥=0.3, σ∥=5) improve text rendering?
//!
//! Comparison:
//! - Control: Isotropic Gaussians everywhere
//! - Treatment: Stroke presets in detected text regions
//!
//! NOTE: Since Kodak images don't contain much text, this test uses
//! images with high coherence thin lines as a proxy for text-like strokes.

use lgi_core::{ImageBuffer, StructureTensorField, text_stroke_detection::TextStrokeDetector};
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
    println!("   ISOLATED TOOLKIT TEST: Text Stroke Detection");
    println!("   Question: Do stroke presets (σ⊥:σ∥ = 1:16.7) help thin lines?");
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

    // Test on images with thin linear structures (text-like)
    let images = vec![
        ("kodim19.png", "fence - thin lines like text strokes"),
        ("kodim01.png", "lighthouse - some thin structures"),
        ("kodim08.png", "castle - windows like rectangles"),
        ("kodim23.png", "houses - rooflines"),
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

        // Compute structure tensor and detect strokes
        println!("  Computing structure tensor and detecting strokes...");
        let tensor_field = StructureTensorField::compute(&target, 1.2, 2.5)
            .expect("Failed to compute structure tensor");
        let detector = TextStrokeDetector::default();
        let stroke_mask = detector.detect_strokes(&target, &tensor_field);

        // Analyze stroke distribution
        let stroke_count = stroke_mask.iter().filter(|&&s| s).count();
        let stroke_pct = stroke_count as f32 / stroke_mask.len() as f32;
        println!("  Detected stroke pixels: {:.2}%", stroke_pct * 100.0);

        // ═══════════════════════════════════════════════════════════════
        // CONTROL: Isotropic initialization
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [CONTROL] Isotropic Gaussians everywhere:");
        let control_gaussians = init_isotropic(width, height, n_gaussians, &target, base_scale);
        let control_result = run_optimization(
            &device, &queue, control_gaussians, &target, iterations,
            &format!("isolated_text_stroke_control_{}", img_name.replace(".png", "")),
            &image_path, "Control: Isotropic"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 control_result.0, control_result.1, control_result.2);

        // ═══════════════════════════════════════════════════════════════
        // TREATMENT: Stroke presets for detected strokes
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [TREATMENT] Stroke presets (σ⊥:σ∥ = 1:16.7) for detected strokes:");
        let treatment_gaussians = init_stroke_aware(
            width, height, n_gaussians, &target, base_scale, &stroke_mask, &tensor_field
        );
        let treatment_result = run_optimization(
            &device, &queue, treatment_gaussians, &target, iterations,
            &format!("isolated_text_stroke_treatment_{}", img_name.replace(".png", "")),
            &image_path, "Treatment: Stroke presets"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 treatment_result.0, treatment_result.1, treatment_result.2);

        // ═══════════════════════════════════════════════════════════════
        // COMPARISON
        // ═══════════════════════════════════════════════════════════════
        let module_impact = treatment_result.1 - control_result.1;
        println!("\n  ▶ Text Stroke Detection Impact: {:+.3}dB", module_impact);

        results.push((
            img_name.to_string(),
            control_result.1,
            treatment_result.1,
            module_impact,
            stroke_pct,
        ));
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("        TEXT STROKE DETECTION MODULE - ISOLATED TEST SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:15} {:>12} {:>12} {:>12} {:>10}",
             "Image", "Control", "Treatment", "Impact", "Stroke %");
    println!("  {:->15} {:->12} {:->12} {:->12} {:->10}", "", "", "", "", "");

    let mut total_impact = 0.0;
    for (name, control, treatment, impact, stroke_pct) in &results {
        println!("  {:15} {:>10.2}dB {:>10.2}dB {:>+10.3}dB {:>8.2}%",
                 name, control, treatment, impact, stroke_pct * 100.0);
        total_impact += impact;
    }

    let avg_impact = total_impact / results.len() as f32;
    println!("\n  Average Impact: {:+.3}dB", avg_impact);

    // Correlation analysis
    println!("\n  Correlation with Stroke Content:");
    let (high_stroke, low_stroke): (Vec<_>, Vec<_>) = results.iter()
        .partition(|(_, _, _, _, s)| *s > 0.02);

    if !high_stroke.is_empty() && !low_stroke.is_empty() {
        let high_stroke_avg: f32 = high_stroke.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / high_stroke.len() as f32;
        let low_stroke_avg: f32 = low_stroke.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / low_stroke.len() as f32;
        println!("    High stroke content (>2%): {:+.3}dB avg impact", high_stroke_avg);
        println!("    Low stroke content  (<2%): {:+.3}dB avg impact", low_stroke_avg);
    }

    // Conclusion
    println!("\n  TOOLKIT FINDING:");
    if avg_impact > 0.1 {
        println!("    ✓ Text stroke presets HELP for thin lines ({:+.2}dB average)", avg_impact);
        println!("    → Recommended for: text, diagrams, line art, thin structures");
    } else if avg_impact < -0.1 {
        println!("    ✗ Text stroke presets HURT quality ({:+.2}dB average)", avg_impact);
        println!("    → Extreme anisotropy (1:16.7) may be too aggressive");
    } else {
        println!("    ○ Text stroke presets have MINIMAL impact ({:+.2}dB average)", avg_impact);
        println!("    → Most useful for actual text; natural images less affected");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
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

fn init_stroke_aware(
    width: u32, height: u32, count: usize, target: &ImageBuffer<f32>, base_scale: f32,
    stroke_mask: &[bool], tensor_field: &StructureTensorField
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

        // Check if this position is in a stroke region
        let idx = (py * width + px) as usize;
        let is_stroke = stroke_mask.get(idx).copied().unwrap_or(false);

        // Use TEXT STROKE PRESET: σ⊥=0.3, σ∥=5 (ratio 1:16.7)
        let (scale_x, scale_y, rotation) = if is_stroke {
            // Get orientation from structure tensor
            let tensor = tensor_field.get(px, py);
            let tangent = tensor.eigenvector_minor;
            let angle = tangent.y.atan2(tangent.x);

            // Text stroke preset: extreme anisotropy
            // σ⊥ = 0.3px, σ∥ = 5px in pixel space
            // Convert to normalized coordinates
            let image_size = width.max(height) as f32;
            let sigma_perp = 0.3 / image_size;  // Very thin across
            let sigma_para = 5.0 / image_size;  // Long along stroke

            (sigma_para, sigma_perp, angle)
        } else {
            // Isotropic for non-stroke regions
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
    result.set_param("module_under_test", "text_stroke_detection".to_string());
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (init_psnr, final_psnr, final_psnr - init_psnr)
}
