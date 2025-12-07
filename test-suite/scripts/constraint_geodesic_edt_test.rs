//! CONSTRAINT-Based Geodesic EDT Test
//!
//! Tests the geodesic EDT module as a CONTINUOUS CONSTRAINT during optimization.
//! Unlike the isolated init test, this applies sigma clamping EVERY iteration.
//!
//! Question: Does enforcing σ ≤ edge_distance prevent color bleeding?
//!
//! Comparison:
//! - Control: Standard Adam with no constraints
//! - Treatment: Adam + sigma clamped to geodesic edge distance every update
//!
//! This tests the module as an OPTIMIZATION-TIME constraint, not just init.

use lgi_core::{ImageBuffer, StructureTensorField, geodesic_edt::GeodesicEDT};
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_gpu::gradient::GpuGradientComputer;
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
    println!("   CONSTRAINT TEST: Geodesic EDT Sigma Clamping");
    println!("   Question: Does continuous sigma clamping prevent bleeding?");
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

    // Test on images with strong edges where bleeding is visible
    let images = vec![
        ("kodim03.png", "parrots - complex color boundaries"),
        ("kodim08.png", "castle - architectural edges"),
        ("kodim19.png", "fence - fine detail boundaries"),
        ("kodim23.png", "houses - strong contrast edges"),
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

        // Analyze how much clamping would occur
        let clamp_stats = analyze_clamp_potential(&gedt, width, height, base_scale);
        println!("  Potential clamp reduction: {:.1}%", clamp_stats * 100.0);

        // ═══════════════════════════════════════════════════════════════
        // CONTROL: Standard Adam (no constraints)
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [CONTROL] Standard Adam (no sigma constraints):");
        let control_gaussians = init_gaussians(width, height, n_gaussians, &target, base_scale);
        let control_result = run_unconstrained(
            &device, &queue, control_gaussians, &target, iterations,
            &format!("constraint_gedt_control_{}", img_name.replace(".png", "")),
            &image_path, "Control: No constraints"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 control_result.0, control_result.1, control_result.2);

        // ═══════════════════════════════════════════════════════════════
        // TREATMENT: Adam + Geodesic sigma clamping every iteration
        // ═══════════════════════════════════════════════════════════════
        println!("\n  [TREATMENT] Adam + Geodesic sigma clamping (every iteration):");
        let treatment_gaussians = init_gaussians(width, height, n_gaussians, &target, base_scale);
        let treatment_result = run_with_sigma_constraint(
            &device, &queue, treatment_gaussians, &target, iterations, &gedt,
            &format!("constraint_gedt_treatment_{}", img_name.replace(".png", "")),
            &image_path, "Treatment: Geodesic clamping"
        );
        println!("    Initial: {:.2}dB → Final: {:.2}dB (Δ={:+.2}dB)",
                 treatment_result.0, treatment_result.1, treatment_result.2);

        // ═══════════════════════════════════════════════════════════════
        // COMPARISON
        // ═══════════════════════════════════════════════════════════════
        let module_impact = treatment_result.1 - control_result.1;
        println!("\n  ▶ Geodesic EDT Constraint Impact: {:+.3}dB", module_impact);

        results.push((
            img_name.to_string(),
            control_result.1,
            treatment_result.1,
            module_impact,
            clamp_stats,
        ));
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("    GEODESIC EDT CONSTRAINT - OPTIMIZATION-TIME TEST SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:15} {:>12} {:>12} {:>12} {:>10}",
             "Image", "Control", "Constrained", "Impact", "Clamp %");
    println!("  {:->15} {:->12} {:->12} {:->12} {:->10}", "", "", "", "", "");

    let mut total_impact = 0.0;
    for (name, control, treatment, impact, clamp_pct) in &results {
        println!("  {:15} {:>10.2}dB {:>10.2}dB {:>+10.3}dB {:>8.1}%",
                 name, control, treatment, impact, clamp_pct * 100.0);
        total_impact += impact;
    }

    let avg_impact = total_impact / results.len() as f32;
    println!("\n  Average Impact: {:+.3}dB", avg_impact);

    // Correlation analysis
    println!("\n  Correlation with Clamp Amount:");
    let (high_clamp, low_clamp): (Vec<_>, Vec<_>) = results.iter()
        .partition(|(_, _, _, _, c)| *c > 0.2);

    if !high_clamp.is_empty() && !low_clamp.is_empty() {
        let high_clamp_avg: f32 = high_clamp.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / high_clamp.len() as f32;
        let low_clamp_avg: f32 = low_clamp.iter().map(|(_, _, _, i, _)| i).sum::<f32>() / low_clamp.len() as f32;
        println!("    High clamping (>20%): {:+.3}dB avg impact", high_clamp_avg);
        println!("    Low clamping  (<20%): {:+.3}dB avg impact", low_clamp_avg);
    }

    // Conclusion
    println!("\n  TOOLKIT FINDING:");
    if avg_impact > 0.1 {
        println!("    ✓ Geodesic EDT constraint HELPS ({:+.2}dB average)", avg_impact);
        println!("    → Sigma clamping prevents color bleeding at edges");
    } else if avg_impact < -0.1 {
        println!("    ✗ Geodesic EDT constraint HURTS ({:+.2}dB average)", avg_impact);
        println!("    → Constraint may be too restrictive, limiting optimization freedom");
    } else {
        println!("    ○ Geodesic EDT constraint has MINIMAL PSNR impact ({:+.2}dB average)", avg_impact);
        println!("    → May still help perceptually (sharper edges) even if PSNR unchanged");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn analyze_clamp_potential(gedt: &GeodesicEDT, width: u32, height: u32, desired_scale: f32) -> f32 {
    let mut total_reduction = 0.0;
    let mut count = 0;

    // Sample grid
    let cols = 16;
    let rows = 16;
    for row in 0..rows {
        for col in 0..cols {
            let x = (col * width / cols).min(width - 1);
            let y = (row * height / rows).min(height - 1);

            let clamped = gedt.clamp_sigma(x, y, desired_scale);
            total_reduction += 1.0 - (clamped / desired_scale);
            count += 1;
        }
    }

    total_reduction / count as f32
}

fn init_gaussians(
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

/// Run optimization WITHOUT any constraints (standard Adam)
fn run_unconstrained(
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

    // Standard Adam - no modifications
    let mut optimizer = lgi_gpu::GpuAdamOptimizer::new(device).expect("Failed to create Adam");
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
    result.set_param("module_under_test", "geodesic_edt_constraint".to_string());
    result.set_param("constraint_type", "none".to_string());
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (init_psnr, final_psnr, final_psnr - init_psnr)
}

/// Run optimization WITH geodesic sigma clamping constraint applied every iteration
fn run_with_sigma_constraint(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    iterations: usize,
    gedt: &GeodesicEDT,
    test_name: &str,
    image_path: &str,
    description: &str,
) -> (f32, f32, f32) {
    let width = target.width;
    let height = target.height;

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(target, &init_render);

    let start = Instant::now();

    // Create GPU gradient computer for manual optimization loop
    let gradient_computer = GpuGradientComputer::new(device).expect("Failed to create gradient computer");

    // Adam hyperparameters
    let lr_position = 0.0002;
    let lr_color = 0.02;
    let lr_scale = 0.005;
    let lr_opacity = 0.05;
    let lr_rotation = 0.01;
    let beta1: f32 = 0.9;
    let beta2: f32 = 0.999;
    let epsilon: f32 = 1e-8;

    // Adam state - per Gaussian: position(2) + scale(2) + rotation(1) + color(4) + opacity(1) = 10
    let n_gaussians = gaussians.len();
    let mut m_pos = vec![[0.0f32; 2]; n_gaussians];
    let mut v_pos = vec![[0.0f32; 2]; n_gaussians];
    let mut m_scale = vec![[0.0f32; 2]; n_gaussians];
    let mut v_scale = vec![[0.0f32; 2]; n_gaussians];
    let mut m_rot = vec![0.0f32; n_gaussians];
    let mut v_rot = vec![0.0f32; n_gaussians];
    let mut m_color = vec![[0.0f32; 4]; n_gaussians];
    let mut v_color = vec![[0.0f32; 4]; n_gaussians];
    let mut m_opacity = vec![0.0f32; n_gaussians];
    let mut v_opacity = vec![0.0f32; n_gaussians];

    let mut last_psnr = init_psnr;

    for iter in 0..iterations {
        // Render current state
        let rendered = RendererV2::render(&gaussians, width, height);

        // Compute gradients using GPU
        let gpu_gradients = gradient_computer.compute_gradients(device, queue, &gaussians, &rendered, target)
            .expect("Gradient computation failed");

        // Bias correction factors
        let bc1 = 1.0 - beta1.powi(iter as i32 + 1);
        let bc2 = 1.0 - beta2.powi(iter as i32 + 1);

        // Apply Adam update with geodesic sigma constraint
        for (i, gaussian) in gaussians.iter_mut().enumerate() {
            let grad = &gpu_gradients[i];

            // Position update
            for j in 0..2 {
                let g = grad.d_position[j];
                m_pos[i][j] = beta1 * m_pos[i][j] + (1.0 - beta1) * g;
                v_pos[i][j] = beta2 * v_pos[i][j] + (1.0 - beta2) * g * g;

                let m_hat = m_pos[i][j] / bc1;
                let v_hat = v_pos[i][j] / bc2;
                let update = lr_position * m_hat / (v_hat.sqrt() + epsilon);

                if j == 0 {
                    gaussian.position.x -= update;
                    gaussian.position.x = gaussian.position.x.clamp(0.0, 1.0);
                } else {
                    gaussian.position.y -= update;
                    gaussian.position.y = gaussian.position.y.clamp(0.0, 1.0);
                }
            }

            // Scale update
            let scale_grads = [grad.d_scale_x, grad.d_scale_y];
            for j in 0..2 {
                let g = scale_grads[j];
                m_scale[i][j] = beta1 * m_scale[i][j] + (1.0 - beta1) * g;
                v_scale[i][j] = beta2 * v_scale[i][j] + (1.0 - beta2) * g * g;

                let m_hat = m_scale[i][j] / bc1;
                let v_hat = v_scale[i][j] / bc2;
                let update = lr_scale * m_hat / (v_hat.sqrt() + epsilon);

                if j == 0 {
                    gaussian.shape.scale_x -= update;
                    gaussian.shape.scale_x = gaussian.shape.scale_x.max(0.001);
                } else {
                    gaussian.shape.scale_y -= update;
                    gaussian.shape.scale_y = gaussian.shape.scale_y.max(0.001);
                }
            }

            // ═══════════════════════════════════════════════════════════
            // GEODESIC SIGMA CONSTRAINT: Clamp scale to edge distance
            // ═══════════════════════════════════════════════════════════
            let px = (gaussian.position.x * width as f32).clamp(0.0, width as f32 - 1.0) as u32;
            let py = (gaussian.position.y * height as f32).clamp(0.0, height as f32 - 1.0) as u32;

            // Convert to pixel scale for clamping
            let image_size = width.max(height) as f32;
            let pixel_scale_x = gaussian.shape.scale_x * image_size;
            let pixel_scale_y = gaussian.shape.scale_y * image_size;

            let clamped_x = gedt.clamp_sigma(px, py, pixel_scale_x);
            let clamped_y = gedt.clamp_sigma(px, py, pixel_scale_y);

            // Convert back to normalized
            gaussian.shape.scale_x = clamped_x / image_size;
            gaussian.shape.scale_y = clamped_y / image_size;

            // Rotation update
            let g = grad.d_rotation;
            m_rot[i] = beta1 * m_rot[i] + (1.0 - beta1) * g;
            v_rot[i] = beta2 * v_rot[i] + (1.0 - beta2) * g * g;
            let m_hat = m_rot[i] / bc1;
            let v_hat = v_rot[i] / bc2;
            gaussian.shape.rotation -= lr_rotation * m_hat / (v_hat.sqrt() + epsilon);

            // Color update
            for j in 0..4 {
                let g = grad.d_color[j];
                m_color[i][j] = beta1 * m_color[i][j] + (1.0 - beta1) * g;
                v_color[i][j] = beta2 * v_color[i][j] + (1.0 - beta2) * g * g;

                let m_hat = m_color[i][j] / bc1;
                let v_hat = v_color[i][j] / bc2;
                let update = lr_color * m_hat / (v_hat.sqrt() + epsilon);

                match j {
                    0 => { gaussian.color.r -= update; gaussian.color.r = gaussian.color.r.clamp(0.0, 1.0); }
                    1 => { gaussian.color.g -= update; gaussian.color.g = gaussian.color.g.clamp(0.0, 1.0); }
                    2 => { gaussian.color.b -= update; gaussian.color.b = gaussian.color.b.clamp(0.0, 1.0); }
                    _ => { gaussian.color.a -= update; gaussian.color.a = gaussian.color.a.clamp(0.0, 1.0); }
                }
            }

            // Opacity update
            let g = grad.d_opacity;
            m_opacity[i] = beta1 * m_opacity[i] + (1.0 - beta1) * g;
            v_opacity[i] = beta2 * v_opacity[i] + (1.0 - beta2) * g * g;
            let m_hat = m_opacity[i] / bc1;
            let v_hat = v_opacity[i] / bc2;
            gaussian.opacity -= lr_opacity * m_hat / (v_hat.sqrt() + epsilon);
            gaussian.opacity = gaussian.opacity.clamp(0.01, 1.0);
        }

        // Progress reporting
        if iter % 50 == 0 || iter == iterations - 1 {
            let current_render = RendererV2::render(&gaussians, width, height);
            let current_psnr = lgi_core::error_metrics::compute_psnr(target, &current_render);
            let direction = if current_psnr > last_psnr { "↑" } else { "↓" };
            println!("      [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                     iter, iterations, current_psnr, direction, current_psnr - last_psnr);
            last_psnr = current_psnr;
        }
    }

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
    result.set_param("module_under_test", "geodesic_edt_constraint".to_string());
    result.set_param("constraint_type", "continuous_sigma_clamp".to_string());
    result.set_param("lr_position", lr_position);
    result.set_param("lr_color", lr_color);
    result.set_param("lr_scale", lr_scale);
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (init_psnr, final_psnr, final_psnr - init_psnr)
}
