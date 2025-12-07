//! Rotation Initialization Sweep
//!
//! Gaussian rotation is often under-utilized in optimization.
//! This experiment tests different initial rotation strategies:
//! - **Zero rotation**: All Gaussians start axis-aligned
//! - **Random rotation**: Each Gaussian gets random rotation
//! - **Gradient-aligned**: Rotation aligned with local image gradient
//! - **Structure-aligned**: Rotation based on local structure tensor
//!
//! Hypothesis: Gradient-aligned rotation may provide better starting point
//! for representing edges and oriented textures.

use lgi_core::ImageBuffer;
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
    println!("   ROTATION INITIALIZATION SWEEP");
    println!("   Testing different initial rotation strategies");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Initialize GPU
    println!("Initializing GPU...");
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
        },
        None,
    )).expect("Failed to create GPU device");

    println!("  GPU: {}\n", adapter.get_info().name);

    let image_path = "test-data/kodak-dataset/kodim03.png";
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

    println!("Image: kodim03 ({}x{})\n", width, height);

    // Compute image gradients for gradient-aligned initialization
    let gradients = compute_image_gradients(&target);

    let n_gaussians = 256;
    let iterations = 200;

    // Rotation strategies to test
    let strategies: Vec<(&str, Box<dyn Fn(u32, u32, &[(f32, f32)]) -> f32>)> = vec![
        ("Zero rotation", Box::new(|_px, _py, _grads| 0.0f32)),
        ("Random rotation", Box::new(|_px, _py, _grads| {
            let mut rng = StdRng::seed_from_u64((_px as u64) * 1000 + (_py as u64));
            rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI)
        })),
        ("Random small", Box::new(|_px, _py, _grads| {
            let mut rng = StdRng::seed_from_u64((_px as u64) * 1000 + (_py as u64));
            rng.gen_range(-0.5..0.5)
        })),
        ("Gradient-aligned", Box::new(|px, py, grads| {
            let idx = (py * 768 + px) as usize; // Assuming kodim03 size
            if idx < grads.len() {
                let (gx, gy) = grads[idx];
                gy.atan2(gx) // Rotation perpendicular to gradient
            } else {
                0.0
            }
        })),
        ("Gradient-perpendicular", Box::new(|px, py, grads| {
            let idx = (py * 768 + px) as usize;
            if idx < grads.len() {
                let (gx, gy) = grads[idx];
                gy.atan2(gx) + std::f32::consts::FRAC_PI_2 // Perpendicular
            } else {
                0.0
            }
        })),
        ("Diagonal (+45°)", Box::new(|_px, _py, _grads| std::f32::consts::FRAC_PI_4)),
        ("Diagonal (-45°)", Box::new(|_px, _py, _grads| -std::f32::consts::FRAC_PI_4)),
    ];

    let mut results = Vec::new();

    for (name, rot_fn) in &strategies {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Strategy: {}", name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let gaussians = init_with_rotation(width, height, n_gaussians, &target, &gradients, rot_fn);
        let rendered = RendererV2::render(&gaussians, width, height);
        let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
        println!("  Initial PSNR: {:.2}dB", init_psnr);

        let mut gaussians = gaussians;
        let start = Instant::now();

        let mut optimizer = GpuAdamOptimizer::new(&device).expect("Failed to create Adam");
        optimizer.max_iterations = iterations;

        // Run with progress reporting
        let mut last_psnr = init_psnr;
        for phase in 0..4 {
            optimizer.max_iterations = 50;
            let _ = optimizer.optimize(&device, &queue, &mut gaussians, &target);

            let mid_render = RendererV2::render(&gaussians, width, height);
            let mid_psnr = lgi_core::error_metrics::compute_psnr(&target, &mid_render);

            let direction = if mid_psnr > last_psnr { "↑" } else { "↓" };
            println!("  [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                     (phase + 1) * 50, iterations, mid_psnr, direction, mid_psnr - last_psnr);
            last_psnr = mid_psnr;
        }

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

        // Save result
        let clean_name = name.to_lowercase()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "plus")
            .replace("-", "minus")
            .replace("°", "deg");
        let mut result = TestResult::new(&format!("rotation_init_{}", clean_name), image_path);
        result.optimizer = format!("GPU Adam (rotation={})", name);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f64() as f32;
        result.image_dimensions = (width, height);
        result.set_param("rotation_strategy", clean_name.clone());
        result.complete();
        let _ = result.save_with_image(&final_render, RESULTS_DIR);

        println!("\n  Final: {:.2}dB (Δ={:+.2}dB from init) in {:.1}s\n",
                 final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

        results.push((name.to_string(), init_psnr, final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64()));
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    ROTATION INITIALIZATION SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:25} {:>10} {:>10} {:>12} {:>10}",
             "Strategy", "Init PSNR", "Final", "Improvement", "Time");
    println!("  {:->25} {:->10} {:->10} {:->12} {:->10}", "", "", "", "", "");

    for (name, init_psnr, final_psnr, improvement, time) in &results {
        println!("  {:25} {:>8.2}dB {:>8.2}dB {:>+10.2}dB {:>8.1}s",
                 name, init_psnr, final_psnr, improvement, time);
    }

    // Find best final PSNR
    let best = results.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
    let zero_rot = results.iter().find(|(n, _, _, _, _)| n.contains("Zero"));

    println!("\n  Best strategy: {} ({:.2}dB final)", best.0, best.2);
    if let Some(z) = zero_rot {
        println!("  Improvement over zero rotation: {:+.2}dB", best.2 - z.2);
    }

    // Analysis
    println!("\n  Analysis:");
    let gradient_aligned = results.iter().find(|(n, _, _, _, _)| n.contains("Gradient-aligned"));
    if let Some((_, _, ga_psnr, _, _)) = gradient_aligned {
        if let Some((_, _, z_psnr, _, _)) = zero_rot {
            if ga_psnr > z_psnr + 0.1 {
                println!("  → Gradient-aligned rotation helps (+{:.2}dB)", ga_psnr - z_psnr);
            } else if z_psnr > ga_psnr + 0.1 {
                println!("  → Zero rotation is better - gradient alignment not beneficial");
            } else {
                println!("  → Rotation initialization has minimal impact");
            }
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn compute_image_gradients(target: &ImageBuffer<f32>) -> Vec<(f32, f32)> {
    let width = target.width;
    let height = target.height;
    let mut gradients = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            // Sobel-like gradient computation
            let get_lum = |px: i32, py: i32| -> f32 {
                if px < 0 || px >= width as i32 || py < 0 || py >= height as i32 {
                    return 0.5;
                }
                let c = target.get_pixel(px as u32, py as u32).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
                0.299 * c.r + 0.587 * c.g + 0.114 * c.b
            };

            let ix = x as i32;
            let iy = y as i32;

            // Horizontal gradient
            let gx = get_lum(ix + 1, iy) - get_lum(ix - 1, iy);
            // Vertical gradient
            let gy = get_lum(ix, iy + 1) - get_lum(ix, iy - 1);

            gradients.push((gx, gy));
        }
    }

    gradients
}

fn init_with_rotation(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    gradients: &[(f32, f32)],
    rotation_fn: &dyn Fn(u32, u32, &[(f32, f32)]) -> f32,
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
        let px = ((col as f32 + 0.5 + jitter_x) * cell_w) as i32;
        let py = ((row as f32 + 0.5 + jitter_y) * cell_h) as i32;
        let px = px.clamp(0, width as i32 - 1) as u32;
        let py = py.clamp(0, height as i32 - 1) as u32;

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        let rotation = rotation_fn(px, py, gradients);
        let shape = Euler {
            scale_x: 0.04,
            scale_y: 0.04,
            rotation,
        };
        gaussians.push(Gaussian2D::new(position, shape, color, 1.0));
    }

    gaussians
}
