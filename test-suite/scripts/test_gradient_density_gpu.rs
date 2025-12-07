//! GPU-Accelerated Gradient-Guided Placement Density Test
//!
//! Same experiment as test_gradient_density_placement but using GPU Adam optimizer
//! for faster optimization. This allows testing more variations quickly.

use lgi_core::ImageBuffer;
use lgi_encoder_v2::test_results::TestResult;
use lgi_gpu::GpuAdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
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
    println!("   GPU GRADIENT-GUIDED PLACEMENT DENSITY TEST");
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

    println!("  GPU: {}", adapter.get_info().name);
    println!();

    let image_path = "test-data/kodak-dataset/kodim03.png";
    let full_path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), image_path);

    // Load image
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

    // Compute gradient magnitude map
    println!("Computing gradient magnitude...");
    let gradient_map = compute_gradient_magnitude(&target);
    let max_grad = gradient_map.iter().cloned().fold(0.0_f32, f32::max);
    println!("  Max gradient: {:.4}\n", max_grad);

    let n_gaussians = 256;

    // Create GPU optimizer
    let mut optimizer = GpuAdamOptimizer::new(&device).expect("Failed to create GPU optimizer");
    optimizer.max_iterations = 200;

    // Test 1: Uniform grid
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: UNIFORM GRID - Equal density");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let uniform_result = run_gpu_encoding(
        "gpu_uniform_grid",
        &target,
        init_uniform_grid(width, height, n_gaussians, &target),
        &device,
        &queue,
        &optimizer,
        image_path,
    );

    // Test 2: Gradient-weighted placement
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: GRADIENT-WEIGHTED - Higher density in details");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let gradient_result = run_gpu_encoding(
        "gpu_gradient_weighted",
        &target,
        init_gradient_weighted(width, height, n_gaussians, &target, &gradient_map, max_grad, false),
        &device,
        &queue,
        &optimizer,
        image_path,
    );

    // Test 3: Gradient-adaptive sigma
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: GRADIENT-ADAPTIVE SIGMA - Placement + size");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let adaptive_result = run_gpu_encoding(
        "gpu_gradient_adaptive",
        &target,
        init_gradient_adaptive(width, height, n_gaussians, &target, &gradient_map, max_grad),
        &device,
        &queue,
        &optimizer,
        image_path,
    );

    // Test 4: Higher Gaussian count (GPU makes this fast)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 4: GRADIENT-ADAPTIVE 512 - More Gaussians");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let high_count_result = run_gpu_encoding(
        "gpu_gradient_adaptive_512",
        &target,
        init_gradient_adaptive(width, height, 512, &target, &gradient_map, max_grad),
        &device,
        &queue,
        &optimizer,
        image_path,
    );

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    GPU PLACEMENT DENSITY SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:30} {:>8} {:>12} {:>12} {:>10}", "Strategy", "N", "Initial", "Final", "Time");
    println!("  {:->30} {:->8} {:->12} {:->12} {:->10}", "", "", "", "", "");

    for (name, (n, init_psnr, final_psnr, time)) in [
        ("Uniform grid", uniform_result),
        ("Gradient-weighted", gradient_result),
        ("Gradient + adaptive σ", adaptive_result),
        ("Gradient + adaptive 512", high_count_result),
    ] {
        println!("  {:30} {:>8} {:>10.2}dB {:>10.2}dB {:>8.1}s", name, n, init_psnr, final_psnr, time);
    }

    let gradient_advantage = gradient_result.2 - uniform_result.2;
    let adaptive_advantage = adaptive_result.2 - uniform_result.2;

    println!("\n  Gradient-weighted advantage: {:+.2}dB vs uniform", gradient_advantage);
    println!("  Adaptive sigma advantage: {:+.2}dB vs uniform", adaptive_advantage);

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn run_gpu_encoding(
    name: &str,
    target: &ImageBuffer<f32>,
    mut gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    optimizer: &GpuAdamOptimizer,
    image_path: &str,
) -> (usize, f32, f32, f64) {
    let width = target.width;
    let height = target.height;
    let n_gaussians = gaussians.len();

    let mut result = TestResult::new(&format!("gradient_density_{}", name), image_path);
    result.optimizer = "GPU Adam".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = optimizer.max_iterations;

    // Compute initial PSNR
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
    result.initial_psnr = init_psnr;

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n_gaussians, init_psnr);

    // GPU optimization
    let start = Instant::now();
    let _ = optimizer.optimize(device, queue, &mut gaussians, target);
    let elapsed = start.elapsed();

    // Final PSNR
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = lgi_core::error_metrics::compute_psnr(target, &final_render);

    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f64() as f32;
    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    println!("  Final: PSNR={:.2}dB (improvement: {:+.2}dB)", final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.1}s", elapsed.as_secs_f64());

    (n_gaussians, init_psnr, final_psnr, elapsed.as_secs_f64())
}

fn compute_gradient_magnitude(target: &ImageBuffer<f32>) -> Vec<f32> {
    let width = target.width as i32;
    let height = target.height as i32;
    let mut gradients = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let get_lum = |px: i32, py: i32| -> f32 {
                let px = px.clamp(0, width - 1) as u32;
                let py = py.clamp(0, height - 1) as u32;
                let c = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
                0.299 * c.r + 0.587 * c.g + 0.114 * c.b
            };

            let gx = get_lum(x + 1, y) - get_lum(x - 1, y);
            let gy = get_lum(x, y + 1) - get_lum(x, y - 1);
            gradients.push((gx * gx + gy * gy).sqrt());
        }
    }

    gradients
}

fn init_uniform_grid(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
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

        gaussians.push(Gaussian2D::new(position, Euler::isotropic(0.04), color, 1.0));
    }

    gaussians
}

fn init_gradient_weighted(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    gradient_map: &[f32],
    max_gradient: f32,
    _inverse: bool,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    let mut weights: Vec<f32> = gradient_map.iter().map(|&g| {
        let normalized = (g / max_gradient.max(0.001)).clamp(0.0, 1.0);
        normalized + 0.1
    }).collect();

    let total: f32 = weights.iter().sum();
    for w in &mut weights {
        *w /= total;
    }

    let mut cdf = Vec::with_capacity(weights.len());
    let mut cumsum = 0.0;
    for w in &weights {
        cumsum += w;
        cdf.push(cumsum);
    }

    let min_dist_sq = 0.02_f32.powi(2);
    let mut selected_positions = Vec::with_capacity(count);

    for _ in 0..count * 10 {
        if selected_positions.len() >= count {
            break;
        }

        let r: f32 = rng.gen();
        let idx = cdf.iter().position(|&c| c >= r).unwrap_or(cdf.len() - 1);

        let px = (idx as u32) % width;
        let py = (idx as u32) / width;
        let x = px as f32 / width as f32;
        let y = py as f32 / height as f32;

        let too_close = selected_positions.iter().any(|&(ex, ey): &(f32, f32)| {
            (x - ex).powi(2) + (y - ey).powi(2) < min_dist_sq
        });

        if !too_close {
            selected_positions.push((x, y));
            let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
            gaussians.push(Gaussian2D::new(Vector2::new(x, y), Euler::isotropic(0.04), color, 1.0));
        }
    }

    while gaussians.len() < count {
        let x: f32 = rng.gen();
        let y: f32 = rng.gen();
        let px = (x * width as f32).min(width as f32 - 1.0) as u32;
        let py = (y * height as f32).min(height as f32 - 1.0) as u32;
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
        gaussians.push(Gaussian2D::new(Vector2::new(x, y), Euler::isotropic(0.04), color, 1.0));
    }

    gaussians
}

fn init_gradient_adaptive(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    gradient_map: &[f32],
    max_gradient: f32,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    let mut weights: Vec<f32> = gradient_map.iter().map(|&g| {
        let normalized = (g / max_gradient.max(0.001)).clamp(0.0, 1.0);
        normalized + 0.1
    }).collect();

    let total: f32 = weights.iter().sum();
    for w in &mut weights {
        *w /= total;
    }

    let mut cdf = Vec::with_capacity(weights.len());
    let mut cumsum = 0.0;
    for w in &weights {
        cumsum += w;
        cdf.push(cumsum);
    }

    let sigma_min = 0.02;
    let sigma_max = 0.06;
    let min_dist_sq = 0.02_f32.powi(2);
    let mut selected_positions = Vec::with_capacity(count);

    for _ in 0..count * 10 {
        if selected_positions.len() >= count {
            break;
        }

        let r: f32 = rng.gen();
        let idx = cdf.iter().position(|&c| c >= r).unwrap_or(cdf.len() - 1);

        let px = (idx as u32) % width;
        let py = (idx as u32) / width;
        let x = px as f32 / width as f32;
        let y = py as f32 / height as f32;

        let too_close = selected_positions.iter().any(|&(ex, ey): &(f32, f32)| {
            (x - ex).powi(2) + (y - ey).powi(2) < min_dist_sq
        });

        if !too_close {
            selected_positions.push((x, y));
            let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
            let gradient = gradient_map[idx];
            let norm_grad = (gradient / max_gradient.max(0.001)).clamp(0.0, 1.0);
            let sigma = sigma_max - norm_grad * (sigma_max - sigma_min);
            gaussians.push(Gaussian2D::new(Vector2::new(x, y), Euler::isotropic(sigma), color, 1.0));
        }
    }

    while gaussians.len() < count {
        let x: f32 = rng.gen();
        let y: f32 = rng.gen();
        let px = (x * width as f32).min(width as f32 - 1.0) as u32;
        let py = (y * height as f32).min(height as f32 - 1.0) as u32;
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
        gaussians.push(Gaussian2D::new(Vector2::new(x, y), Euler::isotropic(0.04), color, 1.0));
    }

    gaussians
}
