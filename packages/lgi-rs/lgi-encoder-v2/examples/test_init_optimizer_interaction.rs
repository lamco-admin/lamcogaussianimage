//! Initialization × Optimizer Interaction Test
//!
//! Explores the interaction between initialization strategies and optimizers.
//! Key hypotheses:
//!
//! 1. Different initializations may favor different optimizers:
//!    - Uniform grid: Clean gradients, might favor L-BFGS
//!    - Jittered grid: More exploration, might favor Adam's adaptivity
//!    - Gradient-weighted: Starts closer to solution, might benefit from L-M's curvature
//!
//! 2. The "best" optimizer might depend on initialization quality:
//!    - Good init: Second-order methods can converge faster
//!    - Poor init: First-order methods more robust to bad starting points
//!
//! 3. Time-to-quality trade-offs differ across combinations

use lgi_core::ImageBuffer;
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_gpu::{GpuAdamOptimizer, GpuLbfgsOptimizer, GpuLMOptimizer};
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
    println!("   INITIALIZATION × OPTIMIZER INTERACTION TEST");
    println!("   Exploring synergies between init strategies and optimizers");
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

    let n_gaussians = 256;

    // Initialization strategies
    let init_strategies: Vec<(&str, Box<dyn Fn() -> Vec<Gaussian2D<f32, Euler<f32>>>>)> = vec![
        ("Uniform Grid", Box::new(|| init_uniform_grid(width, height, n_gaussians, &target))),
        ("Jittered Grid", Box::new(|| init_jittered_grid(width, height, n_gaussians, &target))),
        ("Gradient-Weighted", Box::new(|| init_gradient_weighted(width, height, n_gaussians, &target))),
    ];

    // Optimizer configs: (name, iterations, run_fn)
    type OptFn = Box<dyn Fn(&wgpu::Device, &wgpu::Queue, &mut Vec<Gaussian2D<f32, Euler<f32>>>, &ImageBuffer<f32>)>;
    let optimizers: Vec<(&str, usize, OptFn)> = vec![
        ("Adam", 200, Box::new(|dev, queue, g, t| {
            let mut opt = GpuAdamOptimizer::new(dev).unwrap();
            opt.max_iterations = 200;
            let _ = opt.optimize(dev, queue, g, t);
        })),
        ("L-BFGS", 100, Box::new(|dev, queue, g, t| {
            let mut opt = GpuLbfgsOptimizer::new(dev).unwrap();
            opt.config.max_iterations = 100;
            let _ = opt.optimize(dev, queue, g, t);
        })),
        ("L-M", 50, Box::new(|dev, queue, g, t| {
            let opt = GpuLMOptimizer::new(dev).unwrap();
            let _ = opt.optimize(dev, queue, g, t);
        })),
    ];

    // Results matrix
    let mut results: Vec<(String, String, f32, f32, f32, f64)> = Vec::new();

    for (init_name, init_fn) in &init_strategies {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Initialization: {}", init_name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // Create base init and measure initial quality
        let base_gaussians = init_fn();
        let rendered = RendererV2::render(&base_gaussians, width, height);
        let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
        println!("  Initial PSNR: {:.2}dB\n", init_psnr);

        for (opt_name, iterations, opt_fn) in &optimizers {
            println!("  Testing {} ({} iterations)...", opt_name, iterations);

            let mut gaussians = base_gaussians.clone();
            let start = Instant::now();
            opt_fn(&device, &queue, &mut gaussians, &target);
            let elapsed = start.elapsed();

            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

            // Save result
            let clean_init = init_name.to_lowercase().replace(" ", "_").replace("-", "_");
            let clean_opt = opt_name.to_lowercase().replace(" ", "_").replace("-", "_");
            let mut result = TestResult::new(
                &format!("init_opt_{}_{}", clean_init, clean_opt),
                image_path
            );
            result.optimizer = format!("{} + {}", init_name, opt_name);
            result.n_gaussians = n_gaussians;
            result.iterations = *iterations;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = final_psnr - init_psnr;
            result.elapsed_seconds = elapsed.as_secs_f64() as f32;
            result.image_dimensions = (width, height);
            result.set_param("init_strategy", init_name.to_string());
            result.complete();
            let _ = result.save_with_image(&final_render, RESULTS_DIR);

            println!("    Result: {:.2}dB (Δ={:+.2}dB) in {:.1}s",
                     final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

            results.push((
                init_name.to_string(),
                opt_name.to_string(),
                init_psnr,
                final_psnr,
                final_psnr - init_psnr,
                elapsed.as_secs_f64()
            ));
        }
        println!();
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    INTERACTION MATRIX SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Matrix display
    println!("  Final PSNR (dB):");
    println!("  {:20} {:>12} {:>12} {:>12}", "Init \\ Optimizer", "Adam", "L-BFGS", "L-M");
    println!("  {:->20} {:->12} {:->12} {:->12}", "", "", "", "");

    for init_name in ["Uniform Grid", "Jittered Grid", "Gradient-Weighted"] {
        let adam = results.iter().find(|(i, o, _, _, _, _)| i == init_name && o == "Adam");
        let lbfgs = results.iter().find(|(i, o, _, _, _, _)| i == init_name && o == "L-BFGS");
        let lm = results.iter().find(|(i, o, _, _, _, _)| i == init_name && o == "L-M");

        let adam_psnr = adam.map(|r| r.3).unwrap_or(0.0);
        let lbfgs_psnr = lbfgs.map(|r| r.3).unwrap_or(0.0);
        let lm_psnr = lm.map(|r| r.3).unwrap_or(0.0);

        println!("  {:20} {:>10.2}dB {:>10.2}dB {:>10.2}dB",
                 init_name, adam_psnr, lbfgs_psnr, lm_psnr);
    }

    // Improvement matrix
    println!("\n  Improvement (dB):");
    println!("  {:20} {:>12} {:>12} {:>12}", "Init \\ Optimizer", "Adam", "L-BFGS", "L-M");
    println!("  {:->20} {:->12} {:->12} {:->12}", "", "", "", "");

    for init_name in ["Uniform Grid", "Jittered Grid", "Gradient-Weighted"] {
        let adam = results.iter().find(|(i, o, _, _, _, _)| i == init_name && o == "Adam");
        let lbfgs = results.iter().find(|(i, o, _, _, _, _)| i == init_name && o == "L-BFGS");
        let lm = results.iter().find(|(i, o, _, _, _, _)| i == init_name && o == "L-M");

        let adam_imp = adam.map(|r| r.4).unwrap_or(0.0);
        let lbfgs_imp = lbfgs.map(|r| r.4).unwrap_or(0.0);
        let lm_imp = lm.map(|r| r.4).unwrap_or(0.0);

        println!("  {:20} {:>+10.2}dB {:>+10.2}dB {:>+10.2}dB",
                 init_name, adam_imp, lbfgs_imp, lm_imp);
    }

    // Find best combinations
    println!("\n  Analysis:");

    // Best overall
    let best = results.iter().max_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap();
    println!("  Best overall: {} + {} = {:.2}dB", best.0, best.1, best.3);

    // Best per init
    for init_name in ["Uniform Grid", "Jittered Grid", "Gradient-Weighted"] {
        let best_for_init = results.iter()
            .filter(|(i, _, _, _, _, _)| i == init_name)
            .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap());
        if let Some(b) = best_for_init {
            println!("  Best for {}: {} ({:.2}dB)", init_name, b.1, b.3);
        }
    }

    // Best per optimizer
    println!();
    for opt_name in ["Adam", "L-BFGS", "L-M"] {
        let best_for_opt = results.iter()
            .filter(|(_, o, _, _, _, _)| o == opt_name)
            .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap());
        if let Some(b) = best_for_opt {
            println!("  Best for {}: {} ({:.2}dB)", opt_name, b.0, b.3);
        }
    }

    // Synergy analysis
    println!("\n  Synergy observations:");

    // Check if gradient-weighted helps L-M more than Adam
    let gw_adam = results.iter().find(|(i, o, _, _, _, _)| i == "Gradient-Weighted" && o == "Adam");
    let gw_lm = results.iter().find(|(i, o, _, _, _, _)| i == "Gradient-Weighted" && o == "L-M");
    let uni_adam = results.iter().find(|(i, o, _, _, _, _)| i == "Uniform Grid" && o == "Adam");
    let uni_lm = results.iter().find(|(i, o, _, _, _, _)| i == "Uniform Grid" && o == "L-M");

    if let (Some(ga), Some(gm), Some(ua), Some(um)) = (gw_adam, gw_lm, uni_adam, uni_lm) {
        let gw_lift_adam = ga.3 - ua.3;
        let gw_lift_lm = gm.3 - um.3;
        if gw_lift_lm > gw_lift_adam + 0.1 {
            println!("  → Gradient-weighted init helps L-M more than Adam");
            println!("    (L-M: {:+.2}dB vs Adam: {:+.2}dB)", gw_lift_lm, gw_lift_adam);
        } else if gw_lift_adam > gw_lift_lm + 0.1 {
            println!("  → Gradient-weighted init helps Adam more than L-M");
            println!("    (Adam: {:+.2}dB vs L-M: {:+.2}dB)", gw_lift_adam, gw_lift_lm);
        } else {
            println!("  → Gradient-weighted init benefits all optimizers similarly");
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn init_uniform_grid(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut gaussians = Vec::with_capacity(count);
    let cols = (count as f32).sqrt().ceil() as u32;
    let rows = (count as f32 / cols as f32).ceil() as u32;
    let cell_w = width as f32 / cols as f32;
    let cell_h = height as f32 / rows as f32;

    for i in 0..count {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;
        let px = ((col as f32 + 0.5) * cell_w) as u32;
        let py = ((row as f32 + 0.5) * cell_h) as u32;
        let px = px.min(width - 1);
        let py = py.min(height - 1);

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
        gaussians.push(Gaussian2D::new(position, Euler::isotropic(0.04), color, 1.0));
    }
    gaussians
}

fn init_jittered_grid(
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
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);

    // Compute gradient magnitude for each pixel
    let mut gradient_sum = 0.0f32;
    let mut gradient_map = vec![0.0f32; (width * height) as usize];

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = (y * width + x) as usize;
            let c = target.get_pixel(x, y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
            let cx1 = target.get_pixel(x + 1, y).unwrap_or(c);
            let cx0 = target.get_pixel(x - 1, y).unwrap_or(c);
            let cy1 = target.get_pixel(x, y + 1).unwrap_or(c);
            let cy0 = target.get_pixel(x, y - 1).unwrap_or(c);

            let gx = ((cx1.r - cx0.r).powi(2) + (cx1.g - cx0.g).powi(2) + (cx1.b - cx0.b).powi(2)).sqrt();
            let gy = ((cy1.r - cy0.r).powi(2) + (cy1.g - cy0.g).powi(2) + (cy1.b - cy0.b).powi(2)).sqrt();
            let grad = (gx * gx + gy * gy).sqrt() + 0.01; // Small bias for uniform coverage

            gradient_map[idx] = grad;
            gradient_sum += grad;
        }
    }

    // Sample positions weighted by gradient
    let mut gaussians = Vec::with_capacity(count);
    for _ in 0..count {
        let r = rng.gen_range(0.0..gradient_sum);
        let mut cumsum = 0.0f32;
        let mut px = width / 2;
        let mut py = height / 2;

        'outer: for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                cumsum += gradient_map[(y * width + x) as usize];
                if cumsum >= r {
                    px = x;
                    py = y;
                    break 'outer;
                }
            }
        }

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));
        gaussians.push(Gaussian2D::new(position, Euler::isotropic(0.04), color, 1.0));
    }

    gaussians
}
