//! Optimized Configuration Test
//!
//! Tests combinations of the best hyperparameters discovered from sweeps:
//! - Best initial scale
//! - Best learning rate ratios
//! - Best Adam beta values
//! - Optimal iteration count
//!
//! This validates that improvements from individual sweeps are additive.

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
    println!("   OPTIMIZED CONFIGURATION TEST");
    println!("   Testing combinations of best-known hyperparameters");
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

    // Test on multiple Kodak images for validation
    let images = vec![
        ("kodim03.png", "parrots"),
        ("kodim01.png", "lighthouse"),
        ("kodim05.png", "flowers"),
        ("kodim08.png", "castle"),
    ];

    let n_gaussians = 256;
    let iterations = 300; // Extended iterations for quality

    // Configuration variants
    struct Config {
        name: &'static str,
        init_scale: f32,
        lr_position: f32,
        lr_position_final: f32,
        lr_color: f32,
        lr_scale: f32,
        lr_opacity: f32,
        lr_rotation: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    }

    let configs = vec![
        Config {
            name: "Baseline (default)",
            init_scale: 0.04,
            lr_position: 0.0002,
            lr_position_final: 0.00002,
            lr_color: 0.02,
            lr_scale: 0.005,
            lr_opacity: 0.05,
            lr_rotation: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        Config {
            name: "Small init scale",
            init_scale: 0.02,  // From scale sweep - smaller sometimes better
            lr_position: 0.0002,
            lr_position_final: 0.00002,
            lr_color: 0.02,
            lr_scale: 0.005,
            lr_opacity: 0.05,
            lr_rotation: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        Config {
            name: "Higher position LR",
            init_scale: 0.04,
            lr_position: 0.001,  // 5x higher position LR
            lr_position_final: 0.0001,
            lr_color: 0.02,
            lr_scale: 0.005,
            lr_opacity: 0.05,
            lr_rotation: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        Config {
            name: "3DGS-inspired",
            init_scale: 0.04,
            lr_position: 0.00016,
            lr_position_final: 0.0000016,
            lr_color: 0.025,
            lr_scale: 0.005,
            lr_opacity: 0.05,
            lr_rotation: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-15,  // 3DGS uses very small epsilon
        },
        Config {
            name: "High momentum (β1=0.95)",
            init_scale: 0.04,
            lr_position: 0.0002,
            lr_position_final: 0.00002,
            lr_color: 0.02,
            lr_scale: 0.005,
            lr_opacity: 0.05,
            lr_rotation: 0.01,
            beta1: 0.95,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        Config {
            name: "Optimized combo A",
            init_scale: 0.02,  // Small init
            lr_position: 0.0005,  // Higher position
            lr_position_final: 0.00005,
            lr_color: 0.03,  // Slightly higher color
            lr_scale: 0.005,
            lr_opacity: 0.05,
            lr_rotation: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        Config {
            name: "Optimized combo B",
            init_scale: 0.03,  // Medium init
            lr_position: 0.0003,
            lr_position_final: 0.00003,
            lr_color: 0.02,
            lr_scale: 0.008,  // Higher scale LR
            lr_opacity: 0.08,  // Higher opacity LR
            lr_rotation: 0.02,  // Higher rotation LR
            beta1: 0.9,
            beta2: 0.99,  // Lower β2 for faster adaptation
            epsilon: 1e-7,
        },
        Config {
            name: "Aggressive learning",
            init_scale: 0.04,
            lr_position: 0.001,
            lr_position_final: 0.0001,
            lr_color: 0.05,
            lr_scale: 0.01,
            lr_opacity: 0.1,
            lr_rotation: 0.02,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
    ];

    let mut all_results: Vec<(String, String, f32, f32, f32)> = Vec::new();

    for (img_name, img_desc) in &images {
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

        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Image: {} ({}) - {}x{}", img_name, img_desc, width, height);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for config in &configs {
            print!("  {:<25} ", config.name);

            let gaussians = init_with_scale(width, height, n_gaussians, &target, config.init_scale);
            let rendered = RendererV2::render(&gaussians, width, height);
            let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);

            let mut gaussians = gaussians;
            let start = Instant::now();

            let mut optimizer = GpuAdamOptimizer::new(&device).expect("Failed to create Adam");
            optimizer.max_iterations = iterations;
            optimizer.learning_rates.position = config.lr_position;
            optimizer.learning_rates.position_final = config.lr_position_final;
            optimizer.learning_rates.color = config.lr_color;
            optimizer.learning_rates.scale = config.lr_scale;
            optimizer.learning_rates.opacity = config.lr_opacity;
            optimizer.learning_rates.rotation = config.lr_rotation;
            optimizer.beta1 = config.beta1;
            optimizer.beta2 = config.beta2;
            optimizer.epsilon = config.epsilon;

            let _ = optimizer.optimize(&device, &queue, &mut gaussians, &target);

            let elapsed = start.elapsed();
            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

            println!("{:.2}dB → {:.2}dB (Δ={:+.2}dB) in {:.1}s",
                     init_psnr, final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

            // Save result
            let clean_name = config.name.to_lowercase()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("=", "_")
                .replace(".", "p");
            let clean_img = img_name.replace(".png", "");
            let mut result = TestResult::new(
                &format!("optimized_{}_{}", clean_name, clean_img),
                &image_path
            );
            result.optimizer = format!("GPU Adam ({})", config.name);
            result.n_gaussians = n_gaussians;
            result.iterations = iterations;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = final_psnr - init_psnr;
            result.elapsed_seconds = elapsed.as_secs_f64() as f32;
            result.image_dimensions = (width, height);
            result.set_param("init_scale", config.init_scale as f64);
            result.set_param("lr_position", config.lr_position as f64);
            result.set_param("lr_color", config.lr_color as f64);
            result.set_param("lr_scale", config.lr_scale as f64);
            result.set_param("beta1", config.beta1 as f64);
            result.set_param("beta2", config.beta2 as f64);
            result.set_param("epsilon", config.epsilon as f64);
            result.complete();
            let _ = result.save_with_image(&final_render, RESULTS_DIR);

            all_results.push((
                config.name.to_string(),
                img_name.to_string(),
                init_psnr,
                final_psnr,
                final_psnr - init_psnr
            ));
        }
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    OPTIMIZED CONFIGURATION SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Average results per config
    let config_names: Vec<&str> = configs.iter().map(|c| c.name).collect();

    println!("  Average improvement across all test images:\n");
    println!("  {:25} {:>12} {:>12}", "Config", "Avg Final", "Avg Δ");
    println!("  {:->25} {:->12} {:->12}", "", "", "");

    for name in &config_names {
        let matching: Vec<_> = all_results.iter()
            .filter(|(n, _, _, _, _)| n == *name)
            .collect();

        if !matching.is_empty() {
            let avg_final: f32 = matching.iter().map(|(_, _, _, f, _)| f).sum::<f32>()
                / matching.len() as f32;
            let avg_imp: f32 = matching.iter().map(|(_, _, _, _, i)| i).sum::<f32>()
                / matching.len() as f32;

            println!("  {:25} {:>10.2}dB {:>+10.2}dB", name, avg_final, avg_imp);
        }
    }

    // Find best overall config
    let mut config_scores: Vec<(&str, f32)> = Vec::new();
    for name in &config_names {
        let avg: f32 = all_results.iter()
            .filter(|(n, _, _, _, _)| n == *name)
            .map(|(_, _, _, f, _)| *f)
            .sum::<f32>() / images.len() as f32;
        config_scores.push((name, avg));
    }

    config_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n  Best configuration: {} (avg {:.2}dB)",
             config_scores[0].0, config_scores[0].1);

    let baseline_avg = config_scores.iter()
        .find(|(n, _)| n.contains("Baseline"))
        .map(|(_, a)| *a)
        .unwrap_or(0.0);

    println!("  Improvement over baseline: {:+.2}dB",
             config_scores[0].1 - baseline_avg);

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn init_with_scale(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    scale: f32,
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

        gaussians.push(Gaussian2D::new(position, Euler::isotropic(scale), color, 1.0));
    }

    gaussians
}
