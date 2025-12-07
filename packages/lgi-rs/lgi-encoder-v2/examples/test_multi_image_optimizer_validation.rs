//! Multi-Image Optimizer Validation
//!
//! Tests all three GPU optimizers across multiple Kodak images to get
//! robust statistics about which optimizer truly performs best.
//!
//! Single-image tests can be misleading - one image might favor one optimizer
//! due to specific content characteristics. This test validates across:
//! - kodim03 (parrots - vibrant colors, fine feather detail)
//! - kodim04 (barn - natural textures, gradients)
//! - kodim05 (architecture - sharp edges, geometric)
//! - kodim23 (lighthouse - sky gradients, hard edges)

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
    println!("   MULTI-IMAGE OPTIMIZER VALIDATION");
    println!("   Comparing Adam, L-BFGS, L-M across diverse images");
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

    // Test images with different characteristics
    let test_images = vec![
        ("kodim03.png", "parrots - fine detail, vibrant colors"),
        ("kodim04.png", "barn - natural textures"),
        ("kodim05.png", "architecture - geometric, sharp edges"),
        ("kodim23.png", "lighthouse - gradients, hard edges"),
    ];

    let n_gaussians = 256;
    let iterations_adam = 200;
    let iterations_lbfgs = 100;
    let iterations_lm = 50;

    // Results storage: (image, optimizer) -> (init_psnr, final_psnr, improvement, time)
    let mut all_results: Vec<(String, String, f32, f32, f32, f64)> = Vec::new();

    for (image_name, description) in &test_images {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Image: {} ({})", image_name, description);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let image_path = format!("test-data/kodak-dataset/{}", image_name);
        let full_path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), image_path);

        let img = match image::open(&full_path) {
            Ok(img) => img.to_rgb32f(),
            Err(e) => {
                println!("  Skipping {} - failed to load: {}", image_name, e);
                continue;
            }
        };

        let width = img.width();
        let height = img.height();

        let mut target = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                target.set_pixel(x, y, Color4::new(pixel[0], pixel[1], pixel[2], 1.0));
            }
        }

        // Create base initialization (same seed for all optimizers)
        let base_gaussians = init_jittered_grid(width, height, n_gaussians, &target, 42);
        let rendered = RendererV2::render(&base_gaussians, width, height);
        let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
        println!("  Initial: {} Gaussians, PSNR={:.2}dB\n", n_gaussians, init_psnr);

        // Test Adam
        println!("  Testing GPU Adam ({} iterations)...", iterations_adam);
        let (adam_final, adam_time) = {
            let mut gaussians = base_gaussians.clone();
            let mut optimizer = GpuAdamOptimizer::new(&device).expect("Failed to create Adam");
            optimizer.max_iterations = iterations_adam;

            let start = Instant::now();
            let _ = optimizer.optimize(&device, &queue, &mut gaussians, &target);
            let elapsed = start.elapsed();

            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

            // Save result
            let mut result = TestResult::new(
                &format!("multi_img_adam_{}", image_name.replace(".png", "")),
                &image_path
            );
            result.optimizer = "GPU Adam".to_string();
            result.n_gaussians = n_gaussians;
            result.iterations = iterations_adam;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = final_psnr - init_psnr;
            result.elapsed_seconds = elapsed.as_secs_f64() as f32;
            result.image_dimensions = (width, height);
            result.complete();
            let _ = result.save_with_image(&final_render, RESULTS_DIR);

            println!("    Result: {:.2}dB (Δ={:+.2}dB) in {:.1}s",
                     final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());
            (final_psnr, elapsed.as_secs_f64())
        };
        all_results.push((image_name.to_string(), "Adam".to_string(),
                         init_psnr, adam_final, adam_final - init_psnr, adam_time));

        // Test L-BFGS
        println!("  Testing GPU L-BFGS ({} iterations)...", iterations_lbfgs);
        let (lbfgs_final, lbfgs_time) = {
            let mut gaussians = base_gaussians.clone();
            let mut optimizer = GpuLbfgsOptimizer::new(&device).expect("Failed to create L-BFGS");
            optimizer.config.max_iterations = iterations_lbfgs;

            let start = Instant::now();
            let _ = optimizer.optimize(&device, &queue, &mut gaussians, &target);
            let elapsed = start.elapsed();

            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

            let mut result = TestResult::new(
                &format!("multi_img_lbfgs_{}", image_name.replace(".png", "")),
                &image_path
            );
            result.optimizer = "GPU L-BFGS".to_string();
            result.n_gaussians = n_gaussians;
            result.iterations = iterations_lbfgs;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = final_psnr - init_psnr;
            result.elapsed_seconds = elapsed.as_secs_f64() as f32;
            result.image_dimensions = (width, height);
            result.set_param("history_size", optimizer.config.history_size as f64);
            result.complete();
            let _ = result.save_with_image(&final_render, RESULTS_DIR);

            println!("    Result: {:.2}dB (Δ={:+.2}dB) in {:.1}s",
                     final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());
            (final_psnr, elapsed.as_secs_f64())
        };
        all_results.push((image_name.to_string(), "L-BFGS".to_string(),
                         init_psnr, lbfgs_final, lbfgs_final - init_psnr, lbfgs_time));

        // Test L-M
        println!("  Testing GPU L-M ({} iterations)...", iterations_lm);
        let (lm_final, lm_time) = {
            let mut gaussians = base_gaussians.clone();
            let mut optimizer = GpuLMOptimizer::new(&device).expect("Failed to create L-M");

            let start = Instant::now();
            let _ = optimizer.optimize(&device, &queue, &mut gaussians, &target);
            let elapsed = start.elapsed();

            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

            let mut result = TestResult::new(
                &format!("multi_img_lm_{}", image_name.replace(".png", "")),
                &image_path
            );
            result.optimizer = "GPU L-M".to_string();
            result.n_gaussians = n_gaussians;
            result.iterations = iterations_lm;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = final_psnr - init_psnr;
            result.elapsed_seconds = elapsed.as_secs_f64() as f32;
            result.image_dimensions = (width, height);
            result.complete();
            let _ = result.save_with_image(&final_render, RESULTS_DIR);

            println!("    Result: {:.2}dB (Δ={:+.2}dB) in {:.1}s",
                     final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());
            (final_psnr, elapsed.as_secs_f64())
        };
        all_results.push((image_name.to_string(), "L-M".to_string(),
                         init_psnr, lm_final, lm_final - init_psnr, lm_time));

        println!();
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    MULTI-IMAGE SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Per-image comparison
    println!("  Per-Image Results:");
    println!("  {:12} {:>10} {:>10} {:>10} {:>10}", "Image", "Adam", "L-BFGS", "L-M", "Winner");
    println!("  {:->12} {:->10} {:->10} {:->10} {:->10}", "", "", "", "", "");

    let images_seen: Vec<String> = test_images.iter().map(|(n, _)| n.to_string()).collect();
    for image_name in &images_seen {
        let adam = all_results.iter()
            .find(|(img, opt, _, _, _, _)| img == image_name && opt == "Adam")
            .map(|(_, _, _, final_psnr, _, _)| *final_psnr);
        let lbfgs = all_results.iter()
            .find(|(img, opt, _, _, _, _)| img == image_name && opt == "L-BFGS")
            .map(|(_, _, _, final_psnr, _, _)| *final_psnr);
        let lm = all_results.iter()
            .find(|(img, opt, _, _, _, _)| img == image_name && opt == "L-M")
            .map(|(_, _, _, final_psnr, _, _)| *final_psnr);

        if let (Some(a), Some(l), Some(m)) = (adam, lbfgs, lm) {
            let winner = if a > l && a > m { "Adam" }
                        else if l > a && l > m { "L-BFGS" }
                        else { "L-M" };
            println!("  {:12} {:>8.2}dB {:>8.2}dB {:>8.2}dB {:>10}",
                     image_name.replace(".png", ""), a, l, m, winner);
        }
    }

    // Aggregate statistics
    println!("\n  Aggregate Statistics:");
    for optimizer in &["Adam", "L-BFGS", "L-M"] {
        let results: Vec<_> = all_results.iter()
            .filter(|(_, opt, _, _, _, _)| opt == *optimizer)
            .collect();

        if results.is_empty() { continue; }

        let avg_improvement: f32 = results.iter()
            .map(|(_, _, _, _, imp, _)| imp)
            .sum::<f32>() / results.len() as f32;
        let avg_time: f64 = results.iter()
            .map(|(_, _, _, _, _, time)| time)
            .sum::<f64>() / results.len() as f64;
        let max_improvement = results.iter()
            .map(|(_, _, _, _, imp, _)| *imp)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_improvement = results.iter()
            .map(|(_, _, _, _, imp, _)| *imp)
            .fold(f32::INFINITY, f32::min);

        println!("  {:8}: avg={:+.2}dB, range=[{:+.2}, {:+.2}], time={:.1}s",
                 optimizer, avg_improvement, min_improvement, max_improvement, avg_time);
    }

    // Winner analysis
    let adam_wins = all_results.iter()
        .filter(|(_, opt, _, _, _, _)| opt == "Adam")
        .count();
    let _lbfgs_wins = all_results.iter()
        .filter(|(_, opt, _, _, _, _)| opt == "L-BFGS")
        .count();
    let _lm_wins = all_results.iter()
        .filter(|(_, opt, _, _, _, _)| opt == "L-M")
        .count();

    // Count wins per optimizer across images
    let mut adam_win_count = 0;
    let mut lbfgs_win_count = 0;
    let mut lm_win_count = 0;

    for image_name in &images_seen {
        let adam = all_results.iter()
            .find(|(img, opt, _, _, _, _)| img == image_name && opt == "Adam")
            .map(|(_, _, _, final_psnr, _, _)| *final_psnr);
        let lbfgs = all_results.iter()
            .find(|(img, opt, _, _, _, _)| img == image_name && opt == "L-BFGS")
            .map(|(_, _, _, final_psnr, _, _)| *final_psnr);
        let lm = all_results.iter()
            .find(|(img, opt, _, _, _, _)| img == image_name && opt == "L-M")
            .map(|(_, _, _, final_psnr, _, _)| *final_psnr);

        if let (Some(a), Some(l), Some(m)) = (adam, lbfgs, lm) {
            if a > l && a > m { adam_win_count += 1; }
            else if l > a && l > m { lbfgs_win_count += 1; }
            else { lm_win_count += 1; }
        }
    }

    println!("\n  Win Count: Adam={}, L-BFGS={}, L-M={}",
             adam_win_count, lbfgs_win_count, lm_win_count);

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn init_jittered_grid(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    seed: u64,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(seed);
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
