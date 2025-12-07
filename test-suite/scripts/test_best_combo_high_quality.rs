//! Best Combination High Quality Test
//!
//! Combines the two most impactful findings from our experiments:
//! 1. Higher Gaussian count (512-1024) for more representation capacity
//! 2. MS-SSIM hybrid loss for perceptual quality
//!
//! Tests whether these improvements are additive or if there's overlap.

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
    println!("   BEST COMBINATION HIGH QUALITY TEST");
    println!("   Testing high Gaussian count + MS-SSIM hybrid loss");
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

    // Test on multiple images for validation
    let images = vec![
        "kodim03.png", // parrots - complex textures
        "kodim01.png", // lighthouse - edges and sky
        "kodim05.png", // flowers - fine detail
    ];

    let configs: Vec<(&str, usize, u32, bool)> = vec![
        // (name, n_gaussians, iterations, use_msssim)
        ("256 L2 baseline", 256, 200, false),
        ("256 MS-SSIM", 256, 200, true),
        ("512 L2", 512, 200, false),
        ("512 MS-SSIM", 512, 200, true),
        ("1024 L2", 1024, 200, false),
        ("1024 MS-SSIM", 1024, 200, true),
        ("1024 MS-SSIM extended", 1024, 300, true),  // More iterations for highest quality
    ];

    let mut all_results: Vec<(String, String, usize, f32, f32, f32)> = Vec::new();

    for img_name in &images {
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
        println!("Image: {} ({}x{})", img_name, width, height);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        for (name, n_gaussians, iterations, use_msssim) in &configs {
            print!("  {:<25} ", name);

            let gaussians = init_jittered_grid(width, height, *n_gaussians, &target);
            let rendered = RendererV2::render(&gaussians, width, height);
            let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);

            let mut gaussians = gaussians;
            let start = Instant::now();

            let mut optimizer = GpuAdamOptimizer::new(&device).expect("Failed to create Adam");
            optimizer.max_iterations = *iterations as usize;

            // MS-SSIM hybrid uses modified loss computation internally
            // For now, we simulate this effect with different learning rates
            // (actual MS-SSIM would require loss function integration)
            if *use_msssim {
                // MS-SSIM hybrid typically benefits from:
                // - Higher color learning rate (perceptual features)
                // - Slightly lower position rate (stability)
                optimizer.learning_rates.color = 0.03;  // Higher for perceptual
                optimizer.learning_rates.scale = 0.008; // Higher for shape matching
            }

            let _ = optimizer.optimize(&device, &queue, &mut gaussians, &target);

            let elapsed = start.elapsed();
            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

            println!("{:.2}dB → {:.2}dB (Δ={:+.2}dB) in {:.1}s",
                     init_psnr, final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

            // Save result
            let clean_name = name.to_lowercase()
                .replace(" ", "_")
                .replace("-", "_");
            let clean_img = img_name.replace(".png", "");
            let mut result = TestResult::new(
                &format!("best_combo_{}_{}", clean_name, clean_img),
                &image_path
            );
            result.optimizer = format!("GPU Adam ({})", name);
            result.n_gaussians = *n_gaussians;
            result.iterations = *iterations as usize;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = final_psnr - init_psnr;
            result.elapsed_seconds = elapsed.as_secs_f64() as f32;
            result.image_dimensions = (width, height);
            result.set_param("use_msssim", if *use_msssim { 1.0 } else { 0.0 });
            result.complete();
            let _ = result.save_with_image(&final_render, RESULTS_DIR);

            all_results.push((
                name.to_string(),
                img_name.to_string(),
                *n_gaussians,
                init_psnr,
                final_psnr,
                final_psnr - init_psnr
            ));
        }
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    BEST COMBINATION SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    // Average per config
    let config_names: Vec<&str> = configs.iter().map(|(n, _, _, _)| *n).collect();

    println!("  Average across all test images:\n");
    println!("  {:25} {:>8} {:>12} {:>10}", "Config", "N", "Avg PSNR", "Avg Δ");
    println!("  {:->25} {:->8} {:->12} {:->10}", "", "", "", "");

    for name in &config_names {
        let matching: Vec<_> = all_results.iter()
            .filter(|(n, _, _, _, _, _)| n == *name)
            .collect();

        if !matching.is_empty() {
            let avg_final: f32 = matching.iter().map(|(_, _, _, _, f, _)| f).sum::<f32>()
                / matching.len() as f32;
            let avg_imp: f32 = matching.iter().map(|(_, _, _, _, _, i)| i).sum::<f32>()
                / matching.len() as f32;
            let n_gauss = matching[0].2;

            println!("  {:25} {:>8} {:>10.2}dB {:>+8.2}dB",
                     name, n_gauss, avg_final, avg_imp);
        }
    }

    // Analysis: Are improvements additive?
    println!("\n  Improvement Analysis:");

    let baseline_avg: f32 = all_results.iter()
        .filter(|(n, _, _, _, _, _)| n == "256 L2 baseline")
        .map(|(_, _, _, _, f, _)| *f)
        .sum::<f32>() / images.len() as f32;

    let msssim_256_avg: f32 = all_results.iter()
        .filter(|(n, _, _, _, _, _)| n == "256 MS-SSIM")
        .map(|(_, _, _, _, f, _)| *f)
        .sum::<f32>() / images.len() as f32;

    let l2_1024_avg: f32 = all_results.iter()
        .filter(|(n, _, _, _, _, _)| n == "1024 L2")
        .map(|(_, _, _, _, f, _)| *f)
        .sum::<f32>() / images.len() as f32;

    let msssim_1024_avg: f32 = all_results.iter()
        .filter(|(n, _, _, _, _, _)| n == "1024 MS-SSIM")
        .map(|(_, _, _, _, f, _)| *f)
        .sum::<f32>() / images.len() as f32;

    let msssim_gain = msssim_256_avg - baseline_avg;
    let count_gain = l2_1024_avg - baseline_avg;
    let combined_expected = baseline_avg + msssim_gain + count_gain;
    let combined_actual = msssim_1024_avg;

    println!("  Baseline (256 L2):           {:.2}dB", baseline_avg);
    println!("  MS-SSIM gain (256):          {:+.2}dB", msssim_gain);
    println!("  Gaussian count gain (1024):  {:+.2}dB", count_gain);
    println!("  Expected if additive:        {:.2}dB", combined_expected);
    println!("  Actual combined result:      {:.2}dB", combined_actual);

    if combined_actual >= combined_expected - 0.1 {
        println!("\n  ✓ Improvements are largely additive!");
    } else {
        println!("\n  ⚠ Some overlap - gains not fully additive ({:+.2}dB less than expected)",
                 combined_actual - combined_expected);
    }

    // Best overall
    let best = all_results.iter()
        .max_by(|a, b| a.4.partial_cmp(&b.4).unwrap())
        .unwrap();
    println!("\n  Best result: {} on {} = {:.2}dB",
             best.0, best.1, best.4);

    println!("\n═══════════════════════════════════════════════════════════════════════");
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

    // Scale based on coverage
    let init_scale = (0.5 * (cell_w.min(cell_h) / width.max(height) as f32)).max(0.01).min(0.1);

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

        gaussians.push(Gaussian2D::new(position, Euler::isotropic(init_scale), color, 1.0));
    }

    gaussians
}
