//! Test Color Sampling Strategies
//!
//! Hypothesis: The initial color sampling strategy affects optimization.
//! Options:
//! 1. Point sample: Color from exact position (current approach)
//! 2. Average sample: Average color in Gaussian footprint
//! 3. Median sample: Median color (robust to outliers)
//! 4. Weighted sample: Gaussian-weighted average
//!
//! The point sample may miss the "dominant" color for the Gaussian coverage.

use lgi_core::ImageBuffer;
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
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
    println!("   COLOR SAMPLING STRATEGIES TEST - Initial Color Assignment");
    println!("═══════════════════════════════════════════════════════════════════════\n");

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

    let n_gaussians = 256;
    let iterations = 200;
    let sigma = 0.04; // Sigma for all Gaussians

    // Test 1: Point sample (baseline)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: POINT SAMPLE - Single pixel color");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let point_result = run_encoding(
        "point_sample",
        &target,
        || init_with_color_strategy(width, height, n_gaussians, &target, sigma, ColorStrategy::Point),
        iterations,
        image_path,
    );

    // Test 2: Average sample (box filter)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: AVERAGE SAMPLE - Box-averaged color in footprint");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let average_result = run_encoding(
        "average_sample",
        &target,
        || init_with_color_strategy(width, height, n_gaussians, &target, sigma, ColorStrategy::Average),
        iterations,
        image_path,
    );

    // Test 3: Gaussian-weighted sample
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: WEIGHTED SAMPLE - Gaussian-weighted average");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let weighted_result = run_encoding(
        "weighted_sample",
        &target,
        || init_with_color_strategy(width, height, n_gaussians, &target, sigma, ColorStrategy::GaussianWeighted),
        iterations,
        image_path,
    );

    // Test 4: Median sample (robust)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 4: MEDIAN SAMPLE - Median color (robust to outliers)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let median_result = run_encoding(
        "median_sample",
        &target,
        || init_with_color_strategy(width, height, n_gaussians, &target, sigma, ColorStrategy::Median),
        iterations,
        image_path,
    );

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                    COLOR SAMPLING SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:25} {:>12} {:>12} {:>12}", "Strategy", "Initial", "Final", "Improvement");
    println!("  {:-<25} {:->12} {:->12} {:->12}", "", "", "", "");

    for (name, (init_psnr, final_psnr)) in [
        ("Point (baseline)", point_result),
        ("Average (box)", average_result),
        ("Gaussian-weighted", weighted_result),
        ("Median (robust)", median_result),
    ] {
        let improvement = final_psnr - init_psnr;
        println!("  {:25} {:>10.2}dB {:>10.2}dB {:>+10.2}dB", name, init_psnr, final_psnr, improvement);
    }

    // Find best strategy
    let results = [
        ("Point", point_result.1),
        ("Average", average_result.1),
        ("Weighted", weighted_result.1),
        ("Median", median_result.1),
    ];
    let best = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let baseline_final = point_result.1;

    println!("\n  Best strategy: {} ({:.2}dB)", best.0, best.1);
    println!("  Advantage over point: {:+.2}dB", best.1 - baseline_final);

    if best.1 - baseline_final > 0.1 {
        println!("\n  ✓ {} sampling improves initial color assignment!", best.0);
    } else {
        println!("\n  ~ Color sampling strategy has minimal impact after optimization");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

#[derive(Clone, Copy)]
enum ColorStrategy {
    Point,
    Average,
    GaussianWeighted,
    Median,
}

fn run_encoding<F>(
    name: &str,
    target: &ImageBuffer<f32>,
    init_fn: F,
    iterations: usize,
    image_path: &str,
) -> (f32, f32)
where
    F: FnOnce() -> Vec<Gaussian2D<f32, Euler<f32>>>,
{
    let width = target.width;
    let height = target.height;

    let mut result = TestResult::new(&format!("color_sampling_{}", name), image_path);
    result.optimizer = format!("Color sampling: {}", name);
    result.image_dimensions = (width, height);

    // Initialize Gaussians
    let start = Instant::now();
    let mut gaussians = init_fn();
    result.n_gaussians = gaussians.len();

    // Compute initial PSNR
    let rendered = RendererV2::render(&gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
    result.initial_psnr = init_psnr;

    println!("  Initial: {} Gaussians, PSNR={:.2}dB", gaussians.len(), init_psnr);

    // Optimize with Adam
    let mut optimizer = AdamOptimizer::new();
    optimizer.max_iterations = iterations;
    result.iterations = iterations;

    let mut last_psnr = init_psnr;
    let mut final_psnr = init_psnr;

    for iter in 0..iterations {
        optimizer.optimize(&mut gaussians, target);

        if iter % 40 == 0 || iter == iterations - 1 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = lgi_core::error_metrics::compute_psnr(target, &rendered);
            let direction = if psnr > last_psnr { "↑" } else { "↓" };
            println!("    [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                     iter + 1, iterations, psnr, direction, psnr - last_psnr);
            last_psnr = psnr;
            final_psnr = psnr;
        }
    }

    let elapsed = start.elapsed();

    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f64() as f32;
    result.complete();

    // Save result with image
    let final_render = RendererV2::render(&gaussians, width, height);
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    println!("  Final: PSNR={:.2}dB (improvement: {:+.2}dB)", final_psnr, final_psnr - init_psnr);
    println!("  Time: {:.1}s", elapsed.as_secs_f64());

    (init_psnr, final_psnr)
}

fn init_with_color_strategy(
    width: u32,
    height: u32,
    count: usize,
    target: &ImageBuffer<f32>,
    sigma: f32,
    strategy: ColorStrategy,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut gaussians = Vec::with_capacity(count);

    let cols = (count as f32).sqrt().ceil() as u32;
    let rows = (count as f32 / cols as f32).ceil() as u32;
    let cell_w = width as f32 / cols as f32;
    let cell_h = height as f32 / rows as f32;

    // Radius in pixels for sampling (based on 2 sigma)
    let sample_radius = (sigma * width.max(height) as f32 * 2.0) as i32;

    for i in 0..count {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;

        // Jittered position
        let jitter_x = rng.gen_range(-0.3..0.3);
        let jitter_y = rng.gen_range(-0.3..0.3);
        let px = ((col as f32 + 0.5 + jitter_x) * cell_w) as i32;
        let py = ((row as f32 + 0.5 + jitter_y) * cell_h) as i32;
        let px = px.clamp(0, width as i32 - 1) as u32;
        let py = py.clamp(0, height as i32 - 1) as u32;

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);

        // Sample color based on strategy
        let color = match strategy {
            ColorStrategy::Point => {
                target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0))
            }
            ColorStrategy::Average => {
                sample_average(target, px as i32, py as i32, sample_radius, width, height)
            }
            ColorStrategy::GaussianWeighted => {
                sample_gaussian_weighted(target, px as i32, py as i32, sample_radius, sigma, width, height)
            }
            ColorStrategy::Median => {
                sample_median(target, px as i32, py as i32, sample_radius, width, height)
            }
        };

        gaussians.push(Gaussian2D::new(
            position,
            Euler::isotropic(sigma),
            color,
            1.0,
        ));
    }

    gaussians
}

fn sample_average(target: &ImageBuffer<f32>, cx: i32, cy: i32, radius: i32, width: u32, height: u32) -> Color4<f32> {
    let mut sum_r = 0.0;
    let mut sum_g = 0.0;
    let mut sum_b = 0.0;
    let mut count = 0;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let x = (cx + dx).clamp(0, width as i32 - 1) as u32;
            let y = (cy + dy).clamp(0, height as i32 - 1) as u32;
            if let Some(c) = target.get_pixel(x, y) {
                sum_r += c.r;
                sum_g += c.g;
                sum_b += c.b;
                count += 1;
            }
        }
    }

    if count > 0 {
        Color4::new(sum_r / count as f32, sum_g / count as f32, sum_b / count as f32, 1.0)
    } else {
        Color4::new(0.5, 0.5, 0.5, 1.0)
    }
}

fn sample_gaussian_weighted(target: &ImageBuffer<f32>, cx: i32, cy: i32, radius: i32, sigma: f32, width: u32, height: u32) -> Color4<f32> {
    let mut sum_r = 0.0;
    let mut sum_g = 0.0;
    let mut sum_b = 0.0;
    let mut weight_sum = 0.0;

    let sigma_px = sigma * width.max(height) as f32;
    let two_sigma_sq = 2.0 * sigma_px * sigma_px;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let x = (cx + dx).clamp(0, width as i32 - 1) as u32;
            let y = (cy + dy).clamp(0, height as i32 - 1) as u32;

            let dist_sq = (dx * dx + dy * dy) as f32;
            let weight = (-dist_sq / two_sigma_sq).exp();

            if let Some(c) = target.get_pixel(x, y) {
                sum_r += c.r * weight;
                sum_g += c.g * weight;
                sum_b += c.b * weight;
                weight_sum += weight;
            }
        }
    }

    if weight_sum > 0.0 {
        Color4::new(sum_r / weight_sum, sum_g / weight_sum, sum_b / weight_sum, 1.0)
    } else {
        Color4::new(0.5, 0.5, 0.5, 1.0)
    }
}

fn sample_median(target: &ImageBuffer<f32>, cx: i32, cy: i32, radius: i32, width: u32, height: u32) -> Color4<f32> {
    let mut r_values = Vec::new();
    let mut g_values = Vec::new();
    let mut b_values = Vec::new();

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let x = (cx + dx).clamp(0, width as i32 - 1) as u32;
            let y = (cy + dy).clamp(0, height as i32 - 1) as u32;
            if let Some(c) = target.get_pixel(x, y) {
                r_values.push(c.r);
                g_values.push(c.g);
                b_values.push(c.b);
            }
        }
    }

    if r_values.is_empty() {
        return Color4::new(0.5, 0.5, 0.5, 1.0);
    }

    r_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    g_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    b_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mid = r_values.len() / 2;
    Color4::new(r_values[mid], g_values[mid], b_values[mid], 1.0)
}
