//! Test Coherence-Adaptive Anisotropy Encoding
//!
//! Hypothesis: Using structure tensor coherence to set Gaussian anisotropy
//! at initialization improves encoding quality, especially for directional features.
//!
//! Key insight from today's findings:
//! - α=0.84 hybrid loss is optimal (84% MSE + 16% MS-SSIM)
//! - Coherent regions (edges, strokes) benefit from elongated Gaussians
//! - Incoherent regions (corners, texture) need more isotropic Gaussians
//!
//! This test:
//! 1. Computes structure tensor for coherence + orientation
//! 2. Initializes Gaussians with coherence-driven anisotropy
//! 3. Orients anisotropic Gaussians along edge tangent
//! 4. Optimizes with hybrid loss (α=0.84)
//! 5. Compares against baseline (random orientation, fixed anisotropy)

use lgi_core::{
    ImageBuffer,
    structure_tensor::StructureTensorField,
};
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
    println!("   COHERENCE-ADAPTIVE ANISOTROPY TEST - Structure Tensor Guidance");
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

    // Compute structure tensor
    println!("Step 1: Structure Tensor Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let tensor_field = StructureTensorField::compute(&target, 1.5, 2.5).expect("Structure tensor failed");

    // Analyze coherence distribution
    let mut coherences: Vec<f32> = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            coherences.push(tensor_field.get(x, y).coherence);
        }
    }

    let mean_coherence: f32 = coherences.iter().sum::<f32>() / coherences.len() as f32;
    let high_coherence_count = coherences.iter().filter(|&&c| c > 0.5).count();
    let high_coherence_pct = 100.0 * high_coherence_count as f32 / coherences.len() as f32;

    println!("  Structure tensor computed ({} pixels)", coherences.len());
    println!("  Mean coherence: {:.3}", mean_coherence);
    println!("  High coherence regions (c > 0.5): {:.1}%\n", high_coherence_pct);

    let n_gaussians = 256;
    let iterations = 200;

    // Test 1: Baseline (random orientation, fixed anisotropy)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 1: BASELINE - Random Orientation, Fixed Anisotropy");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let baseline_result = run_encoding(
        "baseline_random",
        &target,
        || init_baseline_random(width, height, n_gaussians, &target),
        iterations,
        image_path,
    );

    // Test 2: Coherence-adaptive
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 2: COHERENCE-ADAPTIVE - Structure Tensor Guidance");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let adaptive_result = run_encoding(
        "coherence_adaptive",
        &target,
        || init_coherence_adaptive(&tensor_field, width, height, n_gaussians, &target),
        iterations,
        image_path,
    );

    // Test 3: Strong coherence weighting
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Test 3: STRONG COHERENCE - Higher Anisotropy Range");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let strong_result = run_encoding(
        "strong_coherence",
        &target,
        || init_strong_coherence(&tensor_field, width, height, n_gaussians, &target),
        iterations,
        image_path,
    );

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                      COHERENCE ENCODING SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  {:25} {:>12} {:>12} {:>12}", "Method", "Initial", "Final", "Improvement");
    println!("  {:-<25} {:->12} {:->12} {:->12}", "", "", "", "");

    for (name, (init_psnr, final_psnr)) in [
        ("Baseline (random)", baseline_result),
        ("Coherence-adaptive", adaptive_result),
        ("Strong coherence", strong_result),
    ] {
        let improvement = final_psnr - init_psnr;
        println!("  {:25} {:>10.2}dB {:>10.2}dB {:>+10.2}dB", name, init_psnr, final_psnr, improvement);
    }

    let best_improvement = adaptive_result.1 - baseline_result.1;
    let strong_improvement = strong_result.1 - baseline_result.1;

    println!("\n  Coherence-adaptive vs baseline: {:+.2}dB", best_improvement);
    println!("  Strong coherence vs baseline: {:+.2}dB", strong_improvement);

    if best_improvement > 0.1 || strong_improvement > 0.1 {
        println!("\n  ✓ Coherence-adaptive anisotropy improves encoding!");
    } else if best_improvement < -0.1 && strong_improvement < -0.1 {
        println!("\n  ⚠ Baseline performs better - review coherence logic");
    } else {
        println!("\n  ~ Results comparable - optimization dominates initialization");
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
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

    let mut result = TestResult::new(&format!("coherence_encoding_{}", name), image_path);
    result.optimizer = format!("Coherence-adaptive: {}", name);
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

fn init_baseline_random(
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
    let base_sigma = 0.04;  // Fixed anisotropy ratio of 1.5

    for i in 0..count {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;

        // Jittered position
        let jitter_x = rng.gen_range(-0.3..0.3);
        let jitter_y = rng.gen_range(-0.3..0.3);
        let px = ((col as f32 + 0.5 + jitter_x) * cell_w) as u32;
        let py = ((row as f32 + 0.5 + jitter_y) * cell_h) as u32;
        let px = px.min(width - 1);
        let py = py.min(height - 1);

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        // Fixed anisotropy, random orientation
        let angle = rng.gen_range(0.0..std::f32::consts::PI);
        let sigma_major = base_sigma;
        let sigma_minor = base_sigma / 1.5;

        gaussians.push(Gaussian2D::new(
            position,
            Euler::new(sigma_major, sigma_minor, angle),
            color.clone(),
            1.0,
        ));
    }

    gaussians
}

fn init_coherence_adaptive(
    tensor_field: &lgi_core::structure_tensor::StructureTensorField,
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

    // Anisotropy range: 1.0 (isotropic) to 3.0 (elongated)
    let min_anisotropy = 1.0;
    let max_anisotropy = 3.0;
    let base_sigma = 0.035;

    for i in 0..count {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;

        // Jittered position
        let jitter_x = rng.gen_range(-0.3..0.3);
        let jitter_y = rng.gen_range(-0.3..0.3);
        let px = ((col as f32 + 0.5 + jitter_x) * cell_w) as u32;
        let py = ((row as f32 + 0.5 + jitter_y) * cell_h) as u32;
        let px = px.min(width - 1);
        let py = py.min(height - 1);

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        // Get structure tensor at this location
        let tensor = tensor_field.get(px, py);

        // Coherence-adaptive anisotropy: more coherent = more elongated
        let anisotropy = min_anisotropy + tensor.coherence * (max_anisotropy - min_anisotropy);

        // Orientation from eigenvector (edge tangent = minor eigenvector)
        let angle = tensor.eigenvector_minor.y.atan2(tensor.eigenvector_minor.x);

        let sigma_major = base_sigma;
        let sigma_minor = base_sigma / anisotropy;

        gaussians.push(Gaussian2D::new(
            position,
            Euler::new(sigma_major, sigma_minor, angle),
            color.clone(),
            1.0,
        ));
    }

    gaussians
}

fn init_strong_coherence(
    tensor_field: &lgi_core::structure_tensor::StructureTensorField,
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

    // Stronger anisotropy range: 1.0 to 5.0
    let min_anisotropy = 1.0;
    let max_anisotropy = 5.0;
    let base_sigma = 0.035;

    for i in 0..count {
        let col = (i as u32) % cols;
        let row = (i as u32) / cols;

        // Jittered position
        let jitter_x = rng.gen_range(-0.3..0.3);
        let jitter_y = rng.gen_range(-0.3..0.3);
        let px = ((col as f32 + 0.5 + jitter_x) * cell_w) as u32;
        let py = ((row as f32 + 0.5 + jitter_y) * cell_h) as u32;
        let px = px.min(width - 1);
        let py = py.min(height - 1);

        let position = Vector2::new(px as f32 / width as f32, py as f32 / height as f32);
        let color = target.get_pixel(px, py).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        // Get structure tensor at this location
        let tensor = tensor_field.get(px, py);

        // Strong coherence weighting with power function
        // c^2 makes high-coherence regions much more anisotropic
        let coherence_weight = tensor.coherence * tensor.coherence;
        let anisotropy = min_anisotropy + coherence_weight * (max_anisotropy - min_anisotropy);

        // Orientation from eigenvector
        let angle = tensor.eigenvector_minor.y.atan2(tensor.eigenvector_minor.x);

        let sigma_major = base_sigma;
        let sigma_minor = base_sigma / anisotropy;

        gaussians.push(Gaussian2D::new(
            position,
            Euler::new(sigma_major, sigma_minor, angle),
            color.clone(),
            1.0,
        ));
    }

    gaussians
}
