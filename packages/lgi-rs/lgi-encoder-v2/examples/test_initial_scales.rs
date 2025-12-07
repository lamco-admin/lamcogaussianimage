//! Test Different Initial Gaussian Scales
//!
//! We've defaulted to scale=0.02 everywhere. This systematically tests:
//! - Different base scales (0.005 to 0.1)
//! - Different anisotropy ratios
//! - Scale relative to Gaussian count
//!
//! Run: cargo run --release --example test_initial_scales -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   INITIAL GAUSSIAN SCALE EXPLORATION                          ║");
    println!("║   Testing different base scales and anisotropy ratios         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);
    println!("");

    let n_gaussians = 256;

    // Test 1: Different base scales
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Different Base Scales (isotropic)");
    println!("═══════════════════════════════════════════════════════════════");

    let scales = vec![0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1];
    let mut scale_results = Vec::new();

    for &scale in &scales {
        let gaussians = create_grid_with_scale(&target, n_gaussians, scale, 1.0);
        let render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&render, &target);

        // Optimize
        let mut opt_gaussians = gaussians.clone();
        let mut adam = AdamOptimizer::new();
        adam.max_iterations = 100;
        adam.optimize(&mut opt_gaussians, &target);

        let opt_render = RendererV2::render(&opt_gaussians, width, height);
        let final_psnr = compute_psnr(&opt_render, &target);

        println!("  scale={:.3}: init={:.2}dB, final={:.2}dB (+{:.2}dB)",
                 scale, init_psnr, final_psnr, final_psnr - init_psnr);

        scale_results.push((scale, init_psnr, final_psnr));
    }

    // Find best scale
    let best_scale = scale_results.iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();
    println!("\n  Best scale: {:.3} → {:.2} dB", best_scale.0, best_scale.2);

    // Test 2: Different anisotropy ratios (keeping area constant)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: Different Anisotropy Ratios (constant area)");
    println!("═══════════════════════════════════════════════════════════════");

    let base_scale = 0.02;  // Keep geometric mean constant
    let anisotropy_ratios = vec![1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
    let mut aniso_results = Vec::new();

    for &ratio in &anisotropy_ratios {
        // Keep scale_x * scale_y = base_scale^2
        // scale_x = base_scale * sqrt(ratio)
        // scale_y = base_scale / sqrt(ratio)
        let scale_x = base_scale * ratio.sqrt();
        let scale_y = base_scale / ratio.sqrt();

        let gaussians = create_grid_with_anisotropy(&target, n_gaussians, scale_x, scale_y);
        let render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&render, &target);

        // Optimize
        let mut opt_gaussians = gaussians.clone();
        let mut adam = AdamOptimizer::new();
        adam.max_iterations = 100;
        adam.optimize(&mut opt_gaussians, &target);

        let opt_render = RendererV2::render(&opt_gaussians, width, height);
        let final_psnr = compute_psnr(&opt_render, &target);

        println!("  ratio={:.1} (sx={:.3}, sy={:.3}): init={:.2}dB, final={:.2}dB (+{:.2}dB)",
                 ratio, scale_x, scale_y, init_psnr, final_psnr, final_psnr - init_psnr);

        aniso_results.push((ratio, init_psnr, final_psnr));
    }

    // Find best anisotropy
    let best_aniso = aniso_results.iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();
    println!("\n  Best anisotropy: {:.1}× → {:.2} dB", best_aniso.0, best_aniso.2);

    // Test 3: Scale relative to Gaussian count
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 3: Scale Relative to N (coverage-based)");
    println!("═══════════════════════════════════════════════════════════════");

    // Theory: With N Gaussians, each covers ~1/N of the image
    // So ideal scale might be related to 1/sqrt(N)
    let counts = vec![64, 128, 256, 512];
    let mut coverage_results = Vec::new();

    for &n in &counts {
        // Coverage-based scale: each Gaussian covers roughly image/N area
        // Scale = sqrt(1/(2*N)) ≈ 0.7/sqrt(N) (factor of 2 for overlap)
        let coverage_scale = 0.7 / (n as f32).sqrt();
        let fixed_scale = 0.02;

        println!("\n  N={}", n);

        // With coverage-based scale
        let gaussians_cov = create_grid_with_scale(&target, n, coverage_scale, 1.0);
        let render_cov = RendererV2::render(&gaussians_cov, width, height);
        let init_cov = compute_psnr(&render_cov, &target);

        let mut opt_cov = gaussians_cov.clone();
        let mut adam_cov = AdamOptimizer::new();
        adam_cov.max_iterations = 100;
        adam_cov.optimize(&mut opt_cov, &target);
        let final_cov = compute_psnr(&RendererV2::render(&opt_cov, width, height), &target);

        // With fixed scale
        let gaussians_fix = create_grid_with_scale(&target, n, fixed_scale, 1.0);
        let render_fix = RendererV2::render(&gaussians_fix, width, height);
        let init_fix = compute_psnr(&render_fix, &target);

        let mut opt_fix = gaussians_fix.clone();
        let mut adam_fix = AdamOptimizer::new();
        adam_fix.max_iterations = 100;
        adam_fix.optimize(&mut opt_fix, &target);
        let final_fix = compute_psnr(&RendererV2::render(&opt_fix, width, height), &target);

        println!("    Coverage (s={:.3}): init={:.2}dB, final={:.2}dB",
                 coverage_scale, init_cov, final_cov);
        println!("    Fixed (s=0.02):     init={:.2}dB, final={:.2}dB",
                 init_fix, final_fix);
        println!("    Delta: {:+.2}dB (coverage vs fixed)", final_cov - final_fix);

        coverage_results.push((n, coverage_scale, final_cov, final_fix));
    }

    // Summary
    println!("\n\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY                                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("");
    println!("Best base scale (isotropic): {:.3}", best_scale.0);
    println!("Best anisotropy ratio: {:.1}×", best_aniso.0);
    println!("");
    println!("Coverage-based vs Fixed scale:");
    for (n, cov_scale, final_cov, final_fix) in &coverage_results {
        let winner = if final_cov > final_fix { "Coverage" } else { "Fixed" };
        println!("  N={}: coverage({:.3})={:.2}dB, fixed(0.02)={:.2}dB → {}",
                 n, cov_scale, final_cov, final_fix, winner);
    }

    // Save result
    let mut result = TestResult::new("initial_scales", image_path);
    result.optimizer = "Initial Scale Exploration".to_string();
    result.n_gaussians = n_gaussians;
    result.iterations = 100;
    result.initial_psnr = best_scale.1;
    result.final_psnr = best_scale.2;
    result.improvement_db = best_scale.2 - best_scale.1;
    result.image_dimensions = (width, height);
    result.set_param("best_scale", best_scale.0);
    result.set_param("best_anisotropy", best_aniso.0);
    result.set_param("scales_tested", scales.len());
    result.set_param("anisotropy_tested", anisotropy_ratios.len());
    result.complete();

    let _ = result.save(RESULTS_DIR);

    println!("\n✓ Initial scale exploration complete!");
}

fn create_grid_with_scale(
    target: &ImageBuffer<f32>,
    n: usize,
    scale: f32,
    _anisotropy: f32,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let mut gaussians = Vec::with_capacity(n);

    for i in 0..grid_size {
        for j in 0..grid_size {
            if gaussians.len() >= n {
                break;
            }

            let x = (i as f32 + 0.5) / grid_size as f32;
            let y = (j as f32 + 0.5) / grid_size as f32;

            let px = (x * width as f32) as u32;
            let py = (y * height as f32) as u32;

            let color = target.get_pixel(px.min(width - 1), py.min(height - 1))
                .cloned()
                .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            gaussians.push(Gaussian2D {
                position: Vector2::new(x, y),
                shape: Euler {
                    scale_x: scale,
                    scale_y: scale,
                    rotation: 0.0,
                },
                color: Color4::rgb(color.r, color.g, color.b),
                opacity: 0.8,
            });
        }
    }

    gaussians
}

fn create_grid_with_anisotropy(
    target: &ImageBuffer<f32>,
    n: usize,
    scale_x: f32,
    scale_y: f32,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let mut gaussians = Vec::with_capacity(n);
    let mut rng = 42u64;

    for i in 0..grid_size {
        for j in 0..grid_size {
            if gaussians.len() >= n {
                break;
            }

            let x = (i as f32 + 0.5) / grid_size as f32;
            let y = (j as f32 + 0.5) / grid_size as f32;

            let px = (x * width as f32) as u32;
            let py = (y * height as f32) as u32;

            let color = target.get_pixel(px.min(width - 1), py.min(height - 1))
                .cloned()
                .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            // Random rotation for anisotropic Gaussians
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rotation = ((rng >> 33) as f32 / u32::MAX as f32) * std::f32::consts::PI;

            gaussians.push(Gaussian2D {
                position: Vector2::new(x, y),
                shape: Euler {
                    scale_x,
                    scale_y,
                    rotation,
                },
                color: Color4::rgb(color.r, color.g, color.b),
                opacity: 0.8,
            });
        }
    }

    gaussians
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mse: f32 = rendered
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(r, t)| {
            let dr = r.r - t.r;
            let dg = r.g - t.g;
            let db = r.b - t.b;
            (dr * dr + dg * dg + db * db) / 3.0
        })
        .sum::<f32>()
        / rendered.data.len() as f32;

    if mse < 1e-10 {
        100.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}
