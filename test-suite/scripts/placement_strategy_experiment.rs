//! Placement Strategy Experiment
//!
//! Tests different Gaussian placement strategies to find what works best:
//! 1. Uniform grid (baseline)
//! 2. PPM-weighted (gradient + entropy)
//! 3. Laplacian peaks (place at feature points)
//! 4. Structure-tensor-guided (coherence-aware)
//!
//! Each strategy tested with both isotropic and anisotropic shapes.
//! Results saved to test-results/ for corpus building.

use lgi_core::{
    ImageBuffer,
    structure_tensor::StructureTensorField,
    position_probability_map::{PositionProbabilityMap, PPMConfig},
    laplacian::{compute_laplacian, find_local_maxima},
    entropy::compute_image_entropy,
};
use lgi_encoder_v2::{
    renderer_v2::RendererV2,
    adam_optimizer::AdamOptimizer,
    test_results::save_rendered_image,
};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;

const RESULTS_DIR: &str = "/home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs/lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Placement Strategy Experiment                               ║");
    println!("║  Testing: Uniform vs PPM vs Laplacian vs Structure-guided    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let kodak_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = match ImageBuffer::load(kodak_path) {
        Ok(img) => {
            println!("Loaded: {} ({}x{})", kodak_path, img.width, img.height);
            img
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            return;
        }
    };

    // Precompute image analysis
    println!("\nPrecomputing image analysis...");
    let start_analysis = Instant::now();

    let structure_tensor = StructureTensorField::compute(&target, 1.2, 2.5)
        .expect("Structure tensor failed");

    let laplacian = compute_laplacian(&target);
    let laplacian_peaks = find_local_maxima(&laplacian, target.width, target.height, 0.05, 7);

    let gradient_map = compute_gradient_map(&target);
    let entropy_map = compute_per_pixel_entropy(&target, 8);

    let ppm = PositionProbabilityMap::from_entropy_gradient(
        &entropy_map,
        &gradient_map,
        target.width,
        target.height,
        &PPMConfig::default(),
    );

    println!("  Analysis time: {:.2}s", start_analysis.elapsed().as_secs_f32());
    println!("  Image entropy: {:.3}", compute_image_entropy(&target));
    println!("  Laplacian peaks found: {}", laplacian_peaks.len());

    let n_gaussians = 576;  // 24x24 equivalent
    let iterations = 100;

    println!("\nUsing {} Gaussians, {} iterations per test\n", n_gaussians, iterations);

    // Store all results
    let mut all_results: Vec<(String, f32, f32, f32)> = Vec::new();

    // ═══════════════════════════════════════════════════════════════
    // Strategy 1: Uniform Grid (Baseline)
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("Strategy 1: UNIFORM GRID (Baseline)");
    println!("═══════════════════════════════════════════════════════════════");

    let (psnr_uniform, time_uniform) = test_strategy(
        "uniform_grid",
        &target,
        &structure_tensor,
        || create_uniform_grid(&target, n_gaussians, &structure_tensor, false),
        iterations,
    );
    all_results.push(("Uniform (iso)".to_string(), psnr_uniform, time_uniform, 0.0));

    // ═══════════════════════════════════════════════════════════════
    // Strategy 2: PPM-Weighted Sampling
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Strategy 2: PPM-WEIGHTED (gradient + entropy)");
    println!("═══════════════════════════════════════════════════════════════");

    let (psnr_ppm, time_ppm) = test_strategy(
        "ppm_weighted",
        &target,
        &structure_tensor,
        || create_ppm_gaussians(&target, &ppm, &structure_tensor, n_gaussians, false),
        iterations,
    );
    all_results.push(("PPM (iso)".to_string(), psnr_ppm, time_ppm, psnr_ppm - psnr_uniform));

    // ═══════════════════════════════════════════════════════════════
    // Strategy 3: Laplacian Peak Placement
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Strategy 3: LAPLACIAN PEAKS (feature points)");
    println!("═══════════════════════════════════════════════════════════════");

    let (psnr_lap, time_lap) = test_strategy(
        "laplacian_peaks",
        &target,
        &structure_tensor,
        || create_laplacian_gaussians(&target, &laplacian_peaks, &structure_tensor, n_gaussians),
        iterations,
    );
    all_results.push(("Laplacian".to_string(), psnr_lap, time_lap, psnr_lap - psnr_uniform));

    // ═══════════════════════════════════════════════════════════════
    // Strategy 4: Structure-Tensor Anisotropic (Classical)
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Strategy 4: UNIFORM + ANISOTROPIC (Classical)");
    println!("═══════════════════════════════════════════════════════════════");

    let (psnr_aniso, time_aniso) = test_strategy(
        "uniform_anisotropic",
        &target,
        &structure_tensor,
        || create_uniform_grid(&target, n_gaussians, &structure_tensor, true),
        iterations,
    );
    all_results.push(("Uniform (aniso)".to_string(), psnr_aniso, time_aniso, psnr_aniso - psnr_uniform));

    // ═══════════════════════════════════════════════════════════════
    // Strategy 5: PPM + Anisotropic
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Strategy 5: PPM + ANISOTROPIC");
    println!("═══════════════════════════════════════════════════════════════");

    let (psnr_ppm_aniso, time_ppm_aniso) = test_strategy(
        "ppm_anisotropic",
        &target,
        &structure_tensor,
        || create_ppm_gaussians(&target, &ppm, &structure_tensor, n_gaussians, true),
        iterations,
    );
    all_results.push(("PPM (aniso)".to_string(), psnr_ppm_aniso, time_ppm_aniso, psnr_ppm_aniso - psnr_uniform));

    // ═══════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<20} {:>10} {:>10} {:>10}", "Strategy", "PSNR", "Time", "vs Base");
    println!("{:-<20} {:-<10} {:-<10} {:-<10}", "", "", "", "");

    for (name, psnr, time, diff) in &all_results {
        println!("{:<20} {:>10.2} {:>10.2}s {:>+10.2}", name, psnr, time, diff);
    }

    println!("═══════════════════════════════════════════════════════════════");

    // Find winner
    let best = all_results.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    println!("\nBEST: {} at {:.2} dB ({:+.2} vs baseline)", best.0, best.1, best.3);

    // Isotropic vs Anisotropic comparison
    let iso_avg = (psnr_uniform + psnr_ppm + psnr_lap) / 3.0;
    let aniso_avg = (psnr_aniso + psnr_ppm_aniso) / 2.0;
    println!("\nIsotropic avg: {:.2} dB", iso_avg);
    println!("Anisotropic avg: {:.2} dB", aniso_avg);
    println!("Difference: {:+.2} dB ({})",
             iso_avg - aniso_avg,
             if iso_avg > aniso_avg { "ISOTROPIC WINS" } else { "ANISOTROPIC WINS" });

    // Save summary
    save_summary(&all_results, &target);
}

fn test_strategy<F>(
    name: &str,
    target: &ImageBuffer<f32>,
    _structure_tensor: &StructureTensorField,
    init_fn: F,
    iterations: usize,
) -> (f32, f32)
where
    F: Fn() -> Vec<Gaussian2D<f32, Euler<f32>>>,
{
    let mut gaussians = init_fn();
    let n = gaussians.len();

    let init_rendered = RendererV2::render(&gaussians, target.width, target.height);
    let init_psnr = compute_psnr(target, &init_rendered);
    println!("  Initial: {:.2} dB ({} Gaussians)", init_psnr, n);

    let mut adam = AdamOptimizer::new();
    adam.max_iterations = iterations;

    let start = Instant::now();
    let _loss = adam.optimize(&mut gaussians, target);
    let elapsed = start.elapsed().as_secs_f32();

    let final_rendered = RendererV2::render(&gaussians, target.width, target.height);
    let final_psnr = compute_psnr(target, &final_rendered);

    println!("  Final: {:.2} dB ({:.2}s)", final_psnr, elapsed);
    println!("  Improvement: {:+.2} dB", final_psnr - init_psnr);

    // Save rendered image
    let _ = save_rendered_image(&final_rendered, RESULTS_DIR, &format!("placement_{}", name), "final");

    (final_psnr, elapsed)
}

fn create_uniform_grid(
    target: &ImageBuffer<f32>,
    n: usize,
    structure_tensor: &StructureTensorField,
    anisotropic: bool,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32).sqrt().ceil() as u32;
    let mut gaussians = Vec::new();

    let step_x = target.width / grid_size;
    let step_y = target.height / grid_size;
    let sigma_base = 1.0 / grid_size as f32 * 0.8;

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx * step_x + step_x / 2).min(target.width - 1);
            let y = (gy * step_y + step_y / 2).min(target.height - 1);

            let pos = Vector2::new(x as f32 / target.width as f32, y as f32 / target.height as f32);
            let color = target.get_pixel(x, y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            let shape = if anisotropic {
                let tensor = structure_tensor.get(x, y);
                if tensor.coherence > 0.2 {
                    // Anisotropic: elongate along edge tangent
                    let angle = tensor.eigenvector_minor.y.atan2(tensor.eigenvector_minor.x);
                    let sigma_perp = sigma_base / (1.0 + 2.0 * tensor.coherence);
                    let sigma_para = sigma_perp * 3.0;
                    Euler::new(sigma_para, sigma_perp, angle)
                } else {
                    Euler::isotropic(sigma_base)
                }
            } else {
                Euler::isotropic(sigma_base)
            };

            gaussians.push(Gaussian2D::new(pos, shape, color, 1.0));
        }
    }

    gaussians
}

fn create_ppm_gaussians(
    target: &ImageBuffer<f32>,
    ppm: &PositionProbabilityMap,
    structure_tensor: &StructureTensorField,
    n: usize,
    anisotropic: bool,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let positions = ppm.sample_positions_unique(n);
    let sigma_base = 0.03;

    let mut gaussians = Vec::new();

    for (x, y) in positions {
        let pos = Vector2::new(x as f32 / target.width as f32, y as f32 / target.height as f32);
        let color = target.get_pixel(x, y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        let shape = if anisotropic {
            let tensor = structure_tensor.get(x, y);
            if tensor.coherence > 0.2 {
                let angle = tensor.eigenvector_minor.y.atan2(tensor.eigenvector_minor.x);
                let sigma_perp = sigma_base / (1.0 + 2.0 * tensor.coherence);
                let sigma_para = sigma_perp * 3.0;
                Euler::new(sigma_para, sigma_perp, angle)
            } else {
                Euler::isotropic(sigma_base)
            }
        } else {
            // Scale based on local complexity (smaller at high-gradient regions)
            let tensor = structure_tensor.get(x, y);
            let energy = (tensor.eigenvalue_major + tensor.eigenvalue_minor).sqrt();
            let adaptive_sigma = sigma_base / (1.0 + energy * 5.0);
            Euler::isotropic(adaptive_sigma.max(0.005))
        };

        gaussians.push(Gaussian2D::new(pos, shape, color, 1.0));
    }

    gaussians
}

fn create_laplacian_gaussians(
    target: &ImageBuffer<f32>,
    peaks: &[(u32, u32, f32)],
    structure_tensor: &StructureTensorField,
    n: usize,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let mut gaussians = Vec::new();

    // Take top peaks
    let peak_count = n.min(peaks.len());
    let sigma_base = 0.025;

    for (x, y, _magnitude) in peaks.iter().take(peak_count) {
        let pos = Vector2::new(*x as f32 / target.width as f32, *y as f32 / target.height as f32);
        let color = target.get_pixel(*x, *y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        // Isotropic shape (quantum-guided finding)
        let shape = Euler::isotropic(sigma_base);

        gaussians.push(Gaussian2D::new(pos, shape, color, 1.0));
    }

    // If we need more Gaussians, fill with uniform grid
    if gaussians.len() < n {
        let remaining = n - gaussians.len();
        let grid_size = (remaining as f32).sqrt().ceil() as u32;

        let step_x = target.width / grid_size;
        let step_y = target.height / grid_size;

        for gy in 0..grid_size {
            for gx in 0..grid_size {
                if gaussians.len() >= n {
                    break;
                }

                let x = (gx * step_x + step_x / 2).min(target.width - 1);
                let y = (gy * step_y + step_y / 2).min(target.height - 1);

                let pos = Vector2::new(x as f32 / target.width as f32, y as f32 / target.height as f32);
                let color = target.get_pixel(x, y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

                gaussians.push(Gaussian2D::new(pos, Euler::isotropic(sigma_base * 1.5), color, 1.0));
            }
        }
    }

    gaussians
}

fn compute_gradient_map(image: &ImageBuffer<f32>) -> Vec<f32> {
    let mut gradient = vec![0.0; (image.width * image.height) as usize];

    for y in 1..(image.height - 1) {
        for x in 1..(image.width - 1) {
            let idx = (y * image.width + x) as usize;

            let left = image.get_pixel(x - 1, y).unwrap();
            let right = image.get_pixel(x + 1, y).unwrap();
            let up = image.get_pixel(x, y - 1).unwrap();
            let down = image.get_pixel(x, y + 1).unwrap();

            let gx = ((right.r - left.r).powi(2) + (right.g - left.g).powi(2) + (right.b - left.b).powi(2)).sqrt();
            let gy = ((down.r - up.r).powi(2) + (down.g - up.g).powi(2) + (down.b - up.b).powi(2)).sqrt();

            gradient[idx] = (gx * gx + gy * gy).sqrt();
        }
    }

    // Normalize
    let max = gradient.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for g in &mut gradient {
            *g /= max;
        }
    }

    gradient
}

fn compute_per_pixel_entropy(image: &ImageBuffer<f32>, window: usize) -> Vec<f32> {
    let mut entropy = vec![0.0; (image.width * image.height) as usize];
    let half = window / 2;

    for y in half..(image.height as usize - half) {
        for x in half..(image.width as usize - half) {
            let idx = y * image.width as usize + x;

            // Compute local variance
            let mut sum_r = 0.0;
            let mut sum_g = 0.0;
            let mut sum_b = 0.0;
            let mut count = 0;

            for dy in 0..window {
                for dx in 0..window {
                    let px = (x - half + dx) as u32;
                    let py = (y - half + dy) as u32;
                    if let Some(p) = image.get_pixel(px, py) {
                        sum_r += p.r;
                        sum_g += p.g;
                        sum_b += p.b;
                        count += 1;
                    }
                }
            }

            let mean_r = sum_r / count as f32;
            let mean_g = sum_g / count as f32;
            let mean_b = sum_b / count as f32;

            let mut var = 0.0;
            for dy in 0..window {
                for dx in 0..window {
                    let px = (x - half + dx) as u32;
                    let py = (y - half + dy) as u32;
                    if let Some(p) = image.get_pixel(px, py) {
                        var += (p.r - mean_r).powi(2) + (p.g - mean_g).powi(2) + (p.b - mean_b).powi(2);
                    }
                }
            }

            entropy[idx] = (var / count as f32).sqrt();
        }
    }

    // Normalize
    let max = entropy.iter().cloned().fold(0.0f32, f32::max);
    if max > 0.0 {
        for e in &mut entropy {
            *e /= max;
        }
    }

    entropy
}

fn compute_psnr(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (original.width * original.height * 3) as f32;

    for (p1, p2) in original.data.iter().zip(rendered.data.iter()) {
        mse += (p1.r - p2.r).powi(2);
        mse += (p1.g - p2.g).powi(2);
        mse += (p1.b - p2.b).powi(2);
    }

    mse /= count;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}

fn save_summary(results: &[(String, f32, f32, f32)], target: &ImageBuffer<f32>) {
    use std::fs::File;
    use std::io::Write;

    let summary = serde_json::json!({
        "experiment": "placement_strategy_comparison",
        "timestamp": chrono_lite(),
        "image": "kodim03.png",
        "image_dimensions": [target.width, target.height],
        "n_gaussians": 576,
        "iterations": 100,
        "results": results.iter().map(|(name, psnr, time, diff)| {
            serde_json::json!({
                "strategy": name,
                "final_psnr": psnr,
                "time_seconds": time,
                "vs_baseline_db": diff
            })
        }).collect::<Vec<_>>()
    });

    let path = format!("{}/2025-12-05/placement_strategy_comparison.json", RESULTS_DIR);
    if let Ok(mut file) = File::create(&path) {
        let _ = file.write_all(serde_json::to_string_pretty(&summary).unwrap().as_bytes());
        println!("\nResults saved to: {}", path);
    }
}

fn chrono_lite() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    format!("2025-12-05T{:02}:{:02}:{:02}", (secs / 3600) % 24, (secs / 60) % 60, secs % 60)
}
