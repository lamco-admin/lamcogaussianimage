//! Overnight Init Recalibration Test
//!
//! KEY HYPOTHESIS: Feature-based initialization methods (K-Means, GDGS, Gradient Peak)
//! may have failed because they use "smart" sizing that constrains Adam.
//!
//! This test tries:
//! 1. Feature-based POSITION selection + UNIFORM sizing (let Adam optimize shape)
//! 2. K-Means with different weight configurations
//! 3. GDGS with uniform Gaussians
//! 4. Gradient peaks as position hints only
//!
//! Run: cargo run --release --example test_overnight_init_recalibration -p lgi-encoder-v2

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::loss_functions::compute_ssim;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   OVERNIGHT INIT RECALIBRATION TEST                          ║");
    println!("║   Feature-based positions + uniform sizing                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let test_images = vec![
        ("kodim03", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png"),
        ("kodim05", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim05.png"),
        ("kodim08", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim08.png"),
    ];

    for (image_name, image_path) in &test_images {
        println!("\n{}", "=".repeat(70));
        println!("Processing image: {}", image_name);
        println!("{}\n", "=".repeat(70));

        let target = match ImageBuffer::load(image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Failed to load {}: {:?}", image_path, e);
                continue;
            }
        };

        let structure_tensor = StructureTensorField::compute(&target, 1.2, 1.0)
            .expect("Failed to compute structure tensor");

        // BASELINE
        println!("═══════════════════════════════════════════════════════════════");
        println!("BASELINE: Jittered Grid + Adam");
        println!("═══════════════════════════════════════════════════════════════");
        test_config(&target, image_path, image_name, &structure_tensor,
                    "baseline_jittered", InitMethod::JitteredGrid, 256);

        // K-MEANS RECALIBRATION
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("K-MEANS RECALIBRATION: Position as hints, uniform sizing");
        println!("═══════════════════════════════════════════════════════════════");

        // K-Means: color-heavy (position weight very low)
        test_config(&target, image_path, image_name, &structure_tensor,
                    "kmeans_color_heavy", InitMethod::KMeansUniform {
                        weight_color: 1.0,
                        weight_position: 0.1,  // Very low position weight
                        weight_gradient: 0.1,
                    }, 256);

        // K-Means: balanced low weights
        test_config(&target, image_path, image_name, &structure_tensor,
                    "kmeans_balanced_low", InitMethod::KMeansUniform {
                        weight_color: 0.5,
                        weight_position: 0.3,
                        weight_gradient: 0.2,
                    }, 256);

        // K-Means: gradient-heavy (should cluster around edges)
        test_config(&target, image_path, image_name, &structure_tensor,
                    "kmeans_gradient_heavy", InitMethod::KMeansUniform {
                        weight_color: 0.3,
                        weight_position: 0.2,
                        weight_gradient: 1.0,  // Heavy gradient weight
                    }, 256);

        // GRADIENT PEAK RECALIBRATION
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("GRADIENT PEAK RECALIBRATION: Peak positions, uniform sizing");
        println!("═══════════════════════════════════════════════════════════════");

        // Gradient peaks as position hints only (no fancy sizing)
        test_config(&target, image_path, image_name, &structure_tensor,
                    "gradient_peak_uniform", InitMethod::GradientPeakUniform {
                        threshold: 0.01,
                        max_peaks: 256,
                    }, 256);

        // Lower threshold = more peaks
        test_config(&target, image_path, image_name, &structure_tensor,
                    "gradient_peak_low_thresh", InitMethod::GradientPeakUniform {
                        threshold: 0.005,
                        max_peaks: 256,
                    }, 256);

        // GDGS RECALIBRATION
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("GDGS RECALIBRATION: Laplacian peaks, uniform sizing");
        println!("═══════════════════════════════════════════════════════════════");

        test_config(&target, image_path, image_name, &structure_tensor,
                    "gdgs_uniform", InitMethod::GDGSUniform {
                        lap_threshold: 0.01,
                        max_gaussians: 256,
                    }, 256);

        // Lower threshold for more Gaussians
        test_config(&target, image_path, image_name, &structure_tensor,
                    "gdgs_low_thresh", InitMethod::GDGSUniform {
                        lap_threshold: 0.005,
                        max_gaussians: 256,
                    }, 256);

        // HYBRID: Gradient peaks + background grid
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("HYBRID: Peaks + sparse background grid");
        println!("═══════════════════════════════════════════════════════════════");

        test_config(&target, image_path, image_name, &structure_tensor,
                    "hybrid_peaks_plus_grid", InitMethod::HybridPeaksGrid {
                        n_peaks: 200,
                        n_background: 56,  // 200 + 56 = 256
                    }, 256);
    }

    println!("\n{}", "=".repeat(70));
    println!("OVERNIGHT INIT RECALIBRATION TEST COMPLETE");
    println!("{}", "=".repeat(70));
}

#[derive(Debug, Clone)]
enum InitMethod {
    JitteredGrid,
    KMeansUniform { weight_color: f32, weight_position: f32, weight_gradient: f32 },
    GradientPeakUniform { threshold: f32, max_peaks: usize },
    GDGSUniform { lap_threshold: f32, max_gaussians: usize },
    HybridPeaksGrid { n_peaks: usize, n_background: usize },
}

fn test_config(
    target: &ImageBuffer<f32>,
    image_path: &str,
    image_name: &str,
    structure_tensor: &StructureTensorField,
    config_name: &str,
    init_method: InitMethod,
    target_n: usize,
) {
    println!("───────────────────────────────────────────────────────────────");
    println!("Config: {}", config_name);
    println!("───────────────────────────────────────────────────────────────");

    let start = Instant::now();
    let (width, height) = (target.width, target.height);

    // Initialize Gaussians
    let mut gaussians = match &init_method {
        InitMethod::JitteredGrid => create_jittered_grid(target, target_n),
        InitMethod::KMeansUniform { weight_color, weight_position, weight_gradient } => {
            create_kmeans_uniform(target, structure_tensor, target_n,
                                  *weight_color, *weight_position, *weight_gradient)
        },
        InitMethod::GradientPeakUniform { threshold, max_peaks } => {
            create_gradient_peak_uniform(target, structure_tensor, *threshold, *max_peaks)
        },
        InitMethod::GDGSUniform { lap_threshold, max_gaussians } => {
            create_gdgs_uniform(target, *lap_threshold, *max_gaussians)
        },
        InitMethod::HybridPeaksGrid { n_peaks, n_background } => {
            create_hybrid_peaks_grid(target, structure_tensor, *n_peaks, *n_background)
        },
    };

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);
    let init_ssim = compute_ssim(&init_render, target);
    println!("  Initial: N={}, PSNR={:.2}dB, SSIM={:.4}", gaussians.len(), init_psnr, init_ssim);

    // Optimize with Adam
    let mut adam = AdamOptimizer::new();
    adam.max_iterations = 150;
    let _loss = adam.optimize(&mut gaussians, target);

    let elapsed = start.elapsed();
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);
    let final_ssim = compute_ssim(&final_render, target);

    println!("  Final: PSNR={:.2}dB ({:+.2}dB), SSIM={:.4} in {:.1}s",
             final_psnr, final_psnr - init_psnr, final_ssim, elapsed.as_secs_f32());

    // Save result
    let exp_name = format!("overnight_recalib_{}_{}", config_name, image_name);
    let mut result = TestResult::new(&exp_name, image_path);
    result.optimizer = format!("Adam with {:?}", init_method);
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.iterations = 150;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.extra_metrics = Some(json!({
        "init_method": config_name,
        "target_n": target_n,
        "actual_n": gaussians.len(),
        "initial_ssim": init_ssim,
        "final_ssim": final_ssim,
    }));
    let _ = result.save(RESULTS_DIR);
}

fn create_jittered_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n);
    let cell_size = 1.0 / grid_size as f32;

    for i in 0..n {
        let gx = i % grid_size;
        let gy = i / grid_size;

        let jx = ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5;
        let jy = ((i * 11 + 7) % 100) as f32 / 100.0 - 0.5;

        let x = ((gx as f32 + 0.5 + jx * 0.4) / grid_size as f32).clamp(0.01, 0.99);
        let y = ((gy as f32 + 0.5 + jy * 0.4) / grid_size as f32).clamp(0.01, 0.99);

        let px = ((x * width as f32) as u32).min(width - 1);
        let py = ((y * height as f32) as u32).min(height - 1);
        let color = target.get_pixel(px, py).unwrap();

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(cell_size, cell_size, 0.0),
            color,
            1.0,
        ));
    }

    gaussians
}

fn create_kmeans_uniform(
    target: &ImageBuffer<f32>,
    structure_tensor: &StructureTensorField,
    k: usize,
    weight_color: f32,
    weight_position: f32,
    weight_gradient: f32,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    // Extract pixel features
    let mut features = Vec::new();
    for y in 0..target.height {
        for x in 0..target.width {
            let pixel = target.get_pixel(x, y).unwrap();
            let tensor = structure_tensor.get(x, y);
            let gradient = tensor.eigenvalue_major.sqrt();

            features.push(PixelFeature {
                r: pixel.r * weight_color,
                g: pixel.g * weight_color,
                b: pixel.b * weight_color,
                x: (x as f32 / target.width as f32) * weight_position,
                y: (y as f32 / target.height as f32) * weight_position,
                gradient: gradient * weight_gradient,
                orig_x: x,
                orig_y: y,
            });
        }
    }

    // Simple k-means
    let (centroids, _) = kmeans_simple(&features, k, 30);

    // UNIFORM SIZING: All Gaussians get the same size
    let uniform_size = 1.0 / (k as f32).sqrt();

    let mut gaussians = Vec::new();
    for centroid in &centroids {
        let x = (centroid.x / weight_position.max(0.001)).clamp(0.01, 0.99);
        let y = (centroid.y / weight_position.max(0.001)).clamp(0.01, 0.99);

        let px = ((x * target.width as f32) as u32).min(target.width - 1);
        let py = ((y * target.height as f32) as u32).min(target.height - 1);
        let color = target.get_pixel(px, py).unwrap();

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(uniform_size, uniform_size, 0.0),  // UNIFORM!
            color,
            1.0,
        ));
    }

    gaussians
}

fn create_gradient_peak_uniform(
    target: &ImageBuffer<f32>,
    structure_tensor: &StructureTensorField,
    threshold: f32,
    max_peaks: usize,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    // Find gradient peaks
    let mut gradient_map = vec![0.0f32; (target.width * target.height) as usize];
    for y in 0..target.height {
        for x in 0..target.width {
            let tensor = structure_tensor.get(x, y);
            gradient_map[(y * target.width + x) as usize] = tensor.eigenvalue_major.sqrt();
        }
    }

    let peaks = find_local_maxima(&gradient_map, target.width, target.height, threshold, 5);

    // UNIFORM SIZING
    let uniform_size = 1.0 / (max_peaks as f32).sqrt();

    let mut gaussians = Vec::new();
    for (x, y, _) in peaks.iter().take(max_peaks) {
        let nx = *x as f32 / target.width as f32;
        let ny = *y as f32 / target.height as f32;
        let color = target.get_pixel(*x, *y).unwrap();

        gaussians.push(Gaussian2D::new(
            Vector2::new(nx, ny),
            Euler::new(uniform_size, uniform_size, 0.0),  // UNIFORM!
            color,
            1.0,
        ));
    }

    // If not enough peaks, fill with grid
    if gaussians.len() < max_peaks {
        let remaining = max_peaks - gaussians.len();
        let grid_gaussians = create_sparse_background(target, remaining, &gaussians);
        gaussians.extend(grid_gaussians);
    }

    gaussians
}

fn create_gdgs_uniform(
    target: &ImageBuffer<f32>,
    lap_threshold: f32,
    max_gaussians: usize,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    // Compute Laplacian
    let laplacian = compute_laplacian(target);

    // Find peaks
    let peaks = find_local_maxima(&laplacian, target.width, target.height, lap_threshold, 5);

    // UNIFORM SIZING
    let uniform_size = 1.0 / (max_gaussians as f32).sqrt();

    let mut gaussians = Vec::new();
    for (x, y, _) in peaks.iter().take(max_gaussians) {
        let nx = *x as f32 / target.width as f32;
        let ny = *y as f32 / target.height as f32;
        let color = target.get_pixel(*x, *y).unwrap();

        gaussians.push(Gaussian2D::new(
            Vector2::new(nx, ny),
            Euler::new(uniform_size, uniform_size, 0.0),  // UNIFORM!
            color,
            1.0,
        ));
    }

    // Fill with background if needed
    if gaussians.len() < max_gaussians {
        let remaining = max_gaussians - gaussians.len();
        let grid_gaussians = create_sparse_background(target, remaining, &gaussians);
        gaussians.extend(grid_gaussians);
    }

    gaussians
}

fn create_hybrid_peaks_grid(
    target: &ImageBuffer<f32>,
    structure_tensor: &StructureTensorField,
    n_peaks: usize,
    n_background: usize,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let total = n_peaks + n_background;
    let uniform_size = 1.0 / (total as f32).sqrt();

    // Get gradient peaks
    let mut gradient_map = vec![0.0f32; (target.width * target.height) as usize];
    for y in 0..target.height {
        for x in 0..target.width {
            let tensor = structure_tensor.get(x, y);
            gradient_map[(y * target.width + x) as usize] = tensor.eigenvalue_major.sqrt();
        }
    }

    let peaks = find_local_maxima(&gradient_map, target.width, target.height, 0.01, 5);

    let mut gaussians = Vec::new();

    // Add peak Gaussians
    for (x, y, _) in peaks.iter().take(n_peaks) {
        let nx = *x as f32 / target.width as f32;
        let ny = *y as f32 / target.height as f32;
        let color = target.get_pixel(*x, *y).unwrap();

        gaussians.push(Gaussian2D::new(
            Vector2::new(nx, ny),
            Euler::new(uniform_size, uniform_size, 0.0),
            color,
            1.0,
        ));
    }

    // Add sparse background
    let bg = create_sparse_background(target, n_background, &gaussians);
    gaussians.extend(bg);

    gaussians
}

fn create_sparse_background(
    target: &ImageBuffer<f32>,
    n: usize,
    existing: &[Gaussian2D<f32, Euler<f32>>],
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32 * 2.0).sqrt().ceil() as usize;  // Oversample and filter
    let uniform_size = 1.0 / ((existing.len() + n) as f32).sqrt();
    let min_dist = uniform_size * 0.5;

    let mut bg = Vec::new();

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            if bg.len() >= n { break; }

            let x = (gx as f32 + 0.5) / grid_size as f32;
            let y = (gy as f32 + 0.5) / grid_size as f32;

            // Check distance from existing
            let too_close = existing.iter().chain(bg.iter()).any(|g| {
                let dx = g.position.x - x;
                let dy = g.position.y - y;
                (dx * dx + dy * dy).sqrt() < min_dist
            });

            if too_close { continue; }

            let px = ((x * target.width as f32) as u32).min(target.width - 1);
            let py = ((y * target.height as f32) as u32).min(target.height - 1);
            let color = target.get_pixel(px, py).unwrap();

            bg.push(Gaussian2D::new(
                Vector2::new(x, y),
                Euler::new(uniform_size, uniform_size, 0.0),
                color,
                1.0,
            ));
        }
    }

    bg
}

// Helper types and functions
struct PixelFeature {
    r: f32, g: f32, b: f32,
    x: f32, y: f32,
    gradient: f32,
    orig_x: u32, orig_y: u32,
}

fn kmeans_simple(features: &[PixelFeature], k: usize, max_iters: usize) -> (Vec<PixelFeature>, Vec<usize>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Random initialization
    let mut centroids: Vec<PixelFeature> = (0..k)
        .map(|_| {
            let f = &features[rng.gen_range(0..features.len())];
            PixelFeature {
                r: f.r, g: f.g, b: f.b, x: f.x, y: f.y,
                gradient: f.gradient, orig_x: 0, orig_y: 0
            }
        })
        .collect();

    let mut assignments = vec![0usize; features.len()];

    for _ in 0..max_iters {
        // Assign
        for (i, f) in features.iter().enumerate() {
            let mut best_d = f32::INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let d = (f.r - c.r).powi(2) + (f.g - c.g).powi(2) + (f.b - c.b).powi(2)
                    + (f.x - c.x).powi(2) + (f.y - c.y).powi(2) + (f.gradient - c.gradient).powi(2);
                if d < best_d {
                    best_d = d;
                    assignments[i] = j;
                }
            }
        }

        // Update
        let mut sums = vec![(0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0usize); k];
        for (i, f) in features.iter().enumerate() {
            let c = assignments[i];
            sums[c].0 += f.r;
            sums[c].1 += f.g;
            sums[c].2 += f.b;
            sums[c].3 += f.x;
            sums[c].4 += f.y;
            sums[c].5 += f.gradient;
            sums[c].6 += 1;
        }

        for (j, s) in sums.iter().enumerate() {
            if s.6 > 0 {
                let n = s.6 as f32;
                centroids[j] = PixelFeature {
                    r: s.0 / n, g: s.1 / n, b: s.2 / n,
                    x: s.3 / n, y: s.4 / n, gradient: s.5 / n,
                    orig_x: 0, orig_y: 0,
                };
            }
        }
    }

    (centroids, assignments)
}

fn find_local_maxima(map: &[f32], width: u32, height: u32, threshold: f32, window: usize) -> Vec<(u32, u32, f32)> {
    let half = window / 2;
    let mut peaks = Vec::new();

    for y in half..(height as usize - half) {
        for x in half..(width as usize - half) {
            let idx = y * width as usize + x;
            let val = map[idx];
            if val < threshold { continue; }

            let mut is_max = true;
            for dy in -(half as i32)..=(half as i32) {
                for dx in -(half as i32)..=(half as i32) {
                    if dx == 0 && dy == 0 { continue; }
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    if map[ny * width as usize + nx] > val {
                        is_max = false;
                        break;
                    }
                }
                if !is_max { break; }
            }

            if is_max {
                peaks.push((x as u32, y as u32, val));
            }
        }
    }

    peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    peaks
}

fn compute_laplacian(image: &ImageBuffer<f32>) -> Vec<f32> {
    let w = image.width as usize;
    let h = image.height as usize;
    let mut laplacian = vec![0.0f32; w * h];

    // Laplacian kernel: [0,-1,0], [-1,4,-1], [0,-1,0]
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let get_lum = |px: usize, py: usize| -> f32 {
                let c = image.get_pixel(px as u32, py as u32).unwrap();
                0.299 * c.r + 0.587 * c.g + 0.114 * c.b
            };

            let center = get_lum(x, y);
            let north = get_lum(x, y - 1);
            let south = get_lum(x, y + 1);
            let west = get_lum(x - 1, y);
            let east = get_lum(x + 1, y);

            laplacian[y * w + x] = (4.0 * center - north - south - west - east).abs();
        }
    }

    laplacian
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height * 3) as f32;
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= count;
    if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}
