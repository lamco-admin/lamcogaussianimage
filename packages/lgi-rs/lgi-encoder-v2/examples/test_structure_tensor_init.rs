//! Test Structure Tensor-Guided Initialization
//!
//! The structure tensor provides edge orientation and coherence,
//! allowing us to create anisotropic Gaussians with shapes derived
//! from image analysis rather than gradient descent.
//!
//! This tests:
//! 1. Structure tensor computation
//! 2. Anisotropic Gaussian shape from eigenanalysis
//! 3. Comparison with isotropic initialization
//! 4. Different optimizer compatibility (Adam vs direct placement)
//!
//! Run: cargo run --release --example test_structure_tensor_init -p lgi-encoder-v2

use lgi_core::{
    ImageBuffer,
    structure_tensor::StructureTensorField,
    initializer::{Initializer, InitStrategy},
    laplacian::{compute_laplacian, find_local_maxima},
};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   STRUCTURE TENSOR-GUIDED INITIALIZATION                      ║");
    println!("║   Anisotropic Gaussians from image analysis                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);
    println!("");

    // Step 1: Compute structure tensor field
    println!("═══════════════════════════════════════════════════════════════");
    println!("Step 1: Computing Structure Tensor Field");
    println!("═══════════════════════════════════════════════════════════════");

    let tensor_start = Instant::now();
    let tensor_field = StructureTensorField::compute(&target, 1.2, 2.5)
        .expect("Failed to compute structure tensor");
    let tensor_time = tensor_start.elapsed().as_secs_f32();

    // Analyze tensor statistics
    let mut avg_coherence = 0.0;
    let mut edge_count = 0;
    let mut corner_count = 0;
    let mut flat_count = 0;

    for y in 0..height {
        for x in 0..width {
            let tensor = tensor_field.get(x, y);
            avg_coherence += tensor.coherence;

            if tensor.is_edge(0.7) {
                edge_count += 1;
            } else if tensor.is_corner(0.3, 0.01) {
                corner_count += 1;
            } else if tensor.is_flat(0.001) {
                flat_count += 1;
            }
        }
    }

    let total_pixels = (width * height) as f32;
    avg_coherence /= total_pixels;

    println!("  Computation time: {:.3}s", tensor_time);
    println!("  Average coherence: {:.3}", avg_coherence);
    println!("  Edge pixels: {} ({:.1}%)", edge_count, 100.0 * edge_count as f32 / total_pixels);
    println!("  Corner pixels: {} ({:.1}%)", corner_count, 100.0 * corner_count as f32 / total_pixels);
    println!("  Flat pixels: {} ({:.1}%)", flat_count, 100.0 * flat_count as f32 / total_pixels);

    // Step 2: Find Laplacian peaks for placement
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 2: Finding Laplacian Peaks for Gaussian Placement");
    println!("═══════════════════════════════════════════════════════════════");

    let laplacian = compute_laplacian(&target);
    let peaks = find_local_maxima(&laplacian, width, height, 0.02, 8);

    println!("  Found {} Laplacian peaks", peaks.len());
    if peaks.len() > 5 {
        println!("  Top 5 peaks: {:?}", &peaks[..5]);
    }

    // Step 3: Create structure tensor-guided Gaussians
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 3: Creating Structure Tensor-Guided Gaussians");
    println!("═══════════════════════════════════════════════════════════════");

    let n_gaussians = 256;
    let st_gaussians = create_structure_tensor_gaussians(&target, &tensor_field, &peaks, n_gaussians);

    // Render and measure
    let st_render = RendererV2::render(&st_gaussians, width, height);
    let st_psnr = compute_psnr(&st_render, &target);

    println!("  Created {} anisotropic Gaussians", st_gaussians.len());
    println!("  Initial PSNR: {:.2} dB", st_psnr);

    // Analyze anisotropy
    let anisotropy_stats = analyze_anisotropy(&st_gaussians);
    println!("  Anisotropy stats:");
    println!("    Min: {:.3}", anisotropy_stats.0);
    println!("    Max: {:.3}", anisotropy_stats.1);
    println!("    Mean: {:.3}", anisotropy_stats.2);
    println!("    % highly anisotropic (>2×): {:.1}%", anisotropy_stats.3);

    // Step 4: Compare with isotropic initialization
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 4: Comparing with Isotropic Grid Initialization");
    println!("═══════════════════════════════════════════════════════════════");

    let iso_gaussians = create_isotropic_grid(&target, n_gaussians);
    let iso_render = RendererV2::render(&iso_gaussians, width, height);
    let iso_psnr = compute_psnr(&iso_render, &target);

    println!("  Created {} isotropic Gaussians", iso_gaussians.len());
    println!("  Initial PSNR: {:.2} dB", iso_psnr);
    println!("  Difference: {:+.2} dB (ST vs ISO)", st_psnr - iso_psnr);

    // Step 5: Optimize both with Adam
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 5: Optimizing Both with Adam (100 iterations)");
    println!("═══════════════════════════════════════════════════════════════");

    // Optimize ST-guided
    let mut st_opt = st_gaussians.clone();
    let mut adam_st = AdamOptimizer::new();
    adam_st.max_iterations = 100;

    println!("\n  [ST-Guided + Adam]");
    let st_opt_start = Instant::now();
    adam_st.optimize(&mut st_opt, &target);
    let st_opt_time = st_opt_start.elapsed().as_secs_f32();

    let st_opt_render = RendererV2::render(&st_opt, width, height);
    let st_opt_psnr = compute_psnr(&st_opt_render, &target);
    println!("    Final PSNR: {:.2} dB (+{:.2} dB from init)", st_opt_psnr, st_opt_psnr - st_psnr);
    println!("    Time: {:.2}s", st_opt_time);

    // Optimize isotropic
    let mut iso_opt = iso_gaussians.clone();
    let mut adam_iso = AdamOptimizer::new();
    adam_iso.max_iterations = 100;

    println!("\n  [Isotropic Grid + Adam]");
    let iso_opt_start = Instant::now();
    adam_iso.optimize(&mut iso_opt, &target);
    let iso_opt_time = iso_opt_start.elapsed().as_secs_f32();

    let iso_opt_render = RendererV2::render(&iso_opt, width, height);
    let iso_opt_psnr = compute_psnr(&iso_opt_render, &target);
    println!("    Final PSNR: {:.2} dB (+{:.2} dB from init)", iso_opt_psnr, iso_opt_psnr - iso_psnr);
    println!("    Time: {:.2}s", iso_opt_time);

    // Step 6: Test WITHOUT optimization (closed-form only)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 6: Closed-Form Color Optimization Only (No Adam)");
    println!("═══════════════════════════════════════════════════════════════");

    // This is key: can we get good results without any gradient descent?
    let st_closed = create_structure_tensor_gaussians_with_optimal_colors(&target, &tensor_field, &peaks, n_gaussians);
    let st_closed_render = RendererV2::render(&st_closed, width, height);
    let st_closed_psnr = compute_psnr(&st_closed_render, &target);

    println!("  Structure Tensor + Closed-Form Colors");
    println!("    PSNR: {:.2} dB", st_closed_psnr);
    println!("    vs Adam-optimized ST: {:+.2} dB", st_closed_psnr - st_opt_psnr);
    println!("    vs Adam-optimized ISO: {:+.2} dB", st_closed_psnr - iso_opt_psnr);

    // Summary
    println!("\n\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY                                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("");
    println!("{:<35} {:>10} {:>10}", "Method", "Init dB", "Final dB");
    println!("{}", "─".repeat(60));
    println!("{:<35} {:>10.2} {:>10.2}", "ST-Guided + Adam", st_psnr, st_opt_psnr);
    println!("{:<35} {:>10.2} {:>10.2}", "Isotropic Grid + Adam", iso_psnr, iso_opt_psnr);
    println!("{:<35} {:>10.2} {:>10.2}", "ST-Guided + Closed-Form (No Adam)", st_closed_psnr, st_closed_psnr);
    println!("");
    println!("Key insight: Does structure tensor init help Adam converge better?");
    println!("  ST vs ISO improvement after Adam: {:+.2} dB", st_opt_psnr - iso_opt_psnr);

    // Save results
    let mut result = TestResult::new("structure_tensor_init", image_path);
    result.optimizer = "Structure Tensor-Guided Init".to_string();
    result.n_gaussians = n_gaussians;
    result.iterations = 100;
    result.initial_psnr = st_psnr;
    result.final_psnr = st_opt_psnr;
    result.improvement_db = st_opt_psnr - st_psnr;
    result.elapsed_seconds = tensor_time + st_opt_time;
    result.image_dimensions = (width, height);
    result.set_param("tensor_time_s", tensor_time);
    result.set_param("avg_coherence", avg_coherence);
    result.set_param("edge_percent", 100.0 * edge_count as f32 / total_pixels);
    result.set_param("iso_init_psnr", iso_psnr);
    result.set_param("iso_final_psnr", iso_opt_psnr);
    result.set_param("st_vs_iso_delta", st_opt_psnr - iso_opt_psnr);
    result.set_param("closed_form_psnr", st_closed_psnr);
    result.set_param("anisotropy_mean", anisotropy_stats.2);
    result.set_param("anisotropy_high_percent", anisotropy_stats.3);
    result.complete();

    let _ = result.save_with_image(&st_opt_render, RESULTS_DIR);

    println!("\n✓ Structure tensor initialization test complete!");
}

/// Create Gaussians with shape derived from structure tensor
fn create_structure_tensor_gaussians(
    target: &ImageBuffer<f32>,
    tensor_field: &StructureTensorField,
    peaks: &[(u32, u32, f32)],
    n: usize,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n);

    // Use top N peaks as placement positions
    for (i, &(px, py, _strength)) in peaks.iter().take(n).enumerate() {
        let tensor = tensor_field.get(px, py);

        // Normalized position
        let x = (px as f32 + 0.5) / width as f32;
        let y = (py as f32 + 0.5) / height as f32;

        // Get color from image
        let color = target.get_pixel(px, py)
            .cloned()
            .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        // Derive shape from structure tensor eigenanalysis
        // - Major eigenvalue → thin along edge normal
        // - Minor eigenvalue → long along edge tangent
        // - Orientation from eigenvector

        let base_scale = 0.02;  // Base scale in normalized coords

        // Anisotropy from eigenvalue ratio (clamped)
        let anisotropy = if tensor.eigenvalue_minor > 1e-6 {
            (tensor.eigenvalue_major / tensor.eigenvalue_minor).sqrt().clamp(1.0, 5.0)
        } else {
            1.0
        };

        // Scale based on coherence: high coherence = more anisotropic
        let scale_factor = 1.0 + tensor.coherence;

        // For edges: thin across edge, long along edge
        // Major eigenvector is edge NORMAL, minor is edge TANGENT
        // So scale_x (along rotation direction) should be LARGER for edge tangent
        let scale_x = base_scale * scale_factor * anisotropy;  // Along tangent (longer)
        let scale_y = base_scale * scale_factor / anisotropy;  // Across edge (thinner)

        // Rotation = angle of edge tangent (minor eigenvector)
        let rotation = tensor.eigenvector_minor.y.atan2(tensor.eigenvector_minor.x);

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

    // If we need more Gaussians than peaks, fill with grid
    if gaussians.len() < n {
        let remaining = n - gaussians.len();
        let grid_size = (remaining as f32).sqrt().ceil() as usize;

        for i in 0..grid_size {
            for j in 0..grid_size {
                if gaussians.len() >= n {
                    break;
                }

                let x = (i as f32 + 0.5) / grid_size as f32;
                let y = (j as f32 + 0.5) / grid_size as f32;

                let px = (x * width as f32) as u32;
                let py = (y * height as f32) as u32;

                let tensor = tensor_field.get(px.min(width - 1), py.min(height - 1));
                let color = target.get_pixel(px.min(width - 1), py.min(height - 1))
                    .cloned()
                    .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

                let rotation = tensor.eigenvector_minor.y.atan2(tensor.eigenvector_minor.x);

                gaussians.push(Gaussian2D {
                    position: Vector2::new(x, y),
                    shape: Euler {
                        scale_x: 0.02,
                        scale_y: 0.02,
                        rotation,
                    },
                    color: Color4::rgb(color.r, color.g, color.b),
                    opacity: 0.8,
                });
            }
        }
    }

    gaussians
}

/// Create Gaussians with closed-form optimal colors
fn create_structure_tensor_gaussians_with_optimal_colors(
    target: &ImageBuffer<f32>,
    tensor_field: &StructureTensorField,
    peaks: &[(u32, u32, f32)],
    n: usize,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    // Start with structure tensor-guided placement
    let mut gaussians = create_structure_tensor_gaussians(target, tensor_field, peaks, n);

    // Optimize colors using weighted average of covered pixels
    // This is closed-form, no gradient descent
    let (width, height) = (target.width, target.height);

    for g in &mut gaussians {
        let cx = g.position.x * width as f32;
        let cy = g.position.y * height as f32;
        let sx = g.shape.scale_x * width as f32;
        let sy = g.shape.scale_y * height as f32;

        // Sample region around Gaussian
        let radius = (sx.max(sy) * 3.0) as i32;
        let mut sum_r = 0.0;
        let mut sum_g = 0.0;
        let mut sum_b = 0.0;
        let mut sum_weight = 0.0;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let px = (cx as i32 + dx).clamp(0, width as i32 - 1) as u32;
                let py = (cy as i32 + dy).clamp(0, height as i32 - 1) as u32;

                if let Some(pixel) = target.get_pixel(px, py) {
                    // Gaussian weight
                    let x_dist = dx as f32 / sx;
                    let y_dist = dy as f32 / sy;
                    let weight = (-0.5 * (x_dist * x_dist + y_dist * y_dist)).exp();

                    sum_r += pixel.r * weight;
                    sum_g += pixel.g * weight;
                    sum_b += pixel.b * weight;
                    sum_weight += weight;
                }
            }
        }

        if sum_weight > 0.0 {
            g.color = Color4::rgb(
                sum_r / sum_weight,
                sum_g / sum_weight,
                sum_b / sum_weight,
            );
        }
    }

    gaussians
}

/// Create isotropic grid Gaussians for comparison
fn create_isotropic_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
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
                    scale_x: 0.02,
                    scale_y: 0.02,
                    rotation: 0.0,
                },
                color: Color4::rgb(color.r, color.g, color.b),
                opacity: 0.8,
            });
        }
    }

    gaussians
}

/// Analyze anisotropy statistics
fn analyze_anisotropy(gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> (f32, f32, f32, f32) {
    let mut min_ratio = f32::INFINITY;
    let mut max_ratio = 0.0;
    let mut sum_ratio = 0.0;
    let mut high_aniso_count = 0;

    for g in gaussians {
        let ratio = (g.shape.scale_x / g.shape.scale_y.max(0.001)).max(g.shape.scale_y / g.shape.scale_x.max(0.001));
        min_ratio = min_ratio.min(ratio);
        max_ratio = max_ratio.max(ratio);
        sum_ratio += ratio;

        if ratio > 2.0 {
            high_aniso_count += 1;
        }
    }

    let mean_ratio = sum_ratio / gaussians.len() as f32;
    let high_percent = 100.0 * high_aniso_count as f32 / gaussians.len() as f32;

    (min_ratio, max_ratio, mean_ratio, high_percent)
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
