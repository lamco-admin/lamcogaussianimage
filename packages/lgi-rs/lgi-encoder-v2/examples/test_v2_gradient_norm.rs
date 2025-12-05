//! Test V2 with gradient normalization
//!
//! Hypothesis: V2's gradients are smaller in magnitude than Adam's.
//! If we normalize gradients to unit length before applying LR,
//! this should compensate for magnitude differences.
//!
//! Run: cargo run --release --example test_v2_gradient_norm -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::{TestResult, save_rendered_image};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   V2 WITH GRADIENT NORMALIZATION                             ║");
    println!("║   Testing if normalizing gradient magnitude helps            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load");
    let (width, height) = (target.width, target.height);

    let grid_size = 16;
    let n_gaussians = grid_size * grid_size;
    let iterations = 100;

    println!("Image: {}x{}, Gaussians: {}, Iterations: {}\n", width, height, n_gaussians, iterations);

    // Test configurations
    let configs = vec![
        ("No normalization (baseline)", false, 0.1, 0.6, 0.1),
        ("Gradient norm + default LR", true, 0.1, 0.6, 0.1),
        ("Gradient norm + lower LR", true, 0.01, 0.1, 0.01),
        ("Gradient norm + Adam-like LR", true, 0.001, 0.02, 0.005),
    ];

    for (name, normalize, lr_pos, lr_color, lr_scale) in &configs {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Testing: {}", name);
        println!("  normalize={}, lr_pos={}, lr_color={}, lr_scale={}",
                 normalize, lr_pos, lr_color, lr_scale);
        println!("═══════════════════════════════════════════════════════════════");

        let mut gaussians = create_gaussians(&target, grid_size);

        let init_render = RendererV2::render(&gaussians, width, height);
        let init_psnr = compute_psnr(&init_render, &target);

        let start = Instant::now();
        let final_loss = optimize_with_norm(
            &mut gaussians,
            &target,
            iterations,
            *normalize,
            *lr_pos,
            *lr_color,
            *lr_scale,
        );
        let elapsed = start.elapsed();

        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(&final_render, &target);

        println!("  Initial PSNR: {:.2} dB", init_psnr);
        println!("  Final PSNR:   {:.2} dB", final_psnr);
        println!("  Improvement:  {:+.2} dB", final_psnr - init_psnr);
        println!("  Time:         {:.1}s\n", elapsed.as_secs_f32());

        // Save result
        let exp_name = format!("v2_grad_norm_{}", name.replace(" ", "_").replace("(", "").replace(")", ""));
        let mut result = TestResult::new(&exp_name, image_path);
        result.optimizer = format!("V2 Gradient Norm: {}", name);
        result.image_dimensions = (width, height);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.final_loss = final_loss;
        result.elapsed_seconds = elapsed.as_secs_f32();
        result.extra_metrics = Some(json!({
            "normalize": normalize,
            "lr_position": lr_pos,
            "lr_color": lr_color,
            "lr_scale": lr_scale,
        }));
        let _ = result.save(RESULTS_DIR);
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("HYPOTHESIS: If gradient normalization helps, normalized configs");
    println!("should outperform baseline with appropriate LR tuning.");
    println!("═══════════════════════════════════════════════════════════════");
}

fn optimize_with_norm(
    gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    max_iterations: usize,
    normalize: bool,
    lr_pos: f32,
    lr_color: f32,
    lr_scale: f32,
) -> f32 {
    let width = target.width;
    let height = target.height;
    let mut best_loss = f32::INFINITY;
    let mut patience = 0;
    let mut best_gaussians = gaussians.to_vec();

    for iter in 0..max_iterations {
        let rendered = RendererV2::render(gaussians, width, height);
        let loss = compute_loss(&rendered, target);

        if iter % 10 == 0 {
            println!("  Iteration {}: loss = {:.6}", iter, loss);
        }

        if loss < best_loss {
            best_loss = loss;
            best_gaussians.clone_from_slice(gaussians);
            patience = 0;
        } else {
            patience += 1;
            if patience >= 20 {
                println!("  Early stop at iteration {}", iter);
                break;
            }
        }

        // Compute V2-style gradients
        let grads = compute_v2_gradients(gaussians, &rendered, target);

        // Apply updates with optional normalization
        for (gaussian, grad) in gaussians.iter_mut().zip(grads.iter()) {
            let (pos_x, pos_y, color_r, color_g, color_b, scale_x, scale_y) = if normalize {
                // Normalize each parameter group
                let pos_mag = (grad.position_x.powi(2) + grad.position_y.powi(2)).sqrt().max(1e-8);
                let color_mag = (grad.color_r.powi(2) + grad.color_g.powi(2) + grad.color_b.powi(2)).sqrt().max(1e-8);
                let scale_mag = (grad.scale_x.powi(2) + grad.scale_y.powi(2)).sqrt().max(1e-8);

                (
                    grad.position_x / pos_mag,
                    grad.position_y / pos_mag,
                    grad.color_r / color_mag,
                    grad.color_g / color_mag,
                    grad.color_b / color_mag,
                    grad.scale_x / scale_mag,
                    grad.scale_y / scale_mag,
                )
            } else {
                (
                    grad.position_x,
                    grad.position_y,
                    grad.color_r,
                    grad.color_g,
                    grad.color_b,
                    grad.scale_x,
                    grad.scale_y,
                )
            };

            // Apply with sign corrections (position.x needs MINUS)
            gaussian.position.x -= lr_pos * pos_x;
            gaussian.position.y += lr_pos * pos_y;
            gaussian.color.r -= lr_color * color_r;
            gaussian.color.g -= lr_color * color_g;
            gaussian.color.b -= lr_color * color_b;
            gaussian.shape.scale_x -= lr_scale * scale_x;
            gaussian.shape.scale_y -= lr_scale * scale_y;

            // Clamp
            gaussian.position.x = gaussian.position.x.clamp(0.0, 1.0);
            gaussian.position.y = gaussian.position.y.clamp(0.0, 1.0);
            gaussian.color.r = gaussian.color.r.clamp(0.0, 1.0);
            gaussian.color.g = gaussian.color.g.clamp(0.0, 1.0);
            gaussian.color.b = gaussian.color.b.clamp(0.0, 1.0);
            gaussian.shape.scale_x = gaussian.shape.scale_x.clamp(0.01, 0.25);
            gaussian.shape.scale_y = gaussian.shape.scale_y.clamp(0.01, 0.25);
        }
    }

    gaussians.copy_from_slice(&best_gaussians);
    best_loss
}

#[derive(Clone)]
struct V2Gradient {
    position_x: f32,
    position_y: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    scale_x: f32,
    scale_y: f32,
}

fn compute_v2_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
) -> Vec<V2Gradient> {
    let width = target.width;
    let height = target.height;
    let mut gradients: Vec<V2Gradient> = gaussians.iter().map(|_| V2Gradient {
        position_x: 0.0, position_y: 0.0,
        color_r: 0.0, color_g: 0.0, color_b: 0.0,
        scale_x: 0.0, scale_y: 0.0,
    }).collect();

    for y in 0..height {
        for x in 0..width {
            let px = x as f32 / width as f32;
            let py = y as f32 / height as f32;

            let rendered_color = rendered.get_pixel(x, y).unwrap();
            let target_color = target.get_pixel(x, y).unwrap();

            let error_r = 2.0 * (rendered_color.r - target_color.r);
            let error_g = 2.0 * (rendered_color.g - target_color.g);
            let error_b = 2.0 * (rendered_color.b - target_color.b);

            for (i, gaussian) in gaussians.iter().enumerate() {
                let dx = px - gaussian.position.x;
                let dy = py - gaussian.position.y;

                let sx = gaussian.shape.scale_x.max(1e-6);
                let sy = gaussian.shape.scale_y.max(1e-6);
                let theta = gaussian.shape.rotation;
                let cos_t = theta.cos();
                let sin_t = theta.sin();

                // Rotated coordinates
                let dx_rot = dx * cos_t + dy * sin_t;
                let dy_rot = -dx * sin_t + dy * cos_t;

                // Mahalanobis distance
                let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);
                if dist_sq > 16.0 { continue; }

                let weight = (-0.5 * dist_sq).exp();
                if weight < 1e-6 { continue; }

                // Color gradient
                gradients[i].color_r += error_r * weight;
                gradients[i].color_g += error_g * weight;
                gradients[i].color_b += error_b * weight;

                // Error weighted by color
                let error_weighted = error_r * gaussian.color.r +
                                    error_g * gaussian.color.g +
                                    error_b * gaussian.color.b;

                // Position gradients (V2 style with rotation)
                let grad_weight_x = weight * (dx_rot * cos_t / (sx * sx) - dy_rot * sin_t / (sy * sy));
                let grad_weight_y = weight * (dx_rot * sin_t / (sx * sx) + dy_rot * cos_t / (sy * sy));

                gradients[i].position_x += error_weighted * grad_weight_x;
                gradients[i].position_y += error_weighted * grad_weight_y;

                // Scale gradients
                gradients[i].scale_x += error_weighted * weight * (dx_rot * dx_rot) / (sx * sx * sx);
                gradients[i].scale_y += error_weighted * weight * (dy_rot * dy_rot) / (sy * sy * sy);
            }
        }
    }

    // Normalize by pixel count
    let pixel_count = (width * height) as f32;
    for grad in &mut gradients {
        grad.position_x /= pixel_count;
        grad.position_y /= pixel_count;
        grad.color_r /= pixel_count;
        grad.color_g /= pixel_count;
        grad.color_b /= pixel_count;
        grad.scale_x /= pixel_count;
        grad.scale_y /= pixel_count;
    }

    gradients
}

fn create_gaussians(target: &ImageBuffer<f32>, grid_size: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(grid_size * grid_size);
    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx as f32 + 0.5) / grid_size as f32;
            let y = (gy as f32 + 0.5) / grid_size as f32;
            let px = ((x * width as f32) as u32).min(width - 1);
            let py = ((y * height as f32) as u32).min(height - 1);
            let color = target.get_pixel(px, py).unwrap();
            let scale = 1.0 / grid_size as f32;
            gaussians.push(Gaussian2D::new(
                Vector2::new(x, y),
                Euler::new(scale, scale, 0.0),
                color,
                1.0,
            ));
        }
    }
    gaussians
}

fn compute_loss(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height) as f32;
    let mut loss = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        loss += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    loss / count
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
