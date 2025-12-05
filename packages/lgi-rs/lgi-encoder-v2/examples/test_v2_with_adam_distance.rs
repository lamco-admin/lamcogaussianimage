//! Test V2's structure but with Adam's simplified isotropic distance
//!
//! Hypothesis: V2 fails because rotated Mahalanobis distance != renderer
//! If we use Adam's simplified distance but keep everything else, should work.
//!
//! Run: cargo run --release --example test_v2_with_adam_distance -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::{TestResult, save_rendered_image};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   V2 STRUCTURE + ADAM DISTANCE EXPERIMENT                    ║");
    println!("║   Testing if simplified isotropic distance fixes V2          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load");
    let (width, height) = (target.width, target.height);

    let grid_size = 16;
    let n_gaussians = grid_size * grid_size;
    let iterations = 100;

    println!("Image: {}x{}, Gaussians: {}, Iterations: {}\n", width, height, n_gaussians, iterations);

    // Create initial Gaussians
    let mut gaussians = create_gaussians(&target, grid_size);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, &target);

    println!("Initial PSNR: {:.2} dB\n", init_psnr);

    // Adam parameters
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let epsilon = 1e-8f32;
    let lr = 0.01f32;

    let n_params = n_gaussians * 6;  // position(2), color(3), scale(1 - product)
    let mut m = vec![0.0f32; n_params];
    let mut v = vec![0.0f32; n_params];

    let start = Instant::now();
    let mut best_loss = f32::INFINITY;
    let mut best_gaussians = gaussians.clone();
    let mut patience = 0;

    for iter in 0..iterations {
        let rendered = RendererV2::render(&gaussians, width, height);
        let loss = compute_loss(&rendered, &target);

        if iter % 10 == 0 {
            let psnr = compute_psnr(&rendered, &target);
            println!("  Iteration {}: loss = {:.6}, PSNR = {:.2} dB", iter, loss, psnr);
        }

        if loss < best_loss {
            best_loss = loss;
            best_gaussians.clone_from_slice(&gaussians);
            patience = 0;
        } else {
            patience += 1;
            if patience >= 20 {
                println!("  Early stop at iteration {}", iter);
                break;
            }
        }

        // Compute gradients using ADAM'S SIMPLIFIED DISTANCE (isotropic)
        let grads = compute_adam_style_gradients(&gaussians, &rendered, &target);

        // Apply Adam update
        let bias_correction1 = 1.0 - beta1.powi(iter as i32 + 1);
        let bias_correction2 = 1.0 - beta2.powi(iter as i32 + 1);

        for (i, (gaussian, grad)) in gaussians.iter_mut().zip(grads.iter()).enumerate() {
            let base_idx = i * 6;

            // Position X
            let g = grad.position_x;
            m[base_idx] = beta1 * m[base_idx] + (1.0 - beta1) * g;
            v[base_idx] = beta2 * v[base_idx] + (1.0 - beta2) * g * g;
            let m_hat = m[base_idx] / bias_correction1;
            let v_hat = v[base_idx] / bias_correction2;
            gaussian.position.x -= lr * m_hat / (v_hat.sqrt() + epsilon);
            gaussian.position.x = gaussian.position.x.clamp(0.0, 1.0);

            // Position Y
            let g = grad.position_y;
            m[base_idx + 1] = beta1 * m[base_idx + 1] + (1.0 - beta1) * g;
            v[base_idx + 1] = beta2 * v[base_idx + 1] + (1.0 - beta2) * g * g;
            let m_hat = m[base_idx + 1] / bias_correction1;
            let v_hat = v[base_idx + 1] / bias_correction2;
            gaussian.position.y -= lr * m_hat / (v_hat.sqrt() + epsilon);
            gaussian.position.y = gaussian.position.y.clamp(0.0, 1.0);

            // Color R
            let g = grad.color_r;
            m[base_idx + 2] = beta1 * m[base_idx + 2] + (1.0 - beta1) * g;
            v[base_idx + 2] = beta2 * v[base_idx + 2] + (1.0 - beta2) * g * g;
            let m_hat = m[base_idx + 2] / bias_correction1;
            let v_hat = v[base_idx + 2] / bias_correction2;
            gaussian.color.r -= lr * m_hat / (v_hat.sqrt() + epsilon);
            gaussian.color.r = gaussian.color.r.clamp(0.0, 1.0);

            // Color G
            let g = grad.color_g;
            m[base_idx + 3] = beta1 * m[base_idx + 3] + (1.0 - beta1) * g;
            v[base_idx + 3] = beta2 * v[base_idx + 3] + (1.0 - beta2) * g * g;
            let m_hat = m[base_idx + 3] / bias_correction1;
            let v_hat = v[base_idx + 3] / bias_correction2;
            gaussian.color.g -= lr * m_hat / (v_hat.sqrt() + epsilon);
            gaussian.color.g = gaussian.color.g.clamp(0.0, 1.0);

            // Color B
            let g = grad.color_b;
            m[base_idx + 4] = beta1 * m[base_idx + 4] + (1.0 - beta1) * g;
            v[base_idx + 4] = beta2 * v[base_idx + 4] + (1.0 - beta2) * g * g;
            let m_hat = m[base_idx + 4] / bias_correction1;
            let v_hat = v[base_idx + 4] / bias_correction2;
            gaussian.color.b -= lr * m_hat / (v_hat.sqrt() + epsilon);
            gaussian.color.b = gaussian.color.b.clamp(0.0, 1.0);

            // Scale (update both together using simplified gradient)
            let g = grad.scale;
            m[base_idx + 5] = beta1 * m[base_idx + 5] + (1.0 - beta1) * g;
            v[base_idx + 5] = beta2 * v[base_idx + 5] + (1.0 - beta2) * g * g;
            let m_hat = m[base_idx + 5] / bias_correction1;
            let v_hat = v[base_idx + 5] / bias_correction2;
            let scale_update = lr * m_hat / (v_hat.sqrt() + epsilon);
            gaussian.shape.scale_x -= scale_update;
            gaussian.shape.scale_y -= scale_update;
            gaussian.shape.scale_x = gaussian.shape.scale_x.clamp(0.01, 0.25);
            gaussian.shape.scale_y = gaussian.shape.scale_y.clamp(0.01, 0.25);
        }
    }

    let elapsed = start.elapsed();

    gaussians.copy_from_slice(&best_gaussians);
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_loss = compute_loss(&final_render, &target);
    let final_psnr = compute_psnr(&final_render, &target);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("RESULTS:");
    println!("  Initial PSNR: {:.2} dB", init_psnr);
    println!("  Final PSNR:   {:.2} dB", final_psnr);
    println!("  Improvement:  {:+.2} dB", final_psnr - init_psnr);
    println!("  Time:         {:.1}s", elapsed.as_secs_f32());
    println!("═══════════════════════════════════════════════════════════════");

    let mut result = TestResult::new("v2_with_adam_distance", image_path);
    result.optimizer = "V2 + Adam-style isotropic distance".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = n_gaussians;
    result.iterations = iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.final_loss = final_loss;
    result.elapsed_seconds = elapsed.as_secs_f32();
    result.extra_metrics = Some(json!({
        "hypothesis": "Adam's simplified distance is key to convergence",
        "distance_type": "isotropic (dx² + dy²)"
    }));
    let _ = result.save(RESULTS_DIR);
    let _ = save_rendered_image(&final_render, RESULTS_DIR, "v2_adam_distance", "final");

    if final_psnr - init_psnr > 1.0 {
        println!("\n✅ HYPOTHESIS CONFIRMED: Simplified distance fixes V2!");
    } else if final_psnr - init_psnr > 0.5 {
        println!("\n🔶 PARTIAL: Better than V2, but not matching Adam");
    } else {
        println!("\n❌ HYPOTHESIS REJECTED: Distance formula is not the issue");
    }
}

#[derive(Clone)]
struct SimpleGradient {
    position_x: f32,
    position_y: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    scale: f32,
}

/// Adam-style gradient using simplified isotropic distance
fn compute_adam_style_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
) -> Vec<SimpleGradient> {
    let width = target.width;
    let height = target.height;
    let mut gradients: Vec<SimpleGradient> = gaussians.iter().map(|_| SimpleGradient {
        position_x: 0.0, position_y: 0.0,
        color_r: 0.0, color_g: 0.0, color_b: 0.0,
        scale: 0.0,
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

                // ADAM'S SIMPLIFIED ISOTROPIC DISTANCE (no rotation)
                let dist_sq = dx * dx + dy * dy;
                let scale_product = (gaussian.shape.scale_x * gaussian.shape.scale_y).max(1e-6);

                // Early cutoff (Adam uses 0.1 in normalized coords)
                if dist_sq > 0.1 { continue; }

                // Adam's weight formula
                let weight = (-0.5 * dist_sq / scale_product).exp();
                if weight < 1e-6 { continue; }

                // Color gradient
                gradients[i].color_r += error_r * weight;
                gradients[i].color_g += error_g * weight;
                gradients[i].color_b += error_b * weight;

                let error_weighted = error_r * gaussian.color.r +
                                    error_g * gaussian.color.g +
                                    error_b * gaussian.color.b;

                // Position gradient (Adam's simple form)
                gradients[i].position_x += error_weighted * weight * dx;
                gradients[i].position_y += error_weighted * weight * dy;

                // Scale gradient (Adam's simplified form)
                gradients[i].scale += error_weighted * weight * dist_sq / scale_product;
            }
        }
    }

    // Normalize
    let pixel_count = (width * height) as f32;
    for grad in &mut gradients {
        grad.position_x /= pixel_count;
        grad.position_y /= pixel_count;
        grad.color_r /= pixel_count;
        grad.color_g /= pixel_count;
        grad.color_b /= pixel_count;
        grad.scale /= pixel_count;
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
