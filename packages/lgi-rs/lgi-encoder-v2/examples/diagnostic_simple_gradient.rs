//! Diagnostic: Simple gradient with detailed logging
//!
//! Debug Plan Section 0 - Prove the loop is alive

use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════╗");
    println!("║  DIAGNOSTIC: Gradient with 64 Gaussians     ║");
    println!("║  Debug Plan Section 0 - Prove Loop Lives    ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Create linear gradient (blue → red)
    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let t = x as f32 / 255.0;
            target.set_pixel(x, y, Color4::new(t, 0.0, 1.0 - t, 1.0));
        }
    }

    println!("Test: 8×8 grid (64 Gaussians)\n");

    let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
    let mut gaussians = encoder.initialize_gaussians(8);

    println!("Initialization check:");
    println!("  Number of Gaussians: {}", gaussians.len());
    println!("  Gaussian[0] position: ({:.3}, {:.3})", gaussians[0].position.x, gaussians[0].position.y);
    println!("  Gaussian[0] scales: sx={:.4}, sy={:.4}", gaussians[0].shape.scale_x, gaussians[0].shape.scale_y);
    println!("  Gaussian[0] color: ({:.3}, {:.3}, {:.3})", gaussians[0].color.r, gaussians[0].color.g, gaussians[0].color.b);

    // Check coverage
    let rendered_init = RendererV2::render(&gaussians, 256, 256);
    let init_psnr = compute_psnr(&target, &rendered_init);
    println!("  Initial PSNR: {:.2} dB\n", init_psnr);

    // Optimize with detailed logging
    println!("Optimizing with enhanced diagnostics:\n");

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 20;  // Just 20 for diagnosis

    for iter in 0..optimizer.max_iterations {
        // Render
        let rendered = RendererV2::render(&gaussians, 256, 256);

        // Compute loss
        let loss = compute_loss(&rendered, &target);

        // Compute PSNR
        let psnr = compute_psnr(&target, &rendered);

        // Compute parameter changes
        let old_gaussians = gaussians.clone();

        // Single optimization step
        let grads = compute_gradients(&gaussians, &rendered, &target);

        // Apply updates
        for (gaussian, grad) in gaussians.iter_mut().zip(grads.iter()) {
            gaussian.color.r -= optimizer.learning_rate_color * grad.color.r;
            gaussian.color.g -= optimizer.learning_rate_color * grad.color.g;
            gaussian.color.b -= optimizer.learning_rate_color * grad.color.b;
            gaussian.color.r = gaussian.color.r.clamp(0.0, 1.0);
            gaussian.color.g = gaussian.color.g.clamp(0.0, 1.0);
            gaussian.color.b = gaussian.color.b.clamp(0.0, 1.0);

            gaussian.position.x -= optimizer.learning_rate_position * grad.position.x;
            gaussian.position.y -= optimizer.learning_rate_position * grad.position.y;
            gaussian.position.x = gaussian.position.x.clamp(0.0, 1.0);
            gaussian.position.y = gaussian.position.y.clamp(0.0, 1.0);
        }

        // Compute changes (Debug Plan Section 0.4)
        let mut color_change_sum = 0.0;
        let mut pos_change_sum = 0.0;
        for (old, new) in old_gaussians.iter().zip(gaussians.iter()) {
            let dc_r = (new.color.r - old.color.r).abs();
            let dc_g = (new.color.g - old.color.g).abs();
            let dc_b = (new.color.b - old.color.b).abs();
            color_change_sum += dc_r + dc_g + dc_b;

            let dp_x = (new.position.x - old.position.x).abs();
            let dp_y = (new.position.y - old.position.y).abs();
            pos_change_sum += dp_x + dp_y;
        }
        let mean_color_change = color_change_sum / (gaussians.len() as f32 * 3.0);
        let mean_pos_change = pos_change_sum / (gaussians.len() as f32 * 2.0);

        // Compute weight statistics
        let mut weight_stats = compute_weight_stats(&gaussians, 256, 256);

        println!("Iter {:2}: loss={:.6}, PSNR={:5.2} dB, |Δc|={:.6}, |Δμ|={:.6}, W_med={:.3}",
            iter, loss, psnr, mean_color_change, mean_pos_change, weight_stats.median);

        if loss < 1e-4 {
            println!("\nConverged at iteration {}", iter);
            break;
        }
    }

    // Final check
    let rendered_final = RendererV2::render(&gaussians, 256, 256);
    let final_psnr = compute_psnr(&target, &rendered_final);

    println!("\n═══════════════════════════════════════");
    println!("RESULTS:");
    println!("  Initial PSNR: {:.2} dB", init_psnr);
    println!("  Final PSNR:   {:.2} dB", final_psnr);
    println!("  Improvement:  {:+.2} dB", final_psnr - init_psnr);

    if final_psnr >= 20.0 {
        println!("\n✅ PASS: Optimization working!");
    } else if final_psnr > init_psnr + 2.0 {
        println!("\n⚠️  PARTIAL: Some improvement but below target");
    } else {
        println!("\n❌ FAIL: No significant improvement");
    }
}

struct WeightStats {
    min: f32,
    median: f32,
    max: f32,
}

fn compute_weight_stats(gaussians: &[Gaussian2D<f32, Euler<f32>>], width: u32, height: u32) -> WeightStats {
    let mut weight_sums = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let px = x as f32 / width as f32;
            let py = y as f32 / height as f32;

            let mut weight_sum = 0.0;
            for gaussian in gaussians {
                let dx = px - gaussian.position.x;
                let dy = py - gaussian.position.y;

                let sx = gaussian.shape.scale_x;
                let sy = gaussian.shape.scale_y;
                let theta = gaussian.shape.rotation;

                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let dx_rot = dx * cos_t + dy * sin_t;
                let dy_rot = -dx * sin_t + dy * cos_t;

                let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);

                if dist_sq <= 12.25 {
                    weight_sum += gaussian.opacity * (-0.5 * dist_sq).exp();
                }
            }

            weight_sums.push(weight_sum);
        }
    }

    weight_sums.sort_by(|a, b| a.partial_cmp(b).unwrap());

    WeightStats {
        min: *weight_sums.first().unwrap_or(&0.0),
        median: weight_sums[weight_sums.len() / 2],
        max: *weight_sums.last().unwrap_or(&0.0),
    }
}

#[derive(Clone)]
struct GaussianGradient {
    position: Vector2<f32>,
    color: Color4<f32>,
}

impl GaussianGradient {
    fn zero() -> Self {
        Self {
            position: Vector2::zero(),
            color: Color4::new(0.0, 0.0, 0.0, 0.0),
        }
    }
}

fn compute_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
) -> Vec<GaussianGradient> {
    let width = target.width;
    let height = target.height;

    let mut gradients = vec![GaussianGradient::zero(); gaussians.len()];

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

                let cos_t = gaussian.shape.rotation.cos();
                let sin_t = gaussian.shape.rotation.sin();
                let dx_rot = dx * cos_t + dy * sin_t;
                let dy_rot = -dx * sin_t + dy * cos_t;

                let sx = gaussian.shape.scale_x;
                let sy = gaussian.shape.scale_y;
                let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);

                if dist_sq > 12.25 {
                    continue;
                }

                let gaussian_val = (-0.5 * dist_sq).exp();
                let weight = gaussian.opacity * gaussian_val;

                gradients[i].color.r += error_r * weight;
                gradients[i].color.g += error_g * weight;
                gradients[i].color.b += error_b * weight;

                let grad_weight_x = weight * (dx_rot * cos_t / (sx * sx) + dy_rot * (-sin_t) / (sy * sy));
                let grad_weight_y = weight * (dx_rot * sin_t / (sx * sx) + dy_rot * cos_t / (sy * sy));

                let error_weighted = error_r * gaussian.color.r +
                                    error_g * gaussian.color.g +
                                    error_b * gaussian.color.b;

                gradients[i].position.x += error_weighted * grad_weight_x;
                gradients[i].position.y += error_weighted * grad_weight_y;
            }
        }
    }

    let pixel_count = (width * height) as f32;
    for grad in &mut gradients {
        grad.color.r /= pixel_count;
        grad.color.g /= pixel_count;
        grad.color.b /= pixel_count;
        grad.position.x /= pixel_count;
        grad.position.y /= pixel_count;
    }

    gradients
}

fn compute_loss(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut loss = 0.0;
    let count = (rendered.width * rendered.height) as f32;

    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        loss += (r.r - t.r).powi(2);
        loss += (r.g - t.g).powi(2);
        loss += (r.b - t.b).powi(2);
    }

    loss / count
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

    if mse < 1e-10 {
        100.0
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}
