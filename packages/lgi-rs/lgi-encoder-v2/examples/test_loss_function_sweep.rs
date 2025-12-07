//! Loss Function Sweep Test
//!
//! Different loss functions can significantly affect optimization behavior:
//!
//! 1. **L2 (MSE)**: Standard, smooth gradients, sensitive to outliers
//! 2. **L1 (MAE)**: More robust to outliers, constant gradient magnitude
//! 3. **Huber**: L2 for small errors, L1 for large - best of both
//! 4. **Charbonnier**: Smooth approximation to L1, commonly used in vision
//! 5. **Log-Cosh**: Similar to Huber but twice differentiable
//!
//! Hypothesis: Huber or Charbonnier loss might help with difficult regions
//! where pure L2 struggles (e.g., edges, high-contrast areas).

use lgi_core::ImageBuffer;
use lgi_encoder_v2::test_results::TestResult;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
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

#[derive(Clone, Copy)]
enum LossFunction {
    L2,         // MSE: (a-b)^2
    L1,         // MAE: |a-b|
    Huber(f32), // Huber with delta threshold
    Charbonnier(f32), // sqrt(x^2 + eps^2) - eps
    LogCosh,    // log(cosh(x))
}

impl LossFunction {
    fn name(&self) -> &'static str {
        match self {
            LossFunction::L2 => "L2 (MSE)",
            LossFunction::L1 => "L1 (MAE)",
            LossFunction::Huber(_) => "Huber",
            LossFunction::Charbonnier(_) => "Charbonnier",
            LossFunction::LogCosh => "Log-Cosh",
        }
    }

    fn compute(&self, diff: f32) -> f32 {
        match self {
            LossFunction::L2 => diff * diff,
            LossFunction::L1 => diff.abs(),
            LossFunction::Huber(delta) => {
                let abs_diff = diff.abs();
                if abs_diff <= *delta {
                    0.5 * diff * diff
                } else {
                    delta * (abs_diff - 0.5 * delta)
                }
            }
            LossFunction::Charbonnier(eps) => {
                (diff * diff + eps * eps).sqrt() - eps
            }
            LossFunction::LogCosh => {
                let x = diff;
                x.cosh().ln()
            }
        }
    }

    fn gradient(&self, diff: f32) -> f32 {
        match self {
            LossFunction::L2 => 2.0 * diff,
            LossFunction::L1 => if diff > 0.0 { 1.0 } else if diff < 0.0 { -1.0 } else { 0.0 },
            LossFunction::Huber(delta) => {
                let abs_diff = diff.abs();
                if abs_diff <= *delta {
                    diff
                } else {
                    delta * diff.signum()
                }
            }
            LossFunction::Charbonnier(eps) => {
                diff / (diff * diff + eps * eps).sqrt()
            }
            LossFunction::LogCosh => {
                diff.tanh()
            }
        }
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("   LOSS FUNCTION SWEEP TEST");
    println!("   Comparing different loss functions for Gaussian optimization");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let image_path = "test-data/kodak-dataset/kodim03.png";
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

    println!("Image: kodim03 ({}x{})\n", width, height);

    let n_gaussians = 256;
    let iterations = 200;

    let base_gaussians = init_jittered_grid(width, height, n_gaussians, &target);
    let rendered = RendererV2::render(&base_gaussians, width, height);
    let init_psnr = lgi_core::error_metrics::compute_psnr(&target, &rendered);
    println!("Initial: {} Gaussians, PSNR={:.2}dB\n", n_gaussians, init_psnr);

    // Loss functions to test
    let loss_functions = vec![
        LossFunction::L2,
        LossFunction::L1,
        LossFunction::Huber(0.1),
        LossFunction::Huber(0.01),
        LossFunction::Charbonnier(0.01),
        LossFunction::Charbonnier(0.001),
        LossFunction::LogCosh,
    ];

    let mut results = Vec::new();

    for loss_fn in &loss_functions {
        let loss_name = format!("{}{}", loss_fn.name(),
            match loss_fn {
                LossFunction::Huber(d) => format!(" (δ={})", d),
                LossFunction::Charbonnier(e) => format!(" (ε={})", e),
                _ => String::new(),
            });

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Loss: {}", loss_name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let mut gaussians = base_gaussians.clone();
        let start = Instant::now();

        // Custom optimization with specified loss
        optimize_with_loss(&mut gaussians, &target, *loss_fn, iterations);

        let elapsed = start.elapsed();
        let final_render = RendererV2::render(&gaussians, width, height);
        let final_psnr = lgi_core::error_metrics::compute_psnr(&target, &final_render);

        // Save result
        let clean_name = loss_name.to_lowercase()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("=", "")
            .replace("δ", "d")
            .replace("ε", "e");
        let mut result = TestResult::new(&format!("loss_fn_{}", clean_name), image_path);
        result.optimizer = format!("Adam ({})", loss_name);
        result.n_gaussians = n_gaussians;
        result.iterations = iterations;
        result.initial_psnr = init_psnr;
        result.final_psnr = final_psnr;
        result.improvement_db = final_psnr - init_psnr;
        result.elapsed_seconds = elapsed.as_secs_f64() as f32;
        result.image_dimensions = (width, height);
        result.set_param("loss_function", loss_name.clone());
        result.complete();
        let _ = result.save_with_image(&final_render, RESULTS_DIR);

        println!("\n  Final: {:.2}dB (Δ={:+.2}dB) in {:.1}s\n",
                 final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64());

        results.push((loss_name.clone(), final_psnr, final_psnr - init_psnr, elapsed.as_secs_f64()));
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                    LOSS FUNCTION SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    println!("  Starting PSNR: {:.2}dB\n", init_psnr);
    println!("  {:25} {:>12} {:>12} {:>10}", "Loss Function", "Final PSNR", "Improvement", "Time");
    println!("  {:->25} {:->12} {:->12} {:->10}", "", "", "", "");

    for (name, final_psnr, improvement, time) in &results {
        println!("  {:25} {:>10.2}dB {:>+10.2}dB {:>8.1}s", name, final_psnr, improvement, time);
    }

    // Find best
    let best = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let l2 = results.iter().find(|(n, _, _, _)| n.contains("L2"));

    println!("\n  Best loss function: {} ({:.2}dB)", best.0, best.1);
    if let Some(l2_result) = l2 {
        println!("  Improvement over L2: {:+.2}dB", best.1 - l2_result.1);
    }

    // Analysis
    println!("\n  Analysis:");

    // Check if robust losses help
    let robust_avg: f32 = results.iter()
        .filter(|(n, _, _, _)| n.contains("Huber") || n.contains("Charbonnier") || n.contains("Log"))
        .map(|(_, _, i, _)| i)
        .sum::<f32>() / 5.0;

    let l2_imp = l2.map(|(_, _, i, _)| *i).unwrap_or(0.0);

    if robust_avg > l2_imp + 0.1 {
        println!("  → Robust losses outperform L2 - outlier handling beneficial");
        println!("    (robust avg: {:+.2}dB vs L2: {:+.2}dB)", robust_avg, l2_imp);
    } else if l2_imp > robust_avg + 0.1 {
        println!("  → L2 performs best - smooth gradients preferred for this task");
    } else {
        println!("  → Loss function choice has minimal impact");
    }

    // Check Huber sensitivity to delta
    let huber_results: Vec<_> = results.iter()
        .filter(|(n, _, _, _)| n.contains("Huber"))
        .collect();

    if huber_results.len() == 2 {
        let (_, psnr1, _, _) = huber_results[0];
        let (_, psnr2, _, _) = huber_results[1];
        if (psnr1 - psnr2).abs() > 0.2 {
            println!("  → Huber δ parameter has significant impact - tune carefully");
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

fn optimize_with_loss(
    gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    target: &ImageBuffer<f32>,
    loss_fn: LossFunction,
    iterations: usize,
) {
    let width = target.width;
    let height = target.height;

    // Learning rates (same as standard Adam)
    let lr_position = 0.0002;
    let lr_color = 0.02;
    let lr_scale = 0.005;
    let lr_opacity = 0.05;
    let lr_rotation = 0.001;

    // Adam parameters
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;

    // Initialize moment estimates
    let n = gaussians.len();
    let mut m_pos = vec![(0.0f32, 0.0f32); n];
    let mut v_pos = vec![(0.0f32, 0.0f32); n];
    let mut m_color = vec![(0.0f32, 0.0f32, 0.0f32); n];
    let mut v_color = vec![(0.0f32, 0.0f32, 0.0f32); n];
    let mut m_scale = vec![(0.0f32, 0.0f32); n];
    let mut v_scale = vec![(0.0f32, 0.0f32); n];
    let mut m_opacity = vec![0.0f32; n];
    let mut v_opacity = vec![0.0f32; n];
    let mut m_rotation = vec![0.0f32; n];
    let mut v_rotation = vec![0.0f32; n];

    let mut last_psnr = lgi_core::error_metrics::compute_psnr(
        target,
        &RendererV2::render(gaussians, width, height),
    );

    for iter in 0..iterations {
        // Compute gradients using custom loss
        let rendered = RendererV2::render(gaussians, width, height);

        // Compute per-pixel loss gradient
        let mut loss_grads = vec![Color4::new(0.0, 0.0, 0.0, 0.0); (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let r = rendered.get_pixel(x, y).unwrap_or(Color4::black());
                let t = target.get_pixel(x, y).unwrap_or(Color4::black());

                loss_grads[idx] = Color4::new(
                    loss_fn.gradient(r.r - t.r),
                    loss_fn.gradient(r.g - t.g),
                    loss_fn.gradient(r.b - t.b),
                    0.0,
                );
            }
        }

        // Compute parameter gradients using numerical differentiation
        let delta = 0.001;

        for i in 0..n {
            // Position gradients
            let orig_pos = gaussians[i].position;

            gaussians[i].position.x += delta;
            let loss_plus_x = compute_total_loss(&RendererV2::render(gaussians, width, height), target, loss_fn);
            gaussians[i].position.x = orig_pos.x - delta;
            let loss_minus_x = compute_total_loss(&RendererV2::render(gaussians, width, height), target, loss_fn);
            gaussians[i].position.x = orig_pos.x;

            gaussians[i].position.y += delta;
            let loss_plus_y = compute_total_loss(&RendererV2::render(gaussians, width, height), target, loss_fn);
            gaussians[i].position.y = orig_pos.y - delta;
            let loss_minus_y = compute_total_loss(&RendererV2::render(gaussians, width, height), target, loss_fn);
            gaussians[i].position = orig_pos;

            let grad_pos_x = (loss_plus_x - loss_minus_x) / (2.0 * delta);
            let grad_pos_y = (loss_plus_y - loss_minus_y) / (2.0 * delta);

            // Color gradients (simplified - use error at gaussian center)
            let px = (orig_pos.x * width as f32) as u32;
            let py = (orig_pos.y * height as f32) as u32;
            let r_color = rendered.get_pixel(px.min(width-1), py.min(height-1)).unwrap_or(Color4::black());
            let t_color = target.get_pixel(px.min(width-1), py.min(height-1)).unwrap_or(Color4::black());

            let grad_color_r = loss_fn.gradient(r_color.r - t_color.r);
            let grad_color_g = loss_fn.gradient(r_color.g - t_color.g);
            let grad_color_b = loss_fn.gradient(r_color.b - t_color.b);

            // Scale gradients (numerical)
            let orig_scale = gaussians[i].parameterization;
            let (orig_sx, orig_sy) = orig_scale.scales();

            gaussians[i].parameterization = Euler::new(orig_sx + delta, orig_sy, orig_scale.rotation());
            let loss_plus_sx = compute_total_loss(&RendererV2::render(gaussians, width, height), target, loss_fn);
            gaussians[i].parameterization = Euler::new(orig_sx - delta, orig_sy, orig_scale.rotation());
            let loss_minus_sx = compute_total_loss(&RendererV2::render(gaussians, width, height), target, loss_fn);
            gaussians[i].parameterization = orig_scale;

            let grad_scale_x = (loss_plus_sx - loss_minus_sx) / (2.0 * delta);
            let grad_scale_y = grad_scale_x; // Simplified

            // Opacity gradient
            let orig_opacity = gaussians[i].opacity;
            gaussians[i].opacity = (orig_opacity + delta).min(1.0);
            let loss_plus_o = compute_total_loss(&RendererV2::render(gaussians, width, height), target, loss_fn);
            gaussians[i].opacity = (orig_opacity - delta).max(0.0);
            let loss_minus_o = compute_total_loss(&RendererV2::render(gaussians, width, height), target, loss_fn);
            gaussians[i].opacity = orig_opacity;

            let grad_opacity = (loss_plus_o - loss_minus_o) / (2.0 * delta);

            // Adam updates
            let t_f = (iter + 1) as f32;

            // Position
            m_pos[i].0 = beta1 * m_pos[i].0 + (1.0 - beta1) * grad_pos_x;
            m_pos[i].1 = beta1 * m_pos[i].1 + (1.0 - beta1) * grad_pos_y;
            v_pos[i].0 = beta2 * v_pos[i].0 + (1.0 - beta2) * grad_pos_x * grad_pos_x;
            v_pos[i].1 = beta2 * v_pos[i].1 + (1.0 - beta2) * grad_pos_y * grad_pos_y;
            let m_hat_x = m_pos[i].0 / (1.0 - beta1.powf(t_f));
            let m_hat_y = m_pos[i].1 / (1.0 - beta1.powf(t_f));
            let v_hat_x = v_pos[i].0 / (1.0 - beta2.powf(t_f));
            let v_hat_y = v_pos[i].1 / (1.0 - beta2.powf(t_f));
            gaussians[i].position.x -= lr_position * m_hat_x / (v_hat_x.sqrt() + eps);
            gaussians[i].position.y -= lr_position * m_hat_y / (v_hat_y.sqrt() + eps);
            gaussians[i].position.x = gaussians[i].position.x.clamp(0.0, 1.0);
            gaussians[i].position.y = gaussians[i].position.y.clamp(0.0, 1.0);

            // Color
            m_color[i].0 = beta1 * m_color[i].0 + (1.0 - beta1) * grad_color_r;
            m_color[i].1 = beta1 * m_color[i].1 + (1.0 - beta1) * grad_color_g;
            m_color[i].2 = beta1 * m_color[i].2 + (1.0 - beta1) * grad_color_b;
            v_color[i].0 = beta2 * v_color[i].0 + (1.0 - beta2) * grad_color_r * grad_color_r;
            v_color[i].1 = beta2 * v_color[i].1 + (1.0 - beta2) * grad_color_g * grad_color_g;
            v_color[i].2 = beta2 * v_color[i].2 + (1.0 - beta2) * grad_color_b * grad_color_b;
            let m_hat_r = m_color[i].0 / (1.0 - beta1.powf(t_f));
            let m_hat_g = m_color[i].1 / (1.0 - beta1.powf(t_f));
            let m_hat_b = m_color[i].2 / (1.0 - beta1.powf(t_f));
            let v_hat_r = v_color[i].0 / (1.0 - beta2.powf(t_f));
            let v_hat_g = v_color[i].1 / (1.0 - beta2.powf(t_f));
            let v_hat_b = v_color[i].2 / (1.0 - beta2.powf(t_f));
            gaussians[i].color.r -= lr_color * m_hat_r / (v_hat_r.sqrt() + eps);
            gaussians[i].color.g -= lr_color * m_hat_g / (v_hat_g.sqrt() + eps);
            gaussians[i].color.b -= lr_color * m_hat_b / (v_hat_b.sqrt() + eps);
            gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
            gaussians[i].color.g = gaussians[i].color.g.clamp(0.0, 1.0);
            gaussians[i].color.b = gaussians[i].color.b.clamp(0.0, 1.0);

            // Scale
            m_scale[i].0 = beta1 * m_scale[i].0 + (1.0 - beta1) * grad_scale_x;
            m_scale[i].1 = beta1 * m_scale[i].1 + (1.0 - beta1) * grad_scale_y;
            v_scale[i].0 = beta2 * v_scale[i].0 + (1.0 - beta2) * grad_scale_x * grad_scale_x;
            v_scale[i].1 = beta2 * v_scale[i].1 + (1.0 - beta2) * grad_scale_y * grad_scale_y;
            let m_hat_sx = m_scale[i].0 / (1.0 - beta1.powf(t_f));
            let m_hat_sy = m_scale[i].1 / (1.0 - beta1.powf(t_f));
            let v_hat_sx = v_scale[i].0 / (1.0 - beta2.powf(t_f));
            let v_hat_sy = v_scale[i].1 / (1.0 - beta2.powf(t_f));
            let (sx, sy) = gaussians[i].parameterization.scales();
            let new_sx = (sx - lr_scale * m_hat_sx / (v_hat_sx.sqrt() + eps)).clamp(0.001, 0.5);
            let new_sy = (sy - lr_scale * m_hat_sy / (v_hat_sy.sqrt() + eps)).clamp(0.001, 0.5);
            gaussians[i].parameterization = Euler::new(new_sx, new_sy, gaussians[i].parameterization.rotation());

            // Opacity
            m_opacity[i] = beta1 * m_opacity[i] + (1.0 - beta1) * grad_opacity;
            v_opacity[i] = beta2 * v_opacity[i] + (1.0 - beta2) * grad_opacity * grad_opacity;
            let m_hat_o = m_opacity[i] / (1.0 - beta1.powf(t_f));
            let v_hat_o = v_opacity[i] / (1.0 - beta2.powf(t_f));
            gaussians[i].opacity -= lr_opacity * m_hat_o / (v_hat_o.sqrt() + eps);
            gaussians[i].opacity = gaussians[i].opacity.clamp(0.01, 1.0);
        }

        // Progress reporting
        if iter % 40 == 39 || iter == 0 {
            let current_render = RendererV2::render(gaussians, width, height);
            let current_psnr = lgi_core::error_metrics::compute_psnr(target, &current_render);
            let direction = if current_psnr > last_psnr { "↑" } else { "↓" };
            println!("  [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                     iter + 1, iterations, current_psnr, direction, current_psnr - last_psnr);
            last_psnr = current_psnr;
        }
    }
}

fn compute_total_loss(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>, loss_fn: LossFunction) -> f32 {
    let mut total = 0.0;
    for y in 0..target.height {
        for x in 0..target.width {
            let r = rendered.get_pixel(x, y).unwrap_or(Color4::black());
            let t = target.get_pixel(x, y).unwrap_or(Color4::black());
            total += loss_fn.compute(r.r - t.r);
            total += loss_fn.compute(r.g - t.g);
            total += loss_fn.compute(r.b - t.b);
        }
    }
    total / (target.width * target.height * 3) as f32
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

        gaussians.push(Gaussian2D::new(position, Euler::isotropic(0.04), color, 1.0));
    }

    gaussians
}
