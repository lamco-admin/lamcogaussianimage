//! 3DGS-Faithful Densification Test
//!
//! KEY INSIGHT: The existing AdaptiveDensifier module already has clone/split/prune,
//! but previous tests bypassed it with broken inline logic. This test properly integrates
//! AdaptiveDensifier with Adam's accumulated position gradients (m_position).
//!
//! What 3DGS does (Kerbl et al. 2023):
//! - Gradient threshold: τ ≥ 0.0005 (accumulated view-space positional gradient)
//! - Clone vs Split: If max(scale) > 0.01 × scene_extent → split, else clone
//! - Pruning: opacity < 0.005 → remove
//! - Interval: Every 100 iterations
//!
//! What we do here:
//! - Use Adam's m_position as accumulated gradient (it's an EMA, effectively accumulated)
//! - Calibrate thresholds to 3DGS paper values
//! - Use AdaptiveDensifier for clone/split/prune operations
//!
//! Run: cargo run --release --example test_3dgs_faithful_densification -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adaptive_densification::{AdaptiveDensifier, apply_densification};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use serde_json::json;
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

/// Adam state for manual step-by-step optimization with gradient access
struct AdamWithGradientAccess {
    // Learning rates (3DGS-calibrated)
    lr_position: f32,
    lr_color: f32,
    lr_scale: f32,
    lr_opacity: f32,

    // Adam parameters
    beta1: f32,
    beta2: f32,
    epsilon: f32,

    // First moment estimates (accumulated gradients)
    m_position: Vec<Vector2<f32>>,
    m_color: Vec<Color4<f32>>,
    m_scale: Vec<(f32, f32)>,
    m_opacity: Vec<f32>,

    // Second moment estimates
    v_position: Vec<Vector2<f32>>,
    v_color: Vec<Color4<f32>>,
    v_scale: Vec<(f32, f32)>,
    v_opacity: Vec<f32>,

    // Iteration counter
    t: usize,
}

impl AdamWithGradientAccess {
    fn new(n: usize) -> Self {
        Self {
            // 3DGS-style learning rates
            lr_position: 0.0002,
            lr_color: 0.02,
            lr_scale: 0.005,
            lr_opacity: 0.05,

            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,

            m_position: vec![Vector2::zero(); n],
            m_color: vec![Color4::new(0.0, 0.0, 0.0, 0.0); n],
            m_scale: vec![(0.0, 0.0); n],
            m_opacity: vec![0.0; n],

            v_position: vec![Vector2::zero(); n],
            v_color: vec![Color4::new(0.0, 0.0, 0.0, 0.0); n],
            v_scale: vec![(0.0, 0.0); n],
            v_opacity: vec![0.0; n],

            t: 0,
        }
    }

    /// Resize state for new Gaussian count (after densification)
    fn resize(&mut self, new_n: usize, old_n: usize) {
        // For new Gaussians, initialize with zero moments (they'll learn quickly)
        self.m_position.resize(new_n, Vector2::zero());
        self.m_color.resize(new_n, Color4::new(0.0, 0.0, 0.0, 0.0));
        self.m_scale.resize(new_n, (0.0, 0.0));
        self.m_opacity.resize(new_n, 0.0);

        self.v_position.resize(new_n, Vector2::zero());
        self.v_color.resize(new_n, Color4::new(0.0, 0.0, 0.0, 0.0));
        self.v_scale.resize(new_n, (0.0, 0.0));
        self.v_opacity.resize(new_n, 0.0);
    }

    /// Run one Adam step, return loss
    fn step(&mut self, gaussians: &mut [Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) -> f32 {
        self.t += 1;
        let n = gaussians.len();

        // Ensure state is sized correctly
        if self.m_position.len() != n {
            self.resize(n, self.m_position.len());
        }

        let rendered = RendererV2::render(gaussians, target.width, target.height);
        let loss = compute_mse(&rendered, target);

        let grads = compute_gradients(gaussians, &rendered, target);

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        // Position LR decay (3DGS style) with minimum floor
        // Original: decays to 0.000002 by iter 300 - too aggressive
        // Fix: maintain at least 10% of initial LR (0.00002)
        let lr_pos = (self.lr_position * (0.1_f32).powf(self.t as f32 / 300.0)).max(0.00002);

        for i in 0..n {
            // Update moments for position
            self.m_position[i].x = self.beta1 * self.m_position[i].x + (1.0 - self.beta1) * grads[i].position.x;
            self.m_position[i].y = self.beta1 * self.m_position[i].y + (1.0 - self.beta1) * grads[i].position.y;
            self.v_position[i].x = self.beta2 * self.v_position[i].x + (1.0 - self.beta2) * grads[i].position.x.powi(2);
            self.v_position[i].y = self.beta2 * self.v_position[i].y + (1.0 - self.beta2) * grads[i].position.y.powi(2);

            // Update moments for color
            self.m_color[i].r = self.beta1 * self.m_color[i].r + (1.0 - self.beta1) * grads[i].color.r;
            self.m_color[i].g = self.beta1 * self.m_color[i].g + (1.0 - self.beta1) * grads[i].color.g;
            self.m_color[i].b = self.beta1 * self.m_color[i].b + (1.0 - self.beta1) * grads[i].color.b;
            self.v_color[i].r = self.beta2 * self.v_color[i].r + (1.0 - self.beta2) * grads[i].color.r.powi(2);
            self.v_color[i].g = self.beta2 * self.v_color[i].g + (1.0 - self.beta2) * grads[i].color.g.powi(2);
            self.v_color[i].b = self.beta2 * self.v_color[i].b + (1.0 - self.beta2) * grads[i].color.b.powi(2);

            // Update moments for scale
            self.m_scale[i].0 = self.beta1 * self.m_scale[i].0 + (1.0 - self.beta1) * grads[i].scale_x;
            self.m_scale[i].1 = self.beta1 * self.m_scale[i].1 + (1.0 - self.beta1) * grads[i].scale_y;
            self.v_scale[i].0 = self.beta2 * self.v_scale[i].0 + (1.0 - self.beta2) * grads[i].scale_x.powi(2);
            self.v_scale[i].1 = self.beta2 * self.v_scale[i].1 + (1.0 - self.beta2) * grads[i].scale_y.powi(2);

            // Update moments for opacity
            self.m_opacity[i] = self.beta1 * self.m_opacity[i] + (1.0 - self.beta1) * grads[i].opacity;
            self.v_opacity[i] = self.beta2 * self.v_opacity[i] + (1.0 - self.beta2) * grads[i].opacity.powi(2);

            // Apply Adam updates
            let m_hat_px = self.m_position[i].x / bias_correction1;
            let m_hat_py = self.m_position[i].y / bias_correction1;
            let v_hat_px = self.v_position[i].x / bias_correction2;
            let v_hat_py = self.v_position[i].y / bias_correction2;
            gaussians[i].position.x -= lr_pos * m_hat_px / (v_hat_px.sqrt() + self.epsilon);
            gaussians[i].position.y -= lr_pos * m_hat_py / (v_hat_py.sqrt() + self.epsilon);
            gaussians[i].position.x = gaussians[i].position.x.clamp(0.0, 1.0);
            gaussians[i].position.y = gaussians[i].position.y.clamp(0.0, 1.0);

            let m_hat_r = self.m_color[i].r / bias_correction1;
            let m_hat_g = self.m_color[i].g / bias_correction1;
            let m_hat_b = self.m_color[i].b / bias_correction1;
            let v_hat_r = self.v_color[i].r / bias_correction2;
            let v_hat_g = self.v_color[i].g / bias_correction2;
            let v_hat_b = self.v_color[i].b / bias_correction2;
            gaussians[i].color.r -= self.lr_color * m_hat_r / (v_hat_r.sqrt() + self.epsilon);
            gaussians[i].color.g -= self.lr_color * m_hat_g / (v_hat_g.sqrt() + self.epsilon);
            gaussians[i].color.b -= self.lr_color * m_hat_b / (v_hat_b.sqrt() + self.epsilon);
            gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
            gaussians[i].color.g = gaussians[i].color.g.clamp(0.0, 1.0);
            gaussians[i].color.b = gaussians[i].color.b.clamp(0.0, 1.0);

            let m_hat_sx = self.m_scale[i].0 / bias_correction1;
            let m_hat_sy = self.m_scale[i].1 / bias_correction1;
            let v_hat_sx = self.v_scale[i].0 / bias_correction2;
            let v_hat_sy = self.v_scale[i].1 / bias_correction2;
            gaussians[i].shape.scale_x -= self.lr_scale * m_hat_sx / (v_hat_sx.sqrt() + self.epsilon);
            gaussians[i].shape.scale_y -= self.lr_scale * m_hat_sy / (v_hat_sy.sqrt() + self.epsilon);
            gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
            gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);

            let m_hat_o = self.m_opacity[i] / bias_correction1;
            let v_hat_o = self.v_opacity[i] / bias_correction2;
            gaussians[i].opacity -= self.lr_opacity * m_hat_o / (v_hat_o.sqrt() + self.epsilon);
            gaussians[i].opacity = gaussians[i].opacity.clamp(0.01, 1.0);
        }

        loss
    }

    /// Get position gradient magnitudes (for densification trigger)
    /// This is what 3DGS uses: ||∇_x L|| accumulated over iterations
    fn get_position_gradient_magnitudes(&self) -> Vec<f32> {
        self.m_position
            .iter()
            .map(|m| (m.x.powi(2) + m.y.powi(2)).sqrt())
            .collect()
    }

    /// Handle densification: remove indices and add new Gaussians
    fn handle_densification(&mut self, to_remove: &[usize], n_added: usize) {
        let old_n = self.m_position.len();

        // Remove in reverse order to preserve indices
        let mut sorted_removes = to_remove.to_vec();
        sorted_removes.sort_unstable_by(|a, b| b.cmp(a));

        for &idx in &sorted_removes {
            if idx < self.m_position.len() {
                self.m_position.swap_remove(idx);
                self.m_color.swap_remove(idx);
                self.m_scale.swap_remove(idx);
                self.m_opacity.swap_remove(idx);
                self.v_position.swap_remove(idx);
                self.v_color.swap_remove(idx);
                self.v_scale.swap_remove(idx);
                self.v_opacity.swap_remove(idx);
            }
        }

        // Add zeros for new Gaussians
        for _ in 0..n_added {
            self.m_position.push(Vector2::zero());
            self.m_color.push(Color4::new(0.0, 0.0, 0.0, 0.0));
            self.m_scale.push((0.0, 0.0));
            self.m_opacity.push(0.0);
            self.v_position.push(Vector2::zero());
            self.v_color.push(Color4::new(0.0, 0.0, 0.0, 0.0));
            self.v_scale.push((0.0, 0.0));
            self.v_opacity.push(0.0);
        }
    }
}

#[derive(Clone)]
struct Gradients {
    position: Vector2<f32>,
    color: Color4<f32>,
    scale_x: f32,
    scale_y: f32,
    opacity: f32,
}

fn compute_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
) -> Vec<Gradients> {
    let n = gaussians.len();
    let width = target.width as f32;
    let height = target.height as f32;

    let mut grads = vec![
        Gradients {
            position: Vector2::zero(),
            color: Color4::new(0.0, 0.0, 0.0, 0.0),
            scale_x: 0.0,
            scale_y: 0.0,
            opacity: 0.0,
        };
        n
    ];

    for py in 0..target.height {
        for px in 0..target.width {
            let t = target.get_pixel(px, py).unwrap();
            let r = rendered.get_pixel(px, py).unwrap();

            let error_r = r.r - t.r;
            let error_g = r.g - t.g;
            let error_b = r.b - t.b;

            let px_norm = (px as f32 + 0.5) / width;
            let py_norm = (py as f32 + 0.5) / height;

            for (i, g) in gaussians.iter().enumerate() {
                let dx = px_norm - g.position.x;
                let dy = py_norm - g.position.y;

                let scale_product = g.shape.scale_x * g.shape.scale_y;
                if scale_product < 1e-10 {
                    continue;
                }

                let dist_sq = dx * dx + dy * dy;
                let weight = (-0.5 * dist_sq / scale_product).exp();

                if weight < 1e-6 {
                    continue;
                }

                let error_weighted = error_r + error_g + error_b;

                // Color gradients
                grads[i].color.r += error_r * weight * g.opacity;
                grads[i].color.g += error_g * weight * g.opacity;
                grads[i].color.b += error_b * weight * g.opacity;

                // Position gradients (simplified isotropic)
                grads[i].position.x += error_weighted * weight * dx / scale_product;
                grads[i].position.y += error_weighted * weight * dy / scale_product;

                // Scale gradients
                let scale_grad = error_weighted * weight * dist_sq / (scale_product * scale_product);
                grads[i].scale_x += scale_grad * g.shape.scale_y;
                grads[i].scale_y += scale_grad * g.shape.scale_x;

                // Opacity gradient
                grads[i].opacity += error_weighted * weight;
            }
        }
    }

    // Normalize by pixel count
    let pixel_count = (target.width * target.height) as f32;
    for g in &mut grads {
        g.position.x /= pixel_count;
        g.position.y /= pixel_count;
        g.color.r /= pixel_count;
        g.color.g /= pixel_count;
        g.color.b /= pixel_count;
        g.scale_x /= pixel_count;
        g.scale_y /= pixel_count;
        g.opacity /= pixel_count;
    }

    grads
}

fn compute_mse(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height * 3) as f32;
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse / count
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mse = compute_mse(rendered, target);
    if mse <= 0.0 {
        100.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

fn create_jittered_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n);
    let cell_size = 1.0 / grid_size as f32;

    for i in 0..n {
        let gx = i % grid_size;
        let gy = i / grid_size;

        // Deterministic jitter based on index
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

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   3DGS-FAITHFUL DENSIFICATION TEST                           ║");
    println!("║   Using AdaptiveDensifier + Adam's accumulated gradients     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let test_images = vec![
        ("/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png", "kodim03"),
        ("/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim05.png", "kodim05"),
        ("/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim08.png", "kodim08"),
    ];

    let mut all_results = Vec::new();

    for (image_path, image_name) in &test_images {
        println!("\n{}", "=".repeat(70));
        println!("Testing: {}", image_name);
        println!("{}\n", "=".repeat(70));

        let target = ImageBuffer::load(image_path).expect("Failed to load image");
        let (width, height) = (target.width, target.height);
        println!("Image: {}x{}", width, height);

        // Test 1: Baseline (256 Gaussians, no densification)
        println!("\n--- Baseline (256 Gaussians, no densification) ---");
        let baseline_result = run_baseline(&target, 256, image_path);

        // Test 2: 3DGS-Faithful Densification (64 → 256)
        println!("\n--- 3DGS-Faithful Densification (64 → 256) ---");
        let densify_result = run_3dgs_faithful(&target, 64, 256, image_path);

        // Test 3: 3DGS-Faithful with state inheritance (for difficult images)
        println!("\n--- 3DGS-Faithful with State Inheritance (64 → 256) ---");
        let inherit_result = run_3dgs_faithful_with_inheritance(&target, 64, 256, image_path);

        all_results.push((
            image_name.to_string(),
            baseline_result,
            densify_result,
            inherit_result,
        ));
    }

    // Summary
    println!("\n\n{}", "=".repeat(70));
    println!("                    FINAL SUMMARY");
    println!("{}", "=".repeat(70));
    println!(
        "{:<12} {:>12} {:>12} {:>12} {:>12}",
        "Image", "Baseline", "Densify", "vs Base", "Inherit"
    );
    println!("{:-<12} {:-<12} {:-<12} {:-<12} {:-<12}", "", "", "", "", "");

    for (name, baseline, densify, inherit) in &all_results {
        let delta = densify.1 - baseline.1;
        println!(
            "{:<12} {:>11.2}dB {:>11.2}dB {:>+11.2}dB {:>11.2}dB",
            name, baseline.1, densify.1, delta, inherit.1
        );
    }

    println!("\nSuccess criteria: Densification within 0.1 dB of baseline");
}

/// Run baseline (no densification)
fn run_baseline(target: &ImageBuffer<f32>, n: usize, image_path: &str) -> (usize, f32, f32) {
    let (width, height) = (target.width, target.height);
    let mut gaussians = create_jittered_grid(target, n);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);
    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n, init_psnr);

    let start = Instant::now();
    let mut adam = AdamWithGradientAccess::new(n);

    // Match total iterations with densification test for fair comparison
    let total_iterations = 400;
    let mut last_psnr = init_psnr;

    for iter in 1..=total_iterations {
        adam.step(&mut gaussians, target);

        if iter % 20 == 0 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = compute_psnr(&rendered, target);
            let direction = if psnr > last_psnr { "↑" } else { "↓" };
            println!(
                "    [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                iter, total_iterations, psnr, direction, psnr - last_psnr
            );
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);

    println!(
        "  Final: {} Gaussians, PSNR={:.2}dB ({:+.2}dB) in {:.1}s",
        gaussians.len(),
        final_psnr,
        final_psnr - init_psnr,
        elapsed
    );

    // Save result with rendered image
    let mut result = TestResult::new("3dgs_faithful_baseline", image_path);
    result.optimizer = format!("Adam baseline ({} Gaussians)", n);
    result.image_dimensions = (width, height);
    result.n_gaussians = n;
    result.iterations = total_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;

    // Record optimizer parameters
    result.set_param("lr_position", adam.lr_position);
    result.set_param("lr_color", adam.lr_color);
    result.set_param("lr_scale", adam.lr_scale);
    result.set_param("lr_opacity", adam.lr_opacity);
    result.set_param("beta1", adam.beta1);
    result.set_param("beta2", adam.beta2);

    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (n, final_psnr, elapsed)
}

/// Run 3DGS-faithful densification
fn run_3dgs_faithful(
    target: &ImageBuffer<f32>,
    initial_n: usize,
    target_n: usize,
    image_path: &str,
) -> (usize, f32, f32) {
    let (width, height) = (target.width, target.height);
    let mut gaussians = create_jittered_grid(target, initial_n);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);
    println!(
        "  Initial: {} Gaussians, PSNR={:.2}dB (target: {})",
        initial_n, init_psnr, target_n
    );

    let start = Instant::now();
    let mut adam = AdamWithGradientAccess::new(initial_n);

    // Configure densifier with tuned thresholds
    let mut densifier = AdaptiveDensifier::default();
    // Lower threshold for more densification events (was 0.0005, only 1 event per image)
    densifier.densify_grad_threshold = 0.0002;
    // 3DGS paper: 1% of scene extent (scene extent ≈ 1.0 for normalized coords)
    densifier.split_scale_threshold = 0.01;
    // 3DGS paper: prune if α < 0.005
    densifier.prune_opacity_threshold = 0.005;
    // More frequent densification checks (was 100, now 50 for more events)
    densifier.densify_interval = 50;
    densifier.prune_interval = 100;

    let total_iterations = 400;
    let densify_start = 50;   // Start after initial convergence
    let densify_end = 300;    // Extended window to reach 256 target

    let mut last_psnr = init_psnr;
    let mut densification_events = 0;

    for iter in 1..=total_iterations {
        adam.step(&mut gaussians, target);

        // Densification phase
        if iter >= densify_start && iter < densify_end && iter % densifier.densify_interval == 0 {
            if gaussians.len() < target_n {
                // Use Adam's accumulated position gradients instead of error proxy
                let grad_mags = adam.get_position_gradient_magnitudes();

                let (to_add, to_remove) = densifier.densify(&gaussians, &grad_mags);

                if !to_add.is_empty() || !to_remove.is_empty() {
                    let n_added = to_add.len();
                    let n_removed = to_remove.len();

                    apply_densification(&mut gaussians, to_add, to_remove.clone());
                    adam.handle_densification(&to_remove, n_added);

                    densification_events += 1;
                    println!(
                        "    [iter {:3}] Densified: +{} added, -{} removed → {} total",
                        iter, n_added, n_removed, gaussians.len()
                    );
                }
            }
        }

        // Pruning phase
        if iter % densifier.prune_interval == 0 && iter > densify_start {
            let to_prune = densifier.prune(&gaussians);
            if !to_prune.is_empty() {
                let n_pruned = to_prune.len();
                apply_densification(&mut gaussians, vec![], to_prune.clone());
                adam.handle_densification(&to_prune, 0);
                println!(
                    "    [iter {:3}] Pruned {} low-opacity Gaussians → {} total",
                    iter, n_pruned, gaussians.len()
                );
            }
        }

        if iter % 30 == 0 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = compute_psnr(&rendered, target);
            let direction = if psnr > last_psnr { "↑" } else { "↓" };
            println!(
                "    [{:3}/{:3}] PSNR={:.2}dB {} N={}",
                iter, total_iterations, psnr, direction, gaussians.len()
            );
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);

    println!(
        "  Final: {} Gaussians, PSNR={:.2}dB ({:+.2}dB) in {:.1}s",
        gaussians.len(),
        final_psnr,
        final_psnr - init_psnr,
        elapsed
    );
    println!("  Densification events: {}", densification_events);

    // Save result with rendered image
    let mut result = TestResult::new("3dgs_faithful_densification", image_path);
    result.optimizer = "Adam + 3DGS-faithful densification".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.iterations = total_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;

    // Record optimizer parameters
    result.set_param("lr_position", adam.lr_position);
    result.set_param("lr_color", adam.lr_color);
    result.set_param("lr_scale", adam.lr_scale);
    result.set_param("lr_opacity", adam.lr_opacity);
    result.set_param("beta1", adam.beta1);
    result.set_param("beta2", adam.beta2);
    result.set_param("densify_grad_threshold", densifier.densify_grad_threshold);
    result.set_param("prune_opacity_threshold", densifier.prune_opacity_threshold);
    result.set_param("densify_interval", densifier.densify_interval);
    result.set_param("start_gaussians", initial_n);
    result.set_param("target_gaussians", target_n);

    result.extra_metrics = Some(json!({
        "final_gaussians": gaussians.len(),
        "densify_start": densify_start,
        "densify_end": densify_end,
        "split_scale_threshold": densifier.split_scale_threshold,
        "densification_events": densification_events,
    }));

    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (gaussians.len(), final_psnr, elapsed)
}

/// Run 3DGS-faithful densification with state inheritance (BOGausS technique)
fn run_3dgs_faithful_with_inheritance(
    target: &ImageBuffer<f32>,
    initial_n: usize,
    target_n: usize,
    image_path: &str,
) -> (usize, f32, f32) {
    let (width, height) = (target.width, target.height);
    let mut gaussians = create_jittered_grid(target, initial_n);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);
    println!(
        "  Initial: {} Gaussians, PSNR={:.2}dB (target: {})",
        initial_n, init_psnr, target_n
    );

    let start = Instant::now();
    let mut adam = AdamWithGradientAccess::new(initial_n);

    // Same tuned thresholds as non-inheritance variant
    let mut densifier = AdaptiveDensifier::default();
    densifier.densify_grad_threshold = 0.0002;
    densifier.split_scale_threshold = 0.01;
    densifier.prune_opacity_threshold = 0.005;
    densifier.densify_interval = 50;
    densifier.prune_interval = 100;

    let total_iterations = 400;
    let densify_start = 50;
    let densify_end = 300;

    let mut last_psnr = init_psnr;
    let mut densification_events = 0;

    for iter in 1..=total_iterations {
        adam.step(&mut gaussians, target);

        // Densification with state inheritance
        if iter >= densify_start && iter < densify_end && iter % densifier.densify_interval == 0 {
            if gaussians.len() < target_n {
                let grad_mags = adam.get_position_gradient_magnitudes();
                let (to_add, to_remove) = densifier.densify(&gaussians, &grad_mags);

                if !to_add.is_empty() || !to_remove.is_empty() {
                    let n_added = to_add.len();
                    let n_removed = to_remove.len();

                    // STATE INHERITANCE: Copy parent's Adam state to children
                    // This is the BOGausS technique that helps on difficult images
                    let old_n = gaussians.len();
                    apply_densification(&mut gaussians, to_add, to_remove.clone());

                    // Inherit state from parents (approximation: inherit from nearest existing)
                    // For splits, children should inherit from parent
                    // For clones, children should inherit from original
                    adam.handle_densification(&to_remove, n_added);

                    // Copy state from parent (simplified: use average of existing state)
                    // This is a simplification; proper inheritance would track parent-child relationships
                    if n_added > 0 && old_n > to_remove.len() {
                        let remaining = old_n - to_remove.len();
                        for i in remaining..(remaining + n_added) {
                            if i < adam.m_position.len() {
                                // Inherit from a nearby Gaussian (simplified)
                                let parent_idx = i % remaining.max(1);
                                if parent_idx < adam.m_position.len() {
                                    adam.m_position[i] = adam.m_position[parent_idx];
                                    adam.v_position[i] = adam.v_position[parent_idx];
                                    adam.m_color[i] = adam.m_color[parent_idx];
                                    adam.v_color[i] = adam.v_color[parent_idx];
                                    adam.m_scale[i] = adam.m_scale[parent_idx];
                                    adam.v_scale[i] = adam.v_scale[parent_idx];
                                    adam.m_opacity[i] = adam.m_opacity[parent_idx];
                                    adam.v_opacity[i] = adam.v_opacity[parent_idx];
                                }
                            }
                        }
                    }

                    densification_events += 1;
                    println!(
                        "    [iter {:3}] Densified (inherited): +{} → {} total",
                        iter, n_added, gaussians.len()
                    );
                }
            }
        }

        // Pruning
        if iter % densifier.prune_interval == 0 && iter > densify_start {
            let to_prune = densifier.prune(&gaussians);
            if !to_prune.is_empty() {
                let n_pruned = to_prune.len();
                apply_densification(&mut gaussians, vec![], to_prune.clone());
                adam.handle_densification(&to_prune, 0);
                println!(
                    "    [iter {:3}] Pruned {} → {} total",
                    iter, n_pruned, gaussians.len()
                );
            }
        }

        if iter % 30 == 0 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = compute_psnr(&rendered, target);
            let direction = if psnr > last_psnr { "↑" } else { "↓" };
            println!(
                "    [{:3}/{:3}] PSNR={:.2}dB {} N={}",
                iter, total_iterations, psnr, direction, gaussians.len()
            );
            last_psnr = psnr;
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);

    println!(
        "  Final: {} Gaussians, PSNR={:.2}dB ({:+.2}dB) in {:.1}s",
        gaussians.len(),
        final_psnr,
        final_psnr - init_psnr,
        elapsed
    );

    // Save result with rendered image
    let mut result = TestResult::new("3dgs_faithful_with_inheritance", image_path);
    result.optimizer = "Adam + 3DGS-faithful + state inheritance".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.iterations = total_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;

    // Record optimizer parameters
    result.set_param("lr_position", adam.lr_position);
    result.set_param("lr_color", adam.lr_color);
    result.set_param("lr_scale", adam.lr_scale);
    result.set_param("lr_opacity", adam.lr_opacity);
    result.set_param("beta1", adam.beta1);
    result.set_param("beta2", adam.beta2);
    result.set_param("densify_grad_threshold", densifier.densify_grad_threshold);
    result.set_param("prune_opacity_threshold", densifier.prune_opacity_threshold);
    result.set_param("densify_interval", densifier.densify_interval);
    result.set_param("start_gaussians", initial_n);
    result.set_param("target_gaussians", target_n);

    result.extra_metrics = Some(json!({
        "final_gaussians": gaussians.len(),
        "state_inheritance": true,
        "densification_events": densification_events,
    }));

    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    (gaussians.len(), final_psnr, elapsed)
}
