//! Combined Densification Test
//!
//! Tests three approaches:
//! 1. Densification + Per-Gaussian Adam time (proper state for new Gaussians)
//! 2. Aggressive threshold (0.0001) for more densification events
//! 3. Hybrid: Start dense (256) → prune to optimal count

use lgi_core::ImageBuffer;
use lgi_encoder_v2::adaptive_densification::AdaptiveDensifier;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

/// Per-Gaussian Adam state with individual time counters
#[derive(Clone)]
struct PerGaussianAdamState {
    m_position: Vector2<f32>,
    v_position: Vector2<f32>,
    m_color: Color4<f32>,
    v_color: Color4<f32>,
    m_scale: (f32, f32),
    v_scale: (f32, f32),
    m_opacity: f32,
    v_opacity: f32,
    t: u32, // Per-Gaussian time counter
}

impl Default for PerGaussianAdamState {
    fn default() -> Self {
        Self {
            m_position: Vector2::new(0.0, 0.0),
            v_position: Vector2::new(0.0, 0.0),
            m_color: Color4::new(0.0, 0.0, 0.0, 0.0),
            v_color: Color4::new(0.0, 0.0, 0.0, 0.0),
            m_scale: (0.0, 0.0),
            v_scale: (0.0, 0.0),
            m_opacity: 0.0,
            v_opacity: 0.0,
            t: 0,
        }
    }
}

impl PerGaussianAdamState {
    fn inherit_from(parent: &Self) -> Self {
        Self {
            m_position: parent.m_position,
            v_position: parent.v_position,
            m_color: parent.m_color,
            v_color: parent.v_color,
            m_scale: parent.m_scale,
            v_scale: parent.v_scale,
            m_opacity: parent.m_opacity,
            v_opacity: parent.v_opacity,
            t: parent.t, // Inherit time counter
        }
    }
}

/// Adam optimizer with per-Gaussian time tracking
struct PerGaussianAdam {
    states: Vec<PerGaussianAdamState>,
    lr_position: f32,
    lr_color: f32,
    lr_scale: f32,
    lr_opacity: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    global_t: u32, // Global iteration counter for LR decay
}

impl PerGaussianAdam {
    fn new(n: usize) -> Self {
        Self {
            states: vec![PerGaussianAdamState::default(); n],
            // 3DGS-calibrated learning rates (matched to test_3dgs_faithful_densification)
            lr_position: 0.0002,
            lr_color: 0.02,     // Was 0.01 - too low!
            lr_scale: 0.005,    // Was 0.001 - too low!
            lr_opacity: 0.05,   // Was 0.01 - too low!
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            global_t: 0, // Initialize global counter
        }
    }

    fn resize(&mut self, new_n: usize) {
        self.states.resize(new_n, PerGaussianAdamState::default());
    }

    fn step(&mut self, gaussians: &mut [Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) -> f32 {
        let n = gaussians.len();
        if self.states.len() != n {
            self.resize(n);
        }

        self.global_t += 1; // Increment global counter once per step

        let rendered = RendererV2::render(gaussians, target.width, target.height);
        let loss = compute_mse(&rendered, target);
        let grads = compute_gradients(gaussians, &rendered, target);

        // Use GLOBAL time for LR decay (not per-Gaussian!)
        // This ensures LR decays based on total optimization progress, not per-Gaussian age
        let lr_pos = (self.lr_position * (0.1_f32).powf(self.global_t as f32 / 300.0)).max(0.00002);

        for i in 0..n {
            let state = &mut self.states[i];
            state.t += 1; // Per-Gaussian counter for bias correction only

            let bias_correction1 = 1.0 - self.beta1.powi(state.t as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(state.t as i32);

            // Update position moments
            state.m_position.x = self.beta1 * state.m_position.x + (1.0 - self.beta1) * grads[i].position.x;
            state.m_position.y = self.beta1 * state.m_position.y + (1.0 - self.beta1) * grads[i].position.y;
            state.v_position.x = self.beta2 * state.v_position.x + (1.0 - self.beta2) * grads[i].position.x.powi(2);
            state.v_position.y = self.beta2 * state.v_position.y + (1.0 - self.beta2) * grads[i].position.y.powi(2);

            // Update color moments
            state.m_color.r = self.beta1 * state.m_color.r + (1.0 - self.beta1) * grads[i].color.r;
            state.m_color.g = self.beta1 * state.m_color.g + (1.0 - self.beta1) * grads[i].color.g;
            state.m_color.b = self.beta1 * state.m_color.b + (1.0 - self.beta1) * grads[i].color.b;
            state.v_color.r = self.beta2 * state.v_color.r + (1.0 - self.beta2) * grads[i].color.r.powi(2);
            state.v_color.g = self.beta2 * state.v_color.g + (1.0 - self.beta2) * grads[i].color.g.powi(2);
            state.v_color.b = self.beta2 * state.v_color.b + (1.0 - self.beta2) * grads[i].color.b.powi(2);

            // Update scale moments
            state.m_scale.0 = self.beta1 * state.m_scale.0 + (1.0 - self.beta1) * grads[i].scale_x;
            state.m_scale.1 = self.beta1 * state.m_scale.1 + (1.0 - self.beta1) * grads[i].scale_y;
            state.v_scale.0 = self.beta2 * state.v_scale.0 + (1.0 - self.beta2) * grads[i].scale_x.powi(2);
            state.v_scale.1 = self.beta2 * state.v_scale.1 + (1.0 - self.beta2) * grads[i].scale_y.powi(2);

            // Update opacity moments
            state.m_opacity = self.beta1 * state.m_opacity + (1.0 - self.beta1) * grads[i].opacity;
            state.v_opacity = self.beta2 * state.v_opacity + (1.0 - self.beta2) * grads[i].opacity.powi(2);

            // Apply updates with bias correction
            let m_hat_px = state.m_position.x / bias_correction1;
            let m_hat_py = state.m_position.y / bias_correction1;
            let v_hat_px = state.v_position.x / bias_correction2;
            let v_hat_py = state.v_position.y / bias_correction2;
            gaussians[i].position.x -= lr_pos * m_hat_px / (v_hat_px.sqrt() + self.epsilon);
            gaussians[i].position.y -= lr_pos * m_hat_py / (v_hat_py.sqrt() + self.epsilon);
            gaussians[i].position.x = gaussians[i].position.x.clamp(0.0, 1.0);
            gaussians[i].position.y = gaussians[i].position.y.clamp(0.0, 1.0);

            let m_hat_r = state.m_color.r / bias_correction1;
            let m_hat_g = state.m_color.g / bias_correction1;
            let m_hat_b = state.m_color.b / bias_correction1;
            let v_hat_r = state.v_color.r / bias_correction2;
            let v_hat_g = state.v_color.g / bias_correction2;
            let v_hat_b = state.v_color.b / bias_correction2;
            gaussians[i].color.r -= self.lr_color * m_hat_r / (v_hat_r.sqrt() + self.epsilon);
            gaussians[i].color.g -= self.lr_color * m_hat_g / (v_hat_g.sqrt() + self.epsilon);
            gaussians[i].color.b -= self.lr_color * m_hat_b / (v_hat_b.sqrt() + self.epsilon);
            gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
            gaussians[i].color.g = gaussians[i].color.g.clamp(0.0, 1.0);
            gaussians[i].color.b = gaussians[i].color.b.clamp(0.0, 1.0);

            let m_hat_sx = state.m_scale.0 / bias_correction1;
            let m_hat_sy = state.m_scale.1 / bias_correction1;
            let v_hat_sx = state.v_scale.0 / bias_correction2;
            let v_hat_sy = state.v_scale.1 / bias_correction2;
            gaussians[i].shape.scale_x -= self.lr_scale * m_hat_sx / (v_hat_sx.sqrt() + self.epsilon);
            gaussians[i].shape.scale_y -= self.lr_scale * m_hat_sy / (v_hat_sy.sqrt() + self.epsilon);
            gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
            gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);

            let m_hat_op = state.m_opacity / bias_correction1;
            let v_hat_op = state.v_opacity / bias_correction2;
            gaussians[i].opacity -= self.lr_opacity * m_hat_op / (v_hat_op.sqrt() + self.epsilon);
            gaussians[i].opacity = gaussians[i].opacity.clamp(0.0, 1.0);
        }

        loss
    }

    fn get_position_gradient_magnitudes(&self) -> Vec<f32> {
        self.states
            .iter()
            .map(|s| (s.m_position.x.powi(2) + s.m_position.y.powi(2)).sqrt())
            .collect()
    }

    fn handle_densification_with_inheritance(
        &mut self,
        to_remove: &[usize],
        new_gaussians: &[(Gaussian2D<f32, Euler<f32>>, usize)], // (gaussian, parent_idx)
    ) {
        // Remove states for pruned Gaussians (in reverse order)
        let mut sorted_remove: Vec<_> = to_remove.to_vec();
        sorted_remove.sort_by(|a, b| b.cmp(a));
        for idx in sorted_remove {
            if idx < self.states.len() {
                self.states.remove(idx);
            }
        }

        // Add states for new Gaussians with inheritance
        for (_gaussian, parent_idx) in new_gaussians {
            let new_state = if *parent_idx < self.states.len() {
                PerGaussianAdamState::inherit_from(&self.states[*parent_idx])
            } else {
                PerGaussianAdamState::default()
            };
            self.states.push(new_state);
        }
    }
}

struct GaussianGradient {
    position: Vector2<f32>,
    color: Color4<f32>,
    scale_x: f32,
    scale_y: f32,
    opacity: f32,
}

impl Default for GaussianGradient {
    fn default() -> Self {
        Self {
            position: Vector2::new(0.0, 0.0),
            color: Color4::new(0.0, 0.0, 0.0, 0.0),
            scale_x: 0.0,
            scale_y: 0.0,
            opacity: 0.0,
        }
    }
}

fn compute_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
) -> Vec<GaussianGradient> {
    // Use NORMALIZED coordinates (0-1 space) - matching test_3dgs_faithful_densification
    // This ensures gradient magnitudes are independent of image resolution
    let n = gaussians.len();
    let width = target.width as f32;
    let height = target.height as f32;

    let mut grads: Vec<GaussianGradient> = (0..n)
        .map(|_| GaussianGradient::default())
        .collect();

    for py in 0..target.height {
        for px in 0..target.width {
            let idx = (py * target.width + px) as usize;
            let t = &target.data[idx];
            let r = &rendered.data[idx];

            let error_r = r.r - t.r;
            let error_g = r.g - t.g;
            let error_b = r.b - t.b;

            // Normalized pixel coordinates (0-1)
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
    let cell_size = 1.0 / grid_size as f32;
    let mut gaussians = Vec::with_capacity(n);

    for i in 0..n {
        let gx = i % grid_size;
        let gy = i / grid_size;
        let jx = ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5;
        let jy = ((i * 11 + 7) % 100) as f32 / 100.0 - 0.5;
        let x = ((gx as f32 + 0.5 + jx * 0.4) / grid_size as f32).clamp(0.01, 0.99);
        let y = ((gy as f32 + 0.5 + jy * 0.4) / grid_size as f32).clamp(0.01, 0.99);
        let px = ((x * target.width as f32) as u32).min(target.width - 1);
        let py = ((y * target.height as f32) as u32).min(target.height - 1);
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
    println!("║   COMBINED DENSIFICATION TEST                                ║");
    println!("║   1. Densify + Per-Gaussian Adam Time                        ║");
    println!("║   2. Aggressive Threshold (0.0001)                           ║");
    println!("║   3. Hybrid: Start Dense → Prune                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let test_images = vec![
        ("/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png", "kodim03"),
        ("/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim05.png", "kodim05"),
        ("/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim08.png", "kodim08"),
    ];

    let mut all_results: Vec<(String, f32, f32, f32, f32)> = Vec::new();

    for (image_path, image_name) in &test_images {
        println!("\n{}", "=".repeat(70));
        println!("Testing: {}", image_name);
        println!("{}\n", "=".repeat(70));

        let target = ImageBuffer::load(image_path).expect("Failed to load image");
        println!("Image: {}x{}", target.width, target.height);

        // Test 1: Baseline (256 Gaussians, no densification)
        println!("\n--- Baseline (256 Gaussians) ---");
        let baseline = run_baseline(&target, 256, image_path);

        // Test 2: Densify + Per-Gaussian Adam (64→256, threshold 0.0002)
        println!("\n--- Densify + Per-Gaussian Adam (64→256, t=0.0002) ---");
        let densify_standard = run_densify_per_gaussian(&target, 64, 256, 0.0002, image_path);

        // Test 3: Aggressive threshold (64→256, threshold 0.0001)
        println!("\n--- Aggressive Threshold (64→256, t=0.0001) ---");
        let densify_aggressive = run_densify_per_gaussian(&target, 64, 256, 0.0001, image_path);

        // Test 4: Hybrid - Start dense, prune to optimal
        println!("\n--- Hybrid: Dense→Prune (256→optimal) ---");
        let hybrid = run_hybrid_prune(&target, 256, image_path);

        all_results.push((
            image_name.to_string(),
            baseline,
            densify_standard,
            densify_aggressive,
            hybrid,
        ));
    }

    // Summary
    println!("\n\n{}", "=".repeat(70));
    println!("                    FINAL SUMMARY");
    println!("{}", "=".repeat(70));
    println!(
        "{:<12} {:>10} {:>12} {:>12} {:>10}",
        "Image", "Baseline", "Densify", "Aggressive", "Hybrid"
    );
    println!("{:-<12} {:-<10} {:-<12} {:-<12} {:-<10}", "", "", "", "", "");

    for (name, baseline, densify, aggressive, hybrid) in &all_results {
        println!(
            "{:<12} {:>9.2}dB {:>11.2}dB {:>11.2}dB {:>9.2}dB",
            name, baseline, densify, aggressive, hybrid
        );
    }

    // Find best per image
    println!("\nBest per image:");
    for (name, baseline, densify, aggressive, hybrid) in &all_results {
        let methods = [
            ("Baseline", *baseline),
            ("Densify", *densify),
            ("Aggressive", *aggressive),
            ("Hybrid", *hybrid),
        ];
        let best = methods.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        let delta = best.1 - baseline;
        println!("  {}: {} ({:.2}dB, {:+.2}dB vs baseline)", name, best.0, best.1, delta);
    }
}

fn run_baseline(target: &ImageBuffer<f32>, n: usize, image_path: &str) -> f32 {
    let mut gaussians = create_jittered_grid(target, n);
    let init_psnr = compute_psnr(&RendererV2::render(&gaussians, target.width, target.height), target);
    println!("  Initial: {} Gaussians, PSNR={:.2}dB", n, init_psnr);

    let start = Instant::now();
    let mut adam = PerGaussianAdam::new(n);

    // Create TestResult upfront to track state
    let mut result = TestResult::new("combined_baseline", image_path);
    result.optimizer = "Per-Gaussian Adam baseline".to_string();
    result.n_gaussians = n;
    result.iterations = 200;
    result.initial_psnr = init_psnr;
    result.image_dimensions = (target.width, target.height);
    // Record optimizer parameters
    result.set_param("lr_position", adam.lr_position);
    result.set_param("lr_color", adam.lr_color);
    result.set_param("lr_scale", adam.lr_scale);
    result.set_param("lr_opacity", adam.lr_opacity);
    result.set_param("beta1", adam.beta1);
    result.set_param("beta2", adam.beta2);

    for iter in 1..=200 {
        adam.step(&mut gaussians, target);
        if iter % 50 == 0 {
            let psnr = compute_psnr(&RendererV2::render(&gaussians, target.width, target.height), target);
            println!("    [{:3}/200] PSNR={:.2}dB N={}", iter, psnr, gaussians.len());
        }
    }

    let rendered = RendererV2::render(&gaussians, target.width, target.height);
    let final_psnr = compute_psnr(&rendered, target);
    let elapsed = start.elapsed().as_secs_f32();
    println!("  Final: {} Gaussians, PSNR={:.2}dB in {:.1}s", gaussians.len(), final_psnr, elapsed);

    // Complete result and save with image
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;
    result.complete();
    let _ = result.save_with_image(&rendered, RESULTS_DIR);

    final_psnr
}

fn run_densify_per_gaussian(
    target: &ImageBuffer<f32>,
    initial_n: usize,
    target_n: usize,
    grad_threshold: f32,
    image_path: &str,
) -> f32 {
    let mut gaussians = create_jittered_grid(target, initial_n);
    let init_psnr = compute_psnr(&RendererV2::render(&gaussians, target.width, target.height), target);
    println!("  Initial: {} Gaussians, PSNR={:.2}dB (target: {})", initial_n, init_psnr, target_n);

    let start = Instant::now();
    let mut adam = PerGaussianAdam::new(initial_n);

    let mut densifier = AdaptiveDensifier::default();
    densifier.densify_grad_threshold = grad_threshold;
    densifier.split_scale_threshold = 0.01;
    densifier.prune_opacity_threshold = 0.005;
    densifier.densify_interval = 30; // More frequent checks
    densifier.prune_interval = 100;

    let total_iterations = 300;
    let densify_start = 30;
    let densify_end = 250;
    let mut densify_count = 0;

    for iter in 1..=total_iterations {
        adam.step(&mut gaussians, target);

        // Densification phase
        if iter >= densify_start && iter < densify_end && iter % densifier.densify_interval == 0 {
            if gaussians.len() < target_n {
                let grad_mags = adam.get_position_gradient_magnitudes();
                let (to_add, to_remove) = densifier.densify(&gaussians, &grad_mags);

                if !to_add.is_empty() || !to_remove.is_empty() {
                    let n_added = to_add.len();
                    let n_removed = to_remove.len();

                    // Create new Gaussians with parent indices for inheritance
                    let new_with_parents: Vec<_> = to_add.iter().map(|g| {
                        // Find closest existing Gaussian as parent
                        let parent_idx = gaussians.iter().enumerate()
                            .min_by(|(_, a), (_, b)| {
                                let da = (a.position.x - g.position.x).powi(2) + (a.position.y - g.position.y).powi(2);
                                let db = (b.position.x - g.position.x).powi(2) + (b.position.y - g.position.y).powi(2);
                                da.partial_cmp(&db).unwrap()
                            })
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        (g.clone(), parent_idx)
                    }).collect();

                    // Apply to Gaussians
                    let mut sorted_remove: Vec<_> = to_remove.clone();
                    sorted_remove.sort_by(|a, b| b.cmp(a));
                    for idx in &sorted_remove {
                        if *idx < gaussians.len() {
                            gaussians.remove(*idx);
                        }
                    }
                    gaussians.extend(to_add);

                    // Update Adam state with inheritance
                    adam.handle_densification_with_inheritance(&to_remove, &new_with_parents);

                    densify_count += 1;
                    println!(
                        "    [iter {:3}] Densified: +{} -{} → {} total",
                        iter, n_added, n_removed, gaussians.len()
                    );
                }
            }
        }

        // Pruning
        if iter % densifier.prune_interval == 0 && iter > densify_start {
            let to_prune = densifier.prune(&gaussians);
            if !to_prune.is_empty() {
                let n_pruned = to_prune.len();
                adam.handle_densification_with_inheritance(&to_prune, &[]);
                let mut sorted: Vec<_> = to_prune;
                sorted.sort_by(|a, b| b.cmp(a));
                for idx in sorted {
                    if idx < gaussians.len() {
                        gaussians.remove(idx);
                    }
                }
                println!("    [iter {:3}] Pruned {} → {} total", iter, n_pruned, gaussians.len());
            }
        }

        if iter % 50 == 0 {
            let psnr = compute_psnr(&RendererV2::render(&gaussians, target.width, target.height), target);
            println!("    [{:3}/{}] PSNR={:.2}dB N={}", iter, total_iterations, psnr, gaussians.len());
        }
    }

    let rendered = RendererV2::render(&gaussians, target.width, target.height);
    let final_psnr = compute_psnr(&rendered, target);
    let elapsed = start.elapsed().as_secs_f32();
    println!(
        "  Final: {} Gaussians, PSNR={:.2}dB in {:.1}s ({} densify events)",
        gaussians.len(), final_psnr, elapsed, densify_count
    );

    let mut result = TestResult::new(&format!("combined_densify_t{}", grad_threshold), image_path);
    result.optimizer = format!("Per-Gaussian Adam + Densify (t={})", grad_threshold);
    result.n_gaussians = gaussians.len();
    result.iterations = total_iterations;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;
    result.image_dimensions = (target.width, target.height);
    // Record parameters
    result.set_param("lr_position", adam.lr_position);
    result.set_param("lr_color", adam.lr_color);
    result.set_param("lr_scale", adam.lr_scale);
    result.set_param("lr_opacity", adam.lr_opacity);
    result.set_param("densify_grad_threshold", grad_threshold);
    result.set_param("start_gaussians", initial_n);
    result.set_param("target_gaussians", target_n);
    result.set_param("densify_count", densify_count);
    result.complete();
    let _ = result.save_with_image(&rendered, RESULTS_DIR);

    final_psnr
}

fn run_hybrid_prune(target: &ImageBuffer<f32>, initial_n: usize, image_path: &str) -> f32 {
    let mut gaussians = create_jittered_grid(target, initial_n);
    let init_psnr = compute_psnr(&RendererV2::render(&gaussians, target.width, target.height), target);
    println!("  Initial: {} Gaussians, PSNR={:.2}dB", initial_n, init_psnr);

    let start = Instant::now();
    let mut adam = PerGaussianAdam::new(initial_n);

    // Phase 1: Optimize with all Gaussians (100 iterations)
    println!("  Phase 1: Optimize (100 iters)");
    for iter in 1..=100 {
        adam.step(&mut gaussians, target);
        if iter % 50 == 0 {
            let psnr = compute_psnr(&RendererV2::render(&gaussians, target.width, target.height), target);
            println!("    [{:3}/100] PSNR={:.2}dB N={}", iter, psnr, gaussians.len());
        }
    }

    // Phase 2: Prune low-opacity Gaussians
    println!("  Phase 2: Prune low-opacity");
    let prune_threshold = 0.1; // Aggressive pruning
    let n_before = gaussians.len();
    let to_prune: Vec<usize> = gaussians.iter().enumerate()
        .filter(|(_, g)| g.opacity < prune_threshold)
        .map(|(i, _)| i)
        .collect();

    adam.handle_densification_with_inheritance(&to_prune, &[]);
    let mut sorted: Vec<_> = to_prune;
    sorted.sort_by(|a, b| b.cmp(a));
    for idx in sorted {
        if idx < gaussians.len() {
            gaussians.remove(idx);
        }
    }
    println!("    Pruned {} → {} Gaussians", n_before - gaussians.len(), gaussians.len());

    // Phase 3: Continue optimizing with reduced set (100 more iterations)
    println!("  Phase 3: Re-optimize (100 iters)");
    for iter in 1..=100 {
        adam.step(&mut gaussians, target);
        if iter % 50 == 0 {
            let psnr = compute_psnr(&RendererV2::render(&gaussians, target.width, target.height), target);
            println!("    [{:3}/100] PSNR={:.2}dB N={}", iter, psnr, gaussians.len());
        }
    }

    let rendered = RendererV2::render(&gaussians, target.width, target.height);
    let final_psnr = compute_psnr(&rendered, target);
    let elapsed = start.elapsed().as_secs_f32();
    println!(
        "  Final: {} Gaussians, PSNR={:.2}dB in {:.1}s",
        gaussians.len(), final_psnr, elapsed
    );

    let mut result = TestResult::new("combined_hybrid_prune", image_path);
    result.optimizer = "Hybrid: Dense→Prune→Optimize".to_string();
    result.n_gaussians = gaussians.len();
    result.iterations = 200;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;
    result.image_dimensions = (target.width, target.height);
    // Record parameters
    result.set_param("lr_position", adam.lr_position);
    result.set_param("lr_color", adam.lr_color);
    result.set_param("lr_scale", adam.lr_scale);
    result.set_param("lr_opacity", adam.lr_opacity);
    result.set_param("prune_opacity_threshold", prune_threshold);
    result.set_param("start_gaussians", initial_n);
    result.set_param("phase1_iters", 100_usize);
    result.set_param("phase3_iters", 100_usize);
    result.set_param("final_gaussians", gaussians.len());
    result.complete();
    let _ = result.save_with_image(&rendered, RESULTS_DIR);

    final_psnr
}
