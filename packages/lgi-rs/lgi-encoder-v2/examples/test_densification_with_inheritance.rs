//! Densification with State Inheritance Test
//!
//! KEY INSIGHT from ADAM-considerations.txt and BOGausS:
//! When a Gaussian is split, DON'T throw away optimizer state!
//!
//! Instead:
//! - Copy/scale parent's Adam moments (m, v) to children
//! - Inherit time counter t from parent
//!
//! This avoids the "newborn Gaussians with no history" problem
//! and significantly improves convergence after densification.
//!
//! Tests:
//! 1. Baseline densification (reset state on split)
//! 2. Full state inheritance (m, v, t copied)
//! 3. Partial inheritance (only v copied, fresh m)
//!
//! Run: cargo run --release --example test_densification_with_inheritance -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

const TEST_IMAGES: &[(&str, &str)] = &[
    ("kodim03", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png"),
    ("kodim05", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim05.png"),
    ("kodim08", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim08.png"),
];

/// Inheritance mode for densification
#[derive(Clone, Copy, Debug)]
enum InheritanceMode {
    /// Reset all state for new Gaussians
    NoInheritance,
    /// Inherit everything: m, v, t
    FullInheritance,
    /// Only inherit v (variance), reset m (momentum)
    PartialInheritance,
}

/// Per-Gaussian Adam state
#[derive(Clone)]
struct GaussianState {
    m_color: Color4<f32>,
    m_position: Vector2<f32>,
    m_scale: (f32, f32),
    v_color: Color4<f32>,
    v_position: Vector2<f32>,
    v_scale: (f32, f32),
    t: u32,
}

impl GaussianState {
    fn new() -> Self {
        Self {
            m_color: Color4::new(0.0, 0.0, 0.0, 0.0),
            m_position: Vector2::zero(),
            m_scale: (0.0, 0.0),
            v_color: Color4::new(0.0, 0.0, 0.0, 0.0),
            v_position: Vector2::zero(),
            v_scale: (0.0, 0.0),
            t: 0,
        }
    }

    fn inherit_from(parent: &GaussianState, mode: InheritanceMode) -> Self {
        match mode {
            InheritanceMode::NoInheritance => Self::new(),
            InheritanceMode::FullInheritance => {
                // Full inheritance: m/2, v, t
                Self {
                    m_color: Color4::new(
                        parent.m_color.r * 0.5,
                        parent.m_color.g * 0.5,
                        parent.m_color.b * 0.5,
                        parent.m_color.a * 0.5,
                    ),
                    m_position: Vector2::new(
                        parent.m_position.x * 0.5,
                        parent.m_position.y * 0.5,
                    ),
                    m_scale: (parent.m_scale.0 * 0.5, parent.m_scale.1 * 0.5),
                    v_color: parent.v_color,
                    v_position: parent.v_position,
                    v_scale: parent.v_scale,
                    t: parent.t,
                }
            }
            InheritanceMode::PartialInheritance => {
                // Partial: keep v (variance estimate), reset m (momentum)
                Self {
                    m_color: Color4::new(0.0, 0.0, 0.0, 0.0),
                    m_position: Vector2::zero(),
                    m_scale: (0.0, 0.0),
                    v_color: parent.v_color,
                    v_position: parent.v_position,
                    v_scale: parent.v_scale,
                    t: parent.t,
                }
            }
        }
    }
}

/// Adam optimizer with state inheritance support
struct InheritanceAwareAdam {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    lr_position: f32,
    lr_color: f32,
    lr_scale: f32,
    states: Vec<GaussianState>,
    inheritance_mode: InheritanceMode,
}

impl InheritanceAwareAdam {
    fn new(mode: InheritanceMode) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            lr_position: 0.0002,
            lr_color: 0.02,
            lr_scale: 0.005,
            states: Vec::new(),
            inheritance_mode: mode,
        }
    }

    fn ensure_states(&mut self, n: usize) {
        while self.states.len() < n {
            self.states.push(GaussianState::new());
        }
    }

    /// Split a Gaussian with the configured inheritance mode
    fn split_gaussian(&mut self, gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>, idx: usize) {
        let parent = gaussians[idx].clone();
        let parent_state = self.states[idx].clone();

        let angle = parent.shape.rotation;
        let offset = parent.shape.scale_x * 0.3;

        // Child 1 replaces parent
        gaussians[idx] = Gaussian2D::new(
            Vector2::new(
                (parent.position.x + angle.cos() * offset).clamp(0.01, 0.99),
                (parent.position.y + angle.sin() * offset).clamp(0.01, 0.99),
            ),
            Euler::new(parent.shape.scale_x * 0.6, parent.shape.scale_y * 0.6, angle),
            parent.color,
            1.0,  // FULL opacity
        );

        // Child 2 is appended
        gaussians.push(Gaussian2D::new(
            Vector2::new(
                (parent.position.x - angle.cos() * offset).clamp(0.01, 0.99),
                (parent.position.y - angle.sin() * offset).clamp(0.01, 0.99),
            ),
            Euler::new(parent.shape.scale_x * 0.6, parent.shape.scale_y * 0.6, angle),
            parent.color,
            1.0,
        ));

        // Apply inheritance to states
        self.states[idx] = GaussianState::inherit_from(&parent_state, self.inheritance_mode);
        self.states.push(GaussianState::inherit_from(&parent_state, self.inheritance_mode));
    }

    fn step(&mut self, gaussians: &mut [Gaussian2D<f32, Euler<f32>>], target: &ImageBuffer<f32>) -> f32 {
        let n = gaussians.len();
        self.ensure_states(n);

        let width = target.width;
        let height = target.height;

        let rendered = RendererV2::render(gaussians, width, height);
        let grads = self.compute_gradients(gaussians, &rendered, target);

        for i in 0..n {
            let state = &mut self.states[i];
            state.t += 1;

            let bias_correction1 = 1.0 - self.beta1.powi(state.t as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(state.t as i32);

            // Color update
            state.m_color.r = self.beta1 * state.m_color.r + (1.0 - self.beta1) * grads[i].color.r;
            state.m_color.g = self.beta1 * state.m_color.g + (1.0 - self.beta1) * grads[i].color.g;
            state.m_color.b = self.beta1 * state.m_color.b + (1.0 - self.beta1) * grads[i].color.b;
            state.v_color.r = self.beta2 * state.v_color.r + (1.0 - self.beta2) * grads[i].color.r.powi(2);
            state.v_color.g = self.beta2 * state.v_color.g + (1.0 - self.beta2) * grads[i].color.g.powi(2);
            state.v_color.b = self.beta2 * state.v_color.b + (1.0 - self.beta2) * grads[i].color.b.powi(2);

            gaussians[i].color.r -= self.lr_color * (state.m_color.r / bias_correction1) / ((state.v_color.r / bias_correction2).sqrt() + self.epsilon);
            gaussians[i].color.g -= self.lr_color * (state.m_color.g / bias_correction1) / ((state.v_color.g / bias_correction2).sqrt() + self.epsilon);
            gaussians[i].color.b -= self.lr_color * (state.m_color.b / bias_correction1) / ((state.v_color.b / bias_correction2).sqrt() + self.epsilon);
            gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
            gaussians[i].color.g = gaussians[i].color.g.clamp(0.0, 1.0);
            gaussians[i].color.b = gaussians[i].color.b.clamp(0.0, 1.0);

            // Position update
            state.m_position.x = self.beta1 * state.m_position.x + (1.0 - self.beta1) * grads[i].position.x;
            state.m_position.y = self.beta1 * state.m_position.y + (1.0 - self.beta1) * grads[i].position.y;
            state.v_position.x = self.beta2 * state.v_position.x + (1.0 - self.beta2) * grads[i].position.x.powi(2);
            state.v_position.y = self.beta2 * state.v_position.y + (1.0 - self.beta2) * grads[i].position.y.powi(2);

            gaussians[i].position.x -= self.lr_position * (state.m_position.x / bias_correction1) / ((state.v_position.x / bias_correction2).sqrt() + self.epsilon);
            gaussians[i].position.y -= self.lr_position * (state.m_position.y / bias_correction1) / ((state.v_position.y / bias_correction2).sqrt() + self.epsilon);
            gaussians[i].position.x = gaussians[i].position.x.clamp(0.01, 0.99);
            gaussians[i].position.y = gaussians[i].position.y.clamp(0.01, 0.99);

            // Scale update
            state.m_scale.0 = self.beta1 * state.m_scale.0 + (1.0 - self.beta1) * grads[i].scale_x;
            state.m_scale.1 = self.beta1 * state.m_scale.1 + (1.0 - self.beta1) * grads[i].scale_y;
            state.v_scale.0 = self.beta2 * state.v_scale.0 + (1.0 - self.beta2) * grads[i].scale_x.powi(2);
            state.v_scale.1 = self.beta2 * state.v_scale.1 + (1.0 - self.beta2) * grads[i].scale_y.powi(2);

            gaussians[i].shape.scale_x -= self.lr_scale * (state.m_scale.0 / bias_correction1) / ((state.v_scale.0 / bias_correction2).sqrt() + self.epsilon);
            gaussians[i].shape.scale_y -= self.lr_scale * (state.m_scale.1 / bias_correction1) / ((state.v_scale.1 / bias_correction2).sqrt() + self.epsilon);
            gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
            gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);
        }

        // Compute loss
        let mut mse = 0.0;
        for (r, t) in rendered.data.iter().zip(target.data.iter()) {
            mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
        }
        mse / (rendered.width * rendered.height * 3) as f32
    }

    fn compute_gradients(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>], rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> Vec<GaussianGradient> {
        let width = target.width;
        let height = target.height;
        let mut grads = vec![GaussianGradient::zero(); gaussians.len()];

        for y in 0..height {
            for x in 0..width {
                let px = x as f32 / width as f32;
                let py = y as f32 / height as f32;
                let rc = rendered.get_pixel(x, y).unwrap();
                let tc = target.get_pixel(x, y).unwrap();

                let error_r = 2.0 * (rc.r - tc.r);
                let error_g = 2.0 * (rc.g - tc.g);
                let error_b = 2.0 * (rc.b - tc.b);

                for (i, g) in gaussians.iter().enumerate() {
                    let dx = px - g.position.x;
                    let dy = py - g.position.y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq > 0.1 { continue; }

                    let scale_product = (g.shape.scale_x * g.shape.scale_y).max(1e-6);
                    let weight = (-0.5 * dist_sq / scale_product).exp();
                    if weight < 1e-6 { continue; }

                    grads[i].color.r += error_r * weight;
                    grads[i].color.g += error_g * weight;
                    grads[i].color.b += error_b * weight;

                    let error_weighted = error_r * g.color.r + error_g * g.color.g + error_b * g.color.b;
                    grads[i].position.x += error_weighted * weight * dx;
                    grads[i].position.y += error_weighted * weight * dy;

                    grads[i].scale_x += error_weighted * weight * dist_sq / g.shape.scale_x.powi(2).max(1e-6);
                    grads[i].scale_y += error_weighted * weight * dist_sq / g.shape.scale_y.powi(2).max(1e-6);
                }
            }
        }

        let pixel_count = (width * height) as f32;
        for grad in &mut grads {
            grad.color.r /= pixel_count;
            grad.color.g /= pixel_count;
            grad.color.b /= pixel_count;
            grad.position.x /= pixel_count;
            grad.position.y /= pixel_count;
            grad.scale_x /= pixel_count;
            grad.scale_y /= pixel_count;
        }

        grads
    }
}

#[derive(Clone)]
struct GaussianGradient {
    position: Vector2<f32>,
    color: Color4<f32>,
    scale_x: f32,
    scale_y: f32,
}

impl GaussianGradient {
    fn zero() -> Self {
        Self {
            position: Vector2::zero(),
            color: Color4::new(0.0, 0.0, 0.0, 0.0),
            scale_x: 0.0,
            scale_y: 0.0,
        }
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

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height * 3) as f32;
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= count;
    if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}

fn run_test(
    target: &ImageBuffer<f32>,
    initial_n: usize,
    max_n: usize,
    iterations: usize,
    mode: InheritanceMode,
) -> (f32, f32, usize, usize) {
    let (width, height) = (target.width, target.height);
    let mut gaussians = create_jittered_grid(target, initial_n);
    let mut adam = InheritanceAwareAdam::new(mode);

    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, target);

    let densify_delay = 30;
    let densify_interval = 20;
    let error_threshold = 0.001;  // Split Gaussians with high local error
    let mut splits = 0;

    for iter in 0..iterations {
        adam.step(&mut gaussians, target);

        // Densification after delay
        if iter >= densify_delay && iter % densify_interval == 0 && gaussians.len() < max_n {
            let rendered = RendererV2::render(&gaussians, width, height);

            // Find Gaussians covering high-error regions
            let mut to_split = Vec::new();
            for (i, g) in gaussians.iter().enumerate() {
                let px = (g.position.x * width as f32) as u32;
                let py = (g.position.y * height as f32) as u32;

                let mut local_error = 0.0;
                let mut count = 0;
                for dy in -3i32..=3 {
                    for dx in -3i32..=3 {
                        let x = (px as i32 + dx).clamp(0, width as i32 - 1) as u32;
                        let y = (py as i32 + dy).clamp(0, height as i32 - 1) as u32;
                        if let (Some(r), Some(t)) = (rendered.get_pixel(x, y), target.get_pixel(x, y)) {
                            local_error += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
                            count += 1;
                        }
                    }
                }
                if count > 0 && local_error / count as f32 > error_threshold {
                    to_split.push(i);
                }
            }

            // Split worst performers (limited to not exceed max_n)
            let can_add = max_n - gaussians.len();
            for &idx in to_split.iter().take(can_add).rev() {
                adam.split_gaussian(&mut gaussians, idx);
                splits += 1;
            }
        }
    }

    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, target);

    (init_psnr, final_psnr, gaussians.len(), splits)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   DENSIFICATION STATE INHERITANCE TEST - BOGausS Technique   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let initial_n = 64;
    let max_n = 256;
    let iterations = 200;

    let modes = vec![
        ("no_inheritance", InheritanceMode::NoInheritance),
        ("full_inheritance", InheritanceMode::FullInheritance),
        ("partial_inheritance", InheritanceMode::PartialInheritance),
    ];

    for (img_name, img_path) in TEST_IMAGES {
        println!("\n{}", "=".repeat(70));
        println!("IMAGE: {}", img_name);
        println!("{}\n", "=".repeat(70));

        let target = match ImageBuffer::load(img_path) {
            Ok(t) => t,
            Err(e) => {
                println!("Failed to load: {}", e);
                continue;
            }
        };
        let (width, height) = (target.width, target.height);

        for (mode_name, mode) in &modes {
            println!("--- {} ---", mode_name);

            let start = Instant::now();
            let (init_psnr, final_psnr, final_n, splits) = run_test(&target, initial_n, max_n, iterations, *mode);
            let elapsed = start.elapsed().as_secs_f32();

            let improvement = final_psnr - init_psnr;
            println!("  {}→{} Gaussians ({} splits)", initial_n, final_n, splits);
            println!("  {:.2}dB → {:.2}dB ({:+.2}dB) in {:.2}s", init_psnr, final_psnr, improvement, elapsed);

            // Save result
            let exp_name = format!("densification_inheritance_{}_{}", mode_name, img_name);
            let mut result = TestResult::new(&exp_name, img_path);
            result.optimizer = format!("Adam + Densification ({})", mode_name);
            result.image_dimensions = (width, height);
            result.n_gaussians = final_n;
            result.iterations = iterations;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = improvement;
            result.elapsed_seconds = elapsed;
            result.extra_metrics = Some(json!({
                "initial_gaussians": initial_n,
                "max_gaussians": max_n,
                "final_gaussians": final_n,
                "split_count": splits,
                "inheritance_mode": mode_name,
            }));
            let _ = result.save(RESULTS_DIR);
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("All tests complete! Results saved to {}", RESULTS_DIR);
    println!("{}", "=".repeat(70));
}
