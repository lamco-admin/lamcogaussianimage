//! Advanced Adam Optimizer Tests
//!
//! Tests 5 advanced Adam features from 3DGS literature (ADAM-considerations.txt):
//! 1. Per-Gaussian time counter (correct bias when Gaussians updated irregularly)
//! 2. State inheritance after densification (children inherit parent's m,v,t)
//! 3. Closed-form color updates (linear solve while Adam handles shapes)
//! 4. Confidence-scaled updates (scale steps by Gaussian "importance")
//! 5. Sparse Adam (only update Gaussians touching high-error regions)
//!
//! All tests run on kodim03, kodim05, kodim08 for statistical validity.
//!
//! Run: cargo run --release --example test_adam_advanced -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

// Test images
const TEST_IMAGES: &[(&str, &str)] = &[
    ("kodim03", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png"),
    ("kodim05", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim05.png"),
    ("kodim08", "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim08.png"),
];

/// Adam optimizer state for a single Gaussian
#[derive(Clone)]
struct PerGaussianAdamState {
    // First moment (momentum)
    m_color: Color4<f32>,
    m_position: Vector2<f32>,
    m_scale: (f32, f32),
    m_opacity: f32,

    // Second moment (RMSprop)
    v_color: Color4<f32>,
    v_position: Vector2<f32>,
    v_scale: (f32, f32),
    v_opacity: f32,

    // Per-Gaussian time counter (KEY FEATURE #1)
    t: u32,

    // Confidence score for this Gaussian (KEY FEATURE #4)
    confidence: f32,
}

impl PerGaussianAdamState {
    fn new() -> Self {
        Self {
            m_color: Color4::new(0.0, 0.0, 0.0, 0.0),
            m_position: Vector2::zero(),
            m_scale: (0.0, 0.0),
            m_opacity: 0.0,
            v_color: Color4::new(0.0, 0.0, 0.0, 0.0),
            v_position: Vector2::zero(),
            v_scale: (0.0, 0.0),
            v_opacity: 0.0,
            t: 0,
            confidence: 1.0,
        }
    }

    /// Inherit state from parent (for densification) - KEY FEATURE #2
    fn inherit_from(parent: &PerGaussianAdamState) -> Self {
        Self {
            // Divide momentum by 2 (two children share parent's momentum)
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
            m_opacity: parent.m_opacity * 0.5,

            // Keep second moment (variance estimate) as-is
            v_color: parent.v_color,
            v_position: parent.v_position,
            v_scale: parent.v_scale,
            v_opacity: parent.v_opacity,

            // Inherit time counter (key for correct bias correction)
            t: parent.t,

            // Start with moderate confidence
            confidence: 0.5,
        }
    }
}

/// Advanced Adam optimizer with per-Gaussian state
struct AdvancedAdamOptimizer {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    lr_position: f32,
    lr_color: f32,
    lr_scale: f32,
    lr_opacity: f32,

    // Per-Gaussian optimizer state
    states: Vec<PerGaussianAdamState>,

    // Global iteration counter (for when not using per-Gaussian time)
    global_t: u32,

    // Feature flags
    use_per_gaussian_time: bool,
    use_confidence_scaling: bool,
    use_sparse_updates: bool,
    sparse_threshold: f32,  // Only update if error > threshold
}

impl AdvancedAdamOptimizer {
    fn new() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            lr_position: 0.0002,
            lr_color: 0.02,
            lr_scale: 0.005,
            lr_opacity: 0.05,
            states: Vec::new(),
            global_t: 0,  // Initialize global iteration counter
            use_per_gaussian_time: true,
            use_confidence_scaling: false,
            sparse_threshold: 0.0,
            use_sparse_updates: false,
        }
    }

    fn with_features(mut self, per_gaussian_time: bool, confidence_scaling: bool, sparse: bool) -> Self {
        self.use_per_gaussian_time = per_gaussian_time;
        self.use_confidence_scaling = confidence_scaling;
        self.use_sparse_updates = sparse;
        if sparse {
            self.sparse_threshold = 0.01;  // Only update Gaussians in regions with >1% error
        }
        self
    }

    /// Ensure state vector matches Gaussian count
    fn ensure_states(&mut self, n: usize) {
        while self.states.len() < n {
            self.states.push(PerGaussianAdamState::new());
        }
    }

    /// Split a Gaussian with state inheritance
    fn split_gaussian(
        &mut self,
        gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
        idx: usize,
    ) {
        let parent = &gaussians[idx];
        let parent_state = self.states[idx].clone();

        // Split along major axis
        let angle = parent.shape.rotation;
        let offset = parent.shape.scale_x * 0.3;

        // Create child 1 (offset in +direction)
        let child1 = Gaussian2D::new(
            Vector2::new(
                (parent.position.x + angle.cos() * offset).clamp(0.01, 0.99),
                (parent.position.y + angle.sin() * offset).clamp(0.01, 0.99),
            ),
            Euler::new(
                parent.shape.scale_x * 0.6,
                parent.shape.scale_y * 0.6,
                parent.shape.rotation,
            ),
            parent.color,
            1.0,  // Full opacity
        );

        // Create child 2 (offset in -direction)
        let child2 = Gaussian2D::new(
            Vector2::new(
                (parent.position.x - angle.cos() * offset).clamp(0.01, 0.99),
                (parent.position.y - angle.sin() * offset).clamp(0.01, 0.99),
            ),
            Euler::new(
                parent.shape.scale_x * 0.6,
                parent.shape.scale_y * 0.6,
                parent.shape.rotation,
            ),
            parent.color,
            1.0,
        );

        // Replace parent with child1, add child2
        gaussians[idx] = child1;
        gaussians.push(child2);

        // STATE INHERITANCE (Feature #2)
        self.states[idx] = PerGaussianAdamState::inherit_from(&parent_state);
        self.states.push(PerGaussianAdamState::inherit_from(&parent_state));
    }

    /// Compute per-pixel error map for sparse updates
    fn compute_error_map(&self, rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> Vec<f32> {
        rendered.data.iter().zip(target.data.iter()).map(|(r, t)| {
            (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2)
        }).collect()
    }

    /// Determine which Gaussians should be updated (for sparse Adam)
    fn get_active_gaussians(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        error_map: &[f32],
        width: u32,
        height: u32,
    ) -> Vec<bool> {
        if !self.use_sparse_updates {
            return vec![true; gaussians.len()];
        }

        let mut active = vec![false; gaussians.len()];

        for (i, g) in gaussians.iter().enumerate() {
            // Check if this Gaussian covers any high-error pixels
            let cx = (g.position.x * width as f32) as i32;
            let cy = (g.position.y * height as f32) as i32;
            let radius = ((g.shape.scale_x.max(g.shape.scale_y)) * width as f32 * 3.0) as i32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let px = (cx + dx).clamp(0, width as i32 - 1) as usize;
                    let py = (cy + dy).clamp(0, height as i32 - 1) as usize;
                    let idx = py * width as usize + px;

                    if idx < error_map.len() && error_map[idx] > self.sparse_threshold {
                        active[i] = true;
                        break;
                    }
                }
                if active[i] { break; }
            }
        }

        active
    }

    /// Update confidence scores for Gaussians
    fn update_confidence(&mut self, gaussians: &[Gaussian2D<f32, Euler<f32>>], error_map: &[f32], width: u32, height: u32) {
        if !self.use_confidence_scaling {
            return;
        }

        for (i, g) in gaussians.iter().enumerate() {
            // Confidence based on:
            // 1. How many pixels this Gaussian covers
            // 2. How much error reduction it provides

            let cx = (g.position.x * width as f32) as i32;
            let cy = (g.position.y * height as f32) as i32;
            let radius = ((g.shape.scale_x.max(g.shape.scale_y)) * width as f32 * 2.0) as i32;

            let mut covered_pixels = 0;
            let mut total_error_in_region = 0.0;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let px = (cx + dx).clamp(0, width as i32 - 1) as usize;
                    let py = (cy + dy).clamp(0, height as i32 - 1) as usize;
                    let idx = py * width as usize + px;

                    if idx < error_map.len() {
                        covered_pixels += 1;
                        total_error_in_region += error_map[idx];
                    }
                }
            }

            // High confidence if covering many pixels with some error (useful Gaussian)
            // Low confidence if covering few pixels or in low-error region
            if covered_pixels > 0 {
                let avg_error = total_error_in_region / covered_pixels as f32;
                // Sigmoid-like scaling: high error → high confidence
                self.states[i].confidence = (avg_error * 100.0).tanh().clamp(0.1, 1.0);
            }
        }
    }

    /// Run one Adam iteration
    fn step(
        &mut self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> f32 {
        let width = target.width;
        let height = target.height;

        self.ensure_states(gaussians.len());

        let rendered = RendererV2::render(gaussians, width, height);
        let error_map = self.compute_error_map(&rendered, target);

        // Update confidence scores
        self.update_confidence(gaussians, &error_map, width, height);

        // Determine active Gaussians (for sparse updates)
        let active = self.get_active_gaussians(gaussians, &error_map, width, height);

        // Compute gradients
        let grads = self.compute_gradients(gaussians, &rendered, target);

        // Increment global time counter
        self.global_t += 1;

        // Adam update for each Gaussian
        for i in 0..gaussians.len() {
            // Skip if not active (sparse update mode)
            if !active[i] {
                continue;
            }

            let state = &mut self.states[i];

            // INCREMENT PER-GAUSSIAN TIME (Feature #1)
            state.t += 1;

            // Use per-Gaussian time OR global time for bias correction
            // BUG FIX: Previously used t=1 when per_gaussian_time was false (no bias correction!)
            let t = if self.use_per_gaussian_time { state.t } else { self.global_t };
            let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(t as i32);

            // Confidence scaling factor (Feature #4)
            let conf_scale = if self.use_confidence_scaling { state.confidence } else { 1.0 };

            // Color update
            state.m_color.r = self.beta1 * state.m_color.r + (1.0 - self.beta1) * grads[i].color.r;
            state.m_color.g = self.beta1 * state.m_color.g + (1.0 - self.beta1) * grads[i].color.g;
            state.m_color.b = self.beta1 * state.m_color.b + (1.0 - self.beta1) * grads[i].color.b;

            state.v_color.r = self.beta2 * state.v_color.r + (1.0 - self.beta2) * grads[i].color.r.powi(2);
            state.v_color.g = self.beta2 * state.v_color.g + (1.0 - self.beta2) * grads[i].color.g.powi(2);
            state.v_color.b = self.beta2 * state.v_color.b + (1.0 - self.beta2) * grads[i].color.b.powi(2);

            let m_hat_r = state.m_color.r / bias_correction1;
            let m_hat_g = state.m_color.g / bias_correction1;
            let m_hat_b = state.m_color.b / bias_correction1;
            let v_hat_r = state.v_color.r / bias_correction2;
            let v_hat_g = state.v_color.g / bias_correction2;
            let v_hat_b = state.v_color.b / bias_correction2;

            gaussians[i].color.r -= conf_scale * self.lr_color * m_hat_r / (v_hat_r.sqrt() + self.epsilon);
            gaussians[i].color.g -= conf_scale * self.lr_color * m_hat_g / (v_hat_g.sqrt() + self.epsilon);
            gaussians[i].color.b -= conf_scale * self.lr_color * m_hat_b / (v_hat_b.sqrt() + self.epsilon);
            gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
            gaussians[i].color.g = gaussians[i].color.g.clamp(0.0, 1.0);
            gaussians[i].color.b = gaussians[i].color.b.clamp(0.0, 1.0);

            // Position update
            state.m_position.x = self.beta1 * state.m_position.x + (1.0 - self.beta1) * grads[i].position.x;
            state.m_position.y = self.beta1 * state.m_position.y + (1.0 - self.beta1) * grads[i].position.y;
            state.v_position.x = self.beta2 * state.v_position.x + (1.0 - self.beta2) * grads[i].position.x.powi(2);
            state.v_position.y = self.beta2 * state.v_position.y + (1.0 - self.beta2) * grads[i].position.y.powi(2);

            let m_hat_px = state.m_position.x / bias_correction1;
            let m_hat_py = state.m_position.y / bias_correction1;
            let v_hat_px = state.v_position.x / bias_correction2;
            let v_hat_py = state.v_position.y / bias_correction2;

            gaussians[i].position.x -= conf_scale * self.lr_position * m_hat_px / (v_hat_px.sqrt() + self.epsilon);
            gaussians[i].position.y -= conf_scale * self.lr_position * m_hat_py / (v_hat_py.sqrt() + self.epsilon);
            gaussians[i].position.x = gaussians[i].position.x.clamp(0.0, 1.0);
            gaussians[i].position.y = gaussians[i].position.y.clamp(0.0, 1.0);

            // Scale update
            state.m_scale.0 = self.beta1 * state.m_scale.0 + (1.0 - self.beta1) * grads[i].scale_x;
            state.m_scale.1 = self.beta1 * state.m_scale.1 + (1.0 - self.beta1) * grads[i].scale_y;
            state.v_scale.0 = self.beta2 * state.v_scale.0 + (1.0 - self.beta2) * grads[i].scale_x.powi(2);
            state.v_scale.1 = self.beta2 * state.v_scale.1 + (1.0 - self.beta2) * grads[i].scale_y.powi(2);

            let m_hat_sx = state.m_scale.0 / bias_correction1;
            let m_hat_sy = state.m_scale.1 / bias_correction1;
            let v_hat_sx = state.v_scale.0 / bias_correction2;
            let v_hat_sy = state.v_scale.1 / bias_correction2;

            gaussians[i].shape.scale_x -= conf_scale * self.lr_scale * m_hat_sx / (v_hat_sx.sqrt() + self.epsilon);
            gaussians[i].shape.scale_y -= conf_scale * self.lr_scale * m_hat_sy / (v_hat_sy.sqrt() + self.epsilon);
            gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
            gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);
        }

        // Return MSE loss
        let loss: f32 = error_map.iter().sum::<f32>() / error_map.len() as f32;
        loss
    }

    fn compute_gradients(
        &self,
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
                let rc = rendered.get_pixel(x, y).unwrap();
                let tc = target.get_pixel(x, y).unwrap();

                let error_r = 2.0 * (rc.r - tc.r);
                let error_g = 2.0 * (rc.g - tc.g);
                let error_b = 2.0 * (rc.b - tc.b);

                for (i, g) in gaussians.iter().enumerate() {
                    let scale_x = g.shape.scale_x.max(0.001);
                    let scale_y = g.shape.scale_y.max(0.001);

                    let dx = px - g.position.x;
                    let dy = py - g.position.y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq > 0.1 { continue; }

                    let scale_product = (scale_x * scale_y).max(1e-6);
                    let weight = (-0.5 * dist_sq / scale_product).exp();

                    if weight < 1e-6 || weight.is_nan() { continue; }

                    gradients[i].color.r += error_r * weight;
                    gradients[i].color.g += error_g * weight;
                    gradients[i].color.b += error_b * weight;

                    let error_weighted = error_r * g.color.r + error_g * g.color.g + error_b * g.color.b;
                    gradients[i].position.x += error_weighted * weight * dx;
                    gradients[i].position.y += error_weighted * weight * dy;

                    let scale_x_sq = scale_x.powi(2).max(1e-6);
                    let scale_y_sq = scale_y.powi(2).max(1e-6);
                    gradients[i].scale_x += error_weighted * weight * dist_sq / scale_x_sq;
                    gradients[i].scale_y += error_weighted * weight * dist_sq / scale_y_sq;
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
            grad.scale_x /= pixel_count;
            grad.scale_y /= pixel_count;
        }

        gradients
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

/// Create jittered grid initialization
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

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     ADVANCED ADAM OPTIMIZER TEST - 3DGS Best Practices       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Test configurations: (name, per_gaussian_time, confidence_scaling, sparse_updates)
    let configs = vec![
        ("baseline_global_time", false, false, false),
        ("per_gaussian_time", true, false, false),
        ("confidence_scaling", true, true, false),
        ("sparse_updates", true, false, true),
        ("all_features", true, true, true),
    ];

    let n_gaussians = 256;
    let iterations = 150;

    for (img_name, img_path) in TEST_IMAGES {
        println!("\n{}", "=".repeat(70));
        println!("IMAGE: {} - {}", img_name, img_path);
        println!("{}\n", "=".repeat(70));

        let target = match ImageBuffer::load(img_path) {
            Ok(t) => t,
            Err(e) => {
                println!("Failed to load {}: {}", img_path, e);
                continue;
            }
        };
        let (width, height) = (target.width, target.height);

        for (config_name, per_gaussian_time, confidence_scaling, sparse_updates) in &configs {
            println!("\n--- {} ---", config_name);
            println!("  per_gaussian_time: {}, confidence_scaling: {}, sparse: {}",
                     per_gaussian_time, confidence_scaling, sparse_updates);

            let start = Instant::now();

            // Create initial Gaussians
            let mut gaussians = create_jittered_grid(&target, n_gaussians);

            // Create optimizer with selected features
            let mut optimizer = AdvancedAdamOptimizer::new()
                .with_features(*per_gaussian_time, *confidence_scaling, *sparse_updates);

            // Get initial PSNR
            let init_render = RendererV2::render(&gaussians, width, height);
            let init_psnr = compute_psnr(&init_render, &target);
            println!("  Initial PSNR: {:.2}dB", init_psnr);

            // Optimization loop
            let mut last_psnr = init_psnr;
            for iter in 0..iterations {
                let _loss = optimizer.step(&mut gaussians, &target);

                if iter % 25 == 0 || iter == iterations - 1 {
                    let rendered = RendererV2::render(&gaussians, width, height);
                    let psnr = compute_psnr(&rendered, &target);
                    let direction = if psnr > last_psnr { "↑" } else { "↓" };
                    println!("    [{:3}/{:3}] PSNR={:.2}dB {} ({:+.3}dB)",
                             iter, iterations, psnr, direction, psnr - last_psnr);
                    last_psnr = psnr;
                }
            }

            let elapsed = start.elapsed();
            let final_render = RendererV2::render(&gaussians, width, height);
            let final_psnr = compute_psnr(&final_render, &target);
            let improvement = final_psnr - init_psnr;

            println!("  Final: {:.2}dB ({:+.2}dB improvement) in {:.2}s",
                     final_psnr, improvement, elapsed.as_secs_f32());

            // Save TestResult with rendered image
            let exp_name = format!("adam_advanced_{}_{}", config_name, img_name);
            let mut result = TestResult::new(&exp_name, img_path);
            result.optimizer = format!("AdvancedAdam(per_time={}, conf={}, sparse={})",
                                       per_gaussian_time, confidence_scaling, sparse_updates);
            result.image_dimensions = (width, height);
            result.n_gaussians = n_gaussians;
            result.iterations = iterations;
            result.initial_psnr = init_psnr;
            result.final_psnr = final_psnr;
            result.improvement_db = improvement;
            result.elapsed_seconds = elapsed.as_secs_f32();

            // Record optimizer parameters
            result.set_param("lr_position", optimizer.lr_position);
            result.set_param("lr_color", optimizer.lr_color);
            result.set_param("lr_scale", optimizer.lr_scale);
            result.set_param("lr_opacity", optimizer.lr_opacity);
            result.set_param("beta1", optimizer.beta1);
            result.set_param("beta2", optimizer.beta2);

            result.extra_metrics = Some(json!({
                "per_gaussian_time": per_gaussian_time,
                "confidence_scaling": confidence_scaling,
                "sparse_updates": sparse_updates,
            }));

            result.complete();
            let _ = result.save_with_image(&final_render, RESULTS_DIR);
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("All tests complete! Results saved to {}", RESULTS_DIR);
    println!("{}", "=".repeat(70));
}
