//! Baseline 1: GaussianImage (ECCV 2024) Approach
//!
//! Reference: "Representing Gaussian Splatting for 2D Image Compression" (ECCV 2024)
//!
//! Key characteristics:
//! - Random initialization (positions, colors, scales)
//! - Fixed N (no densification)
//! - Loss: 0.2×L1 + 0.8×L2 + 0.1×(1-SSIM)
//! - Optimizer: Adan (we use Adam as close approximation)
//! - Learning rate: 1e-3, halved every 20k steps
//! - Iterations: 50,000 (paper) → we test with 100-1000
//!
//! Expected: Slow convergence but proven to work

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_encoder_v2::{renderer_v2::RendererV2, loss_functions, correct_gradients};
use std::time::Instant;

struct GaussianImageOptimizer {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    // Adam state
    m_color: Vec<Color4<f32>>,
    v_color: Vec<Color4<f32>>,
    m_position: Vec<Vector2<f32>>,
    v_position: Vec<Vector2<f32>>,
    m_scale: Vec<(f32, f32)>,
    v_scale: Vec<(f32, f32)>,
}

impl GaussianImageOptimizer {
    fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m_color: Vec::new(),
            v_color: Vec::new(),
            m_position: Vec::new(),
            v_position: Vec::new(),
            m_scale: Vec::new(),
            v_scale: Vec::new(),
        }
    }

    /// Run optimization for specified iterations
    fn optimize(
        &mut self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
        max_iterations: usize,
        lr_schedule: bool,
    ) -> Vec<(usize, f32, f32, f32, f32)> {
        let n = gaussians.len();
        self.m_color.resize(n, Color4::new(0.0, 0.0, 0.0, 0.0));
        self.v_color.resize(n, Color4::new(0.0, 0.0, 0.0, 0.0));
        self.m_position.resize(n, Vector2::zero());
        self.v_position.resize(n, Vector2::zero());
        self.m_scale.resize(n, (0.0, 0.0));
        self.v_scale.resize(n, (0.0, 0.0));

        let mut history = Vec::new();
        let initial_lr = self.learning_rate;

        println!("\n🚀 Starting GaussianImage optimization");
        println!("   N = {} Gaussians", n);
        println!("   Initial LR = {}", initial_lr);
        println!("   Max iterations = {}", max_iterations);
        println!("   LR schedule = {}", lr_schedule);

        for t in 1..=max_iterations {
            // Learning rate schedule (halve every 20k iterations)
            if lr_schedule && t % 20000 == 0 {
                self.learning_rate *= 0.5;
                println!("   [Iter {}] LR → {}", t, self.learning_rate);
            }

            // Render
            let rendered = RendererV2::render(gaussians, target.width, target.height);

            // Compute loss components
            let l1 = loss_functions::compute_l1_loss(&rendered, target);
            let l2 = loss_functions::compute_l2_loss(&rendered, target);
            let ssim = loss_functions::compute_ssim(&rendered, target);

            // Combined loss (GaussianImage paper formula)
            let loss = 0.2 * l1 + 0.8 * l2 + 0.1 * (1.0 - ssim);

            // Log every power of 2 or every 10 iterations early on
            let should_log = t <= 10 || t.is_power_of_two() || t % 10 == 0 && t <= 100;

            if should_log {
                println!("   Iter {:4}: loss={:.6} (L1={:.6}, L2={:.6}, SSIM={:.4})",
                    t, loss, l1, l2, ssim);
            }

            // Store history
            history.push((t, loss, l1, l2, ssim));

            // Compute gradients (using corrected implementation)
            let grads = correct_gradients::compute_gradients_correct(gaussians, &rendered, target);

            // Adam updates
            let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(t as i32);

            for i in 0..n {
                // Color update
                self.m_color[i].r = self.beta1 * self.m_color[i].r + (1.0 - self.beta1) * grads[i].color.r;
                self.m_color[i].g = self.beta1 * self.m_color[i].g + (1.0 - self.beta1) * grads[i].color.g;
                self.m_color[i].b = self.beta1 * self.m_color[i].b + (1.0 - self.beta1) * grads[i].color.b;

                self.v_color[i].r = self.beta2 * self.v_color[i].r + (1.0 - self.beta2) * grads[i].color.r.powi(2);
                self.v_color[i].g = self.beta2 * self.v_color[i].g + (1.0 - self.beta2) * grads[i].color.g.powi(2);
                self.v_color[i].b = self.beta2 * self.v_color[i].b + (1.0 - self.beta2) * grads[i].color.b.powi(2);

                let m_hat_r = self.m_color[i].r / bias_correction1;
                let m_hat_g = self.m_color[i].g / bias_correction1;
                let m_hat_b = self.m_color[i].b / bias_correction1;

                let v_hat_r = self.v_color[i].r / bias_correction2;
                let v_hat_g = self.v_color[i].g / bias_correction2;
                let v_hat_b = self.v_color[i].b / bias_correction2;

                gaussians[i].color.r -= self.learning_rate * m_hat_r / (v_hat_r.sqrt() + self.epsilon);
                gaussians[i].color.g -= self.learning_rate * m_hat_g / (v_hat_g.sqrt() + self.epsilon);
                gaussians[i].color.b -= self.learning_rate * m_hat_b / (v_hat_b.sqrt() + self.epsilon);

                gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
                gaussians[i].color.g = gaussians[i].color.g.clamp(0.0, 1.0);
                gaussians[i].color.b = gaussians[i].color.b.clamp(0.0, 1.0);

                // Position update
                self.m_position[i].x = self.beta1 * self.m_position[i].x + (1.0 - self.beta1) * grads[i].position.x;
                self.m_position[i].y = self.beta1 * self.m_position[i].y + (1.0 - self.beta1) * grads[i].position.y;

                self.v_position[i].x = self.beta2 * self.v_position[i].x + (1.0 - self.beta2) * grads[i].position.x.powi(2);
                self.v_position[i].y = self.beta2 * self.v_position[i].y + (1.0 - self.beta2) * grads[i].position.y.powi(2);

                let m_hat_px = self.m_position[i].x / bias_correction1;
                let m_hat_py = self.m_position[i].y / bias_correction1;
                let v_hat_px = self.v_position[i].x / bias_correction2;
                let v_hat_py = self.v_position[i].y / bias_correction2;

                gaussians[i].position.x -= self.learning_rate * m_hat_px / (v_hat_px.sqrt() + self.epsilon);
                gaussians[i].position.y -= self.learning_rate * m_hat_py / (v_hat_py.sqrt() + self.epsilon);

                gaussians[i].position.x = gaussians[i].position.x.clamp(0.0, 1.0);
                gaussians[i].position.y = gaussians[i].position.y.clamp(0.0, 1.0);

                // Scale update
                self.m_scale[i].0 = self.beta1 * self.m_scale[i].0 + (1.0 - self.beta1) * grads[i].scale_x;
                self.m_scale[i].1 = self.beta1 * self.m_scale[i].1 + (1.0 - self.beta1) * grads[i].scale_y;

                self.v_scale[i].0 = self.beta2 * self.v_scale[i].0 + (1.0 - self.beta2) * grads[i].scale_x.powi(2);
                self.v_scale[i].1 = self.beta2 * self.v_scale[i].1 + (1.0 - self.beta2) * grads[i].scale_y.powi(2);

                let m_hat_sx = self.m_scale[i].0 / bias_correction1;
                let m_hat_sy = self.m_scale[i].1 / bias_correction1;
                let v_hat_sx = self.v_scale[i].0 / bias_correction2;
                let v_hat_sy = self.v_scale[i].1 / bias_correction2;

                gaussians[i].shape.scale_x -= self.learning_rate * m_hat_sx / (v_hat_sx.sqrt() + self.epsilon);
                gaussians[i].shape.scale_y -= self.learning_rate * m_hat_sy / (v_hat_sy.sqrt() + self.epsilon);

                gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
                gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);
            }

            // Early stopping if loss is very small
            if loss < 1e-6 {
                println!("   ✅ Converged at iteration {}", t);
                break;
            }
        }

        history
    }
}

/// Random initialization (GaussianImage approach)
fn initialize_random(n: usize, seed: u64) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut gaussians = Vec::new();

    for _ in 0..n {
        let position = Vector2::new(rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0));
        let color = Color4::new(
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            1.0,
        );
        let scale = Euler::isotropic(rng.gen_range(0.01..0.1));

        gaussians.push(Gaussian2D::new(position, scale, color, 1.0));
    }

    gaussians
}

fn create_test_image(size: u32) -> ImageBuffer<f32> {
    let mut img = ImageBuffer::new(size, size);

    // Create a simple gradient + checkerboard test pattern
    for y in 0..size {
        for x in 0..size {
            let gradient = x as f32 / size as f32;
            let checker = if (x / 8 + y / 8) % 2 == 0 { 0.2 } else { 0.0 };

            let value = gradient * 0.7 + checker;
            img.set_pixel(x, y, Color4::new(value, value, value, 1.0));
        }
    }

    img
}

fn main() {
    let sep = "=".repeat(80);
    println!("{}", sep);
    println!("BASELINE 1: GaussianImage (ECCV 2024) Approach");
    println!("{}", sep);

    // Test parameters
    let image_size = 64;
    let n_gaussians = 50;
    let iterations = 100; // Start with 100, paper uses 50k
    let learning_rate = 0.001; // Paper uses 1e-3
    let use_lr_schedule = false; // Don't use schedule for short runs

    println!("\n📊 Test Parameters:");
    println!("   Image size: {}×{}", image_size, image_size);
    println!("   N Gaussians: {}", n_gaussians);
    println!("   Iterations: {}", iterations);
    println!("   Learning rate: {}", learning_rate);

    // Create test image
    let target = create_test_image(image_size);
    println!("\n✅ Created test image");

    // Random initialization
    let mut gaussians = initialize_random(n_gaussians, 42);
    println!("✅ Initialized {} Gaussians (random)", gaussians.len());

    // Initial render
    let initial_render = RendererV2::render(&gaussians, target.width, target.height);
    let initial_loss = loss_functions::compute_combined_loss(&initial_render, &target, 0.2, 0.8, 0.1);
    println!("\n📈 Initial loss: {:.6}", initial_loss);

    // Optimize
    let start = Instant::now();
    let mut optimizer = GaussianImageOptimizer::new(learning_rate);
    let history = optimizer.optimize(&mut gaussians, &target, iterations, use_lr_schedule);
    let elapsed = start.elapsed();

    // Final render
    let final_render = RendererV2::render(&gaussians, target.width, target.height);
    let final_loss = loss_functions::compute_combined_loss(&final_render, &target, 0.2, 0.8, 0.1);

    println!("\n{}", sep);
    println!("RESULTS");
    println!("{}", sep);
    println!("Initial loss: {:.6}", initial_loss);
    println!("Final loss:   {:.6}", final_loss);
    println!("Improvement:  {:.2}%", (initial_loss - final_loss) / initial_loss * 100.0);
    println!("Time:         {:.2}s", elapsed.as_secs_f64());
    println!("Iter/sec:     {:.1}", iterations as f64 / elapsed.as_secs_f64());

    // Check if loss decreased
    if final_loss < initial_loss {
        println!("\n✅ SUCCESS: Loss decreased!");
    } else {
        println!("\n❌ FAILURE: Loss did not decrease!");
    }

    // Show loss progression
    println!("\n📊 Loss Progression:");
    let checkpoints = [1, 2, 3, 5, 10, 20, 50, 100];
    for &iter in &checkpoints {
        if iter <= iterations {
            if let Some(entry) = history.iter().find(|(i, _, _, _, _)| *i == iter) {
                let (_, loss, l1, l2, ssim) = entry;
                println!("   Iter {:3}: loss={:.6} (L1={:.6}, L2={:.6}, SSIM={:.4})",
                    iter, loss, l1, l2, ssim);
            }
        }
    }

    println!("\n{}", sep);
}
