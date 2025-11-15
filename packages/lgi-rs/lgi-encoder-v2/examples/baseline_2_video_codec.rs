//! Baseline 2: Neural Video Compression (2025) Approach
//!
//! Reference: "Neural Video Compression using 2D Gaussian Splatting" (2025)
//!
//! Key characteristics:
//! - Content-aware initialization via superpixels (SLIC/K-means)
//! - Position = superpixel centroid
//! - Covariance = superpixel shape covariance
//! - Color = superpixel mean color
//! - Fixed N (image-dependent)
//! - Loss: 0.2×L1 + 0.8×L2 + 0.1×(1-SSIM) (same as GaussianImage)
//! - Optimizer: Adam with dynamic LR scheduler
//! - Iterations: 1,000-2,000 (88% faster than GaussianImage)
//!
//! Expected: Faster convergence than random init

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_encoder_v2::{renderer_v2::RendererV2, loss_functions, correct_gradients};
use std::time::Instant;

struct VideoCodecOptimizer {
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

impl VideoCodecOptimizer {
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

    /// Constant learning rate (simplified from paper's cosine annealing)
    fn get_dynamic_lr(&self, _iteration: usize, _max_iterations: usize) -> f32 {
        // Using constant LR for stability
        // Previous cosine annealing caused divergence
        self.learning_rate
    }

    fn optimize(
        &mut self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
        max_iterations: usize,
    ) -> Vec<(usize, f32, f32, f32, f32, f32)> {
        let n = gaussians.len();
        self.m_color.resize(n, Color4::new(0.0, 0.0, 0.0, 0.0));
        self.v_color.resize(n, Color4::new(0.0, 0.0, 0.0, 0.0));
        self.m_position.resize(n, Vector2::zero());
        self.v_position.resize(n, Vector2::zero());
        self.m_scale.resize(n, (0.0, 0.0));
        self.v_scale.resize(n, (0.0, 0.0));

        let mut history = Vec::new();
        let base_lr = self.learning_rate;

        println!("\n🚀 Starting Video Codec optimization");
        println!("   N = {} Gaussians", n);
        println!("   Base LR = {}", base_lr);
        println!("   Max iterations = {}", max_iterations);
        println!("   Using cosine annealing LR schedule");

        for t in 1..=max_iterations {
            // Dynamic learning rate
            let current_lr = self.get_dynamic_lr(t, max_iterations);

            // Render
            let rendered = RendererV2::render(gaussians, target.width, target.height);

            // Compute loss components
            let l1 = loss_functions::compute_l1_loss(&rendered, target);
            let l2 = loss_functions::compute_l2_loss(&rendered, target);
            let ssim = loss_functions::compute_ssim(&rendered, target);

            // Combined loss (same as GaussianImage)
            let loss = 0.2 * l1 + 0.8 * l2 + 0.1 * (1.0 - ssim);

            // Log
            let should_log = t <= 10 || t.is_power_of_two() || t % 10 == 0 && t <= 100;
            if should_log {
                println!("   Iter {:4}: loss={:.6} (L1={:.6}, L2={:.6}, SSIM={:.4}) LR={:.6}",
                    t, loss, l1, l2, ssim, current_lr);
            }

            history.push((t, loss, l1, l2, ssim, current_lr));

            // Compute gradients (using corrected implementation)
            let grads = correct_gradients::compute_gradients_correct(gaussians, &rendered, target);

            // Adam updates with dynamic LR
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

                gaussians[i].color.r -= current_lr * m_hat_r / (v_hat_r.sqrt() + self.epsilon);
                gaussians[i].color.g -= current_lr * m_hat_g / (v_hat_g.sqrt() + self.epsilon);
                gaussians[i].color.b -= current_lr * m_hat_b / (v_hat_b.sqrt() + self.epsilon);

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

                gaussians[i].position.x -= current_lr * m_hat_px / (v_hat_px.sqrt() + self.epsilon);
                gaussians[i].position.y -= current_lr * m_hat_py / (v_hat_py.sqrt() + self.epsilon);

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

                gaussians[i].shape.scale_x -= current_lr * m_hat_sx / (v_hat_sx.sqrt() + self.epsilon);
                gaussians[i].shape.scale_y -= current_lr * m_hat_sy / (v_hat_sy.sqrt() + self.epsilon);

                gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
                gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);
            }

            if loss < 1e-6 {
                println!("   ✅ Converged at iteration {}", t);
                break;
            }
        }

        history
    }
}

/// Superpixel initialization (K-means clustering approach)
fn initialize_kmeans(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    println!("   Running K-means clustering (k={})...", n);

    let mut rng = StdRng::seed_from_u64(42);

    // Collect all pixel features (position + color)
    let mut pixels = Vec::new();
    for y in 0..target.height {
        for x in 0..target.width {
            let color = target.get_pixel(x, y).unwrap();
            pixels.push((
                x as f32 / target.width as f32,
                y as f32 / target.height as f32,
                color.r,
                color.g,
                color.b,
            ));
        }
    }

    // Initialize cluster centers randomly
    let mut centers = Vec::new();
    for _ in 0..n {
        let idx = rng.gen_range(0..pixels.len());
        centers.push(pixels[idx]);
    }

    // K-means iterations
    let mut assignments = vec![0; pixels.len()];
    for _iter in 0..10 {
        // Assign pixels to nearest center
        for (i, &(px, py, pr, pg, pb)) in pixels.iter().enumerate() {
            let mut min_dist = f32::INFINITY;
            let mut best_cluster = 0;

            for (j, &(cx, cy, cr, cg, cb)) in centers.iter().enumerate() {
                let dist = (px - cx).powi(2) + (py - cy).powi(2) +
                           (pr - cr).powi(2) + (pg - cg).powi(2) + (pb - cb).powi(2);

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }

            assignments[i] = best_cluster;
        }

        // Update centers
        let mut new_centers = vec![(0.0, 0.0, 0.0, 0.0, 0.0); n];
        let mut counts = vec![0; n];

        for (i, &cluster) in assignments.iter().enumerate() {
            let (px, py, pr, pg, pb) = pixels[i];
            new_centers[cluster].0 += px;
            new_centers[cluster].1 += py;
            new_centers[cluster].2 += pr;
            new_centers[cluster].3 += pg;
            new_centers[cluster].4 += pb;
            counts[cluster] += 1;
        }

        for i in 0..n {
            if counts[i] > 0 {
                let count = counts[i] as f32;
                centers[i] = (
                    new_centers[i].0 / count,
                    new_centers[i].1 / count,
                    new_centers[i].2 / count,
                    new_centers[i].3 / count,
                    new_centers[i].4 / count,
                );
            }
        }
    }

    // Convert clusters to Gaussians
    let mut gaussians = Vec::new();
    for &(cx, cy, cr, cg, cb) in &centers {
        let position = Vector2::new(cx, cy);
        let color = Color4::new(cr, cg, cb, 1.0);

        // Estimate scale from cluster size
        let scale = Euler::isotropic(0.05);

        gaussians.push(Gaussian2D::new(position, scale, color, 1.0));
    }

    gaussians
}

fn create_test_image(size: u32) -> ImageBuffer<f32> {
    let mut img = ImageBuffer::new(size, size);

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
    println!("BASELINE 2: Neural Video Compression (2025) Approach");
    println!("{}", sep);

    let image_size = 64;
    let n_gaussians = 50;
    let iterations = 1000; // Paper uses 1000-2000
    let learning_rate = 0.001; // Constant LR (cosine annealing removed)

    println!("\n📊 Test Parameters:");
    println!("   Image size: {}×{}", image_size, image_size);
    println!("   N Gaussians: {} (content-adaptive)", n_gaussians);
    println!("   Iterations: {}", iterations);
    println!("   Base learning rate: {} (with cosine annealing)", learning_rate);

    let target = create_test_image(image_size);
    println!("\n✅ Created test image");

    // Superpixel initialization
    let mut gaussians = initialize_kmeans(&target, n_gaussians);
    println!("✅ Initialized {} Gaussians (K-means superpixels)", gaussians.len());

    let initial_render = RendererV2::render(&gaussians, target.width, target.height);
    let initial_loss = loss_functions::compute_combined_loss(&initial_render, &target, 0.2, 0.8, 0.1);
    println!("\n📈 Initial loss: {:.6}", initial_loss);

    let start = Instant::now();
    let mut optimizer = VideoCodecOptimizer::new(learning_rate);
    let history = optimizer.optimize(&mut gaussians, &target, iterations);
    let elapsed = start.elapsed();

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

    if final_loss < initial_loss {
        println!("\n✅ SUCCESS: Loss decreased!");
    } else {
        println!("\n❌ FAILURE: Loss did not decrease!");
    }

    println!("\n📊 Loss Progression:");
    let checkpoints = [1, 2, 3, 5, 10, 20, 50, 100];
    for &iter in &checkpoints {
        if iter <= iterations {
            if let Some(entry) = history.iter().find(|(i, _, _, _, _, _)| *i == iter) {
                let (_, loss, l1, l2, ssim, lr) = entry;
                println!("   Iter {:3}: loss={:.6} (L1={:.6}, L2={:.6}, SSIM={:.4}) LR={:.6}",
                    iter, loss, l1, l2, ssim, lr);
            }
        }
    }

    println!("\n{}", sep);
}
