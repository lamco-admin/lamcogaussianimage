//! Baseline 3: 3D Gaussian Splatting (2023) Approach
//!
//! Reference: "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (2023)
//! Adapted to 2D
//!
//! Key characteristics:
//! - Random/sparse initialization
//! - Dynamic N (grows and shrinks during training)
//! - Loss: L1 + D-SSIM (different from GaussianImage!)
//! - Optimizer: Adam
//! - Iterations: 7,000-30,000 typical
//! - Densification: Split/clone/prune every 100 iterations
//! - Gradient accumulation tracking
//!
//! Expected: Adaptive structure, fewer Gaussians for same quality

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_encoder_v2::{renderer_v2::RendererV2, loss_functions, correct_gradients};
use std::time::Instant;

struct AdaptiveGSOptimizer {
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
    // Gradient accumulation for densification
    grad_accum: Vec<f32>,
}

impl AdaptiveGSOptimizer {
    fn new(learning_rate: f32, n: usize) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m_color: vec![Color4::new(0.0, 0.0, 0.0, 0.0); n],
            v_color: vec![Color4::new(0.0, 0.0, 0.0, 0.0); n],
            m_position: vec![Vector2::zero(); n],
            v_position: vec![Vector2::zero(); n],
            m_scale: vec![(0.0, 0.0); n],
            v_scale: vec![(0.0, 0.0); n],
            grad_accum: vec![0.0; n],
        }
    }

    fn resize(&mut self, n: usize) {
        self.m_color.resize(n, Color4::new(0.0, 0.0, 0.0, 0.0));
        self.v_color.resize(n, Color4::new(0.0, 0.0, 0.0, 0.0));
        self.m_position.resize(n, Vector2::zero());
        self.v_position.resize(n, Vector2::zero());
        self.m_scale.resize(n, (0.0, 0.0));
        self.v_scale.resize(n, (0.0, 0.0));
        self.grad_accum.resize(n, 0.0);
    }

    fn optimize(
        &mut self,
        gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
        target: &ImageBuffer<f32>,
        max_iterations: usize,
    ) -> Vec<(usize, f32, f32, f32, usize)> {
        let mut history = Vec::new();

        println!("\n🚀 Starting 3D-GS adaptive optimization");
        println!("   Initial N = {}", gaussians.len());
        println!("   Base LR = {}", self.learning_rate);
        println!("   Max iterations = {}", max_iterations);
        println!("   Densification every 100 iterations");

        for t in 1..=max_iterations {
            let n = gaussians.len();

            // Render
            let rendered = RendererV2::render(gaussians, target.width, target.height);

            // Compute loss components (3D-GS uses L1 + D-SSIM)
            let l1 = loss_functions::compute_l1_loss(&rendered, target);
            let dssim = loss_functions::compute_dssim_loss(&rendered, target);
            let loss = l1 + dssim;

            // Compute SSIM for logging
            let ssim = 1.0 - dssim;

            // Log
            let should_log = t <= 10 || t.is_power_of_two() || t % 10 == 0 && t <= 100 || t % 100 == 0;
            if should_log {
                println!("   Iter {:4}: loss={:.6} (L1={:.6}, D-SSIM={:.6}, SSIM={:.4}) N={}",
                    t, loss, l1, dssim, ssim, n);
            }

            history.push((t, loss, l1, dssim, n));

            // Compute gradients (using corrected implementation)
            let grads = correct_gradients::compute_gradients_correct(gaussians, &rendered, target);

            // Update gradient accumulation
            for (i, grad) in grads.iter().enumerate() {
                let grad_mag = (grad.position.x.powi(2) + grad.position.y.powi(2)).sqrt();
                self.grad_accum[i] += grad_mag;
            }

            // Adam updates
            let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(t as i32);

            for i in 0..n {
                // Color
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

                // Position
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

                // Scale
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

            // Densification every 100 iterations
            if t % 100 == 0 && t < max_iterations - 100 {
                let n_before = gaussians.len();
                self.densify(gaussians);
                let n_after = gaussians.len();

                println!("   [Densification] N: {} → {} (added {}, removed {})",
                    n_before, n_after,
                    if n_after > n_before { n_after - n_before } else { 0 },
                    if n_before > n_after { n_before - n_after } else { 0 });

                // Reset gradient accumulation and momentum state for stability
                self.grad_accum.fill(0.0);
                // Reset momentum for new Gaussians to prevent instability
                for i in n_before..gaussians.len() {
                    self.m_color[i] = Color4::new(0.0, 0.0, 0.0, 0.0);
                    self.v_color[i] = Color4::new(0.0, 0.0, 0.0, 0.0);
                    self.m_position[i] = Vector2::zero();
                    self.v_position[i] = Vector2::zero();
                    self.m_scale[i] = (0.0, 0.0);
                    self.v_scale[i] = (0.0, 0.0);
                }
            }

            if loss < 1e-6 {
                println!("   ✅ Converged at iteration {}", t);
                break;
            }
        }

        history
    }

    fn densify(&mut self, gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>) {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(42);

        // Thresholds from 3D-GS paper
        let grad_threshold = 0.0002;
        let scale_large = 0.01;
        let opacity_threshold = 0.005;
        let scale_max = 0.5;

        let mut to_split = Vec::new();
        let mut to_clone = Vec::new();
        let mut to_prune = Vec::new();

        // Identify Gaussians for densification
        for i in 0..gaussians.len() {
            let grad_accum = self.grad_accum[i];
            let scale_avg = (gaussians[i].shape.scale_x + gaussians[i].shape.scale_y) / 2.0;
            let opacity = gaussians[i].opacity;

            // Split: high gradient + large scale
            if grad_accum > grad_threshold && scale_avg > scale_large {
                to_split.push(i);
            }
            // Clone: high gradient + small scale
            else if grad_accum > grad_threshold && scale_avg <= scale_large {
                to_clone.push(i);
            }

            // Prune: low opacity or too large
            if opacity < opacity_threshold || scale_avg > scale_max {
                to_prune.push(i);
            }
        }

        // Apply operations (split, clone, prune)
        let mut new_gaussians = Vec::new();

        // Split
        for &i in &to_split {
            let g = &gaussians[i];

            // Create 2 smaller Gaussians
            for _ in 0..2 {
                let offset = Vector2::new(
                    rng.gen_range(-0.01..0.01),
                    rng.gen_range(-0.01..0.01),
                );

                let new_pos = Vector2::new(
                    (g.position.x + offset.x).clamp(0.0, 1.0),
                    (g.position.y + offset.y).clamp(0.0, 1.0),
                );

                let new_scale = Euler::new(
                    g.shape.scale_x / 1.6,
                    g.shape.scale_y / 1.6,
                    g.shape.rotation,
                );

                new_gaussians.push(Gaussian2D::new(new_pos, new_scale, g.color, g.opacity));
            }
        }

        // Clone
        for &i in &to_clone {
            let g = &gaussians[i];

            let offset = Vector2::new(
                rng.gen_range(-0.005..0.005),
                rng.gen_range(-0.005..0.005),
            );

            let new_pos = Vector2::new(
                (g.position.x + offset.x).clamp(0.0, 1.0),
                (g.position.y + offset.y).clamp(0.0, 1.0),
            );

            new_gaussians.push(Gaussian2D::new(new_pos, g.shape, g.color, g.opacity));
        }

        // Keep non-pruned, non-split Gaussians
        let to_remove: std::collections::HashSet<usize> = to_prune.iter().chain(to_split.iter()).copied().collect();

        for (i, g) in gaussians.iter().enumerate() {
            if !to_remove.contains(&i) {
                new_gaussians.push(g.clone());
            }
        }

        *gaussians = new_gaussians;
        self.resize(gaussians.len());
    }
}

fn initialize_sparse(n: usize, seed: u64) -> Vec<Gaussian2D<f32, Euler<f32>>> {
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
        let scale = Euler::isotropic(rng.gen_range(0.02..0.08));

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
    println!("BASELINE 3: 3D Gaussian Splatting (2023) Approach");
    println!("{}", sep);

    let image_size = 64;
    let initial_n = 20; // Start sparse
    let iterations = 1000; // Paper uses 7k-30k
    let learning_rate = 0.001;

    println!("\n📊 Test Parameters:");
    println!("   Image size: {}×{}", image_size, image_size);
    println!("   Initial N: {} (sparse, will grow)", initial_n);
    println!("   Iterations: {}", iterations);
    println!("   Learning rate: {}", learning_rate);

    let target = create_test_image(image_size);
    println!("\n✅ Created test image");

    let mut gaussians = initialize_sparse(initial_n, 42);
    println!("✅ Initialized {} Gaussians (sparse random)", gaussians.len());

    let initial_render = RendererV2::render(&gaussians, target.width, target.height);
    let initial_loss = loss_functions::compute_3dgs_loss(&initial_render, &target);
    println!("\n📈 Initial loss: {:.6}", initial_loss);

    let start = Instant::now();
    let mut optimizer = AdaptiveGSOptimizer::new(learning_rate, gaussians.len());
    let history = optimizer.optimize(&mut gaussians, &target, iterations);
    let elapsed = start.elapsed();

    let final_render = RendererV2::render(&gaussians, target.width, target.height);
    let final_loss = loss_functions::compute_3dgs_loss(&final_render, &target);

    println!("\n{}", sep);
    println!("RESULTS");
    println!("{}", sep);
    println!("Initial N:    {}", initial_n);
    println!("Final N:      {}", gaussians.len());
    println!("N change:     {:+}", gaussians.len() as i32 - initial_n as i32);
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
    let checkpoints = [1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
    for &iter in &checkpoints {
        if iter <= iterations {
            if let Some(entry) = history.iter().find(|(i, _, _, _, _)| *i == iter) {
                let (_, loss, l1, dssim, n) = entry;
                let ssim = 1.0 - dssim;
                println!("   Iter {:3}: loss={:.6} (L1={:.6}, SSIM={:.4}) N={}",
                    iter, loss, l1, ssim, n);
            }
        }
    }

    println!("\n{}", sep);
}
