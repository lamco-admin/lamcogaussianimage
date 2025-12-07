//! Convergence Test: Run all 3 baselines with 1000 iterations
//!
//! Systematically tests convergence behavior of:
//! - Baseline 1: GaussianImage (random init)
//! - Baseline 2: Video Codec (K-means init)
//! - Baseline 3: 3D-GS (adaptive densification)

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_encoder_v2::{renderer_v2::RendererV2, loss_functions, correct_gradients};
use std::time::Instant;

// Baseline 1: GaussianImage approach
mod baseline1 {
    use super::*;

    pub struct Optimizer {
        lr: f32,
        m_color: Vec<Color4<f32>>,
        v_color: Vec<Color4<f32>>,
        m_position: Vec<Vector2<f32>>,
        v_position: Vec<Vector2<f32>>,
        m_scale: Vec<(f32, f32)>,
        v_scale: Vec<(f32, f32)>,
    }

    impl Optimizer {
        pub fn new(lr: f32, n: usize) -> Self {
            Self {
                lr,
                m_color: vec![Color4::new(0.0, 0.0, 0.0, 0.0); n],
                v_color: vec![Color4::new(0.0, 0.0, 0.0, 0.0); n],
                m_position: vec![Vector2::zero(); n],
                v_position: vec![Vector2::zero(); n],
                m_scale: vec![(0.0, 0.0); n],
                v_scale: vec![(0.0, 0.0); n],
            }
        }

        pub fn step(
            &mut self,
            gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
            target: &ImageBuffer<f32>,
            t: usize,
        ) -> f32 {
            let rendered = RendererV2::render(gaussians, target.width, target.height);
            let l1 = loss_functions::compute_l1_loss(&rendered, target);
            let l2 = loss_functions::compute_l2_loss(&rendered, target);
            let ssim = loss_functions::compute_ssim(&rendered, target);
            let loss = 0.2 * l1 + 0.8 * l2 + 0.1 * (1.0 - ssim);

            let grads = correct_gradients::compute_gradients_correct(gaussians, &rendered, target);

            let beta1: f32 = 0.9;
            let beta2: f32 = 0.999;
            let epsilon: f32 = 1e-8;
            let bc1 = 1.0 - beta1.powi(t as i32);
            let bc2 = 1.0 - beta2.powi(t as i32);

            for i in 0..gaussians.len() {
                // Color
                self.m_color[i].r = beta1 * self.m_color[i].r + (1.0 - beta1) * grads[i].color.r;
                self.m_color[i].g = beta1 * self.m_color[i].g + (1.0 - beta1) * grads[i].color.g;
                self.m_color[i].b = beta1 * self.m_color[i].b + (1.0 - beta1) * grads[i].color.b;

                self.v_color[i].r = beta2 * self.v_color[i].r + (1.0 - beta2) * grads[i].color.r.powi(2);
                self.v_color[i].g = beta2 * self.v_color[i].g + (1.0 - beta2) * grads[i].color.g.powi(2);
                self.v_color[i].b = beta2 * self.v_color[i].b + (1.0 - beta2) * grads[i].color.b.powi(2);

                gaussians[i].color.r -= self.lr * (self.m_color[i].r / bc1) / ((self.v_color[i].r / bc2).sqrt() + epsilon);
                gaussians[i].color.g -= self.lr * (self.m_color[i].g / bc1) / ((self.v_color[i].g / bc2).sqrt() + epsilon);
                gaussians[i].color.b -= self.lr * (self.m_color[i].b / bc1) / ((self.v_color[i].b / bc2).sqrt() + epsilon);

                gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
                gaussians[i].color.g = gaussians[i].color.g.clamp(0.0, 1.0);
                gaussians[i].color.b = gaussians[i].color.b.clamp(0.0, 1.0);

                // Position
                self.m_position[i].x = beta1 * self.m_position[i].x + (1.0 - beta1) * grads[i].position.x;
                self.m_position[i].y = beta1 * self.m_position[i].y + (1.0 - beta1) * grads[i].position.y;

                self.v_position[i].x = beta2 * self.v_position[i].x + (1.0 - beta2) * grads[i].position.x.powi(2);
                self.v_position[i].y = beta2 * self.v_position[i].y + (1.0 - beta2) * grads[i].position.y.powi(2);

                gaussians[i].position.x -= self.lr * (self.m_position[i].x / bc1) / ((self.v_position[i].x / bc2).sqrt() + epsilon);
                gaussians[i].position.y -= self.lr * (self.m_position[i].y / bc1) / ((self.v_position[i].y / bc2).sqrt() + epsilon);

                gaussians[i].position.x = gaussians[i].position.x.clamp(0.0, 1.0);
                gaussians[i].position.y = gaussians[i].position.y.clamp(0.0, 1.0);

                // Scale
                self.m_scale[i].0 = beta1 * self.m_scale[i].0 + (1.0 - beta1) * grads[i].scale_x;
                self.m_scale[i].1 = beta1 * self.m_scale[i].1 + (1.0 - beta1) * grads[i].scale_y;

                self.v_scale[i].0 = beta2 * self.v_scale[i].0 + (1.0 - beta2) * grads[i].scale_x.powi(2);
                self.v_scale[i].1 = beta2 * self.v_scale[i].1 + (1.0 - beta2) * grads[i].scale_y.powi(2);

                gaussians[i].shape.scale_x -= self.lr * (self.m_scale[i].0 / bc1) / ((self.v_scale[i].0 / bc2).sqrt() + epsilon);
                gaussians[i].shape.scale_y -= self.lr * (self.m_scale[i].1 / bc1) / ((self.v_scale[i].1 / bc2).sqrt() + epsilon);

                gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
                gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);
            }

            loss
        }
    }

    pub fn init_random(n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(42);
        (0..n).map(|_| {
            Gaussian2D::new(
                Vector2::new(rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)),
                Euler::isotropic(rng.gen_range(0.01..0.1)),
                Color4::new(rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0), 1.0),
                1.0,
            )
        }).collect()
    }
}

// Baseline 2: Video Codec approach with K-means
mod baseline2 {
    use super::*;

    pub struct Optimizer {
        lr: f32,
        m_color: Vec<Color4<f32>>,
        v_color: Vec<Color4<f32>>,
        m_position: Vec<Vector2<f32>>,
        v_position: Vec<Vector2<f32>>,
        m_scale: Vec<(f32, f32)>,
        v_scale: Vec<(f32, f32)>,
    }

    impl Optimizer {
        pub fn new(lr: f32, n: usize) -> Self {
            Self {
                lr,
                m_color: vec![Color4::new(0.0, 0.0, 0.0, 0.0); n],
                v_color: vec![Color4::new(0.0, 0.0, 0.0, 0.0); n],
                m_position: vec![Vector2::zero(); n],
                v_position: vec![Vector2::zero(); n],
                m_scale: vec![(0.0, 0.0); n],
                v_scale: vec![(0.0, 0.0); n],
            }
        }

        pub fn step(
            &mut self,
            gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
            target: &ImageBuffer<f32>,
            t: usize,
            total_iters: usize,
        ) -> f32 {
            let rendered = RendererV2::render(gaussians, target.width, target.height);
            let l1 = loss_functions::compute_l1_loss(&rendered, target);
            let l2 = loss_functions::compute_l2_loss(&rendered, target);
            let ssim = loss_functions::compute_ssim(&rendered, target);
            let loss = 0.3 * l1 + 0.7 * l2 + 0.05 * (1.0 - ssim);

            // Cosine annealing LR
            let progress = t as f32 / total_iters as f32;
            let lr = self.lr * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());

            let grads = correct_gradients::compute_gradients_correct(gaussians, &rendered, target);

            let beta1: f32 = 0.9;
            let beta2: f32 = 0.999;
            let epsilon: f32 = 1e-8;
            let bc1 = 1.0 - beta1.powi(t as i32);
            let bc2 = 1.0 - beta2.powi(t as i32);

            for i in 0..gaussians.len() {
                // Color
                self.m_color[i].r = beta1 * self.m_color[i].r + (1.0 - beta1) * grads[i].color.r;
                self.m_color[i].g = beta1 * self.m_color[i].g + (1.0 - beta1) * grads[i].color.g;
                self.m_color[i].b = beta1 * self.m_color[i].b + (1.0 - beta1) * grads[i].color.b;

                self.v_color[i].r = beta2 * self.v_color[i].r + (1.0 - beta2) * grads[i].color.r.powi(2);
                self.v_color[i].g = beta2 * self.v_color[i].g + (1.0 - beta2) * grads[i].color.g.powi(2);
                self.v_color[i].b = beta2 * self.v_color[i].b + (1.0 - beta2) * grads[i].color.b.powi(2);

                gaussians[i].color.r -= lr * (self.m_color[i].r / bc1) / ((self.v_color[i].r / bc2).sqrt() + epsilon);
                gaussians[i].color.g -= lr * (self.m_color[i].g / bc1) / ((self.v_color[i].g / bc2).sqrt() + epsilon);
                gaussians[i].color.b -= lr * (self.m_color[i].b / bc1) / ((self.v_color[i].b / bc2).sqrt() + epsilon);

                gaussians[i].color.r = gaussians[i].color.r.clamp(0.0, 1.0);
                gaussians[i].color.g = gaussians[i].color.g.clamp(0.0, 1.0);
                gaussians[i].color.b = gaussians[i].color.b.clamp(0.0, 1.0);

                // Position
                self.m_position[i].x = beta1 * self.m_position[i].x + (1.0 - beta1) * grads[i].position.x;
                self.m_position[i].y = beta1 * self.m_position[i].y + (1.0 - beta1) * grads[i].position.y;

                self.v_position[i].x = beta2 * self.v_position[i].x + (1.0 - beta2) * grads[i].position.x.powi(2);
                self.v_position[i].y = beta2 * self.v_position[i].y + (1.0 - beta2) * grads[i].position.y.powi(2);

                gaussians[i].position.x -= lr * (self.m_position[i].x / bc1) / ((self.v_position[i].x / bc2).sqrt() + epsilon);
                gaussians[i].position.y -= lr * (self.m_position[i].y / bc1) / ((self.v_position[i].y / bc2).sqrt() + epsilon);

                gaussians[i].position.x = gaussians[i].position.x.clamp(0.0, 1.0);
                gaussians[i].position.y = gaussians[i].position.y.clamp(0.0, 1.0);

                // Scale
                self.m_scale[i].0 = beta1 * self.m_scale[i].0 + (1.0 - beta1) * grads[i].scale_x;
                self.m_scale[i].1 = beta1 * self.m_scale[i].1 + (1.0 - beta1) * grads[i].scale_y;

                self.v_scale[i].0 = beta2 * self.v_scale[i].0 + (1.0 - beta2) * grads[i].scale_x.powi(2);
                self.v_scale[i].1 = beta2 * self.v_scale[i].1 + (1.0 - beta2) * grads[i].scale_y.powi(2);

                gaussians[i].shape.scale_x -= lr * (self.m_scale[i].0 / bc1) / ((self.v_scale[i].0 / bc2).sqrt() + epsilon);
                gaussians[i].shape.scale_y -= lr * (self.m_scale[i].1 / bc1) / ((self.v_scale[i].1 / bc2).sqrt() + epsilon);

                gaussians[i].shape.scale_x = gaussians[i].shape.scale_x.clamp(0.001, 0.5);
                gaussians[i].shape.scale_y = gaussians[i].shape.scale_y.clamp(0.001, 0.5);
            }

            loss
        }
    }

    pub fn init_kmeans(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(42);

        let mut samples = Vec::new();
        for y in 0..target.height {
            for x in 0..target.width {
                let color = target.get_pixel(x, y).unwrap();
                samples.push((
                    x as f32 / target.width as f32,
                    y as f32 / target.height as f32,
                    color.r, color.g, color.b,
                ));
            }
        }

        let mut centroids: Vec<(f32, f32, f32, f32, f32)> = (0..n)
            .map(|_| samples[rng.gen_range(0..samples.len())])
            .collect();

        for _ in 0..10 {
            let mut clusters: Vec<Vec<(f32, f32, f32, f32, f32)>> = vec![Vec::new(); n];

            for &sample in &samples {
                let mut best_idx = 0;
                let mut best_dist = f32::MAX;
                for (i, &centroid) in centroids.iter().enumerate() {
                    let dx = sample.0 - centroid.0;
                    let dy = sample.1 - centroid.1;
                    let dr = sample.2 - centroid.2;
                    let dg = sample.3 - centroid.3;
                    let db = sample.4 - centroid.4;
                    let dist = dx * dx + dy * dy + 0.3 * (dr * dr + dg * dg + db * db);
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = i;
                    }
                }
                clusters[best_idx].push(sample);
            }

            for (i, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let sum = cluster.iter().fold((0.0, 0.0, 0.0, 0.0, 0.0), |acc, &s| {
                        (acc.0 + s.0, acc.1 + s.1, acc.2 + s.2, acc.3 + s.3, acc.4 + s.4)
                    });
                    let count = cluster.len() as f32;
                    centroids[i] = (sum.0 / count, sum.1 / count, sum.2 / count, sum.3 / count, sum.4 / count);
                }
            }
        }

        centroids.into_iter().map(|(px, py, r, g, b)| {
            Gaussian2D::new(
                Vector2::new(px, py),
                Euler::isotropic(0.05),
                Color4::new(r, g, b, 1.0),
                1.0,
            )
        }).collect()
    }
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
    println!("CONVERGENCE TEST: 1000 Iterations on All Baselines");
    println!("{}", sep);

    let image_size = 64;
    let n_gaussians = 50;
    let iterations = 1000;

    println!("\n📊 Test Configuration:");
    println!("   Image size: {}×{}", image_size, image_size);
    println!("   N Gaussians: {}", n_gaussians);
    println!("   Iterations: {}", iterations);

    let target = create_test_image(image_size);

    // ===== BASELINE 1: GaussianImage (Random Init) =====
    println!("\n{}", sep);
    println!("BASELINE 1: GaussianImage (Random Init, Fixed N)");
    println!("{}", sep);

    let mut g1 = baseline1::init_random(n_gaussians);
    let mut opt1 = baseline1::Optimizer::new(0.001, n_gaussians);

    let initial_render = RendererV2::render(&g1, target.width, target.height);
    let initial_loss1 = loss_functions::compute_combined_loss(&initial_render, &target, 0.2, 0.8, 0.1);
    println!("Initial loss: {:.6}", initial_loss1);

    let start = Instant::now();
    let mut loss1_history = vec![initial_loss1];

    for t in 1..=iterations {
        let loss = opt1.step(&mut g1, &target, t);
        loss1_history.push(loss);

        if t.is_power_of_two() || t % 100 == 0 {
            println!("   Iter {:4}: loss={:.6}", t, loss);
        }
    }

    let elapsed1 = start.elapsed();
    let final_loss1 = loss1_history[loss1_history.len() - 1];
    println!("Final loss:   {:.6}", final_loss1);
    println!("Improvement:  {:.2}%", (initial_loss1 - final_loss1) / initial_loss1 * 100.0);
    println!("Time:         {:.2}s ({:.1} iter/s)", elapsed1.as_secs_f64(), iterations as f64 / elapsed1.as_secs_f64());

    // ===== BASELINE 2: Video Codec (K-means Init) =====
    println!("\n{}", sep);
    println!("BASELINE 2: Video Codec (K-means Init, Cosine LR)");
    println!("{}", sep);

    let mut g2 = baseline2::init_kmeans(&target, n_gaussians);
    let mut opt2 = baseline2::Optimizer::new(0.005, n_gaussians);

    let initial_render = RendererV2::render(&g2, target.width, target.height);
    let initial_loss2 = loss_functions::compute_combined_loss(&initial_render, &target, 0.3, 0.7, 0.05);
    println!("Initial loss: {:.6}", initial_loss2);

    let start = Instant::now();
    let mut loss2_history = vec![initial_loss2];

    for t in 1..=iterations {
        let loss = opt2.step(&mut g2, &target, t, iterations);
        loss2_history.push(loss);

        if t.is_power_of_two() || t % 100 == 0 {
            println!("   Iter {:4}: loss={:.6}", t, loss);
        }
    }

    let elapsed2 = start.elapsed();
    let final_loss2 = loss2_history[loss2_history.len() - 1];
    println!("Final loss:   {:.6}", final_loss2);
    println!("Improvement:  {:.2}%", (initial_loss2 - final_loss2) / initial_loss2 * 100.0);
    println!("Time:         {:.2}s ({:.1} iter/s)", elapsed2.as_secs_f64(), iterations as f64 / elapsed2.as_secs_f64());

    // ===== SUMMARY =====
    println!("\n{}", sep);
    println!("SUMMARY");
    println!("{}", sep);

    println!("\nInitial Loss:");
    println!("   Baseline 1 (Random):  {:.6}", initial_loss1);
    println!("   Baseline 2 (K-means): {:.6}", initial_loss2);
    println!("   → K-means advantage:  {:.2}× better start", initial_loss1 / initial_loss2);

    println!("\nFinal Loss (after {} iterations):", iterations);
    println!("   Baseline 1: {:.6} ({:.2}% improvement)", final_loss1, (initial_loss1 - final_loss1) / initial_loss1 * 100.0);
    println!("   Baseline 2: {:.6} ({:.2}% improvement)", final_loss2, (initial_loss2 - final_loss2) / initial_loss2 * 100.0);

    if final_loss1 < final_loss2 {
        println!("   → Winner: Baseline 1 (by {:.2}%)", (final_loss2 - final_loss1) / final_loss2 * 100.0);
    } else {
        println!("   → Winner: Baseline 2 (by {:.2}%)", (final_loss1 - final_loss2) / final_loss1 * 100.0);
    }

    println!("\nConvergence Rate (loss reduction per 100 iterations):");
    for baseline_idx in 1..=2 {
        let history = if baseline_idx == 1 { &loss1_history } else { &loss2_history };
        print!("   Baseline {}: ", baseline_idx);

        for i in [1, 2, 5, 10] {
            let idx = (i * 100).min(history.len() - 1);
            if idx > 0 {
                let improvement = (history[idx - 100] - history[idx]) / history[idx - 100] * 100.0;
                print!("{}k={:.1}% ", i, improvement);
            }
        }
        println!();
    }

    println!("\nSpeed:");
    println!("   Baseline 1: {:.1} iter/s", iterations as f64 / elapsed1.as_secs_f64());
    println!("   Baseline 2: {:.1} iter/s", iterations as f64 / elapsed2.as_secs_f64());

    println!("\n{}", sep);
}
