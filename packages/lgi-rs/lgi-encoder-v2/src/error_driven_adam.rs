//! Error-driven Gaussian placement with Adam optimizer
//!
//! Key improvements over base ErrorDrivenEncoder:
//! 1. Uses Adam instead of OptimizerV2 (proven 10x+ faster)
//! 2. Fresh optimizer state when Gaussians are added (no stale momentum)
//! 3. Early stopping when improvement plateaus
//! 4. Spatial clustering for hotspot selection (prevents redundant placement)
//!
//! Based on analysis showing OptimizerV2 achieves 19.90 dB with 500 Gaussians
//! while Adam achieves 20.99 dB with just 256 Gaussians.

use crate::adam_optimizer::{AdamOptimizer, LearningRates};
use crate::renderer_v2::RendererV2;
use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};

/// Configuration for Adam-based error-driven encoder
#[derive(Clone)]
pub struct ErrorDrivenAdamConfig {
    /// Target MSE loss threshold (default: 0.001)
    pub target_error: f32,
    /// Initial number of Gaussians (default: 64)
    pub initial_gaussians: usize,
    /// Maximum allowed Gaussians (default: 512)
    pub max_gaussians: usize,
    /// Fraction of high-error pixels to consider for placement (default: 0.05)
    pub split_percentile: f32,
    /// Adam iterations per refinement pass (default: 50)
    pub iterations_per_pass: usize,
    /// Minimum PSNR improvement to continue (default: 0.05 dB)
    pub min_improvement_db: f32,
    /// Maximum refinement passes (default: 8)
    pub max_passes: usize,
    /// Minimum distance between new Gaussians (normalized, default: 0.02)
    pub min_gaussian_spacing: f32,
    /// Adam learning rates
    pub learning_rates: LearningRates,
}

impl Default for ErrorDrivenAdamConfig {
    fn default() -> Self {
        Self {
            target_error: 0.001,
            initial_gaussians: 64,
            max_gaussians: 512,
            split_percentile: 0.05,         // Top 5% error pixels
            iterations_per_pass: 50,        // Fewer, faster passes
            min_improvement_db: 0.05,       // Early stopping threshold
            max_passes: 8,
            min_gaussian_spacing: 0.02,     // ~15 pixels apart at 768px
            learning_rates: LearningRates::default(),
        }
    }
}

/// Error-driven encoder using Adam optimizer
pub struct ErrorDrivenAdamEncoder {
    pub config: ErrorDrivenAdamConfig,
}

impl Default for ErrorDrivenAdamEncoder {
    fn default() -> Self {
        Self {
            config: ErrorDrivenAdamConfig::default(),
        }
    }
}

impl ErrorDrivenAdamEncoder {
    pub fn new(config: ErrorDrivenAdamConfig) -> Self {
        Self { config }
    }

    /// Encode image with adaptive Gaussian placement using Adam
    pub fn encode(&self, target: &ImageBuffer<f32>) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        let (width, height) = (target.width, target.height);

        // Initialize with uniform grid
        let grid_size = (self.config.initial_gaussians as f32).sqrt().ceil() as u32;
        let mut gaussians = self.initialize_uniform_grid(target, grid_size);

        println!("ErrorDrivenAdam encoding:");
        println!("  Image: {}×{}", width, height);
        println!("  Initial: {} Gaussians ({}×{} grid)", gaussians.len(), grid_size, grid_size);
        println!("  Max: {} Gaussians", self.config.max_gaussians);
        println!("  Adam iters/pass: {}", self.config.iterations_per_pass);

        let mut last_psnr = 0.0;
        let mut stall_count = 0;

        for pass in 0..self.config.max_passes {
            // Create FRESH Adam optimizer for each pass
            // This is critical: when we add Gaussians, old momentum is invalid
            let mut adam = AdamOptimizer::new();
            adam.learning_rates = self.config.learning_rates.clone();
            adam.max_iterations = self.config.iterations_per_pass;

            // Optimize with progress tracking
            let mut best_psnr = 0.0;
            for iter in 0..self.config.iterations_per_pass {
                adam.max_iterations = 1;
                adam.optimize(&mut gaussians, target);

                if iter % 10 == 0 || iter == self.config.iterations_per_pass - 1 {
                    let rendered = RendererV2::render(&gaussians, width, height);
                    let psnr = compute_psnr(target, &rendered);
                    if psnr > best_psnr {
                        best_psnr = psnr;
                    }
                }
            }

            let rendered = RendererV2::render(&gaussians, width, height);
            let loss = compute_mse(target, &rendered);
            let psnr = compute_psnr(target, &rendered);

            let improvement = psnr - last_psnr;
            let direction = if improvement > 0.0 { "↑" } else { "↓" };

            println!(
                "  Pass {}: N={:4}, PSNR={:.2}dB {} ({:+.2}dB), loss={:.6}",
                pass,
                gaussians.len(),
                psnr,
                direction,
                improvement,
                loss
            );

            // Check convergence conditions
            if loss < self.config.target_error {
                println!("  ✓ Converged to target error!");
                break;
            }

            if gaussians.len() >= self.config.max_gaussians {
                println!("  ✓ Reached max Gaussians limit");
                break;
            }

            // Early stopping: check for plateau
            if pass > 0 && improvement < self.config.min_improvement_db {
                stall_count += 1;
                if stall_count >= 2 {
                    println!("  ✓ Improvement stalled ({:.3}dB < {:.3}dB threshold)",
                             improvement, self.config.min_improvement_db);
                    break;
                }
            } else {
                stall_count = 0;
            }

            last_psnr = psnr;

            // Find high-error regions with spatial clustering
            let error_map = self.compute_error_map(target, &rendered);
            let hotspots = self.find_clustered_hotspots(&error_map, target, &gaussians);

            if hotspots.is_empty() {
                println!("  ✓ No suitable hotspots remaining");
                break;
            }

            // Add Gaussians at hotspots
            let added = self.add_gaussians_at_hotspots(&mut gaussians, target, &hotspots);
            println!("    → Added {} Gaussians at high-error locations", added);
        }

        // Final polish pass
        println!("  Final polish (20 iterations)...");
        let mut adam = AdamOptimizer::new();
        adam.learning_rates = self.config.learning_rates.clone();
        adam.max_iterations = 20;
        adam.optimize(&mut gaussians, target);

        let final_rendered = RendererV2::render(&gaussians, width, height);
        let final_psnr = compute_psnr(target, &final_rendered);
        println!("  Final: {} Gaussians, {:.2}dB", gaussians.len(), final_psnr);

        gaussians
    }

    /// Initialize uniform grid with color sampling
    fn initialize_uniform_grid(
        &self,
        target: &ImageBuffer<f32>,
        grid_size: u32,
    ) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        let mut gaussians = Vec::new();
        let num_gaussians = grid_size * grid_size;

        // Coverage-based scale (gamma=0.8 provides good overlap)
        let gamma = 0.8;
        let width = target.width as f32;
        let height = target.height as f32;
        let sigma_base_px = gamma * ((width * height) / num_gaussians as f32).sqrt();
        let sigma_norm = (sigma_base_px / width).clamp(0.01, 0.25);

        let step_x = target.width / grid_size;
        let step_y = target.height / grid_size;

        for gy in 0..grid_size {
            for gx in 0..grid_size {
                let x = (gx * step_x + step_x / 2).min(target.width - 1);
                let y = (gy * step_y + step_y / 2).min(target.height - 1);

                let position = Vector2::new(x as f32 / width, y as f32 / height);

                let color = target
                    .get_pixel(x, y)
                    .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

                gaussians.push(Gaussian2D::new(
                    position,
                    Euler::isotropic(sigma_norm),
                    color,
                    1.0,
                ));
            }
        }

        gaussians
    }

    /// Compute per-pixel MSE error map
    fn compute_error_map(&self, target: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> Vec<f32> {
        target
            .data
            .iter()
            .zip(rendered.data.iter())
            .map(|(t, r)| (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2))
            .collect()
    }

    /// Find hotspots with spatial clustering to avoid redundant placements
    fn find_clustered_hotspots(
        &self,
        error_map: &[f32],
        target: &ImageBuffer<f32>,
        existing: &[Gaussian2D<f32, Euler<f32>>],
    ) -> Vec<Hotspot> {
        let width = target.width;
        let height = target.height;
        let w_f = width as f32;
        let h_f = height as f32;

        // Find error threshold
        let mut sorted_errors = error_map.to_vec();
        sorted_errors.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold_idx = (sorted_errors.len() as f32 * self.config.split_percentile) as usize;
        let threshold = sorted_errors.get(threshold_idx).copied().unwrap_or(0.0);

        // Collect candidate hotspots above threshold
        let mut candidates: Vec<Hotspot> = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if error_map[idx] > threshold {
                    candidates.push(Hotspot {
                        x,
                        y,
                        error: error_map[idx],
                    });
                }
            }
        }

        // Sort by error (highest first)
        candidates.sort_by(|a, b| b.error.partial_cmp(&a.error).unwrap());

        // Greedy clustering: only keep hotspots that are far enough from:
        // 1. Existing Gaussians
        // 2. Already-selected hotspots
        let min_dist_sq = self.config.min_gaussian_spacing.powi(2);
        let mut selected: Vec<Hotspot> = Vec::new();

        for candidate in candidates {
            let cx = candidate.x as f32 / w_f;
            let cy = candidate.y as f32 / h_f;

            // Check distance to existing Gaussians
            let too_close_to_existing = existing.iter().any(|g| {
                let dx = g.position.x - cx;
                let dy = g.position.y - cy;
                dx * dx + dy * dy < min_dist_sq
            });

            if too_close_to_existing {
                continue;
            }

            // Check distance to already-selected hotspots
            let too_close_to_selected = selected.iter().any(|h| {
                let hx = h.x as f32 / w_f;
                let hy = h.y as f32 / h_f;
                let dx = hx - cx;
                let dy = hy - cy;
                dx * dx + dy * dy < min_dist_sq
            });

            if too_close_to_selected {
                continue;
            }

            selected.push(candidate);

            // Limit per pass to avoid overwhelming optimizer
            if selected.len() >= 30 {
                break;
            }
        }

        selected
    }

    /// Add Gaussians at hotspot locations
    fn add_gaussians_at_hotspots(
        &self,
        gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
        target: &ImageBuffer<f32>,
        hotspots: &[Hotspot],
    ) -> usize {
        let width = target.width as f32;
        let height = target.height as f32;
        let mut added = 0;

        for hotspot in hotspots {
            if gaussians.len() >= self.config.max_gaussians {
                break;
            }

            let position = Vector2::new(hotspot.x as f32 / width, hotspot.y as f32 / height);

            let color = target
                .get_pixel(hotspot.x, hotspot.y)
                .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            // Adaptive sigma based on error magnitude (higher error = smaller detail needed)
            // Base sigma around 0.02-0.05 for detail work
            let sigma = 0.03; // Could be made adaptive

            gaussians.push(Gaussian2D::new(
                position,
                Euler::isotropic(sigma),
                color,
                1.0,
            ));
            added += 1;
        }

        added
    }
}

struct Hotspot {
    x: u32,
    y: u32,
    error: f32,
}

fn compute_mse(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let count = (original.width * original.height * 3) as f32;
    let mse: f32 = original
        .data
        .iter()
        .zip(rendered.data.iter())
        .map(|(p1, p2)| (p1.r - p2.r).powi(2) + (p1.g - p2.g).powi(2) + (p1.b - p2.b).powi(2))
        .sum();
    mse / count
}

fn compute_psnr(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mse = compute_mse(original, rendered);
    if mse < 1e-10 {
        100.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}
