//! Error-driven Gaussian placement
//!
//! Adaptively place Gaussians where reconstruction error is high
//! Data shows: +4.3 dB potential vs uniform grid placement

use crate::renderer_v2::RendererV2;
use crate::optimizer_v2::OptimizerV2;
use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

/// Error-driven encoder
pub struct ErrorDrivenEncoder {
    pub target_error: f32,       // Stop when loss < this
    pub initial_gaussians: usize, // Start with this many
    pub max_gaussians: usize,    // Don't exceed this
    pub split_percentile: f32,   // Split top X% error regions
}

impl Default for ErrorDrivenEncoder {
    fn default() -> Self {
        Self {
            target_error: 0.001,
            initial_gaussians: 100,
            max_gaussians: 2000,
            split_percentile: 0.10,  // Top 10% error
        }
    }
}

impl ErrorDrivenEncoder {
    /// Encode image with adaptive Gaussian placement
    pub fn encode(&self, target: &ImageBuffer<f32>) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        // Start with coarse uniform grid
        let grid_size = (self.initial_gaussians as f32).sqrt() as u32;
        let mut gaussians = Self::initialize_uniform_grid(target, grid_size);

        println!("Error-driven encoding:");
        println!("  Initial Gaussians: {}", gaussians.len());

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 100;

        for pass in 0..10 {  // Max 10 refinement passes
            // Optimize current set
            let loss = optimizer.optimize(&mut gaussians, target);

            let rendered = RendererV2::render(&gaussians, target.width, target.height);
            let psnr = compute_psnr(target, &rendered);

            println!("  Pass {}: N={}, PSNR={:.2} dB, loss={:.6}",
                pass, gaussians.len(), psnr, loss);

            // Check convergence
            if loss < self.target_error {
                println!("  Converged to target error!");
                break;
            }

            if gaussians.len() >= self.max_gaussians {
                println!("  Reached max Gaussians limit");
                break;
            }

            // Find high-error regions
            let error_map = Self::compute_error_map(target, &rendered);
            let hotspots = Self::find_hotspots(&error_map, target, self.split_percentile);

            if hotspots.is_empty() {
                println!("  No more hotspots to split");
                break;
            }

            println!("    Adding {} Gaussians at high-error locations", hotspots.len());

            // Add Gaussians at hotspots
            for hotspot in hotspots {
                if gaussians.len() >= self.max_gaussians {
                    break;
                }

                let new_gaussian = Self::create_gaussian_at(target, &hotspot);
                gaussians.push(new_gaussian);
            }
        }

        gaussians
    }

    /// Create uniform grid initialization
    fn initialize_uniform_grid(target: &ImageBuffer<f32>, grid_size: u32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        let mut gaussians = Vec::new();
        let num_gaussians = grid_size * grid_size;

        // Coverage-based scale
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

                let position = Vector2::new(
                    x as f32 / target.width as f32,
                    y as f32 / target.height as f32,
                );

                let color = target.get_pixel(x, y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

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

    /// Compute per-pixel error map
    fn compute_error_map(target: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> Vec<f32> {
        let mut errors = Vec::with_capacity((target.width * target.height) as usize);

        for (t, r) in target.data.iter().zip(rendered.data.iter()) {
            let err = (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
            errors.push(err);
        }

        errors
    }

    /// Find high-error regions (hotspots)
    fn find_hotspots(error_map: &[f32], target: &ImageBuffer<f32>, percentile: f32) -> Vec<Hotspot> {
        // Sort errors to find threshold
        let mut sorted_errors = error_map.to_vec();
        sorted_errors.sort_by(|a, b| b.partial_cmp(a).unwrap());  // Descending

        let threshold_idx = (sorted_errors.len() as f32 * percentile) as usize;
        let threshold = sorted_errors[threshold_idx];

        let mut hotspots = Vec::new();

        for y in 0..target.height {
            for x in 0..target.width {
                let idx = (y * target.width + x) as usize;
                if error_map[idx] > threshold {
                    hotspots.push(Hotspot {
                        x,
                        y,
                        error: error_map[idx],
                    });
                }
            }
        }

        // Limit number of hotspots per pass
        hotspots.truncate(50);
        hotspots
    }

    /// Create a Gaussian at a hotspot location
    fn create_gaussian_at(target: &ImageBuffer<f32>, hotspot: &Hotspot) -> Gaussian2D<f32, Euler<f32>> {
        let width = target.width as f32;
        let height = target.height as f32;

        let position = Vector2::new(
            hotspot.x as f32 / width,
            hotspot.y as f32 / height,
        );

        let color = target.get_pixel(hotspot.x, hotspot.y)
            .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

        // Small scale for refinement
        let sigma = 0.02;  // Small for detail

        Gaussian2D::new(
            position,
            Euler::isotropic(sigma),
            color,
            1.0,
        )
    }
}

struct Hotspot {
    x: u32,
    y: u32,
    error: f32,
}

fn compute_psnr(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (original.width * original.height * 3) as f32;

    for (p1, p2) in original.data.iter().zip(rendered.data.iter()) {
        mse += (p1.r - p2.r).powi(2);
        mse += (p1.g - p2.g).powi(2);
        mse += (p1.b - p2.b).powi(2);
    }

    mse /= count;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
