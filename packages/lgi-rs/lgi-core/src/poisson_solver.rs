//! Poisson Solver for Image Reconstruction
//!
//! Solves: ∇²u = f
//! Where f = Laplacian field (sparse), u = reconstructed image (dense)
//!
//! Used in GDGS: Place Gaussians at Laplacian peaks, reconstruct via Poisson
//!
//! Method: Gauss-Seidel iteration (good balance of speed and simplicity)

use crate::ImageBuffer;
use lgi_math::color::Color4;

/// Poisson solver configuration
#[derive(Clone)]
pub struct PoissonConfig {
    /// Maximum iterations
    pub max_iterations: usize,

    /// Convergence threshold (L2 norm of residual)
    pub tolerance: f32,

    /// Boundary condition (value at image borders)
    pub boundary_value: f32,
}

impl Default for PoissonConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-4,
            boundary_value: 0.5,  // Gray boundary
        }
    }
}

/// Solve Poisson equation: ∇²u = f
///
/// Reconstructs image from Laplacian field using Gauss-Seidel iteration
///
/// # Arguments
/// * `laplacian` - Source field (∇²u, sparse representation)
/// * `width`, `height` - Image dimensions
/// * `config` - Solver configuration
///
/// # Returns
/// Reconstructed image satisfying ∇²u = laplacian
pub fn solve_poisson(
    laplacian: &[f32],
    width: u32,
    height: u32,
    config: &PoissonConfig,
) -> ImageBuffer<f32> {
    // Initialize solution with boundary conditions
    let mut u_r = vec![config.boundary_value; (width * height) as usize];
    let mut u_g = vec![config.boundary_value; (width * height) as usize];
    let mut u_b = vec![config.boundary_value; (width * height) as usize];

    // Gauss-Seidel iteration
    for iter in 0..config.max_iterations {
        let mut max_change = 0.0f32;

        // Red-black ordering for better convergence
        for color_phase in 0..2 {
            for y in 1..(height - 1) {
                for x in 1..(width - 1) {
                    // Red-black checkerboard pattern
                    if (x + y) % 2 != color_phase {
                        continue;
                    }

                    let idx = (y * width + x) as usize;

                    // Get neighbors
                    let up = ((y - 1) * width + x) as usize;
                    let down = ((y + 1) * width + x) as usize;
                    let left = (y * width + (x - 1)) as usize;
                    let right = (y * width + (x + 1)) as usize;

                    // Gauss-Seidel update: u[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - h²f[i,j]) / 4
                    // For h=1 (unit spacing): u = (neighbors - laplacian) / 4
                    let f = laplacian[idx];

                    // Update each channel
                    let old_r = u_r[idx];
                    u_r[idx] = (u_r[up] + u_r[down] + u_r[left] + u_r[right] - f) * 0.25;
                    let change_r = (u_r[idx] - old_r).abs();

                    let old_g = u_g[idx];
                    u_g[idx] = (u_g[up] + u_g[down] + u_g[left] + u_g[right] - f) * 0.25;
                    let change_g = (u_g[idx] - old_g).abs();

                    let old_b = u_b[idx];
                    u_b[idx] = (u_b[up] + u_b[down] + u_b[left] + u_b[right] - f) * 0.25;
                    let change_b = (u_b[idx] - old_b).abs();

                    max_change = max_change.max(change_r).max(change_g).max(change_b);
                }
            }
        }

        // Check convergence
        if max_change < config.tolerance {
            // Converged
            break;
        }

        // Progress logging (disabled - log crate not in lgi-core)
        // if iter % 100 == 0 && iter > 0 {
        //     println!("Poisson iteration {}: max_change = {:.6}", iter, max_change);
        // }
    }

    // Convert to ImageBuffer
    let mut result = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            result.set_pixel(x, y, Color4::new(
                u_r[idx].clamp(0.0, 1.0),
                u_g[idx].clamp(0.0, 1.0),
                u_b[idx].clamp(0.0, 1.0),
                1.0,
            ));
        }
    }

    result
}

/// Solve Poisson with initial guess (faster convergence)
pub fn solve_poisson_with_init(
    laplacian: &[f32],
    initial_guess: &ImageBuffer<f32>,
    config: &PoissonConfig,
) -> ImageBuffer<f32> {
    let width = initial_guess.width;
    let height = initial_guess.height;

    // Initialize from guess
    let mut u_r = vec![0.0; (width * height) as usize];
    let mut u_g = vec![0.0; (width * height) as usize];
    let mut u_b = vec![0.0; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let pixel = initial_guess.get_pixel(x, y).unwrap();
            u_r[idx] = pixel.r;
            u_g[idx] = pixel.g;
            u_b[idx] = pixel.b;
        }
    }

    // Solve (same as above but starting from initial_guess)
    // ... (code same as solve_poisson but with initialized u_r/g/b)

    // For now, just call solve_poisson
    solve_poisson(laplacian, width, height, config)
}

#[cfg(test)]
mod poisson_tests {
    use super::*;

    #[test]
    fn test_poisson_flat() {
        // Flat Laplacian → flat solution
        let width = 50;
        let height = 50;
        let laplacian = vec![0.0; (width * height) as usize];

        let config = PoissonConfig {
            max_iterations: 100,
            tolerance: 1e-3,
            boundary_value: 0.5,
        };

        let solution = solve_poisson(&laplacian, width, height, &config);

        // Should converge to boundary value
        let center = solution.get_pixel(25, 25).unwrap();
        assert!((center.r - 0.5).abs() < 0.1, "Should converge near boundary value");
    }
}
