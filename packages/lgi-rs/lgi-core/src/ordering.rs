//! Gaussian ordering strategies for progressive rendering

use lgi_math::{Float, gaussian::Gaussian2D, parameterization::Parameterization, color::Color4};

/// Ordering strategy for progressive rendering
#[derive(Debug, Clone, Copy)]
pub enum OrderStrategy {
    /// Original order (no reordering)
    None,
    /// Energy-based (high to low)
    Energy,
    /// Distance from center (inside out)
    Radial,
    /// Custom importance weights
    Custom,
}

/// Compute energy/importance of a Gaussian
///
/// Energy = opacity × scale × color_magnitude
pub fn compute_energy<T: Float, P: Parameterization<T>>(gaussian: &Gaussian2D<T, P>) -> T {
    // Scale contribution: Use determinant of covariance (area of ellipse)
    let cov = gaussian.covariance();
    let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    let area = det.sqrt(); // sqrt(det(Σ)) = area of ellipse

    // Color magnitude
    let color_mag = (
        gaussian.color.r * gaussian.color.r +
        gaussian.color.g * gaussian.color.g +
        gaussian.color.b * gaussian.color.b
    ).sqrt();

    // Energy = opacity × area × color_magnitude
    gaussian.opacity * area * color_mag
}

/// Order Gaussians by strategy
pub fn order_gaussians<T: Float, P: Parameterization<T>>(
    gaussians: &mut [Gaussian2D<T, P>],
    strategy: OrderStrategy,
) {
    match strategy {
        OrderStrategy::None => {
            // No reordering
        }
        OrderStrategy::Energy => {
            // Sort by energy (high to low for front-to-back rendering)
            gaussians.sort_by(|a, b| {
                let energy_a = compute_energy(a);
                let energy_b = compute_energy(b);
                energy_b.partial_cmp(&energy_a).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        OrderStrategy::Radial => {
            // Sort by distance from center
            gaussians.sort_by(|a, b| {
                let dist_a = (a.position.x - T::one() / T::two()) * (a.position.x - T::one() / T::two()) +
                             (a.position.y - T::one() / T::two()) * (a.position.y - T::one() / T::two());
                let dist_b = (b.position.x - T::one() / T::two()) * (b.position.x - T::one() / T::two()) +
                             (b.position.y - T::one() / T::two()) * (b.position.y - T::one() / T::two());
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        OrderStrategy::Custom => {
            // Sort by custom weight if present
            gaussians.sort_by(|a, b| {
                let weight_a = a.weight.unwrap_or(T::one());
                let weight_b = b.weight.unwrap_or(T::one());
                weight_b.partial_cmp(&weight_a).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }
}

/// Create level-of-detail hierarchy
///
/// Returns indices into the Gaussian array for each LOD level
pub fn create_lod_levels<T: Float, P: Parameterization<T>>(
    gaussians: &[Gaussian2D<T, P>],
    num_levels: usize,
) -> Vec<Vec<usize>> {
    if num_levels == 0 || gaussians.is_empty() {
        return vec![];
    }

    let mut levels = Vec::with_capacity(num_levels);
    let total = gaussians.len();

    // Energy-based allocation: L0 gets most important, L(n-1) gets least
    // Distribution: [10%, 30%, 60%] for 3 levels
    let distributions = match num_levels {
        1 => vec![1.0],
        2 => vec![0.3, 0.7],
        3 => vec![0.1, 0.3, 0.6],
        4 => vec![0.05, 0.15, 0.30, 0.50],
        _ => {
            // Exponential distribution
            let mut dist = Vec::with_capacity(num_levels);
            let mut remaining = 1.0;
            for i in 0..(num_levels - 1) {
                let fraction = remaining * 0.3;
                dist.push(fraction);
                remaining -= fraction;
            }
            dist.push(remaining);
            dist
        }
    };

    let mut start = 0;
    for &fraction in &distributions {
        let count = (total as f32 * fraction) as usize;
        let end = (start + count).min(total);

        levels.push((start..end).collect());
        start = end;
    }

    levels
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::{parameterization::Euler, vec::Vector2, color::Color4};

    #[test]
    fn test_energy_computation() {
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::new(0.1, 0.1, 0.0),
            Color4::rgb(1.0, 0.0, 0.0),
            0.8,
        );

        let energy = compute_energy(&gaussian);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_energy_ordering() {
        let mut gaussians = vec![
            Gaussian2D::new(
                Vector2::new(0.5, 0.5),
                Euler::isotropic(0.01), // Small
                Color4::rgb(1.0, 0.0, 0.0),
                0.5,
            ),
            Gaussian2D::new(
                Vector2::new(0.3, 0.3),
                Euler::isotropic(0.1), // Large
                Color4::rgb(1.0, 0.0, 0.0),
                1.0,
            ),
        ];

        order_gaussians(&mut gaussians, OrderStrategy::Energy);

        // Higher energy should come first
        let energy0 = compute_energy(&gaussians[0]);
        let energy1 = compute_energy(&gaussians[1]);
        assert!(energy0 >= energy1);
    }

    #[test]
    fn test_lod_creation() {
        let gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = (0..1000)
            .map(|i| {
                Gaussian2D::new(
                    Vector2::new(i as f32 / 1000.0, 0.5),
                    Euler::isotropic(0.01),
                    Color4::white(),
                    1.0,
                )
            })
            .collect();

        let levels = create_lod_levels(&gaussians, 3);

        assert_eq!(levels.len(), 3);
        // L0: ~10%, L1: ~30%, L2: ~60%
        assert!(levels[0].len() >= 80 && levels[0].len() <= 120);   // ~100
        assert!(levels[1].len() >= 280 && levels[1].len() <= 320);  // ~300
        assert!(levels[2].len() >= 580 && levels[2].len() <= 620);  // ~600
    }
}
