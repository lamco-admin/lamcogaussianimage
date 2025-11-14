//! Energy-based Gaussian selection

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

pub fn compute_energy(gaussian: &Gaussian2D<f32, Euler<f32>>) -> f32 {
    let scale_product = gaussian.shape.scale_x * gaussian.shape.scale_y;
    let color_magnitude = (gaussian.color.r.powi(2) + gaussian.color.g.powi(2) + gaussian.color.b.powi(2)).sqrt();
    gaussian.opacity * scale_product * color_magnitude
}

pub fn select_by_energy(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target_count: usize,
) -> Vec<usize> {
    let mut indexed: Vec<_> = gaussians.iter().enumerate()
        .map(|(i, g)| (i, compute_energy(g)))
        .collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.iter().take(target_count).map(|(i, _)| *i).collect()
}

pub fn compute_cumulative_energy(gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Vec<f32> {
    let energies: Vec<f32> = gaussians.iter().map(compute_energy).collect();
    let total: f32 = energies.iter().sum();

    let mut cumulative = Vec::new();
    let mut sum = 0.0;
    for &e in &energies {
        sum += e;
        cumulative.push(sum / total);
    }

    cumulative
}
