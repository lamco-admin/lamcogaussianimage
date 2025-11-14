//! Hilbert curve ordering for spatial coherence

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

pub fn hilbert_index(x: u32, y: u32, order: u32) -> u32 {
    let mut index = 0;
    let mut s = order / 2;

    while s > 0 {
        let rx = if x & s > 0 { 1 } else { 0 };
        let ry = if y & s > 0 { 1 } else { 0 };
        index += s * s * ((3 * rx) ^ ry);
        s /= 2;
    }

    index
}

pub fn sort_by_hilbert(
    gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
    width: u32,
    height: u32,
) {
    let order = width.max(height).next_power_of_two();

    gaussians.sort_by_cached_key(|g| {
        let x = (g.position.x * width as f32) as u32;
        let y = (g.position.y * height as f32) as u32;
        hilbert_index(x, y, order)
    });
}
