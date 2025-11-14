//! Adaptive Gaussian splitting based on error

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use crate::ImageBuffer;

pub struct AdaptiveSplitter {
    pub error_percentile: f32,  // Top K% error regions to split
    pub max_gaussians: usize,
}

impl Default for AdaptiveSplitter {
    fn default() -> Self {
        Self {
            error_percentile: 0.1,  // Top 10%
            max_gaussians: 10000,
        }
    }
}

impl AdaptiveSplitter {
    pub fn find_split_locations(
        &self,
        error_map: &[f32],
        width: u32,
        height: u32,
    ) -> Vec<(u32, u32)> {
        let mut errors: Vec<_> = error_map.iter().enumerate().map(|(i, &e)| (i, e)).collect();
        errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let k = ((width * height) as f32 * self.error_percentile) as usize;

        errors.iter().take(k).map(|(idx, _)| {
            let x = (idx % width as usize) as u32;
            let y = (idx / width as usize) as u32;
            (x, y)
        }).collect()
    }

    pub fn split_at_locations(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        locations: &[(u32, u32)],
        width: u32,
        height: u32,
    ) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        let mut new_gaussians = Vec::new();

        for &(x, y) in locations {
            if gaussians.len() + new_gaussians.len() >= self.max_gaussians {
                break;
            }

            let px = x as f32 / width as f32;
            let py = y as f32 / height as f32;

            // Find nearest Gaussian
            if let Some(nearest) = gaussians.iter().min_by(|a, b| {
                let dist_a = ((a.position.x - px).powi(2) + (a.position.y - py).powi(2)).sqrt();
                let dist_b = ((b.position.x - px).powi(2) + (b.position.y - py).powi(2)).sqrt();
                dist_a.partial_cmp(&dist_b).unwrap()
            }) {
                // Create smaller Gaussian at error location
                let scale = nearest.shape.scale_x.min(nearest.shape.scale_y) * 0.5;
                new_gaussians.push(Gaussian2D::new(
                    Vector2::new(px, py),
                    Euler::new(scale, scale, 0.0),
                    nearest.color,
                    0.5,
                ));
            }
        }

        new_gaussians
    }
}
