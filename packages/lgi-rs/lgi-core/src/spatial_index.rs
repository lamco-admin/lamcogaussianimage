//! Spatial indexing for fast Gaussian queries

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

pub struct SpatialIndex {
    grid: Vec<Vec<usize>>,  // Grid cells containing Gaussian indices
    grid_size: usize,
}

impl SpatialIndex {
    pub fn new(gaussians: &[Gaussian2D<f32, Euler<f32>>], grid_size: usize) -> Self {
        let mut grid = vec![Vec::new(); grid_size * grid_size];

        for (idx, g) in gaussians.iter().enumerate() {
            let cell_x = (g.position.x * grid_size as f32) as usize;
            let cell_y = (g.position.y * grid_size as f32) as usize;
            let cell_idx = (cell_y.min(grid_size-1) * grid_size + cell_x.min(grid_size-1)).min(grid.len()-1);
            grid[cell_idx].push(idx);
        }

        Self { grid, grid_size }
    }

    pub fn query_region(&self, x: f32, y: f32, radius: f32) -> Vec<usize> {
        let cell_x_min = ((x - radius) * self.grid_size as f32).max(0.0) as usize;
        let cell_x_max = ((x + radius) * self.grid_size as f32).min(self.grid_size as f32 - 1.0) as usize;
        let cell_y_min = ((y - radius) * self.grid_size as f32).max(0.0) as usize;
        let cell_y_max = ((y + radius) * self.grid_size as f32).min(self.grid_size as f32 - 1.0) as usize;

        let mut indices = Vec::new();
        for cy in cell_y_min..=cell_y_max {
            for cx in cell_x_min..=cell_x_max {
                let cell_idx = cy * self.grid_size + cx;
                if cell_idx < self.grid.len() {
                    indices.extend(&self.grid[cell_idx]);
                }
            }
        }

        indices
    }
}
