//! Spatial tiling for efficient rendering and random access

use lgi_math::{Float, gaussian::Gaussian2D, parameterization::Parameterization, vec::Vector2};

/// Tile configuration
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Tile width in pixels
    pub tile_width: u32,
    /// Tile height in pixels
    pub tile_height: u32,
    /// Overlap in sigma units (to handle Gaussian footprints)
    pub overlap_sigma: f32,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_width: 256,
            tile_height: 256,
            overlap_sigma: 3.5,
        }
    }
}

/// Tile index (x, y)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileIndex {
    /// Tile column
    pub x: u32,
    /// Tile row
    pub y: u32,
}

/// Tile manager for spatial partitioning
pub struct TileManager {
    config: TileConfig,
    canvas_width: u32,
    canvas_height: u32,
    tiles_wide: u32,
    tiles_high: u32,
}

impl TileManager {
    /// Create new tile manager
    pub fn new(canvas_width: u32, canvas_height: u32, config: TileConfig) -> Self {
        let tiles_wide = (canvas_width + config.tile_width - 1) / config.tile_width;
        let tiles_high = (canvas_height + config.tile_height - 1) / config.tile_height;

        Self {
            config,
            canvas_width,
            canvas_height,
            tiles_wide,
            tiles_high,
        }
    }

    /// Get total number of tiles
    pub fn num_tiles(&self) -> usize {
        (self.tiles_wide * self.tiles_high) as usize
    }

    /// Get tile index for pixel coordinates
    pub fn tile_for_pixel(&self, x: u32, y: u32) -> TileIndex {
        TileIndex {
            x: x / self.config.tile_width,
            y: y / self.config.tile_height,
        }
    }

    /// Get tile index for normalized coordinates
    pub fn tile_for_normalized(&self, pos: Vector2<f32>) -> TileIndex {
        let x = (pos.x * self.canvas_width as f32) as u32;
        let y = (pos.y * self.canvas_height as f32) as u32;
        self.tile_for_pixel(x, y)
    }

    /// Get tiles that a Gaussian overlaps
    pub fn tiles_for_gaussian<P: Parameterization<f32>>(
        &self,
        gaussian: &Gaussian2D<f32, P>,
    ) -> Vec<TileIndex> {
        let (min, max) = gaussian.bounding_box(self.config.overlap_sigma);

        let min_x = (min.x * self.canvas_width as f32) as u32;
        let min_y = (min.y * self.canvas_height as f32) as u32;
        let max_x = (max.x * self.canvas_width as f32) as u32;
        let max_y = (max.y * self.canvas_height as f32) as u32;

        let tile_min = self.tile_for_pixel(min_x, min_y);
        let tile_max = self.tile_for_pixel(max_x.min(self.canvas_width - 1), max_y.min(self.canvas_height - 1));

        let mut tiles = Vec::new();
        for ty in tile_min.y..=tile_max.y {
            for tx in tile_min.x..=tile_max.x {
                if tx < self.tiles_wide && ty < self.tiles_high {
                    tiles.push(TileIndex { x: tx, y: ty });
                }
            }
        }

        tiles
    }

    /// Get pixel bounds for a tile
    pub fn tile_bounds(&self, tile: TileIndex) -> (u32, u32, u32, u32) {
        let x0 = tile.x * self.config.tile_width;
        let y0 = tile.y * self.config.tile_height;
        let x1 = (x0 + self.config.tile_width).min(self.canvas_width);
        let y1 = (y0 + self.config.tile_height).min(self.canvas_height);

        (x0, y0, x1, y1)
    }
}

/// Spatial index mapping Gaussians to tiles
pub struct SpatialIndex {
    tile_manager: TileManager,
    /// Gaussians assigned to each tile
    tile_gaussians: Vec<Vec<usize>>,
}

impl SpatialIndex {
    /// Build spatial index
    pub fn build<P: Parameterization<f32>>(
        gaussians: &[Gaussian2D<f32, P>],
        canvas_width: u32,
        canvas_height: u32,
        config: TileConfig,
    ) -> Self {
        let tile_manager = TileManager::new(canvas_width, canvas_height, config);
        let num_tiles = tile_manager.num_tiles();

        let mut tile_gaussians = vec![Vec::new(); num_tiles];

        // Assign each Gaussian to its overlapping tiles
        for (idx, gaussian) in gaussians.iter().enumerate() {
            let tiles = tile_manager.tiles_for_gaussian(gaussian);

            for tile in tiles {
                let tile_idx = (tile.y * tile_manager.tiles_wide + tile.x) as usize;
                if tile_idx < num_tiles {
                    tile_gaussians[tile_idx].push(idx);
                }
            }
        }

        Self {
            tile_manager,
            tile_gaussians,
        }
    }

    /// Get Gaussians for a tile
    pub fn gaussians_for_tile(&self, tile: TileIndex) -> &[usize] {
        let tile_idx = (tile.y * self.tile_manager.tiles_wide + tile.x) as usize;
        if tile_idx < self.tile_gaussians.len() {
            &self.tile_gaussians[tile_idx]
        } else {
            &[]
        }
    }

    /// Get tile manager
    pub fn tile_manager(&self) -> &TileManager {
        &self.tile_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::{parameterization::Euler, vec::Vector2, color::Color4};

    #[test]
    fn test_tile_manager() {
        let config = TileConfig {
            tile_width: 100,
            tile_height: 100,
            overlap_sigma: 3.0,
        };

        let manager = TileManager::new(400, 400, config);

        assert_eq!(manager.num_tiles(), 16); // 4x4 grid

        let tile = manager.tile_for_pixel(150, 250);
        assert_eq!(tile.x, 1);
        assert_eq!(tile.y, 2);
    }

    #[test]
    fn test_spatial_index() {
        let gaussians = vec![
            Gaussian2D::new(
                Vector2::new(0.25, 0.25),
                Euler::isotropic(0.05),
                Color4::rgb(1.0, 0.0, 0.0),
                1.0,
            ),
            Gaussian2D::new(
                Vector2::new(0.75, 0.75),
                Euler::isotropic(0.05),
                Color4::rgb(0.0, 0.0, 1.0),
                1.0,
            ),
        ];

        let config = TileConfig {
            tile_width: 100,
            tile_height: 100,
            overlap_sigma: 3.0,
        };

        let index = SpatialIndex::build(&gaussians, 200, 200, config);

        // First Gaussian should be in tile (0, 0)
        let tile00_gaussians = index.gaussians_for_tile(TileIndex { x: 0, y: 0 });
        assert!(tile00_gaussians.contains(&0));

        // Second Gaussian should be in tile (1, 1)
        let tile11_gaussians = index.gaussians_for_tile(TileIndex { x: 1, y: 1 });
        assert!(tile11_gaussians.contains(&1));
    }
}
