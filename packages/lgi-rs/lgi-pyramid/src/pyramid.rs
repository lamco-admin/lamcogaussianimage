//! Multi-level Gaussian pyramid

use crate::{PyramidLevel, Viewport, PyramidError, Result};
use lgi_core::{ImageBuffer, Renderer};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

/// Multi-level Gaussian pyramid for O(1) zoom rendering
pub struct GaussianPyramid {
    /// Pyramid levels (level 0 = full resolution)
    pub levels: Vec<PyramidLevel>,

    /// Base resolution
    pub base_width: u32,
    pub base_height: u32,
}

impl GaussianPyramid {
    /// Create new pyramid from levels
    pub fn new(levels: Vec<PyramidLevel>, base_width: u32, base_height: u32) -> Self {
        Self {
            levels,
            base_width,
            base_height,
        }
    }

    /// Get number of levels
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get specific level
    pub fn get_level(&self, index: usize) -> Result<&PyramidLevel> {
        self.levels.get(index).ok_or(PyramidError::InvalidLevel(index))
    }

    /// Select optimal level for zoom factor
    pub fn select_level_for_zoom(&self, zoom_factor: f32) -> usize {
        // zoom_factor > 1.0 = zoomed in (need finer detail)
        // zoom_factor < 1.0 = zoomed out (can use coarser level)

        // Level 0: zoom 1.0 (full res)
        // Level 1: zoom 0.5 (half res)
        // Level 2: zoom 0.25 (quarter res)

        let level_f = (-zoom_factor.log2()).max(0.0);
        let level = level_f.floor() as usize;

        level.min(self.levels.len() - 1)
    }

    /// Render at specific zoom level
    pub fn render_at_zoom(
        &self,
        zoom_factor: f32,
        viewport: Viewport,
        output_width: u32,
        output_height: u32,
    ) -> Result<ImageBuffer<f32>> {
        let level_idx = self.select_level_for_zoom(zoom_factor);
        let level = self.get_level(level_idx)?;

        // Render using selected level (O(1) in Gaussian count!)
        let renderer = Renderer::new();
        let output = renderer.render(&level.gaussians, output_width, output_height)?;

        Ok(output)
    }

    /// Get total Gaussian count across all levels
    pub fn total_gaussian_count(&self) -> usize {
        self.levels.iter().map(|l| l.gaussian_count()).sum()
    }

    /// Get total size in bytes
    pub fn total_size_bytes(&self) -> usize {
        self.levels.iter().map(|l| l.size_bytes).sum()
    }

    /// Print pyramid statistics
    pub fn print_stats(&self) {
        println!("Pyramid Statistics:");
        println!("  Base Resolution: {}×{}", self.base_width, self.base_height);
        println!("  Levels: {}", self.num_levels());
        println!("  Total Gaussians: {}", self.total_gaussian_count());
        println!("  Total Size: {} KB", self.total_size_bytes() / 1024);
        println!("\n  Level Details:");
        println!("  Idx | Resolution    | Gaussians | PSNR   | Size");
        println!("  --- | ------------- | --------- | ------ | ------");
        for level in &self.levels {
            println!("  {:>3} | {:>5}×{:<5} | {:>9} | {:>5.1} | {:>4} KB",
                level.level_index,
                level.target_width,
                level.target_height,
                level.gaussian_count(),
                level.psnr,
                level.size_bytes / 1024
            );
        }
    }
}
