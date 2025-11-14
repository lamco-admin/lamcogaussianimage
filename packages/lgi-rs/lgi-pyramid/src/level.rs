//! Pyramid level representation

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

/// Single level in pyramid
#[derive(Debug, Clone)]
pub struct PyramidLevel {
    /// Level index (0 = full resolution)
    pub level_index: usize,

    /// Target resolution for this level
    pub target_width: u32,
    pub target_height: u32,

    /// Gaussians optimized for this resolution
    pub gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,

    /// Quality score (PSNR at target resolution)
    pub psnr: f32,

    /// Size in bytes (for storage estimation)
    pub size_bytes: usize,
}

impl PyramidLevel {
    /// Create new pyramid level
    pub fn new(
        level_index: usize,
        target_width: u32,
        target_height: u32,
        gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
    ) -> Self {
        let size_bytes = gaussians.len() * 48; // 48 bytes per Gaussian uncompressed

        Self {
            level_index,
            target_width,
            target_height,
            gaussians,
            psnr: 0.0, // Computed later
            size_bytes,
        }
    }

    /// Set quality score
    pub fn with_psnr(mut self, psnr: f32) -> Self {
        self.psnr = psnr;
        self
    }

    /// Get Gaussian count
    pub fn gaussian_count(&self) -> usize {
        self.gaussians.len()
    }

    /// Get scale factor relative to level 0
    pub fn scale_factor(&self) -> f32 {
        2.0f32.powi(self.level_index as i32)
    }
}
