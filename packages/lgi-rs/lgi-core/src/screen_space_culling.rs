//! Screen-Space Culling
//! Cull sub-pixel Gaussians for view-dependent rendering

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

/// Screen-space culler
pub struct ScreenSpaceCuller {
    /// Minimum screen-space σ (pixels)
    pub min_sigma_px: f32,
}

impl Default for ScreenSpaceCuller {
    fn default() -> Self {
        Self { min_sigma_px: 0.25 }  // From spec
    }
}

impl ScreenSpaceCuller {
    pub fn new(min_sigma_px: f32) -> Self {
        Self { min_sigma_px }
    }

    /// Select visible Gaussians for viewport
    pub fn select_visible<'a>(
        &self,
        gaussians: &'a [Gaussian2D<f32, Euler<f32>>],
        viewport_width: u32,
        viewport_height: u32,
        zoom: f32,
        dpr: f32,
    ) -> Vec<&'a Gaussian2D<f32, Euler<f32>>> {
        gaussians.iter().filter(|g| {
            let sx_screen = g.shape.scale_x * viewport_width as f32 * zoom * dpr;
            let sy_screen = g.shape.scale_y * viewport_height as f32 * zoom * dpr;
            let sigma_min = sx_screen.min(sy_screen);
            sigma_min >= self.min_sigma_px
        }).collect()
    }

    /// Count culled Gaussians
    pub fn count_culled(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        viewport_width: u32,
        viewport_height: u32,
        zoom: f32,
        dpr: f32,
    ) -> usize {
        let total = gaussians.len();
        let visible = self.select_visible(gaussians, viewport_width, viewport_height, zoom, dpr).len();
        total - visible
    }
}
