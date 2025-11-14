//! Viewport frustum culling

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

pub struct ViewportCuller;

impl ViewportCuller {
    pub fn cull_outside_viewport<'a>(
        gaussians: &'a [Gaussian2D<f32, Euler<f32>>],
        viewport_x: f32,
        viewport_y: f32,
        viewport_w: f32,
        viewport_h: f32,
        margin: f32,  // Extra margin for Gaussian footprints
    ) -> Vec<&'a Gaussian2D<f32, Euler<f32>>> {
        gaussians.iter().filter(|g| {
            let gx = g.position.x;
            let gy = g.position.y;
            let radius = (g.shape.scale_x.max(g.shape.scale_y)) * 3.5 + margin;

            gx + radius >= viewport_x &&
            gx - radius <= viewport_x + viewport_w &&
            gy + radius >= viewport_y &&
            gy - radius <= viewport_y + viewport_h
        }).collect()
    }
}
