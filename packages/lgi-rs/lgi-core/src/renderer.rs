//! Gaussian rendering to pixel buffers

use lgi_math::{
    Float, gaussian::Gaussian2D, parameterization::Parameterization,
    evaluation::GaussianEvaluator, compositing::{Compositor, AlphaMode},
    color::Color4, vec::Vector2,
};
use crate::{ImageBuffer, Result};

/// Rendering mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    /// Alpha compositing (standard, physically-based)
    AlphaComposite,
    /// Accumulated summation (GaussianImage ECCV 2024)
    AccumulatedSum,
}

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Rendering configuration
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Background color
    pub background: Color4<f32>,
    /// Alpha compositing mode (for AlphaComposite render mode)
    pub alpha_mode: AlphaMode,
    /// Rendering mode selection
    pub render_mode: RenderMode,
    /// Cutoff threshold for Gaussian evaluation
    pub cutoff_threshold: f32,
    /// Number of sigma for bounding box
    pub n_sigma: f32,
    /// Early termination threshold for alpha (AlphaComposite only)
    pub termination_threshold: f32,
    /// Use parallel rendering
    pub parallel: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            background: Color4::new(0.0, 0.0, 0.0, 0.0),
            alpha_mode: AlphaMode::Straight,
            render_mode: RenderMode::AlphaComposite,  // Default: standard alpha compositing
            cutoff_threshold: 1e-5,
            n_sigma: 3.5,
            termination_threshold: 0.999,
            parallel: true,
        }
    }
}

impl RenderConfig {
    /// Use accumulated summation rendering (GaussianImage ECCV 2024)
    pub fn with_accumulated_sum(mut self) -> Self {
        self.render_mode = RenderMode::AccumulatedSum;
        self
    }

    /// Use alpha compositing rendering (default)
    pub fn with_alpha_composite(mut self) -> Self {
        self.render_mode = RenderMode::AlphaComposite;
        self
    }
}

/// Gaussian renderer
pub struct Renderer {
    config: RenderConfig,
    evaluator: GaussianEvaluator<f32>,
    compositor: Compositor<f32>,
}

impl Renderer {
    /// Create new renderer with default config
    pub fn new() -> Self {
        Self::with_config(RenderConfig::default())
    }

    /// Create renderer with custom config
    pub fn with_config(config: RenderConfig) -> Self {
        let evaluator = GaussianEvaluator::new(config.cutoff_threshold, config.n_sigma);
        let compositor = Compositor::new(config.alpha_mode)
            .with_termination(config.termination_threshold);

        Self {
            config,
            evaluator,
            compositor,
        }
    }

    /// Render Gaussians to image buffer (dispatches based on render_mode)
    pub fn render_basic<P: Parameterization<f32>>(
        &self,
        gaussians: &[Gaussian2D<f32, P>],
        width: u32,
        height: u32,
    ) -> Result<ImageBuffer<f32>> {
        match self.config.render_mode {
            RenderMode::AlphaComposite => self.render_alpha_composite(gaussians, width, height),
            RenderMode::AccumulatedSum => self.render_accumulated_sum(gaussians, width, height),
        }
    }

    /// Render using alpha compositing (standard method)
    fn render_alpha_composite<P: Parameterization<f32>>(
        &self,
        gaussians: &[Gaussian2D<f32, P>],
        width: u32,
        height: u32,
    ) -> Result<ImageBuffer<f32>> {
        let mut buffer = ImageBuffer::new(width, height);

        // For each pixel
        for y in 0..height {
            for x in 0..width {
                let point = Vector2::new(
                    x as f32 / width as f32,
                    y as f32 / height as f32,
                );

                let mut accum_color = Color4::new(0.0, 0.0, 0.0, 0.0);
                let mut accum_alpha = 0.0;

                // Composite all Gaussians
                for gaussian in gaussians {
                    // Bounding box check
                    if let Some(weight) = self.evaluator.evaluate_bounded(gaussian, point) {
                        if self.compositor.composite_over(
                            &mut accum_color,
                            &mut accum_alpha,
                            gaussian.color,
                            gaussian.opacity,
                            weight,
                        ) {
                            break; // Early termination
                        }
                    }
                }

                // Blend with background
                let final_color = self.compositor.blend_background(
                    accum_color,
                    accum_alpha,
                    self.config.background,
                );

                buffer.set_pixel(x, y, final_color);
            }
        }

        Ok(buffer)
    }

    /// Render using accumulated summation (GaussianImage ECCV 2024 method)
    ///
    /// Direct accumulation without alpha tracking:
    /// pixel_color += gaussian.color × gaussian.opacity × weight
    ///
    /// Simpler than alpha compositing, may provide better PSNR for some images
    fn render_accumulated_sum<P: Parameterization<f32>>(
        &self,
        gaussians: &[Gaussian2D<f32, P>],
        width: u32,
        height: u32,
    ) -> Result<ImageBuffer<f32>> {
        let mut buffer = ImageBuffer::new(width, height);

        // For each pixel
        for y in 0..height {
            for x in 0..width {
                let point = Vector2::new(
                    x as f32 / width as f32,
                    y as f32 / height as f32,
                );

                let mut color_sum = Color4::new(0.0, 0.0, 0.0, 0.0);

                // Accumulate contributions from all Gaussians
                for gaussian in gaussians {
                    if let Some(weight) = self.evaluator.evaluate_bounded(gaussian, point) {
                        // Direct accumulation (no alpha tracking)
                        let contribution = weight * gaussian.opacity;
                        color_sum.r += gaussian.color.r * contribution;
                        color_sum.g += gaussian.color.g * contribution;
                        color_sum.b += gaussian.color.b * contribution;
                    }
                }

                // Clamp to [0, 1] range
                color_sum.r = color_sum.r.clamp(0.0, 1.0);
                color_sum.g = color_sum.g.clamp(0.0, 1.0);
                color_sum.b = color_sum.b.clamp(0.0, 1.0);
                color_sum.a = 1.0;

                buffer.set_pixel(x, y, color_sum);
            }
        }

        Ok(buffer)
    }

    /// Render Gaussians (parallel, multi-threaded)
    #[cfg(feature = "rayon")]
    pub fn render_parallel<P: Parameterization<f32> + Send + Sync>(
        &self,
        gaussians: &[Gaussian2D<f32, P>],
        width: u32,
        height: u32,
    ) -> Result<ImageBuffer<f32>> {
        let mut buffer = ImageBuffer::new(width, height);

        // Parallel processing by scanline
        let pixels: Vec<Color4<f32>> = (0..height)
            .into_par_iter()
            .flat_map(|y| {
                (0..width).into_par_iter().map(move |x| {
                    let point = Vector2::new(
                        x as f32 / width as f32,
                        y as f32 / height as f32,
                    );

                    let mut accum_color = Color4::new(0.0, 0.0, 0.0, 0.0);
                    let mut accum_alpha = 0.0;

                    for gaussian in gaussians {
                        if let Some(weight) = self.evaluator.evaluate_bounded(gaussian, point) {
                            if self.compositor.composite_over(
                                &mut accum_color,
                                &mut accum_alpha,
                                gaussian.color,
                                gaussian.opacity,
                                weight,
                            ) {
                                break;
                            }
                        }
                    }

                    self.compositor.blend_background(accum_color, accum_alpha, self.config.background)
                })
            })
            .collect();

        buffer.data = pixels;
        Ok(buffer)
    }

    /// Render with automatic parallel/sequential selection
    pub fn render<P: Parameterization<f32> + Send + Sync>(
        &self,
        gaussians: &[Gaussian2D<f32, P>],
        width: u32,
        height: u32,
    ) -> Result<ImageBuffer<f32>> {
        #[cfg(feature = "rayon")]
        if self.config.parallel {
            return self.render_parallel(gaussians, width, height);
        }

        self.render_basic(gaussians, width, height)
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::parameterization::Euler;

    #[test]
    fn test_render_single_gaussian() {
        let renderer = Renderer::new();

        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.1),
            Color4::rgb(1.0, 0.0, 0.0),
            1.0,
        );

        let buffer = renderer.render(&[gaussian], 100, 100).unwrap();

        // Center pixel should be red
        let center = buffer.get_pixel(50, 50).unwrap();
        assert!(center.r > 0.9); // Nearly full red
        assert!(center.g < 0.1);
        assert!(center.b < 0.1);
    }

    #[test]
    fn test_render_multiple_gaussians() {
        let renderer = Renderer::new();

        let gaussians = vec![
            Gaussian2D::new(
                Vector2::new(0.3, 0.3),
                Euler::isotropic(0.1),
                Color4::rgb(1.0, 0.0, 0.0),
                0.8,
            ),
            Gaussian2D::new(
                Vector2::new(0.7, 0.7),
                Euler::isotropic(0.1),
                Color4::rgb(0.0, 0.0, 1.0),
                0.8,
            ),
        ];

        let buffer = renderer.render(&gaussians, 100, 100).unwrap();

        // Check that Gaussians rendered in different regions
        let top_left = buffer.get_pixel(30, 30).unwrap();
        let bottom_right = buffer.get_pixel(70, 70).unwrap();

        assert!(top_left.r > top_left.b); // More red
        assert!(bottom_right.b > bottom_right.r); // More blue
    }
}
