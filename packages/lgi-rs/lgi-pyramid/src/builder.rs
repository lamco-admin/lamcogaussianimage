//! Pyramid builder

use crate::{GaussianPyramid, PyramidLevel, Result};
use lgi_core::{ImageBuffer, Initializer, Renderer};
use lgi_encoder::{EncoderConfig, OptimizerV2};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

/// Builder for multi-level Gaussian pyramids
pub struct PyramidBuilder {
    num_levels: usize,
    encoder_config: EncoderConfig,
    gaussian_density: f32, // Gaussians per pixel
}

impl Default for PyramidBuilder {
    fn default() -> Self {
        Self {
            num_levels: 4,
            encoder_config: EncoderConfig::balanced(),
            gaussian_density: 0.015, // ~1.5% of pixels
        }
    }
}

impl PyramidBuilder {
    /// Create new builder with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of pyramid levels
    pub fn num_levels(mut self, levels: usize) -> Self {
        self.num_levels = levels.max(1).min(8);
        self
    }

    /// Set encoder configuration
    pub fn encoder_config(mut self, config: EncoderConfig) -> Self {
        self.encoder_config = config;
        self
    }

    /// Set Gaussian density (Gaussians per pixel)
    pub fn gaussian_density(mut self, density: f32) -> Self {
        self.gaussian_density = density;
        self
    }

    /// Build pyramid from image
    pub fn build(&self, image: &ImageBuffer<f32>) -> Result<GaussianPyramid> {
        println!("Building {}-level pyramid for {}×{} image...",
            self.num_levels, image.width, image.height);

        let mut levels = Vec::new();

        for level_idx in 0..self.num_levels {
            let scale_factor = 2u32.pow(level_idx as u32);
            let level_width = (image.width / scale_factor).max(16);
            let level_height = (image.height / scale_factor).max(16);

            println!("  Level {}: {}×{}", level_idx, level_width, level_height);

            // Downsample image for this level
            let downsampled = if level_idx == 0 {
                image.clone()
            } else {
                downsample_image(image, level_width, level_height)
            };

            // Calculate Gaussian count for this level
            let gaussian_count = ((level_width * level_height) as f32 * self.gaussian_density) as usize;
            let gaussian_count = gaussian_count.max(50).min(50000);

            // Optimize Gaussians for this resolution
            let gaussians = self.optimize_for_level(&downsampled, gaussian_count)?;

            // Measure quality
            let psnr = measure_psnr(&gaussians, &downsampled)?;

            println!("    Gaussians: {}, PSNR: {:.2} dB", gaussians.len(), psnr);

            levels.push(PyramidLevel::new(
                level_idx,
                level_width,
                level_height,
                gaussians,
            ).with_psnr(psnr));
        }

        println!("✅ Pyramid built successfully!");

        Ok(GaussianPyramid::new(levels, image.width, image.height))
    }

    /// Optimize Gaussians for specific level
    fn optimize_for_level(
        &self,
        target: &ImageBuffer<f32>,
        gaussian_count: usize,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        // Initialize
        let initializer = Initializer::new(self.encoder_config.init_strategy)
            .with_scale(self.encoder_config.initial_scale);
        let mut gaussians = initializer.initialize(target, gaussian_count)?;

        // Optimize
        let optimizer = OptimizerV2::new(self.encoder_config.clone());
        let _metrics = optimizer.optimize_with_metrics(&mut gaussians, target)?;

        Ok(gaussians)
    }
}

/// Downsample image to target resolution
fn downsample_image(
    source: &ImageBuffer<f32>,
    target_width: u32,
    target_height: u32,
) -> ImageBuffer<f32> {
    let mut target = ImageBuffer::new(target_width, target_height);

    let x_ratio = source.width as f32 / target_width as f32;
    let y_ratio = source.height as f32 / target_height as f32;

    for ty in 0..target_height {
        for tx in 0..target_width {
            // Simple box filter (average source pixels)
            let sx = (tx as f32 * x_ratio) as u32;
            let sy = (ty as f32 * y_ratio) as u32;

            if let Some(pixel) = source.get_pixel(sx.min(source.width - 1), sy.min(source.height - 1)) {
                target.set_pixel(tx, ty, pixel);
            }
        }
    }

    target
}

/// Measure PSNR between Gaussians and target
fn measure_psnr(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
) -> Result<f32> {
    let renderer = Renderer::new();
    let rendered = renderer.render(gaussians, target.width, target.height)?;

    // Compute MSE
    let mut mse = 0.0;
    let count = (target.width * target.height * 3) as f32;

    for (p1, p2) in rendered.data.iter().zip(target.data.iter()) {
        mse += (p1.r - p2.r) * (p1.r - p2.r);
        mse += (p1.g - p2.g) * (p1.g - p2.g);
        mse += (p1.b - p2.b) * (p1.b - p2.b);
    }

    mse /= count;

    if mse < 1e-10 {
        Ok(100.0)
    } else {
        Ok(20.0 * (1.0f32 / mse.sqrt()).log10())
    }
}
