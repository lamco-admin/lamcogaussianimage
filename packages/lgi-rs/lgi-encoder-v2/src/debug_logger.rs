//! Visual Debugging Logger for Optimization
//!
//! Outputs:
//! - Rendered images per iteration
//! - Error heatmaps
//! - Gaussian visualizations (ellipse overlays)
//! - Metrics CSV (iteration, N, loss, PSNR)
//! - Side-by-side comparison grids

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, color::Color4};
use crate::renderer_v2::RendererV2;
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::Write;

/// Debug output configuration
#[derive(Clone)]
pub struct DebugConfig {
    /// Output directory for debug files
    pub output_dir: PathBuf,

    /// Save every N iterations (e.g., 10 = save at 0, 10, 20, ...)
    pub save_every_n_iters: usize,

    /// Save rendered images
    pub save_rendered: bool,

    /// Save error heatmaps
    pub save_error_maps: bool,

    /// Save Gaussian visualizations
    pub save_gaussian_viz: bool,

    /// Save side-by-side comparisons
    pub save_comparison: bool,

    /// Save metrics to CSV
    pub save_metrics_csv: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("debug_output"),
            save_every_n_iters: 10,
            save_rendered: true,
            save_error_maps: true,
            save_gaussian_viz: true,
            save_comparison: true,
            save_metrics_csv: true,
        }
    }
}

/// Visual debugging logger
pub struct DebugLogger {
    config: DebugConfig,
    csv_file: Option<File>,
    iteration_count: usize,
}

impl DebugLogger {
    /// Create new debug logger with config
    pub fn new(config: DebugConfig) -> std::io::Result<Self> {
        // Create output directory
        fs::create_dir_all(&config.output_dir)?;

        // Create CSV file with header
        let csv_file = if config.save_metrics_csv {
            let path = config.output_dir.join("metrics.csv");
            let mut file = File::create(&path)?;
            writeln!(file, "iteration,pass,n_gaussians,loss,psnr,time_ms")?;
            Some(file)
        } else {
            None
        };

        Ok(Self {
            config,
            csv_file,
            iteration_count: 0,
        })
    }

    /// Log single optimization iteration
    pub fn log_iteration(
        &mut self,
        iteration: usize,
        pass: usize,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
        loss: f32,
        psnr: f32,
        elapsed_ms: u64,
    ) -> std::io::Result<()> {
        // Write metrics to CSV always (cheap)
        if let Some(ref mut csv) = self.csv_file {
            writeln!(csv, "{},{},{},{:.6},{:.2},{}",
                iteration, pass, gaussians.len(), loss, psnr, elapsed_ms)?;
            csv.flush()?;
        }

        // Save visual outputs only every N iterations
        if iteration % self.config.save_every_n_iters != 0 {
            return Ok(());
        }

        let rendered = RendererV2::render(gaussians, target.width, target.height);

        // 1. Rendered image
        if self.config.save_rendered {
            let path = self.config.output_dir.join(format!("iter_{:04}_rendered.png", iteration));
            save_image_as_png(&rendered, &path)?;
        }

        // 2. Error heatmap
        if self.config.save_error_maps {
            let error_map = compute_error_heatmap(target, &rendered);
            let path = self.config.output_dir.join(format!("iter_{:04}_error.png", iteration));
            save_image_as_png(&error_map, &path)?;
        }

        // 3. Gaussian visualization
        if self.config.save_gaussian_viz {
            let gaussian_viz = visualize_gaussians(&rendered, gaussians, target.width, target.height);
            let path = self.config.output_dir.join(format!("iter_{:04}_gaussians.png", iteration));
            save_image_as_png(&gaussian_viz, &path)?;
        }

        // 4. Side-by-side comparison
        if self.config.save_comparison {
            let comparison = create_comparison_grid(
                target,
                &rendered,
                &compute_error_heatmap(target, &rendered),
                gaussians,
                iteration,
                psnr,
                loss,
            );
            let path = self.config.output_dir.join(format!("iter_{:04}_comparison.png", iteration));
            save_image_as_png(&comparison, &path)?;
        }

        self.iteration_count += 1;
        Ok(())
    }

    /// Get total iterations logged
    pub fn iteration_count(&self) -> usize {
        self.iteration_count
    }
}

/// Compute error heatmap (red = high error, blue = low)
fn compute_error_heatmap(target: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> ImageBuffer<f32> {
    let mut heatmap = ImageBuffer::new(target.width, target.height);

    for y in 0..target.height {
        for x in 0..target.width {
            let t = target.get_pixel(x, y).unwrap();
            let r = rendered.get_pixel(x, y).unwrap();

            // L1 error per pixel
            let error = ((t.r - r.r).abs() + (t.g - r.g).abs() + (t.b - r.b).abs()) / 3.0;

            // Map error to color (0-0.5 error range)
            let normalized_error = (error * 2.0).min(1.0);

            // Blue (low) → Green (mid) → Red (high)
            let color = if normalized_error < 0.5 {
                // Blue → Green
                let t = normalized_error * 2.0;
                Color4::new(0.0, t, 1.0 - t, 1.0)
            } else {
                // Green → Red
                let t = (normalized_error - 0.5) * 2.0;
                Color4::new(t, 1.0 - t, 0.0, 1.0)
            };

            heatmap.set_pixel(x, y, color);
        }
    }

    heatmap
}

/// Visualize Gaussians as ellipse overlays on rendered image
fn visualize_gaussians(
    base_image: &ImageBuffer<f32>,
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    width: u32,
    height: u32,
) -> ImageBuffer<f32> {
    let mut viz = base_image.clone();

    // Draw each Gaussian as colored ellipse
    for (i, gaussian) in gaussians.iter().enumerate() {
        let px = (gaussian.position.x * width as f32) as u32;
        let py = (gaussian.position.y * height as f32) as u32;

        // Color based on index (cycling through colors)
        let hue = (i as f32 * 137.5) % 360.0;  // Golden angle spacing
        let color = hue_to_rgb(hue);

        // Draw small dot at Gaussian center
        if px < width && py < height {
            viz.set_pixel(px, py, Color4::new(color.0, color.1, color.2, 1.0));
        }

        // TODO: Draw actual ellipse outline (needs imageproc drawing)
        // For now just centers - will add ellipse drawing later
    }

    viz
}

/// Convert HSV hue (0-360) to RGB
fn hue_to_rgb(hue: f32) -> (f32, f32, f32) {
    let h = hue / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();

    let (r, g, b) = match h as i32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };

    (r, g, b)
}

/// Create 2x3 comparison grid
fn create_comparison_grid(
    target: &ImageBuffer<f32>,
    rendered: &ImageBuffer<f32>,
    error_map: &ImageBuffer<f32>,
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    iteration: usize,
    psnr: f32,
    loss: f32,
) -> ImageBuffer<f32> {
    let w = target.width;
    let h = target.height;

    // 2×3 grid with labels
    let grid_width = w * 3;
    let grid_height = h * 2 + 60;  // Extra for text labels

    let mut grid = ImageBuffer::new(grid_width, grid_height);

    // Fill background
    for y in 0..grid_height {
        for x in 0..grid_width {
            grid.set_pixel(x, y, Color4::new(0.1, 0.1, 0.1, 1.0));
        }
    }

    // Row 1: Target | Rendered | Error
    copy_region(&mut grid, target, 0, 30, w, h);
    copy_region(&mut grid, rendered, w, 30, w, h);
    copy_region(&mut grid, error_map, w * 2, 30, w, h);

    // Row 2: Gaussians | Info Panel | Stats
    let gaussian_viz = visualize_gaussians(rendered, gaussians, w, h);
    copy_region(&mut grid, &gaussian_viz, 0, h + 30, w, h);

    // TODO: Add text labels with imageproc::drawing::text
    // For now, grid structure is set up

    grid
}

/// Copy image region to target buffer
fn copy_region(
    target: &mut ImageBuffer<f32>,
    source: &ImageBuffer<f32>,
    offset_x: u32,
    offset_y: u32,
    width: u32,
    height: u32,
) {
    for y in 0..height.min(source.height) {
        for x in 0..width.min(source.width) {
            if let Some(pixel) = source.get_pixel(x, y) {
                let tx = offset_x + x;
                let ty = offset_y + y;
                if tx < target.width && ty < target.height {
                    target.set_pixel(tx, ty, pixel);
                }
            }
        }
    }
}

/// Save ImageBuffer as PNG
fn save_image_as_png(image: &ImageBuffer<f32>, path: &Path) -> std::io::Result<()> {
    let width = image.width;
    let height = image.height;

    // Convert f32 [0,1] to u8 [0,255]
    let mut rgb_buffer = image::RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y).unwrap();

            let r = (pixel.r * 255.0).clamp(0.0, 255.0) as u8;
            let g = (pixel.g * 255.0).clamp(0.0, 255.0) as u8;
            let b = (pixel.b * 255.0).clamp(0.0, 255.0) as u8;

            rgb_buffer.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }

    rgb_buffer.save(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_logger_creation() {
        let config = DebugConfig {
            output_dir: PathBuf::from("/tmp/test_debug"),
            save_every_n_iters: 5,
            ..Default::default()
        };

        let logger = DebugLogger::new(config);
        assert!(logger.is_ok());
    }
}
