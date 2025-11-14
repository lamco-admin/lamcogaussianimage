//! Comprehensive metrics collection during optimization
//!
//! Collects detailed statistics for analysis and visualization

use serde::{Serialize, Deserialize};
use std::time::{Duration, Instant};

/// Metrics collected during a single optimization iteration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationMetrics {
    pub iteration: usize,
    pub timestamp_ms: f64,

    // Loss components
    pub total_loss: f32,
    pub l2_loss: f32,
    pub ssim_loss: f32,

    // Gradient statistics
    pub grad_magnitude_mean: f32,
    pub grad_magnitude_max: f32,
    pub grad_magnitude_min: f32,

    // Per-parameter gradient norms
    pub grad_position_norm: f32,
    pub grad_scale_norm: f32,
    pub grad_rotation_norm: f32,
    pub grad_color_norm: f32,
    pub grad_opacity_norm: f32,

    // Gaussian statistics
    pub avg_opacity: f32,
    pub avg_scale: f32,
    pub num_active_gaussians: usize,  // Opacity > threshold

    // Rendering statistics
    pub render_time_ms: f32,
    pub gradient_time_ms: f32,
    pub update_time_ms: f32,
    pub total_iteration_time_ms: f32,

    // Quality metrics (if computed)
    pub psnr: Option<f32>,
    pub ssim_value: Option<f32>,
}

/// Complete optimization run metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMetrics {
    pub config_name: String,
    pub image_size: (u32, u32),
    pub num_gaussians: usize,
    pub max_iterations: usize,

    // Per-iteration data
    pub iterations: Vec<IterationMetrics>,

    // Final results
    pub final_loss: f32,
    pub final_psnr: f32,
    pub total_time_seconds: f32,
    pub converged_at_iteration: usize,

    // Convergence analysis
    pub convergence_rate: f32,  // Loss decrease per iteration
    pub iterations_to_90_percent: Option<usize>,  // 90% of final quality
}

/// Metrics collector
pub struct MetricsCollector {
    start_time: Instant,
    iterations: Vec<IterationMetrics>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            iterations: Vec::new(),
        }
    }

    /// Record iteration metrics
    pub fn record_iteration(&mut self, metrics: IterationMetrics) {
        self.iterations.push(metrics);
    }

    /// Finalize and return complete metrics
    pub fn finalize(
        self,
        config_name: String,
        image_size: (u32, u32),
        num_gaussians: usize,
        max_iterations: usize,
        final_psnr: f32,
    ) -> OptimizationMetrics {
        let total_time = self.start_time.elapsed().as_secs_f32();
        let final_loss = self.iterations.last().map(|i| i.total_loss).unwrap_or(f32::INFINITY);

        // Find convergence point (where loss stops decreasing significantly)
        let converged_at = find_convergence_point(&self.iterations);

        // Compute convergence rate
        let convergence_rate = if self.iterations.len() > 1 {
            let initial_loss = self.iterations[0].total_loss;
            (initial_loss - final_loss) / self.iterations.len() as f32
        } else {
            0.0
        };

        // Find 90% quality point
        let target_loss = final_loss + (self.iterations[0].total_loss - final_loss) * 0.1;
        let iterations_to_90 = self.iterations.iter()
            .position(|i| i.total_loss < target_loss);

        OptimizationMetrics {
            config_name,
            image_size,
            num_gaussians,
            max_iterations,
            iterations: self.iterations,
            final_loss,
            final_psnr,
            total_time_seconds: total_time,
            converged_at_iteration: converged_at,
            convergence_rate,
            iterations_to_90_percent: iterations_to_90,
        }
    }

    /// Export to JSON
    pub fn export_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.iterations)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Get iterations
    pub fn iterations(&self) -> &[IterationMetrics] {
        &self.iterations
    }

    /// Export to CSV
    pub fn export_csv(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut wtr = csv::Writer::from_path(path)?;

        // Headers
        wtr.write_record(&[
            "iteration", "timestamp_ms", "total_loss", "l2_loss", "ssim_loss",
            "grad_mean", "grad_max", "grad_min",
            "grad_position", "grad_scale", "grad_rotation", "grad_color", "grad_opacity",
            "avg_opacity", "avg_scale", "num_active",
            "render_ms", "gradient_ms", "update_ms", "total_ms",
            "psnr", "ssim",
        ])?;

        // Data
        for iter in &self.iterations {
            wtr.write_record(&[
                iter.iteration.to_string(),
                iter.timestamp_ms.to_string(),
                iter.total_loss.to_string(),
                iter.l2_loss.to_string(),
                iter.ssim_loss.to_string(),
                iter.grad_magnitude_mean.to_string(),
                iter.grad_magnitude_max.to_string(),
                iter.grad_magnitude_min.to_string(),
                iter.grad_position_norm.to_string(),
                iter.grad_scale_norm.to_string(),
                iter.grad_rotation_norm.to_string(),
                iter.grad_color_norm.to_string(),
                iter.grad_opacity_norm.to_string(),
                iter.avg_opacity.to_string(),
                iter.avg_scale.to_string(),
                iter.num_active_gaussians.to_string(),
                iter.render_time_ms.to_string(),
                iter.gradient_time_ms.to_string(),
                iter.update_time_ms.to_string(),
                iter.total_iteration_time_ms.to_string(),
                iter.psnr.map(|p| p.to_string()).unwrap_or_else(|| "".to_string()),
                iter.ssim_value.map(|s| s.to_string()).unwrap_or_else(|| "".to_string()),
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }
}

/// Find iteration where optimization converged
fn find_convergence_point(iterations: &[IterationMetrics]) -> usize {
    if iterations.len() < 10 {
        return iterations.len() - 1;
    }

    // Look for where loss stops decreasing significantly
    let window_size = 10;
    for i in window_size..iterations.len() {
        let recent_avg = iterations[(i - window_size)..i]
            .iter()
            .map(|m| m.total_loss)
            .sum::<f32>() / window_size as f32;

        let current_loss = iterations[i].total_loss;

        // If improvement < 1% over window, consider converged
        if (recent_avg - current_loss) / recent_avg < 0.01 {
            return i;
        }
    }

    iterations.len() - 1
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}
