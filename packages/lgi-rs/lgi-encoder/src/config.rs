//! Encoder configuration

use lgi_core::InitStrategy;
use serde::{Deserialize, Serialize};

/// Encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Initialization strategy
    pub init_strategy: InitStrategy,

    /// Initial Gaussian scale (normalized coordinates)
    /// This should be set automatically based on image size and Gaussian count
    pub initial_scale: f32,

    /// Number of optimization iterations
    pub max_iterations: usize,

    /// Learning rate for position
    pub lr_position: f32,

    /// Learning rate for scale
    pub lr_scale: f32,

    /// Learning rate for rotation
    pub lr_rotation: f32,

    /// Learning rate for color
    pub lr_color: f32,

    /// Learning rate for opacity
    pub lr_opacity: f32,

    /// Loss function weights
    pub loss_l2_weight: f32,
    pub loss_ssim_weight: f32,

    /// Convergence tolerance
    pub convergence_tolerance: f32,

    /// Early stopping patience (iterations without improvement)
    pub early_stopping_patience: usize,

    /// Learning rate decay factor
    pub lr_decay: f32,

    /// Learning rate decay steps
    pub lr_decay_steps: usize,

    /// Enable Quantization-Aware (QA) training (from GaussianImage ECCV 2024)
    pub enable_qa_training: bool,

    /// Iteration to start QA training (typically 70% through optimization)
    pub qa_start_iteration: usize,

    /// VQ codebook size for QA training (256 = 8-bit indices)
    pub qa_codebook_size: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            init_strategy: InitStrategy::Gradient,
            initial_scale: 0.3,  // 30% of image - needed for proper coverage
            max_iterations: 2000,
            lr_position: 0.01,
            lr_scale: 0.005,
            lr_rotation: 0.005,
            lr_color: 0.01,
            lr_opacity: 0.01,
            loss_l2_weight: 0.8,
            loss_ssim_weight: 0.2,
            convergence_tolerance: 1e-6,
            early_stopping_patience: 100,
            lr_decay: 0.1,
            lr_decay_steps: 500,
            enable_qa_training: false,
            qa_start_iteration: 1400,  // 70% of 2000 iterations
            qa_codebook_size: 256,
        }
    }
}

impl EncoderConfig {
    /// Preset: Fast (fewer iterations, lower quality)
    pub fn fast() -> Self {
        Self {
            max_iterations: 500,
            early_stopping_patience: 50,
            ..Default::default()
        }
    }

    /// Preset: Balanced (default)
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Preset: High quality (more iterations)
    pub fn high_quality() -> Self {
        Self {
            max_iterations: 5000,
            early_stopping_patience: 200,
            lr_decay_steps: 1000,
            ..Default::default()
        }
    }

    /// Preset: Ultra (maximum quality, slow)
    pub fn ultra() -> Self {
        Self {
            max_iterations: 10000,
            early_stopping_patience: 500,
            lr_position: 0.005,
            lr_decay_steps: 2000,
            convergence_tolerance: 1e-8,
            ..Default::default()
        }
    }

    /// Calculate appropriate initial scale based on image dimensions and Gaussian count
    ///
    /// Rule of thumb: Gaussians should cover sqrt(pixels/gaussians) area
    /// For 256x256 image with 100 Gaussians: each covers ~256 pixels = 16x16 = scale of 0.06
    pub fn compute_initial_scale(width: u32, height: u32, num_gaussians: usize) -> f32 {
        let pixels_per_gaussian = (width * height) as f32 / num_gaussians.max(1) as f32;
        let coverage_radius = (pixels_per_gaussian.sqrt() / width.min(height) as f32) * 2.0;

        // Clamp to reasonable range (5-30% of image)
        coverage_radius.max(0.05).min(0.3)
    }
}
