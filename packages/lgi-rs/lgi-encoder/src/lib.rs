//! # LGI Encoder
//!
//! Gaussian fitting and optimization for creating LGI images.
//!
//! The encoder performs iterative optimization to fit a set of 2D Gaussians
//! to a target image, minimizing reconstruction error.

#![warn(missing_docs)]

pub mod optimizer;
pub mod optimizer_v2;
pub mod loss;
pub mod config;
pub mod autodiff;
pub mod metrics_collector;
pub mod adaptive;
pub mod lr_scaling;
pub mod vector_quantization;
pub mod adaptive_count;

pub use optimizer::{Optimizer, OptimizerState};
pub use optimizer_v2::OptimizerV2;
pub use loss::{LossFunction, LossFunctions, L2Loss, SSIMLoss};
pub use config::EncoderConfig;
pub use autodiff::{FullGaussianGradient, compute_full_gradients};
pub use metrics_collector::{MetricsCollector, IterationMetrics, OptimizationMetrics};
pub use adaptive::{AdaptiveThresholdController, LifecycleManager};
pub use vector_quantization::{VectorQuantizer, GaussianVector};
pub use adaptive_count::{estimate_gaussian_count, QualityTarget};
use lgi_core::{Result, LgiError, Initializer, InitStrategy, ImageBuffer};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

/// LGI Encoder
pub struct Encoder {
    config: EncoderConfig,
}

impl Encoder {
    /// Create new encoder with default configuration
    pub fn new() -> Self {
        Self::with_config(EncoderConfig::default())
    }

    /// Create encoder with custom configuration
    pub fn with_config(config: EncoderConfig) -> Self {
        Self { config }
    }

    /// Encode an image to Gaussians
    ///
    /// This is the main entry point for encoding.
    pub fn encode(
        &self,
        target: &ImageBuffer<f32>,
        num_gaussians: usize,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        // Step 1: Initialize Gaussians
        println!("Initializing {} Gaussians using {:?} strategy...", num_gaussians, self.config.init_strategy);
        let initializer = Initializer::new(self.config.init_strategy)
            .with_scale(self.config.initial_scale);

        let mut gaussians = initializer.initialize(target, num_gaussians)?;

        // Step 2: Optimize
        println!("Optimizing Gaussians...");
        let optimizer = Optimizer::new(self.config.clone());
        let final_gaussians = optimizer.optimize(&mut gaussians, target)?;

        println!("Encoding complete!");
        Ok(final_gaussians)
    }

    /// Encode with progress callback
    pub fn encode_with_progress<F>(
        &self,
        target: &ImageBuffer<f32>,
        num_gaussians: usize,
        mut progress_callback: F,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>>
    where
        F: FnMut(usize, f32) // (iteration, loss)
    {
        let initializer = Initializer::new(self.config.init_strategy)
            .with_scale(self.config.initial_scale);

        let mut gaussians = initializer.initialize(target, num_gaussians)?;

        let optimizer = Optimizer::new(self.config.clone());
        let final_gaussians = optimizer.optimize_with_callback(
            &mut gaussians,
            target,
            &mut progress_callback,
        )?;

        Ok(final_gaussians)
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}
