//! Global GPU Manager - Single GPU instance shared across entire application
//! Copyright (c) 2025 Lamco Development

use crate::{GpuRenderer, GpuError, GpuGradientComputer, GpuGradient};
use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};
use std::sync::Arc;
use parking_lot::Mutex;
use once_cell::sync::Lazy;

/// Global GPU manager singleton
static GPU_MANAGER: Lazy<GpuManager> = Lazy::new(|| {
    GpuManager {
        renderer: Arc::new(Mutex::new(None)),
        gradient_computer: Arc::new(Mutex::new(None)),
    }
});

/// GPU Manager - provides shared access to single GPU instance
pub struct GpuManager {
    renderer: Arc<Mutex<Option<GpuRenderer>>>,
    gradient_computer: Arc<Mutex<Option<GpuGradientComputer>>>,
}

impl GpuManager {
    /// Get the global GPU manager instance
    pub fn global() -> &'static GpuManager {
        &GPU_MANAGER
    }

    /// Initialize GPU if not already initialized
    pub async fn initialize(&self) -> Result<(), GpuError> {
        let mut renderer_lock = self.renderer.lock();

        if renderer_lock.is_none() {
            tracing::info!("🎮 Initializing global GPU instance...");
            let gpu = GpuRenderer::new().await?;

            // Initialize gradient computer with shared device
            let gradient_computer = GpuGradientComputer::new(gpu.device())?;

            tracing::info!("✅ Global GPU initialized (renderer + gradient computer)");

            *self.gradient_computer.lock() = Some(gradient_computer);
            *renderer_lock = Some(gpu);
        } else {
            tracing::debug!("GPU already initialized, reusing instance");
        }

        Ok(())
    }

    /// Execute a render operation with the GPU
    pub fn render<F, R>(&self, f: F) -> Result<R, GpuError>
    where
        F: FnOnce(&mut GpuRenderer) -> Result<R, GpuError>,
    {
        let mut renderer_lock = self.renderer.lock();
        
        match renderer_lock.as_mut() {
            Some(gpu) => f(gpu),
            None => Err(GpuError::NotInitialized),
        }
    }

    /// Check if GPU is initialized
    pub fn is_initialized(&self) -> bool {
        self.renderer.lock().is_some()
    }

    /// Get GPU info (adapter name, backend, etc.)
    pub fn get_info(&self) -> Option<String> {
        let renderer_lock = self.renderer.lock();

        renderer_lock.as_ref().map(|gpu| {
            // TODO: Add method to GpuRenderer to get adapter info
            "NVIDIA GeForce RTX 4060 (Vulkan)".to_string()
        })
    }

    /// Compute gradients using GPU
    pub fn compute_gradients(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> Result<Vec<GpuGradient>, GpuError> {
        let renderer_lock = self.renderer.lock();
        let gradient_lock = self.gradient_computer.lock();

        match (renderer_lock.as_ref(), gradient_lock.as_ref()) {
            (Some(renderer), Some(gradient_computer)) => {
                gradient_computer.compute_gradients(
                    renderer.device(),
                    renderer.queue(),
                    gaussians,
                    rendered,
                    target,
                )
            }
            _ => Err(GpuError::NotInitialized),
        }
    }
}
