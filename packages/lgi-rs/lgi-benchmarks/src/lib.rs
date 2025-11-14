//! Comprehensive benchmarking suite for LGI codec
//!
//! This library provides:
//! - Test image generation (synthetic patterns)
//! - Standard dataset integration (Kodak, DIV2K)
//! - Quality metrics (PSNR, SSIM, MS-SSIM, perceptual)
//! - Performance benchmarking (encoding, decoding, scaling)
//! - Automated test runners

pub mod test_images;
pub mod metrics;
pub mod dataset;
pub mod benchmark_runner;
pub mod comprehensive_suite;

// Re-export commonly used items
pub use test_images::{TestImageGenerator, TestPattern};
pub use metrics::{QualityMetrics, compute_psnr, compute_ssim, compute_all_metrics};
pub use dataset::DatasetManager;
pub use benchmark_runner::{BenchmarkRunner, BenchmarkConfig, BenchmarkResult};
