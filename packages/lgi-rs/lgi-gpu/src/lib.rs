//! # LGI GPU Rendering
//!
//! High-performance GPU-accelerated rendering for LGI and LGIV formats.
//!
//! ## Features
//!
//! - **Cross-platform**: wgpu supports Vulkan, DirectX 12, Metal, WebGPU
//! - **Auto-detection**: Automatically selects best available backend
//! - **Cutting-edge**: Supports latest GPU features with graceful fallback
//! - **Unified**: Same renderer for LGI (images) and LGIV (video frames)
//! - **Performance**: 1000+ FPS @ 1080p on modern GPUs

#![warn(missing_docs)]

pub mod error;
pub mod backend;
pub mod capabilities;
pub mod buffer;
pub mod pipeline;
pub mod renderer;

pub use error::{GpuError, Result};
pub use capabilities::{BackendInfo, GpuCapabilities, FeatureLevel, BackendType};
pub use renderer::GpuRenderer;

/// Re-export RenderMode from lgi-core
pub use lgi_core::RenderMode;
pub mod manager;
pub use manager::GpuManager;
pub mod gradient;
pub use gradient::{GpuGradientComputer, GpuGradient};

pub mod adam_optimizer;
pub use adam_optimizer::{GpuAdamOptimizer, GpuLearningRates};

pub mod lbfgs_optimizer;
pub use lbfgs_optimizer::{GpuLbfgsOptimizer, LbfgsConfig};

pub mod lm_optimizer;
pub use lm_optimizer::{GpuLMOptimizer, LMConfig};

pub mod jtj_strategy;
pub use jtj_strategy::{
    JtJStrategy, JtJContext, JtJResult, ComplexityClass, SparsityInfo,
    BlockDiagonalJtJ, FullDenseJtJ, SparseJtJ, HierarchicalJtJ, MatrixFreeJtJ,
    JtJStrategyRegistry,
};
