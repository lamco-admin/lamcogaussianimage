//! # LGI Core Library
//!
//! Core encoding and decoding primitives for the LGI format.
//!
//! This crate provides:
//! - Gaussian initialization strategies
//! - Rendering to pixel buffers
//! - Tiling and spatial data structures
//! - Level-of-detail hierarchy
//! - Energy-based ordering

#![warn(missing_docs)]

pub mod error;
pub mod image_buffer;
pub mod initializer;
pub mod renderer;
pub mod tiling;
pub mod ordering;
pub mod entropy;
pub mod structure_tensor;  // NEW: Structure tensor for LGI v2
pub mod gaussian_filter;   // NEW: DoG filters for structure tensor
pub mod geodesic_edt;      // NEW: Geodesic distance transform (anti-bleeding)
pub mod ms_ssim;           // NEW: MS-SSIM perceptual loss
pub mod guided_filter;     // NEW: Edge-preserving smoothing (CRITICAL for photos)
pub mod texture_map;       // NEW: Per-primitive texture mapping for fine detail
pub mod textured_gaussian; // NEW: Textured Gaussian primitives
pub mod blue_noise_residual; // NEW: Procedural micro-detail encoding (90% compression)

pub use error::{LgiError, Result};
pub use image_buffer::ImageBuffer;
pub use initializer::{Initializer, InitStrategy};
pub use renderer::{Renderer, RenderConfig, RenderMode};
pub use entropy::{adaptive_gaussian_count, compute_image_entropy};
pub use structure_tensor::{StructureTensor, StructureTensorField};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::error::*;
    pub use crate::image_buffer::*;
    pub use crate::initializer::*;
    pub use crate::renderer::*;
    pub use crate::tiling::*;
    pub use crate::ordering::*;
    pub use lgi_math::prelude::*;
}
pub mod analytical_triggers; // NEW: Complete analytical trigger framework (7 metrics)
pub mod gradient_upscaling; // NEW: Analytical gradient upscaling (exact interpolation)
pub mod selective_processing; // NEW: Selective processing framework (apply features where needed)
pub mod content_detection; // NEW: Content type detection and adaptive strategies
pub mod lod_system; // NEW: Level-of-Detail system for view-dependent rendering
pub mod trigger_actions; // NEW: Action handlers for analytical triggers (Session 3)
pub mod ms_ssim_loss; // NEW: MS-SSIM as differentiable loss
pub mod ms_ssim_gradients; // NEW: MS-SSIM analytical gradients (Session 4)
pub mod edge_weighted_loss; // NEW: Edge-weighted L2 loss
pub mod screen_space_culling; // NEW: Screen-space culling
pub mod ewa_splatting; // NEW: EWA splatting
pub mod ewa_splatting_v2; // NEW: Full robust EWA per Zwicker et al. (Session 4)
pub mod progressive_loading; // NEW: Progressive loading
pub mod importance_ordering; // NEW: Importance-based ordering
pub mod progressive; // NEW: Progressive format with importance and LOD (Session 6)
pub mod saliency_detection;
pub mod viewport_culling;
pub mod gaussian_splitting;
pub mod gaussian_merging;
pub mod adaptive_splitting;
pub mod tile_cache;
pub mod specular_detection;
pub mod text_stroke_detection;
pub mod rate_distortion;
pub mod perceptual_weighting;
pub mod hilbert_ordering;
pub mod frangi_vesselness;
pub mod coherence_enhancing_diffusion;
pub mod quantization;
pub mod container_format;
pub mod compression_utils;
pub mod error_metrics;
pub mod spatial_index;
pub mod vector_quantization;
pub mod predictive_coding;
pub mod multi_resolution;
pub mod energy_based_selection;
pub mod lbfgs;
pub mod laplacian;  // Laplacian operator for GDGS
pub mod poisson_solver;  // Poisson reconstruction for GDGS
pub mod position_probability_map;  // PPM for non-uniform Gaussian placement (Strategy G)
