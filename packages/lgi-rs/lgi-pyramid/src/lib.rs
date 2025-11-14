//! # LGI Multi-Level Pyramid
//!
//! Resolution-independent zoom rendering with O(1) complexity.

#![warn(missing_docs)]

pub mod pyramid;
pub mod builder;
pub mod level;
pub mod error;

pub use pyramid::GaussianPyramid;
pub use builder::PyramidBuilder;
pub use level::PyramidLevel;
pub use error::{PyramidError, Result};

/// Viewport rectangle for rendering
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    /// X coordinate (normalized [0, 1])
    pub x: f32,
    /// Y coordinate (normalized [0, 1])
    pub y: f32,
    /// Width (normalized [0, 1])
    pub width: f32,
    /// Height (normalized [0, 1])
    pub height: f32,
}

impl Viewport {
    /// Create new viewport
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Full viewport (entire image)
    pub fn full() -> Self {
        Self::new(0.0, 0.0, 1.0, 1.0)
    }
}
