//! # LGI Math Library
//!
//! Core mathematical primitives for the LGI/LGIV Gaussian image format.
//!
//! This library provides:
//! - 2D Gaussian representations with multiple parameterizations
//! - Covariance matrix operations
//! - Gaussian evaluation and rendering
//! - Alpha compositing
//! - SIMD-optimized operations
//!
//! ## Features
//!
//! - `std`: Standard library support (enabled by default)
//! - `f32`: Single-precision floating point (enabled by default)
//! - `f64`: Double-precision floating point
//! - `simd`: SIMD optimizations using `wide` crate
//! - `advanced`: Advanced linear algebra via `nalgebra`
//! - `serde`: Serialization support
//! - `bytemuck`: Zero-copy casting support
//!
//! ## Design Principles
//!
//! 1. **Zero-cost abstractions**: Generic over float types, parameterizations
//! 2. **SIMD-first**: Data layouts optimized for vectorization
//! 3. **Numerically stable**: Careful handling of edge cases
//! 4. **Extensible**: Trait-based design for custom parameterizations
//! 5. **No-std compatible**: Core functionality works without allocator

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::excessive_precision)] // We need precise constants

#[cfg(not(feature = "std"))]
extern crate core as std;

// Re-export glam for convenience (for external use)
pub use glam;

// Public modules
pub mod vec;
pub mod gaussian;
pub mod covariance;
pub mod parameterization;
pub mod evaluation;
pub mod compositing;
pub mod color;
pub mod transform;
pub mod utils;
pub mod eigen2x2;      // NEW: 2×2 eigendecomposition for structure tensor
pub mod log_cholesky;  // NEW: Log-Cholesky covariance parameterization

#[cfg(feature = "simd")]
pub mod simd;

// Prelude for convenient imports
pub mod prelude {
    //! Convenient re-exports of commonly used types and traits.

    pub use crate::vec::*;
    pub use crate::gaussian::*;
    pub use crate::covariance::*;
    pub use crate::parameterization::*;
    pub use crate::evaluation::*;
    pub use crate::compositing::*;
    pub use crate::color::*;
    pub use crate::transform::*;
}

// Type aliases for common use cases
pub use gaussian::Gaussian2D;
pub use evaluation::GaussianEvaluator;
pub use compositing::Compositor;

/// Standard 2D Gaussian using f32 and Euler parameterization
pub type Gaussian2Df32 = gaussian::Gaussian2D<f32, parameterization::Euler<f32>>;

/// High-precision 2D Gaussian using f64
pub type Gaussian2Df64 = gaussian::Gaussian2D<f64, parameterization::Euler<f64>>;

/// Common floating-point trait bounds
pub trait Float:
    Copy
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + PartialOrd
    + approx::AbsDiffEq
    + approx::RelativeEq
{
    /// Zero value
    fn zero() -> Self;
    /// One value
    fn one() -> Self;
    /// Two value
    fn two() -> Self;
    /// PI constant
    fn pi() -> Self;
    /// Square root
    fn sqrt(self) -> Self;
    /// Exponential
    fn exp(self) -> Self;
    /// Natural logarithm
    fn ln(self) -> Self;
    /// Sine
    fn sin(self) -> Self;
    /// Cosine
    fn cos(self) -> Self;
    /// Absolute value
    fn abs(self) -> Self;
    /// Maximum
    fn max(self, other: Self) -> Self;
    /// Minimum
    fn min(self, other: Self) -> Self;
    /// Clamp between min and max
    fn clamp(self, min: Self, max: Self) -> Self;
    /// Arc tangent of y/x
    fn atan2(self, x: Self) -> Self;
}

impl Float for f32 {
    #[inline(always)]
    fn zero() -> Self { 0.0 }
    #[inline(always)]
    fn one() -> Self { 1.0 }
    #[inline(always)]
    fn two() -> Self { 2.0 }
    #[inline(always)]
    fn pi() -> Self { std::f32::consts::PI }
    #[inline(always)]
    fn sqrt(self) -> Self { self.sqrt() }
    #[inline(always)]
    fn exp(self) -> Self { self.exp() }
    #[inline(always)]
    fn ln(self) -> Self { self.ln() }
    #[inline(always)]
    fn sin(self) -> Self { self.sin() }
    #[inline(always)]
    fn cos(self) -> Self { self.cos() }
    #[inline(always)]
    fn abs(self) -> Self { self.abs() }
    #[inline(always)]
    fn max(self, other: Self) -> Self { self.max(other) }
    #[inline(always)]
    fn min(self, other: Self) -> Self { self.min(other) }
    #[inline(always)]
    fn clamp(self, min: Self, max: Self) -> Self { self.clamp(min, max) }
    #[inline(always)]
    fn atan2(self, x: Self) -> Self { self.atan2(x) }
}

impl Float for f64 {
    #[inline(always)]
    fn zero() -> Self { 0.0 }
    #[inline(always)]
    fn one() -> Self { 1.0 }
    #[inline(always)]
    fn two() -> Self { 2.0 }
    #[inline(always)]
    fn pi() -> Self { std::f64::consts::PI }
    #[inline(always)]
    fn sqrt(self) -> Self { self.sqrt() }
    #[inline(always)]
    fn exp(self) -> Self { self.exp() }
    #[inline(always)]
    fn ln(self) -> Self { self.ln() }
    #[inline(always)]
    fn sin(self) -> Self { self.sin() }
    #[inline(always)]
    fn cos(self) -> Self { self.cos() }
    #[inline(always)]
    fn abs(self) -> Self { self.abs() }
    #[inline(always)]
    fn max(self, other: Self) -> Self { self.max(other) }
    #[inline(always)]
    fn min(self, other: Self) -> Self { self.min(other) }
    #[inline(always)]
    fn clamp(self, min: Self, max: Self) -> Self { self.clamp(min, max) }
    #[inline(always)]
    fn atan2(self, x: Self) -> Self { self.atan2(x) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_trait() {
        assert_eq!(f32::zero(), 0.0f32);
        assert_eq!(f32::one(), 1.0f32);
        assert_eq!(f32::two(), 2.0f32);
    }
}
