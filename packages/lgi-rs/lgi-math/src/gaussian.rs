//! Core 2D Gaussian representations
//!
//! This module provides the fundamental Gaussian2D type, which is generic over:
//! - Float type (f32, f64)
//! - Parameterization (Euler, Cholesky, LogRadius, InverseCovariance)
//!
//! The design uses zero-cost abstractions to support multiple parameterizations
//! without runtime overhead.

use crate::{Float, color::Color4, parameterization::Parameterization, vec::Vector2};

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "bytemuck")]
use bytemuck::{Pod, Zeroable};

/// A 2D Gaussian splat
///
/// Generic over float type `T` and parameterization `P`.
///
/// # Type Parameters
///
/// - `T`: Float type (f32 or f64)
/// - `P`: Parameterization scheme (Euler, Cholesky, etc.)
///
/// # Memory Layout
///
/// The struct is designed for cache-friendly access and SIMD operations.
/// All fields are aligned for efficient vectorization.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Gaussian2D<T: Float, P: Parameterization<T>> {
    /// Position in normalized [0,1] × [0,1] coordinates
    pub position: Vector2<T>,

    /// Shape parameters (interpretation depends on parameterization)
    pub shape: P,

    /// Color (RGB or RGBA)
    pub color: Color4<T>,

    /// Opacity [0,1]
    pub opacity: T,

    /// Optional: Importance weight for progressive rendering
    pub weight: Option<T>,
}

impl<T: Float, P: Parameterization<T>> Gaussian2D<T, P> {
    /// Create a new Gaussian
    #[inline]
    pub fn new(position: Vector2<T>, shape: P, color: Color4<T>, opacity: T) -> Self {
        Self {
            position,
            shape,
            color,
            opacity,
            weight: None,
        }
    }

    /// Create with explicit weight
    #[inline]
    pub fn with_weight(position: Vector2<T>, shape: P, color: Color4<T>, opacity: T, weight: T) -> Self {
        Self {
            position,
            shape,
            color,
            opacity,
            weight: Some(weight),
        }
    }

    /// Get the inverse covariance matrix for rendering
    #[inline]
    pub fn inverse_covariance(&self) -> [[T; 2]; 2] {
        self.shape.inverse_covariance()
    }

    /// Get the covariance matrix
    #[inline]
    pub fn covariance(&self) -> [[T; 2]; 2] {
        self.shape.covariance()
    }

    /// Compute bounding box in normalized coordinates
    ///
    /// Returns (min, max) where both are Vector2 in [0,1] space.
    /// Uses `n_sigma` to determine the extent (default 3.5 for 99.9% coverage)
    pub fn bounding_box(&self, n_sigma: T) -> (Vector2<T>, Vector2<T>) {
        let cov = self.covariance();

        // Eigenvalues of 2x2 symmetric matrix
        let trace = cov[0][0] + cov[1][1];
        let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
        let discriminant = (trace * trace - T::two() * T::two() * det).sqrt();

        let lambda1 = (trace + discriminant) / T::two();
        let lambda2 = (trace - discriminant) / T::two();

        // Maximum extent is n_sigma times the sqrt of largest eigenvalue
        let max_sigma = lambda1.max(lambda2).sqrt();
        let radius = n_sigma * max_sigma;

        let min = Vector2::new(
            (self.position.x - radius).max(T::zero()),
            (self.position.y - radius).max(T::zero()),
        );
        let max = Vector2::new(
            (self.position.x + radius).min(T::one()),
            (self.position.y + radius).min(T::one()),
        );

        (min, max)
    }

    /// Convert to a different parameterization
    pub fn convert<Q: Parameterization<T>>(&self) -> Gaussian2D<T, Q>
    where
        P: Into<Q>
    {
        Gaussian2D {
            position: self.position,
            shape: self.shape.into(),
            color: self.color,
            opacity: self.opacity,
            weight: self.weight,
        }
    }
}

/// Structure-of-Arrays layout for SIMD processing
///
/// This layout is optimized for batch processing many Gaussians in parallel.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GaussianSoA<T: Float, P: Parameterization<T>> {
    /// Positions (x, y) interleaved
    pub positions: Vec<Vector2<T>>,

    /// Shape parameters
    pub shapes: Vec<P>,

    /// Colors (r, g, b, a) interleaved
    pub colors: Vec<Color4<T>>,

    /// Opacities
    pub opacities: Vec<T>,

    /// Optional weights
    pub weights: Option<Vec<T>>,
}

impl<T: Float, P: Parameterization<T>> GaussianSoA<T, P> {
    /// Create from a slice of Gaussians
    pub fn from_slice(gaussians: &[Gaussian2D<T, P>]) -> Self {
        let mut positions = Vec::with_capacity(gaussians.len());
        let mut shapes = Vec::with_capacity(gaussians.len());
        let mut colors = Vec::with_capacity(gaussians.len());
        let mut opacities = Vec::with_capacity(gaussians.len());
        let mut has_weights = false;
        let mut weights = Vec::with_capacity(gaussians.len());

        for g in gaussians {
            positions.push(g.position);
            shapes.push(g.shape);
            colors.push(g.color);
            opacities.push(g.opacity);

            if let Some(w) = g.weight {
                has_weights = true;
                weights.push(w);
            } else {
                weights.push(T::one());
            }
        }

        Self {
            positions,
            shapes,
            colors,
            opacities,
            weights: if has_weights { Some(weights) } else { None },
        }
    }

    /// Convert back to Array-of-Structures
    pub fn to_vec(&self) -> Vec<Gaussian2D<T, P>> {
        let len = self.positions.len();
        let mut result = Vec::with_capacity(len);

        for i in 0..len {
            result.push(Gaussian2D {
                position: self.positions[i],
                shape: self.shapes[i],
                color: self.colors[i],
                opacity: self.opacities[i],
                weight: self.weights.as_ref().map(|w| w[i]),
            });
        }

        result
    }

    /// Number of Gaussians
    #[inline]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameterization::Euler;

    #[test]
    fn test_gaussian_creation() {
        let g = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::new(0.1, 0.1, 0.0),
            Color4::new(1.0, 0.0, 0.0, 1.0),
            0.8,
        );

        assert_eq!(g.position, Vector2::new(0.5, 0.5));
        assert_eq!(g.opacity, 0.8);
    }

    #[test]
    fn test_bounding_box() {
        let g = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::new(0.05, 0.05, 0.0),
            Color4::new(1.0, 1.0, 1.0, 1.0),
            1.0,
        );

        let (min, max) = g.bounding_box(3.0);

        // Should be roughly centered with some extent
        assert!(min.x < 0.5);
        assert!(min.y < 0.5);
        assert!(max.x > 0.5);
        assert!(max.y > 0.5);

        // Should be clamped to [0, 1]
        assert!(min.x >= 0.0 && min.x <= 1.0);
        assert!(max.x >= 0.0 && max.x <= 1.0);
    }

    #[test]
    fn test_soa_conversion() {
        let gaussians = vec![
            Gaussian2D::new(
                Vector2::new(0.25, 0.25),
                Euler::new(0.1, 0.1, 0.0),
                Color4::new(1.0, 0.0, 0.0, 1.0),
                0.8,
            ),
            Gaussian2D::new(
                Vector2::new(0.75, 0.75),
                Euler::new(0.05, 0.05, 0.0),
                Color4::new(0.0, 1.0, 0.0, 1.0),
                0.9,
            ),
        ];

        let soa = GaussianSoA::from_slice(&gaussians);
        assert_eq!(soa.len(), 2);

        let back = soa.to_vec();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].position, gaussians[0].position);
        assert_eq!(back[1].position, gaussians[1].position);
    }
}
