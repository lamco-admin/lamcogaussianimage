//! Gaussian shape parameterizations
//!
//! This module provides different ways to represent the covariance matrix of a 2D Gaussian:
//!
//! - **Euler**: Scale (σx, σy) and rotation (θ) - Most intuitive
//! - **Cholesky**: Cholesky decomposition L where Σ = L·Lᵀ - Numerically stable
//! - **LogRadius**: Log radius, eccentricity, and rotation - Good for compression
//! - **InverseCovariance**: Direct inverse covariance Σ⁻¹ - Fastest for rendering

use crate::Float;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "bytemuck")]
use bytemuck::{Pod, Zeroable};

/// Trait for Gaussian shape parameterizations
///
/// All parameterizations must be able to compute:
/// - Covariance matrix Σ
/// - Inverse covariance matrix Σ⁻¹
pub trait Parameterization<T: Float>: Copy + Clone + std::fmt::Debug {
    /// Compute the 2×2 covariance matrix
    fn covariance(&self) -> [[T; 2]; 2];

    /// Compute the 2×2 inverse covariance matrix (for rendering)
    fn inverse_covariance(&self) -> [[T; 2]; 2];

    /// Check if the parameterization is valid (positive definite, etc.)
    fn is_valid(&self) -> bool;
}

/// Euler parameterization: (σx, σy, θ)
///
/// - σx, σy: Standard deviations (scales) along principal axes
/// - θ: Rotation angle in radians
///
/// Covariance: Σ = R(θ) · diag(σx², σy²) · R(θ)ᵀ
///
/// This is the most intuitive parameterization and easiest to work with.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Euler<T: Float> {
    /// Scale along x-axis (σx)
    pub scale_x: T,
    /// Scale along y-axis (σy)
    pub scale_y: T,
    /// Rotation angle in radians
    pub rotation: T,
}

impl<T: Float> Euler<T> {
    /// Create new Euler parameterization
    #[inline]
    pub fn new(scale_x: T, scale_y: T, rotation: T) -> Self {
        Self { scale_x, scale_y, rotation }
    }

    /// Create isotropic (circular) Gaussian
    #[inline]
    pub fn isotropic(scale: T) -> Self {
        Self::new(scale, scale, T::zero())
    }
}

impl<T: Float> Parameterization<T> for Euler<T> {
    #[inline]
    fn covariance(&self) -> [[T; 2]; 2] {
        let sx2 = self.scale_x * self.scale_x;
        let sy2 = self.scale_y * self.scale_y;
        let cos_t = self.rotation.cos();
        let sin_t = self.rotation.sin();

        // Σ = R(θ) · diag(σx², σy²) · R(θ)ᵀ
        let a = sx2 * cos_t * cos_t + sy2 * sin_t * sin_t;
        let b = (sx2 - sy2) * cos_t * sin_t;
        let c = sx2 * sin_t * sin_t + sy2 * cos_t * cos_t;

        [[a, b], [b, c]]
    }

    #[inline]
    fn inverse_covariance(&self) -> [[T; 2]; 2] {
        let sx2 = self.scale_x * self.scale_x;
        let sy2 = self.scale_y * self.scale_y;
        let cos_t = self.rotation.cos();
        let sin_t = self.rotation.sin();

        // Σ⁻¹ = R(θ) · diag(1/σx², 1/σy²) · R(θ)ᵀ
        let inv_sx2 = T::one() / sx2;
        let inv_sy2 = T::one() / sy2;

        let a = inv_sx2 * cos_t * cos_t + inv_sy2 * sin_t * sin_t;
        let b = (inv_sx2 - inv_sy2) * cos_t * sin_t;
        let c = inv_sx2 * sin_t * sin_t + inv_sy2 * cos_t * cos_t;

        [[a, b], [b, c]]
    }

    #[inline]
    fn is_valid(&self) -> bool {
        self.scale_x > T::zero() && self.scale_y > T::zero()
    }
}

/// Cholesky parameterization: L where Σ = L·Lᵀ
///
/// Stores the lower triangular Cholesky factor:
/// ```text
/// L = [L11   0  ]
///     [L21  L22 ]
/// ```
///
/// This is numerically stable and ensures positive definiteness.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Cholesky<T: Float> {
    /// L[0,0] - diagonal element
    pub l11: T,
    /// L[1,0] - off-diagonal element
    pub l21: T,
    /// L[1,1] - diagonal element
    pub l22: T,
}

impl<T: Float> Cholesky<T> {
    /// Create new Cholesky parameterization
    #[inline]
    pub fn new(l11: T, l21: T, l22: T) -> Self {
        Self { l11, l21, l22 }
    }

    /// Create from covariance matrix (performs Cholesky decomposition)
    pub fn from_covariance(cov: [[T; 2]; 2]) -> Option<Self> {
        // Σ = [[a, b], [b, c]]
        let a = cov[0][0];
        let b = cov[0][1];
        let c = cov[1][1];

        // Check positive definiteness
        if a <= T::zero() || c <= T::zero() || a * c - b * b <= T::zero() {
            return None;
        }

        // Cholesky decomposition
        let l11 = a.sqrt();
        let l21 = b / l11;
        let l22 = (c - l21 * l21).sqrt();

        Some(Self::new(l11, l21, l22))
    }
}

impl<T: Float> Parameterization<T> for Cholesky<T> {
    #[inline]
    fn covariance(&self) -> [[T; 2]; 2] {
        // Σ = L·Lᵀ
        let a = self.l11 * self.l11;
        let b = self.l11 * self.l21;
        let c = self.l21 * self.l21 + self.l22 * self.l22;

        [[a, b], [b, c]]
    }

    #[inline]
    fn inverse_covariance(&self) -> [[T; 2]; 2] {
        // Σ⁻¹ = (L·Lᵀ)⁻¹ = L⁻ᵀ·L⁻¹
        let det_l = self.l11 * self.l22;
        let inv_det = T::one() / det_l;

        // L⁻¹
        let inv_l11 = T::one() / self.l11;
        let inv_l21 = -self.l21 / (self.l11 * self.l22);
        let inv_l22 = T::one() / self.l22;

        // (L⁻¹)ᵀ · L⁻¹
        let a = inv_l11 * inv_l11;
        let b = inv_l11 * inv_l21;
        let c = inv_l21 * inv_l21 + inv_l22 * inv_l22;

        [[a, b], [b, c]]
    }

    #[inline]
    fn is_valid(&self) -> bool {
        self.l11 > T::zero() && self.l22 > T::zero()
    }
}

/// Log-radius parameterization: (log r, e, θ)
///
/// - log r: Logarithm of geometric mean radius
/// - e: Eccentricity (ratio σy/σx)
/// - θ: Rotation angle
///
/// Good for compression as log-radius has better dynamic range.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct LogRadius<T: Float> {
    /// Log of geometric mean radius: log(√(σx·σy))
    pub log_radius: T,
    /// Eccentricity: σy/σx
    pub eccentricity: T,
    /// Rotation angle in radians
    pub rotation: T,
}

impl<T: Float> LogRadius<T> {
    /// Create new log-radius parameterization
    #[inline]
    pub fn new(log_radius: T, eccentricity: T, rotation: T) -> Self {
        Self { log_radius, eccentricity, rotation }
    }

    /// Convert to Euler parameterization
    #[inline]
    pub fn to_euler(&self) -> Euler<T> {
        let geom_mean = self.log_radius.exp();
        let scale_x = geom_mean / self.eccentricity.sqrt();
        let scale_y = geom_mean * self.eccentricity.sqrt();
        Euler::new(scale_x, scale_y, self.rotation)
    }

    /// Create from Euler parameterization
    #[inline]
    pub fn from_euler(euler: &Euler<T>) -> Self {
        let geom_mean = (euler.scale_x * euler.scale_y).sqrt();
        let eccentricity = euler.scale_y / euler.scale_x;
        Self::new(geom_mean.ln(), eccentricity, euler.rotation)
    }
}

impl<T: Float> Parameterization<T> for LogRadius<T> {
    #[inline]
    fn covariance(&self) -> [[T; 2]; 2] {
        self.to_euler().covariance()
    }

    #[inline]
    fn inverse_covariance(&self) -> [[T; 2]; 2] {
        self.to_euler().inverse_covariance()
    }

    #[inline]
    fn is_valid(&self) -> bool {
        self.eccentricity > T::zero()
    }
}

/// Inverse covariance parameterization: Direct Σ⁻¹
///
/// Stores the inverse covariance matrix directly:
/// ```text
/// Σ⁻¹ = [a  b]
///       [b  c]
/// ```
///
/// This is the fastest for rendering as no matrix inversion is needed.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct InverseCovariance<T: Float> {
    /// Σ⁻¹[0,0]
    pub a: T,
    /// Σ⁻¹[0,1] = Σ⁻¹[1,0]
    pub b: T,
    /// Σ⁻¹[1,1]
    pub c: T,
}

impl<T: Float> InverseCovariance<T> {
    /// Create new inverse covariance parameterization
    #[inline]
    pub fn new(a: T, b: T, c: T) -> Self {
        Self { a, b, c }
    }

    /// Create from covariance matrix (inverts it)
    pub fn from_covariance(cov: [[T; 2]; 2]) -> Option<Self> {
        let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];

        if det.abs() < T::one() / (T::one() + T::one() + T::one()).exp().exp().exp().exp() { // ~1e-7 for f32
            return None;
        }

        let inv_det = T::one() / det;
        let a = cov[1][1] * inv_det;
        let b = -cov[0][1] * inv_det;
        let c = cov[0][0] * inv_det;

        Some(Self::new(a, b, c))
    }
}

impl<T: Float> Parameterization<T> for InverseCovariance<T> {
    #[inline]
    fn covariance(&self) -> [[T; 2]; 2] {
        let det = self.a * self.c - self.b * self.b;
        let inv_det = T::one() / det;

        let a = self.c * inv_det;
        let b = -self.b * inv_det;
        let c = self.a * inv_det;

        [[a, b], [b, c]]
    }

    #[inline]
    fn inverse_covariance(&self) -> [[T; 2]; 2] {
        [[self.a, self.b], [self.b, self.c]]
    }

    #[inline]
    fn is_valid(&self) -> bool {
        // Must be positive definite: a > 0, c > 0, ac - b² > 0
        self.a > T::zero() && self.c > T::zero() && (self.a * self.c - self.b * self.b) > T::zero()
    }
}

// Conversion implementations
impl<T: Float> From<Euler<T>> for Cholesky<T> {
    fn from(euler: Euler<T>) -> Self {
        let cov = euler.covariance();
        Cholesky::from_covariance(cov).expect("Valid Euler should produce valid Cholesky")
    }
}

impl<T: Float> From<Euler<T>> for LogRadius<T> {
    fn from(euler: Euler<T>) -> Self {
        LogRadius::from_euler(&euler)
    }
}

impl<T: Float> From<Euler<T>> for InverseCovariance<T> {
    fn from(euler: Euler<T>) -> Self {
        let cov = euler.covariance();
        InverseCovariance::from_covariance(cov).expect("Valid Euler should produce valid InverseCovariance")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_euler_covariance() {
        let euler = Euler::new(0.1f32, 0.05f32, 0.0f32);
        let cov = euler.covariance();

        // For zero rotation, should be diagonal
        assert_relative_eq!(cov[0][0], 0.01, epsilon = 1e-6);
        assert_relative_eq!(cov[1][1], 0.0025, epsilon = 1e-6);
        assert_relative_eq!(cov[0][1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cholesky_roundtrip() {
        let euler = Euler::new(0.1f32, 0.05f32, 0.3f32);
        let cov = euler.covariance();

        let cholesky = Cholesky::from_covariance(cov).unwrap();
        let cov2 = cholesky.covariance();

        assert_relative_eq!(cov[0][0], cov2[0][0], epsilon = 1e-5);
        assert_relative_eq!(cov[0][1], cov2[0][1], epsilon = 1e-5);
        assert_relative_eq!(cov[1][1], cov2[1][1], epsilon = 1e-5);
    }

    #[test]
    fn test_inverse_covariance() {
        let euler = Euler::new(0.1f32, 0.1f32, 0.0f32);
        let inv_cov = euler.inverse_covariance();
        let cov = euler.covariance();

        // Σ · Σ⁻¹ = I
        let prod_a = cov[0][0] * inv_cov[0][0] + cov[0][1] * inv_cov[1][0];
        let prod_b = cov[0][0] * inv_cov[0][1] + cov[0][1] * inv_cov[1][1];
        let prod_c = cov[1][0] * inv_cov[0][0] + cov[1][1] * inv_cov[1][0];
        let prod_d = cov[1][0] * inv_cov[0][1] + cov[1][1] * inv_cov[1][1];

        assert_relative_eq!(prod_a, 1.0, epsilon = 1e-5);
        assert_relative_eq!(prod_b, 0.0, epsilon = 1e-5);
        assert_relative_eq!(prod_c, 0.0, epsilon = 1e-5);
        assert_relative_eq!(prod_d, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_log_radius_conversion() {
        let euler = Euler::new(0.1f32, 0.05f32, 0.3f32);
        let log_rad = LogRadius::from_euler(&euler);
        let euler2 = log_rad.to_euler();

        assert_relative_eq!(euler.scale_x, euler2.scale_x, epsilon = 1e-5);
        assert_relative_eq!(euler.scale_y, euler2.scale_y, epsilon = 1e-5);
        assert_relative_eq!(euler.rotation, euler2.rotation, epsilon = 1e-5);
    }
}
