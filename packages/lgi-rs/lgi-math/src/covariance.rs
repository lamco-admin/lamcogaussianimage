//! Covariance matrix utilities

use crate::Float;

/// 2×2 covariance matrix utilities
pub struct CovarianceMatrix;

impl CovarianceMatrix {
    /// Compute eigenvalues of symmetric 2×2 matrix
    ///
    /// Returns (λ₁, λ₂) where λ₁ ≥ λ₂
    #[inline]
    pub fn eigenvalues<T: Float>(cov: [[T; 2]; 2]) -> (T, T) {
        let trace = cov[0][0] + cov[1][1];
        let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
        let discriminant = (trace * trace - T::two() * T::two() * det).sqrt();

        let lambda1 = (trace + discriminant) / T::two();
        let lambda2 = (trace - discriminant) / T::two();

        (lambda1.max(lambda2), lambda1.min(lambda2))
    }

    /// Compute determinant of 2×2 matrix
    #[inline]
    pub fn determinant<T: Float>(cov: [[T; 2]; 2]) -> T {
        cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0]
    }

    /// Invert a 2×2 matrix
    #[inline]
    pub fn invert<T: Float>(m: [[T; 2]; 2]) -> Option<[[T; 2]; 2]> {
        let det = Self::determinant(m);

        if det.abs() < T::one() / (T::one() + T::one()).exp().exp().exp().exp() {
            return None;
        }

        let inv_det = T::one() / det;
        Some([
            [m[1][1] * inv_det, -m[0][1] * inv_det],
            [-m[1][0] * inv_det, m[0][0] * inv_det],
        ])
    }

    /// Check if matrix is positive definite
    #[inline]
    pub fn is_positive_definite<T: Float>(cov: [[T; 2]; 2]) -> bool {
        // For 2×2: a > 0, c > 0, det > 0
        cov[0][0] > T::zero()
            && cov[1][1] > T::zero()
            && Self::determinant(cov) > T::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_eigenvalues() {
        let cov = [[0.04f32, 0.0], [0.0, 0.01]];
        let (l1, l2) = CovarianceMatrix::eigenvalues(cov);

        assert_relative_eq!(l1, 0.04, epsilon = 1e-6);
        assert_relative_eq!(l2, 0.01, epsilon = 1e-6);
    }

    #[test]
    fn test_invert() {
        let m = [[2.0f32, 1.0], [1.0, 3.0]];
        let inv = CovarianceMatrix::invert(m).unwrap();

        // M · M⁻¹ = I
        let i00 = m[0][0] * inv[0][0] + m[0][1] * inv[1][0];
        let i11 = m[1][0] * inv[0][1] + m[1][1] * inv[1][1];

        assert_relative_eq!(i00, 1.0, epsilon = 1e-5);
        assert_relative_eq!(i11, 1.0, epsilon = 1e-5);
    }
}
