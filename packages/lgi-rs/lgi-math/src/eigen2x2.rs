//! Fast 2×2 symmetric matrix eigendecomposition
//!
//! Critical component for structure tensor computation.
//! Eigenvalues and eigenvectors describe edge orientation and strength.

use crate::vec::Vector2;

use crate::Float;

/// Result of 2×2 symmetric matrix eigendecomposition
#[derive(Debug, Clone, Copy)]
pub struct Eigen2x2<T: Float> {
    /// First (larger) eigenvalue
    pub lambda1: T,
    /// Second (smaller) eigenvalue
    pub lambda2: T,
    /// First eigenvector (corresponds to lambda1)
    pub eigenvector1: Vector2<T>,
    /// Second eigenvector (corresponds to lambda2)
    pub eigenvector2: Vector2<T>,
    /// Coherence measure: (λ1 - λ2) / (λ1 + λ2) ∈ [0,1]
    pub coherence: T,
}

impl Eigen2x2<f32> {
    /// Compute eigendecomposition of symmetric 2×2 matrix
    ///
    /// Matrix: [[a, b],
    ///          [b, c]]
    ///
    /// Returns eigenvalues λ1 ≥ λ2 and orthonormal eigenvectors
    ///
    /// # Algorithm
    /// For symmetric 2×2:
    /// - trace = a + c
    /// - det = ac - b²
    /// - discriminant = √(trace² - 4·det)
    /// - λ1 = (trace + discriminant) / 2
    /// - λ2 = (trace - discriminant) / 2
    ///
    /// Eigenvector for λ1:
    /// - If |b| > ε: v1 = [λ1 - c, b] (normalized)
    /// - Else if a > c: v1 = [1, 0]
    /// - Else: v1 = [0, 1]
    /// - v2 = [-v1.y, v1.x] (perpendicular)
    pub fn decompose(a: f32, b: f32, c: f32) -> Self {
        const EPSILON: f32 = 1e-10;

        // Compute eigenvalues
        let trace = a + c;
        let det = a * c - b * b;
        let discriminant_sq = trace * trace - 4.0 * det;
        let discriminant = discriminant_sq.max(0.0).sqrt();

        let lambda1 = (trace + discriminant) / 2.0;
        let lambda2 = (trace - discriminant) / 2.0;

        // Compute first eigenvector (for larger eigenvalue)
        let eigenvector1 = if b.abs() > EPSILON {
            // General case: non-diagonal
            Vector2::new(lambda1 - c, b).normalize()
        } else if a > c {
            // Diagonal, a is larger
            Vector2::new(1.0, 0.0)
        } else {
            // Diagonal, c is larger
            Vector2::new(0.0, 1.0)
        };

        // Second eigenvector is perpendicular (90° rotation)
        let eigenvector2 = Vector2::new(-eigenvector1.y, eigenvector1.x);

        // Coherence measure (edge strength indicator)
        let coherence = if (lambda1 + lambda2).abs() > EPSILON {
            ((lambda1 - lambda2) / (lambda1 + lambda2)).clamp(0.0, 1.0)
        } else {
            0.0  // Flat region, no structure
        };

        Self {
            lambda1,
            lambda2,
            eigenvector1,
            eigenvector2,
            coherence,
        }
    }

    /// Check if eigenvalues are positive (valid covariance)
    pub fn is_positive_definite(&self) -> bool {
        self.lambda1 > 0.0 && self.lambda2 > 0.0
    }

    /// Check if eigenvalues are non-negative (positive semi-definite)
    pub fn is_positive_semidefinite(&self) -> bool {
        self.lambda1 >= 0.0 && self.lambda2 >= 0.0
    }

    /// Reconstruct matrix from eigendecomposition
    ///
    /// Returns: [[a, b], [b, c]] such that
    /// [[a, b], [b, c]] = Q Λ Q^T
    pub fn reconstruct(&self) -> (f32, f32, f32) {
        let e1 = self.eigenvector1;
        let e2 = self.eigenvector2;
        let l1 = self.lambda1;
        let l2 = self.lambda2;

        // Σ = Q Λ Q^T where Q = [e1 | e2]
        let a = e1.x * e1.x * l1 + e2.x * e2.x * l2;
        let b = e1.x * e1.y * l1 + e2.x * e2.y * l2;
        let c = e1.y * e1.y * l1 + e2.y * e2.y * l2;

        (a, b, c)
    }

    /// Condition number: λ1 / λ2 (measures anisotropy)
    pub fn condition_number(&self) -> f32 {
        if self.lambda2.abs() < 1e-10 {
            f32::INFINITY
        } else {
            self.lambda1 / self.lambda2
        }
    }

    /// Orientation angle of major axis (radians)
    pub fn orientation_angle(&self) -> f32 {
        self.eigenvector1.y.atan2(self.eigenvector1.x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagonal_matrix() {
        // Diagonal matrix [[4, 0], [0, 1]]
        let eigen = Eigen2x2::decompose(4.0, 0.0, 1.0);

        assert!((eigen.lambda1 - 4.0).abs() < 1e-6);
        assert!((eigen.lambda2 - 1.0).abs() < 1e-6);
        assert!(eigen.coherence > 0.5);  // Anisotropic
    }

    #[test]
    fn test_isotropic_matrix() {
        // Isotropic matrix [[2, 0], [0, 2]]
        let eigen = Eigen2x2::decompose(2.0, 0.0, 2.0);

        assert!((eigen.lambda1 - 2.0).abs() < 1e-6);
        assert!((eigen.lambda2 - 2.0).abs() < 1e-6);
        assert!(eigen.coherence.abs() < 1e-6);  // No anisotropy
    }

    #[test]
    fn test_rotated_matrix() {
        // Rotated anisotropic matrix
        // [[3, 1], [1, 2]]
        let eigen = Eigen2x2::decompose(3.0, 1.0, 2.0);

        // λ1 + λ2 = trace = 5
        assert!((eigen.lambda1 + eigen.lambda2 - 5.0).abs() < 1e-6);

        // λ1 × λ2 = det = 3×2 - 1² = 5
        assert!((eigen.lambda1 * eigen.lambda2 - 5.0).abs() < 1e-6);

        // Eigenvectors should be orthogonal
        let dot = eigen.eigenvector1.dot(eigen.eigenvector2);
        assert!(dot.abs() < 1e-6);

        // Eigenvectors should be unit length
        assert!((eigen.eigenvector1.length() - 1.0).abs() < 1e-6);
        assert!((eigen.eigenvector2.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reconstruction() {
        // Test that reconstruction gives back original matrix
        let (a, b, c) = (5.0, 2.0, 3.0);
        let eigen = Eigen2x2::decompose(a, b, c);
        let (a_rec, b_rec, c_rec) = eigen.reconstruct();

        assert!((a - a_rec).abs() < 1e-5);
        assert!((b - b_rec).abs() < 1e-5);
        assert!((c - c_rec).abs() < 1e-5);
    }

    #[test]
    fn test_edge_case_zero_matrix() {
        let eigen = Eigen2x2::decompose(0.0, 0.0, 0.0);

        assert!(eigen.lambda1.abs() < 1e-10);
        assert!(eigen.lambda2.abs() < 1e-10);
        assert!(eigen.coherence.abs() < 1e-10);
    }

    #[test]
    fn test_coherence_range() {
        // Coherence should always be in [0, 1]
        for &(a, b, c) in &[(1.0, 0.5, 2.0), (10.0, 1.0, 10.0), (5.0, 4.0, 3.0)] {
            let eigen = Eigen2x2::decompose(a, b, c);
            assert!(eigen.coherence >= -0.001);  // Allow tiny numerical error
            assert!(eigen.coherence <= 1.001);
        }
    }
}
