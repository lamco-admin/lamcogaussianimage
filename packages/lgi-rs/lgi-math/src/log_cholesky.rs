//! Log-Cholesky Parameterization for 2×2 Covariance Matrices
//!
//! Represents positive-definite covariance via Cholesky decomposition
//! with logarithmic parameterization for numerical stability.
//!
//! # Theory
//!
//! Any 2×2 positive-definite matrix Σ can be decomposed:
//! Σ = L L^T where L is lower-triangular with positive diagonal
//!
//! L = [[l11,  0 ],
//!      [l21, l22]]
//!
//! Log-parameterization: θ = [log(l11), l21, log(l22)]
//!
//! Benefits:
//! - Always positive-definite (diagonal elements = exp, always positive)
//! - Unconstrained optimization (no constraints in log-space)
//! - Perceptually uniform quantization
//! - Natural gradient descent on SPD(2) manifold
//!
//! # References
//!
//! - Stan Reference Manual: Cholesky Factors
//! - Amari (1998): Natural Gradient Works Efficiently
//! - Weickert et al: Structure Tensor Applications

use crate::{Float, vec::Vector2};

/// Log-Cholesky parameterization of 2×2 covariance
///
/// Stores covariance as θ = [log(l11), l21, log(l22)]
/// where Σ = L L^T and L is lower-triangular Cholesky factor
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogCholesky {
    /// log(l11): Log of first diagonal element
    pub log_l11: f32,

    /// l21: Off-diagonal element (unconstrained)
    pub l21: f32,

    /// log(l22): Log of second diagonal element
    pub log_l22: f32,
}

impl LogCholesky {
    /// Create from log-Cholesky parameters
    pub fn new(log_l11: f32, l21: f32, log_l22: f32) -> Self {
        Self { log_l11, l21, log_l22 }
    }

    /// Create from covariance matrix [[a, b], [b, c]]
    ///
    /// Computes Cholesky decomposition, then log-parameterizes
    pub fn from_covariance(sigma_xx: f32, sigma_xy: f32, sigma_yy: f32) -> Option<Self> {
        // Check positive-definiteness
        let det = sigma_xx * sigma_yy - sigma_xy * sigma_xy;
        if det <= 0.0 || sigma_xx <= 0.0 || sigma_yy <= 0.0 {
            return None;  // Not positive-definite
        }

        // Cholesky decomposition of [[a, b], [b, c]]
        // L = [[l11, 0], [l21, l22]] such that L L^T = Σ
        let l11 = sigma_xx.sqrt();
        let l21 = sigma_xy / l11;
        let l22 = (sigma_yy - l21 * l21).sqrt();

        Some(Self {
            log_l11: l11.ln(),
            l21,
            log_l22: l22.ln(),
        })
    }

    /// Create from eigendecomposition
    ///
    /// Given eigenvalues λ1, λ2 and rotation angle θ,
    /// construct Σ then convert to log-Cholesky
    pub fn from_eigen(lambda1: f32, lambda2: f32, theta: f32) -> Self {
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Σ = R diag(λ1, λ2) R^T
        let sigma_xx = cos_t * cos_t * lambda1 + sin_t * sin_t * lambda2;
        let sigma_xy = cos_t * sin_t * (lambda1 - lambda2);
        let sigma_yy = sin_t * sin_t * lambda1 + cos_t * cos_t * lambda2;

        Self::from_covariance(sigma_xx, sigma_xy, sigma_yy)
            .expect("Eigendecomposition should give valid covariance")
    }

    /// Create from structure tensor with edge alignment
    ///
    /// Uses structure tensor eigenvectors for orientation
    /// and computes anisotropic scales based on coherence
    pub fn from_structure_tensor(
        eigenvector_major: Vector2<f32>,  // Edge normal
        eigenvector_minor: Vector2<f32>,  // Edge tangent
        coherence: f32,
        sigma_min: f32,
        alpha: f32,        // Coherence weight (typical: 3.0)
        beta: f32,         // Anisotropy ratio (typical: 4.0)
        edt_distance: f32, // Geodesic distance to edge
    ) -> Self {
        // Compute sigma perpendicular to edge (thin)
        let kappa = 2.0;  // Base scale
        let sigma_perp = sigma_min.max(kappa / (1.0 + alpha * coherence));

        // Clamp by edge distance (geodesic EDT constraint)
        let sigma_perp = sigma_perp.min(0.5 + 0.3 * edt_distance);

        // Compute sigma parallel to edge (long)
        let sigma_parallel = beta * sigma_perp;

        // Build covariance from eigenvectors and scales
        // Σ = Q diag(σ_parallel², σ_perp²) Q^T
        // where Q = [e_tangent | e_normal] = [eigenvector_minor | eigenvector_major]

        let e_tangent = eigenvector_minor;  // Long axis
        let e_normal = eigenvector_major;   // Short axis

        let s_tangent_sq = sigma_parallel * sigma_parallel;
        let s_normal_sq = sigma_perp * sigma_perp;

        let sigma_xx = e_tangent.x * e_tangent.x * s_tangent_sq +
                       e_normal.x * e_normal.x * s_normal_sq;

        let sigma_xy = e_tangent.x * e_tangent.y * s_tangent_sq +
                       e_normal.x * e_normal.y * s_normal_sq;

        let sigma_yy = e_tangent.y * e_tangent.y * s_tangent_sq +
                       e_normal.y * e_normal.y * s_normal_sq;

        Self::from_covariance(sigma_xx, sigma_xy, sigma_yy)
            .expect("Structure tensor should give valid covariance")
    }

    /// Reconstruct Cholesky factor L
    pub fn cholesky_factor(&self) -> (f32, f32, f32) {
        let l11 = self.log_l11.exp();
        let l21 = self.l21;
        let l22 = self.log_l22.exp();
        (l11, l21, l22)
    }

    /// Reconstruct covariance matrix [[a, b], [b, c]]
    ///
    /// Σ = L L^T = [[l11², l11·l21], [l11·l21, l21² + l22²]]
    pub fn to_covariance(&self) -> (f32, f32, f32) {
        let (l11, l21, l22) = self.cholesky_factor();

        let sigma_xx = l11 * l11;
        let sigma_xy = l11 * l21;
        let sigma_yy = l21 * l21 + l22 * l22;

        (sigma_xx, sigma_xy, sigma_yy)
    }

    /// Compute inverse covariance (for rendering)
    ///
    /// Σ^-1 needed for Mahalanobis distance in Gaussian evaluation
    pub fn inverse_covariance(&self) -> (f32, f32, f32) {
        let (sigma_xx, sigma_xy, sigma_yy) = self.to_covariance();

        let det = sigma_xx * sigma_yy - sigma_xy * sigma_xy;
        assert!(det > 1e-10, "Singular covariance matrix");

        let inv_xx = sigma_yy / det;
        let inv_xy = -sigma_xy / det;
        let inv_yy = sigma_xx / det;

        (inv_xx, inv_xy, inv_yy)
    }

    /// Determinant of covariance (for scale classification)
    pub fn determinant(&self) -> f32 {
        let (l11, l21, l22) = self.cholesky_factor();
        // det(Σ) = det(L)² = (l11 · l22)²
        (l11 * l22).powi(2)
    }

    /// Check if represents valid (positive-definite) covariance
    ///
    /// By construction, log-Cholesky is always PSD if parameters are finite
    pub fn is_valid(&self) -> bool {
        self.log_l11.is_finite() &&
        self.l21.is_finite() &&
        self.log_l22.is_finite()
    }

    /// Ensure numerical stability
    ///
    /// Clamp log values to prevent extreme scales
    pub fn clamp_for_stability(&mut self, min_log_scale: f32, max_log_scale: f32) {
        self.log_l11 = self.log_l11.clamp(min_log_scale, max_log_scale);
        self.log_l22 = self.log_l22.clamp(min_log_scale, max_log_scale);
    }

    /// Compute eigenvalues (for scale classification)
    pub fn eigenvalues(&self) -> (f32, f32) {
        let (sigma_xx, sigma_xy, sigma_yy) = self.to_covariance();

        let trace = sigma_xx + sigma_yy;
        let det = sigma_xx * sigma_yy - sigma_xy * sigma_xy;
        let discriminant = (trace * trace - 4.0 * det).max(0.0).sqrt();

        let lambda1 = (trace + discriminant) / 2.0;
        let lambda2 = (trace - discriminant) / 2.0;

        (lambda1, lambda2)
    }

    /// Condition number (anisotropy measure)
    pub fn condition_number(&self) -> f32 {
        let (l1, l2) = self.eigenvalues();
        if l2 > 1e-10 {
            l1 / l2
        } else {
            f32::INFINITY
        }
    }

    /// Isotropic covariance (circular Gaussian)
    pub fn isotropic(sigma: f32) -> Self {
        let log_s = sigma.ln();
        Self {
            log_l11: log_s,
            l21: 0.0,        // No correlation
            log_l22: log_s,  // Same scale in both directions
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isotropic_roundtrip() {
        let sigma = 2.0;
        let lc = LogCholesky::isotropic(sigma);
        let (s_xx, s_xy, s_yy) = lc.to_covariance();

        // Should be [[4, 0], [0, 4]]
        assert!((s_xx - 4.0).abs() < 1e-5);
        assert!(s_xy.abs() < 1e-5);
        assert!((s_yy - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_from_covariance_roundtrip() {
        let original = (5.0, 2.0, 3.0);  // [[5, 2], [2, 3]]

        let lc = LogCholesky::from_covariance(original.0, original.1, original.2).unwrap();
        let reconstructed = lc.to_covariance();

        assert!((original.0 - reconstructed.0).abs() < 1e-5);
        assert!((original.1 - reconstructed.1).abs() < 1e-5);
        assert!((original.2 - reconstructed.2).abs() < 1e-5);
    }

    #[test]
    fn test_always_positive_definite() {
        // Even with arbitrary log-parameters, result is always PSD
        let lc = LogCholesky::new(0.5, 1.5, -0.3);

        let (s_xx, s_xy, s_yy) = lc.to_covariance();
        let det = s_xx * s_yy - s_xy * s_xy;

        assert!(det > 0.0, "Should always be positive-definite");
        assert!(s_xx > 0.0 && s_yy > 0.0);
    }

    #[test]
    fn test_inverse_covariance() {
        let lc = LogCholesky::new(0.0, 0.5, 0.0);  // log(1)=0 → l11=l22=1
        let (sigma_xx, sigma_xy, sigma_yy) = lc.to_covariance();
        let (inv_xx, inv_xy, inv_yy) = lc.inverse_covariance();

        // Σ · Σ^-1 should equal identity
        let prod_00 = sigma_xx * inv_xx + sigma_xy * inv_xy;
        let prod_01 = sigma_xx * inv_xy + sigma_xy * inv_yy;
        let prod_11 = sigma_xy * inv_xy + sigma_yy * inv_yy;

        assert!((prod_00 - 1.0).abs() < 1e-4, "Should be identity");
        assert!(prod_01.abs() < 1e-4);
        assert!((prod_11 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_from_structure_tensor() {
        // Horizontal edge: gradient in X direction
        let e_normal = Vector2::new(1.0, 0.0);   // Gradient direction
        let e_tangent = Vector2::new(0.0, 1.0);  // Along edge
        let coherence = 0.9;  // Strong edge

        let lc = LogCholesky::from_structure_tensor(
            e_normal,
            e_tangent,
            coherence,
            0.6,   // sigma_min
            3.0,   // alpha
            4.0,   // beta
            5.0,   // edt_distance
        );

        // Should create thin Gaussian across edge (X), long along edge (Y)
        let (lambda1, lambda2) = lc.eigenvalues();

        // Major eigenvalue (along edge) should be larger
        assert!(lambda1 > lambda2, "Should be anisotropic");

        // Anisotropy should be ~beta
        let anisotropy = (lambda1 / lambda2).sqrt();
        assert!(anisotropy > 2.0 && anisotropy < 8.0, "Should have reasonable anisotropy");
    }

    #[test]
    fn test_determinant() {
        let lc = LogCholesky::new(1.0, 0.5, 0.5);
        let det_direct = lc.determinant();

        // Also compute from reconstructed matrix
        let (s_xx, s_xy, s_yy) = lc.to_covariance();
        let det_matrix = s_xx * s_yy - s_xy * s_xy;

        assert!((det_direct - det_matrix).abs() < 1e-5);
    }

    #[test]
    fn test_stability_clamping() {
        let mut lc = LogCholesky::new(10.0, 5.0, -8.0);  // Extreme values

        lc.clamp_for_stability(-5.0, 5.0);

        assert!(lc.log_l11 >= -5.0 && lc.log_l11 <= 5.0);
        assert!(lc.log_l22 >= -5.0 && lc.log_l22 <= 5.0);
    }
}
