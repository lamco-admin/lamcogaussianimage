//! Gaussian evaluation and rendering primitives

use crate::{Float, gaussian::Gaussian2D, parameterization::Parameterization, vec::Vector2};

/// Gaussian evaluator with optimization options
pub struct GaussianEvaluator<T: Float> {
    /// Cutoff threshold (default: 1e-5)
    pub cutoff: T,
    /// Number of sigma for bounding box (default: 3.5)
    pub n_sigma: T,
}

impl<T: Float> Default for GaussianEvaluator<T> {
    fn default() -> Self {
        Self {
            cutoff: T::one() / ((T::one() + T::one()).exp().exp().exp().exp().exp()), // ~1e-5
            n_sigma: (T::one() + T::one() + T::one()) + (T::one() / T::two()),
        }
    }
}

impl<T: Float> GaussianEvaluator<T> {
    /// Create new evaluator with custom parameters
    pub fn new(cutoff: T, n_sigma: T) -> Self {
        Self { cutoff, n_sigma }
    }

    /// Evaluate Gaussian at a point
    ///
    /// Returns weight in [0, 1] (before applying opacity)
    #[inline]
    pub fn evaluate<P: Parameterization<T>>(
        &self,
        gaussian: &Gaussian2D<T, P>,
        point: Vector2<T>,
    ) -> T {
        // Compute offset from Gaussian center
        let dx = point.x - gaussian.position.x;
        let dy = point.y - gaussian.position.y;

        // Get inverse covariance
        let inv_cov = gaussian.inverse_covariance();

        // Compute Mahalanobis distance squared: dᵀ Σ⁻¹ d
        let mahal_sq =
            inv_cov[0][0] * dx * dx +
            (inv_cov[0][1] + inv_cov[1][0]) * dx * dy +
            inv_cov[1][1] * dy * dy;

        // Gaussian weight: exp(-0.5 * dᵀ Σ⁻¹ d)
        let weight = (-(mahal_sq / T::two())).exp();

        // Apply cutoff
        if weight < self.cutoff {
            T::zero()
        } else {
            weight
        }
    }

    /// Evaluate with early termination check
    ///
    /// Returns None if point is outside bounding box
    #[inline]
    pub fn evaluate_bounded<P: Parameterization<T>>(
        &self,
        gaussian: &Gaussian2D<T, P>,
        point: Vector2<T>,
    ) -> Option<T> {
        // Quick bounding box check
        let (min, max) = gaussian.bounding_box(self.n_sigma);

        if point.x < min.x || point.x > max.x || point.y < min.y || point.y > max.y {
            return None;
        }

        Some(self.evaluate(gaussian, point))
    }

    /// Batch evaluate for SIMD processing
    ///
    /// Evaluates Gaussian at multiple points
    pub fn evaluate_batch<P: Parameterization<T>>(
        &self,
        gaussian: &Gaussian2D<T, P>,
        points: &[Vector2<T>],
        weights: &mut [T],
    ) {
        assert_eq!(points.len(), weights.len());

        let inv_cov = gaussian.inverse_covariance();
        let gx = gaussian.position.x;
        let gy = gaussian.position.y;

        for (point, weight) in points.iter().zip(weights.iter_mut()) {
            let dx = point.x - gx;
            let dy = point.y - gy;

            let mahal_sq =
                inv_cov[0][0] * dx * dx +
                (inv_cov[0][1] + inv_cov[1][0]) * dx * dy +
                inv_cov[1][1] * dy * dy;

            let w = (-(mahal_sq / T::two())).exp();
            *weight = if w < self.cutoff { T::zero() } else { w };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameterization::Euler;
    use crate::color::Color4;
    use approx::assert_relative_eq;

    #[test]
    fn test_evaluate_center() {
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.1),
            Color4::white(),
            1.0,
        );

        let evaluator = GaussianEvaluator::default();
        let weight = evaluator.evaluate(&gaussian, Vector2::new(0.5, 0.5));

        // At center, weight should be ~1.0
        assert_relative_eq!(weight, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_evaluate_falloff() {
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.1),
            Color4::white(),
            1.0,
        );

        let evaluator = GaussianEvaluator::default();

        // At 1 sigma distance, should be exp(-0.5) ≈ 0.606
        let weight = evaluator.evaluate(&gaussian, Vector2::new(0.6, 0.5));
        assert_relative_eq!(weight, 0.606, epsilon = 0.01);

        // At 2 sigma distance, should be exp(-2) ≈ 0.135
        let weight = evaluator.evaluate(&gaussian, Vector2::new(0.7, 0.5));
        assert_relative_eq!(weight, 0.135, epsilon = 0.01);
    }

    #[test]
    fn test_bounded_evaluation() {
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.05),
            Color4::white(),
            1.0,
        );

        let evaluator = GaussianEvaluator::default();

        // Inside bounding box
        assert!(evaluator.evaluate_bounded(&gaussian, Vector2::new(0.5, 0.5)).is_some());

        // Outside bounding box
        assert!(evaluator.evaluate_bounded(&gaussian, Vector2::new(0.9, 0.9)).is_none());
    }
}
