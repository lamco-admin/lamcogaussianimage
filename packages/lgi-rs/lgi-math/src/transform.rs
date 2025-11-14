//! Geometric transformations for Gaussians

use crate::{Float, gaussian::Gaussian2D, parameterization::{Parameterization, Euler}, vec::Vector2};

/// 2D affine transform
#[derive(Debug, Clone, Copy)]
pub struct Affine2<T: Float> {
    /// Linear part (2×2 matrix)
    pub linear: [[T; 2]; 2],
    /// Translation
    pub translation: Vector2<T>,
}

impl<T: Float> Affine2<T> {
    /// Identity transform
    pub fn identity() -> Self {
        Self {
            linear: [[T::one(), T::zero()], [T::zero(), T::one()]],
            translation: Vector2::zero(),
        }
    }

    /// Create translation
    pub fn translation(offset: Vector2<T>) -> Self {
        Self {
            linear: [[T::one(), T::zero()], [T::zero(), T::one()]],
            translation: offset,
        }
    }

    /// Create rotation (angle in radians)
    pub fn rotation(angle: T) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Self {
            linear: [[cos, -sin], [sin, cos]],
            translation: Vector2::zero(),
        }
    }

    /// Create scale
    pub fn scale(sx: T, sy: T) -> Self {
        Self {
            linear: [[sx, T::zero()], [T::zero(), sy]],
            translation: Vector2::zero(),
        }
    }

    /// Transform a point
    #[inline]
    pub fn transform_point(&self, p: Vector2<T>) -> Vector2<T> {
        Vector2::new(
            self.linear[0][0] * p.x + self.linear[0][1] * p.y + self.translation.x,
            self.linear[1][0] * p.x + self.linear[1][1] * p.y + self.translation.y,
        )
    }

    /// Transform a Gaussian
    ///
    /// For Gaussian G(μ, Σ), under affine transform A:
    /// - μ' = A·μ + t
    /// - Σ' = A·Σ·Aᵀ
    pub fn transform_gaussian<P: Parameterization<T>>(
        &self,
        gaussian: &Gaussian2D<T, P>,
    ) -> Gaussian2D<T, Euler<T>> {
        // Transform position
        let new_pos = self.transform_point(gaussian.position);

        // Transform covariance: Σ' = A·Σ·Aᵀ
        let cov = gaussian.covariance();
        let a = &self.linear;

        // Σ' = A·Σ·Aᵀ
        let temp00 = a[0][0] * cov[0][0] + a[0][1] * cov[1][0];
        let temp01 = a[0][0] * cov[0][1] + a[0][1] * cov[1][1];
        let temp10 = a[1][0] * cov[0][0] + a[1][1] * cov[1][0];
        let temp11 = a[1][0] * cov[0][1] + a[1][1] * cov[1][1];

        let new_cov = [
            [temp00 * a[0][0] + temp01 * a[0][1], temp00 * a[1][0] + temp01 * a[1][1]],
            [temp10 * a[0][0] + temp11 * a[0][1], temp10 * a[1][0] + temp11 * a[1][1]],
        ];

        // Convert back to Euler parameterization
        // Extract eigenvalues and eigenvectors
        let trace = new_cov[0][0] + new_cov[1][1];
        let det = new_cov[0][0] * new_cov[1][1] - new_cov[0][1] * new_cov[1][0];
        let disc = (trace * trace - T::two() * T::two() * det).sqrt();

        let lambda1 = (trace + disc) / T::two();
        let lambda2 = (trace - disc) / T::two();

        let scale_x = lambda1.sqrt();
        let scale_y = lambda2.sqrt();

        // Compute rotation from eigenvector
        let rotation = if new_cov[0][1].abs() > T::zero() {
            new_cov[0][1].atan2(lambda1 - new_cov[1][1])
        } else {
            T::zero()
        };

        Gaussian2D::new(
            new_pos,
            Euler::new(scale_x, scale_y, rotation),
            gaussian.color,
            gaussian.opacity,
        )
    }
}

/// Viewport transformation
pub struct ViewportTransform {
    /// Canvas size in normalized coordinates [0,1]
    canvas_size: (u32, u32),
    /// Output size in pixels
    output_size: (u32, u32),
}

impl ViewportTransform {
    pub fn new(canvas_size: (u32, u32), output_size: (u32, u32)) -> Self {
        Self {
            canvas_size,
            output_size,
        }
    }

    /// Convert normalized [0,1] coordinates to pixel coordinates
    #[inline]
    pub fn to_pixels(&self, normalized: Vector2<f32>) -> (i32, i32) {
        let x = (normalized.x * self.output_size.0 as f32) as i32;
        let y = (normalized.y * self.output_size.1 as f32) as i32;
        (x, y)
    }

    /// Convert pixel coordinates to normalized [0,1] coordinates
    #[inline]
    pub fn to_normalized(&self, pixel: (i32, i32)) -> Vector2<f32> {
        Vector2::new(
            pixel.0 as f32 / self.output_size.0 as f32,
            pixel.1 as f32 / self.output_size.1 as f32,
        )
    }

    /// Scale factor from normalized to pixel space
    #[inline]
    pub fn scale_factor(&self) -> (f32, f32) {
        (
            self.output_size.0 as f32 / self.canvas_size.0 as f32,
            self.output_size.1 as f32 / self.canvas_size.1 as f32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color4;
    use approx::assert_relative_eq;

    #[test]
    fn test_affine_translation() {
        let transform = Affine2::translation(Vector2::new(0.5, 0.5));
        let point = Vector2::new(0.2, 0.3);
        let result = transform.transform_point(point);

        assert_relative_eq!(result.x, 0.7, epsilon = 1e-5);
        assert_relative_eq!(result.y, 0.8, epsilon = 1e-5);
    }

    #[test]
    fn test_gaussian_transform() {
        let gaussian = Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::new(0.1, 0.1, 0.0),
            Color4::white(),
            1.0,
        );

        let transform = Affine2::translation(Vector2::new(0.2, 0.3));
        let transformed = transform.transform_gaussian(&gaussian);

        assert_relative_eq!(transformed.position.x, 0.7, epsilon = 1e-5);
        assert_relative_eq!(transformed.position.y, 0.8, epsilon = 1e-5);
    }

    #[test]
    fn test_viewport_transform() {
        let viewport = ViewportTransform::new((1920, 1080), (3840, 2160));

        let (px, py) = viewport.to_pixels(Vector2::new(0.5, 0.5));
        assert_eq!(px, 1920);
        assert_eq!(py, 1080);

        let normalized = viewport.to_normalized((1920, 1080));
        assert_relative_eq!(normalized.x, 0.5, epsilon = 1e-5);
        assert_relative_eq!(normalized.y, 0.5, epsilon = 1e-5);
    }
}
