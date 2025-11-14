//! Alpha compositing operations

use crate::{Float, color::Color4};

/// Alpha compositing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    /// Straight alpha (default)
    Straight,
    /// Premultiplied alpha
    Premultiplied,
}

/// Compositor for accumulating Gaussian contributions
pub struct Compositor<T: Float> {
    mode: AlphaMode,
    /// Early termination threshold (default: 0.999)
    termination_threshold: T,
}

impl<T: Float> Default for Compositor<T> {
    fn default() -> Self {
        Self {
            mode: AlphaMode::Straight,
            termination_threshold: (T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one()) / ((T::one() + T::one()) * (T::one() + T::one() + T::one() + T::one() + T::one())),
        }
    }
}

impl<T: Float> Compositor<T> {
    /// Create new compositor
    pub fn new(mode: AlphaMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Set early termination threshold
    pub fn with_termination(mut self, threshold: T) -> Self {
        self.termination_threshold = threshold;
        self
    }

    /// Front-to-back Porter-Duff over compositing
    ///
    /// Accumulates: C_out = C_out + (1 - A_out) * C_src * A_src * weight
    ///             A_out = A_out + (1 - A_out) * A_src * weight
    ///
    /// Returns true if early termination triggered
    #[inline]
    pub fn composite_over(
        &self,
        accum_color: &mut Color4<T>,
        accum_alpha: &mut T,
        src_color: Color4<T>,
        src_alpha: T,
        weight: T,
    ) -> bool {
        let alpha_contrib = src_alpha * weight;
        let one_minus_accum = T::one() - *accum_alpha;

        match self.mode {
            AlphaMode::Straight => {
                // Standard over: C_out = C_accum + (1 - A_accum) * C_src * α
                accum_color.r = accum_color.r + one_minus_accum * src_color.r * alpha_contrib;
                accum_color.g = accum_color.g + one_minus_accum * src_color.g * alpha_contrib;
                accum_color.b = accum_color.b + one_minus_accum * src_color.b * alpha_contrib;
            },
            AlphaMode::Premultiplied => {
                // Premultiplied: Color already has alpha baked in
                accum_color.r = accum_color.r + one_minus_accum * src_color.r * weight;
                accum_color.g = accum_color.g + one_minus_accum * src_color.g * weight;
                accum_color.b = accum_color.b + one_minus_accum * src_color.b * weight;
            },
        }

        // Accumulate alpha
        *accum_alpha = *accum_alpha + one_minus_accum * alpha_contrib;

        // Early termination check
        *accum_alpha >= self.termination_threshold
    }

    /// Blend with background color
    #[inline]
    pub fn blend_background(
        &self,
        accum_color: Color4<T>,
        accum_alpha: T,
        background: Color4<T>,
    ) -> Color4<T> {
        let one_minus_alpha = T::one() - accum_alpha;

        Color4 {
            r: accum_color.r + one_minus_alpha * background.r,
            g: accum_color.g + one_minus_alpha * background.g,
            b: accum_color.b + one_minus_alpha * background.b,
            a: T::one(), // Fully opaque output
        }
    }

    /// Composite multiple layers efficiently
    ///
    /// Returns (final_color, final_alpha, num_layers_used)
    pub fn composite_layers(
        &self,
        layers: &[(Color4<T>, T, T)], // (color, alpha, weight)
        background: Color4<T>,
    ) -> (Color4<T>, T, usize) {
        let mut accum_color = Color4::new(T::zero(), T::zero(), T::zero(), T::zero());
        let mut accum_alpha = T::zero();
        let mut count = 0;

        for &(color, alpha, weight) in layers {
            if self.composite_over(&mut accum_color, &mut accum_alpha, color, alpha, weight) {
                count += 1;
                break; // Early termination
            }
            count += 1;
        }

        let final_color = self.blend_background(accum_color, accum_alpha, background);
        (final_color, accum_alpha, count)
    }
}

/// Batch compositor for SIMD-friendly processing
pub struct BatchCompositor<T: Float> {
    compositor: Compositor<T>,
}

impl<T: Float> BatchCompositor<T> {
    pub fn new(mode: AlphaMode) -> Self {
        Self {
            compositor: Compositor::new(mode),
        }
    }

    /// Composite a batch of pixels
    ///
    /// All slices must have the same length (num_pixels)
    pub fn composite_batch(
        &self,
        accum_colors: &mut [Color4<T>],
        accum_alphas: &mut [T],
        src_colors: &[Color4<T>],
        src_alphas: &[T],
        weights: &[T],
    ) {
        assert_eq!(accum_colors.len(), accum_alphas.len());
        assert_eq!(accum_colors.len(), src_colors.len());
        assert_eq!(accum_colors.len(), src_alphas.len());
        assert_eq!(accum_colors.len(), weights.len());

        for i in 0..accum_colors.len() {
            self.compositor.composite_over(
                &mut accum_colors[i],
                &mut accum_alphas[i],
                src_colors[i],
                src_alphas[i],
                weights[i],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_basic_compositing() {
        let compositor = Compositor::default();

        let mut accum_color = Color4::new(0.0, 0.0, 0.0, 0.0);
        let mut accum_alpha = 0.0;

        let src_color = Color4::rgb(1.0, 0.0, 0.0);
        let src_alpha = 0.5;
        let weight = 1.0;

        compositor.composite_over(&mut accum_color, &mut accum_alpha, src_color, src_alpha, weight);

        assert_relative_eq!(accum_color.r, 0.5, epsilon = 1e-5);
        assert_relative_eq!(accum_alpha, 0.5, epsilon = 1e-5);
    }

    #[test]
    fn test_early_termination() {
        let compositor = Compositor::default().with_termination(0.99);

        let mut accum_color = Color4::black();
        let mut accum_alpha = 0.0;

        let src_color = Color4::white();
        let src_alpha = 1.0;
        let weight = 1.0;

        let terminated = compositor.composite_over(&mut accum_color, &mut accum_alpha, src_color, src_alpha, weight);

        assert!(terminated);
        assert!(accum_alpha >= 0.99);
    }

    #[test]
    fn test_background_blend() {
        let compositor = Compositor::default();

        let accum_color = Color4::rgb(0.5, 0.0, 0.0);
        let accum_alpha = 0.5;
        let background = Color4::rgb(0.0, 0.0, 1.0);

        let final_color = compositor.blend_background(accum_color, accum_alpha, background);

        // Should be 50% red, 50% blue
        assert_relative_eq!(final_color.r, 0.5, epsilon = 1e-5);
        assert_relative_eq!(final_color.g, 0.0, epsilon = 1e-5);
        assert_relative_eq!(final_color.b, 0.5, epsilon = 1e-5);
    }
}
