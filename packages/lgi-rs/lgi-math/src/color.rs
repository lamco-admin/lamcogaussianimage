//! Color representations and operations

use crate::Float;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 4-channel color (RGBA or RGB + padding)
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Color4<T: Float> {
    pub r: T,
    pub g: T,
    pub b: T,
    pub a: T,
}

impl<T: Float> Color4<T> {
    #[inline]
    pub fn new(r: T, g: T, b: T, a: T) -> Self {
        Self { r, g, b, a }
    }

    #[inline]
    pub fn rgb(r: T, g: T, b: T) -> Self {
        Self::new(r, g, b, T::one())
    }

    #[inline]
    pub fn black() -> Self {
        Self::new(T::zero(), T::zero(), T::zero(), T::one())
    }

    #[inline]
    pub fn white() -> Self {
        Self::new(T::one(), T::one(), T::one(), T::one())
    }

    /// Clamp all channels to [0, 1]
    #[inline]
    pub fn clamp(&self) -> Self {
        Self {
            r: self.r.clamp(T::zero(), T::one()),
            g: self.g.clamp(T::zero(), T::one()),
            b: self.b.clamp(T::zero(), T::one()),
            a: self.a.clamp(T::zero(), T::one()),
        }
    }

    /// Linear to sRGB conversion
    pub fn to_srgb(&self) -> Self {
        fn linear_to_srgb<T: Float>(c: T) -> T {
            if c <= T::zero() {
                T::zero()
            } else if c < (T::one() / (T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one())) {
                // c < 0.0031308
                c * (T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one()) + (T::one() / (T::one() + T::one() + T::one() + T::one() + T::one()))
            } else if c < T::one() {
                // 1.055 * c^(1/2.4) - 0.055
                let exp_val = c.ln() / ((T::one() + T::one()) + (T::one() / (T::one() + T::one() + T::one() + T::one()))).ln();
                ((T::one() + (T::one() / (T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one()))) * exp_val.exp()) - (T::one() / (T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one()))
            } else {
                T::one()
            }
        }

        Self {
            r: linear_to_srgb(self.r),
            g: linear_to_srgb(self.g),
            b: linear_to_srgb(self.b),
            a: self.a,
        }
    }
}

/// Color space enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ColorSpace {
    Linear,
    SRGB,
    DisplayP3,
    BT2020,
}
