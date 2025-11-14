//! Generic vector types
//!
//! We need our own Vec2 because glam::Vec2 is f32-only,
//! but we want to be generic over T: Float

use crate::Float;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Generic 2D vector
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Vector2<T: Float> {
    pub x: T,
    pub y: T,
}

impl<T: Float> Vector2<T> {
    #[inline(always)]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }

    #[inline(always)]
    pub fn one() -> Self {
        Self::new(T::one(), T::one())
    }

    #[inline(always)]
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y
    }

    #[inline(always)]
    pub fn length_squared(self) -> T {
        self.dot(self)
    }

    #[inline(always)]
    pub fn length(self) -> T {
        self.length_squared().sqrt()
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        Self::new(self.x / len, self.y / len)
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y))
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y))
    }

    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self::new(
            self.x.clamp(min.x, max.x),
            self.y.clamp(min.y, max.y),
        )
    }
}

impl<T: Float> std::ops::Add for Vector2<T> {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

impl<T: Float> std::ops::Sub for Vector2<T> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

impl<T: Float> std::ops::Mul<T> for Vector2<T> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, scalar: T) -> Self {
        Self::new(self.x * scalar, self.y * scalar)
    }
}

impl<T: Float> std::ops::Div<T> for Vector2<T> {
    type Output = Self;

    #[inline(always)]
    fn div(self, scalar: T) -> Self {
        Self::new(self.x / scalar, self.y / scalar)
    }
}

impl<T: Float> std::ops::Neg for Vector2<T> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

// Convenient type alias
pub type Vec2f32 = Vector2<f32>;
pub type Vec2f64 = Vector2<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_ops() {
        let v1 = Vector2::new(1.0f32, 2.0);
        let v2 = Vector2::new(3.0, 4.0);

        let sum = v1 + v2;
        assert_eq!(sum.x, 4.0);
        assert_eq!(sum.y, 6.0);

        let diff = v2 - v1;
        assert_eq!(diff.x, 2.0);
        assert_eq!(diff.y, 2.0);

        let scaled = v1 * 2.0;
        assert_eq!(scaled.x, 2.0);
        assert_eq!(scaled.y, 4.0);
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vector2::new(1.0f32, 0.0);
        let v2 = Vector2::new(0.0, 1.0);

        assert_eq!(v1.dot(v2), 0.0);
        assert_eq!(v1.dot(v1), 1.0);
    }
}
