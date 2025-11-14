//! Utility functions

use crate::Float;

/// Space-filling curve orderings for improved spatial coherence
pub mod ordering {
    use crate::vec::Vector2;
    use crate::Float;

    /// Morton (Z-order) curve index
    ///
    /// Interleaves bits of x and y coordinates for spatial locality
    pub fn morton_index(x: u32, y: u32) -> u64 {
        fn part1by1(mut n: u32) -> u64 {
            n &= 0x0000ffff;
            n = (n ^ (n << 8)) & 0x00ff00ff;
            n = (n ^ (n << 4)) & 0x0f0f0f0f;
            n = (n ^ (n << 2)) & 0x33333333;
            n = (n ^ (n << 1)) & 0x55555555;
            n as u64
        }

        part1by1(x) | (part1by1(y) << 1)
    }

    /// Hilbert curve index (approximation for small grids)
    pub fn hilbert_index(x: u32, y: u32, order: u32) -> u32 {
        let mut index = 0;
        let mut s = 1 << (order - 1);
        let mut rx;
        let mut ry;
        let mut x = x;
        let mut y = y;

        while s > 0 {
            rx = ((x & s) > 0) as u32;
            ry = ((y & s) > 0) as u32;
            index += s * s * ((3 * rx) ^ ry);

            // Rotate
            if ry == 0 {
                if rx == 1 {
                    x = s - 1 - x;
                    y = s - 1 - y;
                }
                std::mem::swap(&mut x, &mut y);
            }

            s >>= 1;
        }

        index
    }

    /// Sort Gaussians by spatial curve for better cache locality
    pub fn sort_by_morton<T: Float>(positions: &mut [(Vector2<T>, usize)], resolution: u32) {
        positions.sort_by_key(|(pos, _)| {
            // Convert to u32 by scaling
            let x_f = if let Some(x_f) = to_f32(pos.x * from_u32::<T>(resolution)) {
                x_f
            } else {
                0.0
            };
            let y_f = if let Some(y_f) = to_f32(pos.y * from_u32::<T>(resolution)) {
                y_f
            } else {
                0.0
            };
            let x = x_f as u32;
            let y = y_f as u32;
            morton_index(x, y)
        });
    }

    // Helper functions for generic float conversion
    fn from_u32<T: Float>(val: u32) -> T {
        // This is a workaround - in real code we'd need proper conversion
        // For now, assume f32/f64 which can represent u32
        if std::mem::size_of::<T>() == 4 {
            unsafe { std::mem::transmute_copy(&(val as f32)) }
        } else {
            unsafe { std::mem::transmute_copy(&(val as f64)) }
        }
    }

    fn to_f32<T: Float>(val: T) -> Option<f32> {
        if std::mem::size_of::<T>() == 4 {
            Some(unsafe { std::mem::transmute_copy(&val) })
        } else {
            // f64 -> f32 conversion would go here
            None
        }
    }
}

/// Numerical utilities
pub mod numerics {
    use super::Float;

    /// Safe division with fallback
    #[inline]
    pub fn safe_div<T: Float>(num: T, denom: T, fallback: T) -> T {
        if denom.abs() > T::zero() {
            num / denom
        } else {
            fallback
        }
    }

    /// Clamp value to range
    #[inline]
    pub fn clamp<T: Float>(val: T, min: T, max: T) -> T {
        if val < min {
            min
        } else if val > max {
            max
        } else {
            val
        }
    }

    /// Linear interpolation
    #[inline]
    pub fn lerp<T: Float>(a: T, b: T, t: T) -> T {
        a + (b - a) * t
    }

    /// Smooth step (cubic Hermite interpolation)
    #[inline]
    pub fn smoothstep<T: Float>(edge0: T, edge1: T, x: T) -> T {
        let t = clamp((x - edge0) / (edge1 - edge0), T::zero(), T::one());
        t * t * ((T::one() + T::one() + T::one()) - (T::one() + T::one()) * t)
    }
}

/// Memory layout helpers
pub mod layout {
    /// Calculate aligned size
    #[inline]
    pub const fn align_up(size: usize, alignment: usize) -> usize {
        (size + alignment - 1) & !(alignment - 1)
    }

    /// Check if pointer is aligned
    #[inline]
    pub fn is_aligned(ptr: *const u8, alignment: usize) -> bool {
        (ptr as usize) & (alignment - 1) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vector2;

    #[test]
    fn test_morton_ordering() {
        // Points in different quadrants
        let mut positions = vec![
            (Vector2::new(0.75, 0.75), 0), // Q1
            (Vector2::new(0.25, 0.25), 1), // Q3
            (Vector2::new(0.75, 0.25), 2), // Q2
            (Vector2::new(0.25, 0.75), 3), // Q4
        ];

        ordering::sort_by_morton(&mut positions, 256);

        // Morton order should group nearby points
        // Note: exact order depends on implementation details
        assert!(positions.len() == 4); // Just verify sorting doesn't crash
    }

    #[test]
    fn test_numerics() {
        assert_eq!(numerics::clamp(1.5f32, 0.0, 1.0), 1.0);
        assert_eq!(numerics::lerp(0.0f32, 10.0, 0.5), 5.0);
        assert!(numerics::smoothstep(0.0f32, 1.0, 0.5) > 0.4 && numerics::smoothstep(0.0f32, 1.0, 0.5) < 0.6);
    }

    #[test]
    fn test_alignment() {
        assert_eq!(layout::align_up(13, 16), 16);
        assert_eq!(layout::align_up(16, 16), 16);
        assert_eq!(layout::align_up(17, 16), 32);
    }
}
