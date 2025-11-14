//! Image buffer for storing and manipulating pixel data

use lgi_math::{Float, color::Color4};
use image::{RgbaImage, Rgba};

/// Image buffer storing RGBA pixel data
#[derive(Debug, Clone)]
pub struct ImageBuffer<T: Float> {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Pixel data in row-major order (RGBA interleaved)
    pub data: Vec<Color4<T>>,
}

impl<T: Float> ImageBuffer<T> {
    /// Create new image buffer
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            data: vec![Color4::new(T::zero(), T::zero(), T::zero(), T::zero()); size],
        }
    }

    /// Create from background color
    pub fn with_background(width: u32, height: u32, background: Color4<T>) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            data: vec![background; size],
        }
    }

    /// Get pixel at (x, y)
    #[inline]
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<Color4<T>> {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            Some(self.data[idx])
        } else {
            None
        }
    }

    /// Set pixel at (x, y)
    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, color: Color4<T>) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            self.data[idx] = color;
        }
    }

    /// Get mutable pixel reference
    #[inline]
    pub fn get_pixel_mut(&mut self, x: u32, y: u32) -> Option<&mut Color4<T>> {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            Some(&mut self.data[idx])
        } else {
            None
        }
    }

    /// Fill entire buffer with color
    pub fn fill(&mut self, color: Color4<T>) {
        for pixel in &mut self.data {
            *pixel = color;
        }
    }

    /// Clone a region of the image
    pub fn clone_region(&self, x: u32, y: u32, width: u32, height: u32) -> Self {
        let mut result = Self::new(width, height);

        for dy in 0..height {
            for dx in 0..width {
                if let Some(pixel) = self.get_pixel(x + dx, y + dy) {
                    result.set_pixel(dx, dy, pixel);
                }
            }
        }

        result
    }
}

impl ImageBuffer<f32> {
    /// Convert sRGB to linear color space
    /// JPEG/PNG images are typically in sRGB, but Gaussian blending math requires linear
    fn srgb_to_linear(srgb: f32) -> f32 {
        if srgb <= 0.04045 {
            srgb / 12.92
        } else {
            ((srgb + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Convert linear to sRGB for saving
    fn linear_to_srgb(linear: f32) -> f32 {
        if linear <= 0.0031308 {
            linear * 12.92
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        }
    }

    /// Load from image crate format
    pub fn from_rgba8(img: &RgbaImage) -> Self {
        let width = img.width();
        let height = img.height();
        let size = (width * height) as usize;
        let mut data = Vec::with_capacity(size);

        for pixel in img.pixels() {
            // Convert from sRGB to linear (CRITICAL for proper Gaussian blending!)
            let r_srgb = pixel[0] as f32 / 255.0;
            let g_srgb = pixel[1] as f32 / 255.0;
            let b_srgb = pixel[2] as f32 / 255.0;
            let a_linear = pixel[3] as f32 / 255.0;

            data.push(Color4::new(
                Self::srgb_to_linear(r_srgb),
                Self::srgb_to_linear(g_srgb),
                Self::srgb_to_linear(b_srgb),
                a_linear,
            ));
        }

        Self { width, height, data }
    }

    /// Convert to RGBA8 for saving (linear → sRGB conversion)
    pub fn to_rgba8(&self) -> RgbaImage {
        let mut img = RgbaImage::new(self.width, self.height);

        for (idx, pixel) in self.data.iter().enumerate() {
            let x = (idx as u32) % self.width;
            let y = (idx as u32) / self.width;

            let clamped = pixel.clamp();

            // Convert from linear to sRGB for proper display
            img.put_pixel(
                x,
                y,
                Rgba([
                    (Self::linear_to_srgb(clamped.r) * 255.0) as u8,
                    (Self::linear_to_srgb(clamped.g) * 255.0) as u8,
                    (Self::linear_to_srgb(clamped.b) * 255.0) as u8,
                    (clamped.a * 255.0) as u8,
                ]),
            );
        }

        img
    }

    /// Load from file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> crate::Result<Self> {
        let img = image::open(path)
            .map_err(|e| crate::LgiError::ImageError(e.to_string()))?
            .to_rgba8();
        Ok(Self::from_rgba8(&img))
    }

    /// Save to file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> crate::Result<()> {
        let img = self.to_rgba8();
        img.save(path)
            .map_err(|e| crate::LgiError::ImageError(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_buffer_creation() {
        let buf = ImageBuffer::<f32>::new(100, 100);
        assert_eq!(buf.width, 100);
        assert_eq!(buf.height, 100);
        assert_eq!(buf.data.len(), 10000);
    }

    #[test]
    fn test_pixel_access() {
        let mut buf = ImageBuffer::<f32>::new(10, 10);
        let color = Color4::new(1.0, 0.5, 0.25, 1.0);

        buf.set_pixel(5, 5, color);
        let retrieved = buf.get_pixel(5, 5).unwrap();

        assert_eq!(retrieved.r, 1.0);
        assert_eq!(retrieved.g, 0.5);
    }
}
