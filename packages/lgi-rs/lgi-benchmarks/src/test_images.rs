//! Synthetic test image generation
//!
//! Generates diverse test patterns to stress-test the codec:
//! - Flat colors (worst case for Gaussians)
//! - Gradients (smooth transitions)
//! - High-frequency patterns (edges, textures)
//! - Natural scenes (combinations)

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Test image types
#[derive(Debug, Clone, Copy)]
pub enum TestPattern {
    /// Solid color (simplest case)
    SolidColor,
    /// Linear gradient
    LinearGradient,
    /// Radial gradient
    RadialGradient,
    /// Checkerboard pattern
    Checkerboard,
    /// Concentric circles
    ConcentricCircles,
    /// Frequency sweep (low to high)
    FrequencySweep,
    /// Random noise
    RandomNoise,
    /// Natural-like (combination)
    NaturalScene,
    /// Edges and corners (geometric)
    Geometric,
    /// Text-like patterns (thin lines)
    TextPattern,
}

/// Test image generator
pub struct TestImageGenerator {
    size: u32,
    seed: Option<u64>,
}

impl TestImageGenerator {
    /// Create new generator
    pub fn new(size: u32) -> Self {
        Self { size, seed: None }
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Generate test image
    pub fn generate(&self, pattern: TestPattern) -> ImageBuffer<f32> {
        match pattern {
            TestPattern::SolidColor => self.gen_solid_color(),
            TestPattern::LinearGradient => self.gen_linear_gradient(),
            TestPattern::RadialGradient => self.gen_radial_gradient(),
            TestPattern::Checkerboard => self.gen_checkerboard(),
            TestPattern::ConcentricCircles => self.gen_concentric_circles(),
            TestPattern::FrequencySweep => self.gen_frequency_sweep(),
            TestPattern::RandomNoise => self.gen_random_noise(),
            TestPattern::NaturalScene => self.gen_natural_scene(),
            TestPattern::Geometric => self.gen_geometric(),
            TestPattern::TextPattern => self.gen_text_pattern(),
        }
    }

    fn gen_solid_color(&self) -> ImageBuffer<f32> {
        ImageBuffer::with_background(self.size, self.size, Color4::rgb(0.5, 0.3, 0.7))
    }

    fn gen_linear_gradient(&self) -> ImageBuffer<f32> {
        let mut img = ImageBuffer::new(self.size, self.size);

        for y in 0..self.size {
            for x in 0..self.size {
                let t = x as f32 / self.size as f32;
                img.set_pixel(x, y, Color4::rgb(t, 1.0 - t, 0.5));
            }
        }

        img
    }

    fn gen_radial_gradient(&self) -> ImageBuffer<f32> {
        let mut img = ImageBuffer::new(self.size, self.size);
        let center = self.size as f32 / 2.0;

        for y in 0..self.size {
            for x in 0..self.size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dist = (dx * dx + dy * dy).sqrt() / center;
                let t = dist.min(1.0);

                img.set_pixel(x, y, Color4::rgb(1.0 - t, t * 0.5, t));
            }
        }

        img
    }

    fn gen_checkerboard(&self) -> ImageBuffer<f32> {
        let mut img = ImageBuffer::new(self.size, self.size);
        let square_size = self.size / 8; // 8×8 grid

        for y in 0..self.size {
            for x in 0..self.size {
                let check_x = (x / square_size) % 2;
                let check_y = (y / square_size) % 2;
                let is_white = (check_x + check_y) % 2 == 0;

                let color = if is_white {
                    Color4::white()
                } else {
                    Color4::rgb(0.2, 0.2, 0.2)
                };

                img.set_pixel(x, y, color);
            }
        }

        img
    }

    fn gen_concentric_circles(&self) -> ImageBuffer<f32> {
        let mut img = ImageBuffer::new(self.size, self.size);
        let center = self.size as f32 / 2.0;
        let num_rings = 10;

        for y in 0..self.size {
            for x in 0..self.size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dist = (dx * dx + dy * dy).sqrt();

                let ring = ((dist / center) * num_rings as f32) as u32 % 2;
                let color = if ring == 0 {
                    Color4::rgb(1.0, 0.3, 0.3)
                } else {
                    Color4::rgb(0.3, 0.3, 1.0)
                };

                img.set_pixel(x, y, color);
            }
        }

        img
    }

    fn gen_frequency_sweep(&self) -> ImageBuffer<f32> {
        let mut img = ImageBuffer::new(self.size, self.size);

        for y in 0..self.size {
            for x in 0..self.size {
                let nx = x as f32 / self.size as f32;
                let ny = y as f32 / self.size as f32;

                // Frequency increases with x
                let freq = 1.0 + nx * 20.0;
                let phase = ny * 2.0 * std::f32::consts::PI;
                let wave = (freq * phase).sin() * 0.5 + 0.5;

                img.set_pixel(x, y, Color4::rgb(wave, wave, wave));
            }
        }

        img
    }

    fn gen_random_noise(&self) -> ImageBuffer<f32> {
        let mut img = ImageBuffer::new(self.size, self.size);
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        for y in 0..self.size {
            for x in 0..self.size {
                let r: f32 = rng.gen();
                let g: f32 = rng.gen();
                let b: f32 = rng.gen();

                img.set_pixel(x, y, Color4::rgb(r, g, b));
            }
        }

        img
    }

    fn gen_natural_scene(&self) -> ImageBuffer<f32> {
        let mut img = ImageBuffer::new(self.size, self.size);
        let center = self.size as f32 / 2.0;

        for y in 0..self.size {
            for x in 0..self.size {
                let nx = x as f32 / self.size as f32;
                let ny = y as f32 / self.size as f32;

                // Sky gradient
                let sky_blue = Color4::rgb(0.5, 0.7, 1.0);
                let horizon = Color4::rgb(1.0, 0.9, 0.8);
                let sky_t = ny;
                let mut color = Color4::rgb(
                    sky_blue.r * (1.0 - sky_t) + horizon.r * sky_t,
                    sky_blue.g * (1.0 - sky_t) + horizon.g * sky_t,
                    sky_blue.b * (1.0 - sky_t) + horizon.b * sky_t,
                );

                // Add a "sun"
                let dx = x as f32 - center * 1.5;
                let dy = y as f32 - center * 0.3;
                let sun_dist = (dx * dx + dy * dy).sqrt();

                if sun_dist < 30.0 {
                    let sun_intensity = (1.0 - sun_dist / 30.0).max(0.0);
                    color.r = color.r * (1.0 - sun_intensity) + sun_intensity;
                    color.g = color.g * (1.0 - sun_intensity) + sun_intensity * 0.9;
                    color.b = color.b * (1.0 - sun_intensity) + sun_intensity * 0.3;
                }

                // Add "ground"
                if ny > 0.6 {
                    let ground_green = Color4::rgb(0.3, 0.6, 0.2);
                    let ground_t = (ny - 0.6) / 0.4;
                    color.r = color.r * (1.0 - ground_t) + ground_green.r * ground_t;
                    color.g = color.g * (1.0 - ground_t) + ground_green.g * ground_t;
                    color.b = color.b * (1.0 - ground_t) + ground_green.b * ground_t;
                }

                img.set_pixel(x, y, color);
            }
        }

        img
    }

    fn gen_geometric(&self) -> ImageBuffer<f32> {
        let mut img = ImageBuffer::with_background(self.size, self.size, Color4::rgb(0.9, 0.9, 0.9));

        // Draw rectangle
        for y in self.size / 4..self.size / 2 {
            for x in self.size / 4..self.size / 2 {
                img.set_pixel(x, y, Color4::rgb(1.0, 0.0, 0.0));
            }
        }

        // Draw triangle
        for y in self.size / 2..(3 * self.size / 4) {
            let y_offset = y - self.size / 2;
            let width = y_offset;
            let x_start = self.size / 2;

            for x in x_start..(x_start + width) {
                if x < self.size {
                    img.set_pixel(x, y, Color4::rgb(0.0, 0.0, 1.0));
                }
            }
        }

        img
    }

    fn gen_text_pattern(&self) -> ImageBuffer<f32> {
        let mut img = ImageBuffer::with_background(self.size, self.size, Color4::white());

        // Draw horizontal lines (simulating text)
        let line_height = self.size / 20;
        for line in 0..10 {
            let y_start = line * line_height * 2;

            for y in y_start..(y_start + line_height / 4) {
                for x in (self.size / 10)..(9 * self.size / 10) {
                    if y < self.size && x < self.size {
                        img.set_pixel(x, y, Color4::black());
                    }
                }
            }
        }

        img
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_patterns() {
        let gen = TestImageGenerator::new(256);

        let patterns = [
            TestPattern::SolidColor,
            TestPattern::LinearGradient,
            TestPattern::RadialGradient,
            TestPattern::Checkerboard,
            TestPattern::ConcentricCircles,
            TestPattern::FrequencySweep,
            TestPattern::RandomNoise,
            TestPattern::NaturalScene,
            TestPattern::Geometric,
            TestPattern::TextPattern,
        ];

        for pattern in patterns {
            let img = gen.generate(pattern);
            assert_eq!(img.width, 256);
            assert_eq!(img.height, 256);
        }
    }
}
