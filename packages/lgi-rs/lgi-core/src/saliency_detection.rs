//! Saliency detection for content-adaptive encoding

use crate::ImageBuffer;
use lgi_math::color::Color4;

pub struct SaliencyDetector {
    pub threshold: f32,
}

impl Default for SaliencyDetector {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

impl SaliencyDetector {
    pub fn compute_saliency_map(&self, image: &ImageBuffer<f32>) -> Vec<f32> {
        let mut saliency = vec![0.0; (image.width * image.height) as usize];

        // Simple gradient-based saliency
        for y in 1..(image.height-1) {
            for x in 1..(image.width-1) {
                if let (Some(c), Some(left), Some(right), Some(up), Some(down)) = (
                    image.get_pixel(x, y),
                    image.get_pixel(x-1, y),
                    image.get_pixel(x+1, y),
                    image.get_pixel(x, y-1),
                    image.get_pixel(x, y+1),
                ) {
                    let grad_x = (right.r - left.r).abs() + (right.g - left.g).abs() + (right.b - left.b).abs();
                    let grad_y = (down.r - up.r).abs() + (down.g - up.g).abs() + (down.b - up.b).abs();
                    saliency[(y * image.width + x) as usize] = (grad_x + grad_y) / 6.0;
                }
            }
        }

        saliency
    }

    pub fn get_salient_regions(&self, saliency_map: &[f32]) -> Vec<bool> {
        saliency_map.iter().map(|&s| s > self.threshold).collect()
    }
}
