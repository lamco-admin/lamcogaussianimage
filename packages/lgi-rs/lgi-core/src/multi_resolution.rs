//! Multi-resolution pyramid for rendering

use crate::ImageBuffer;

pub struct MultiResolutionPyramid {
    levels: Vec<ImageBuffer<f32>>,
}

impl MultiResolutionPyramid {
    pub fn build(image: &ImageBuffer<f32>, num_levels: usize) -> Self {
        let mut levels = vec![image.clone()];

        for _ in 1..num_levels {
            let prev = levels.last().unwrap();
            let downsampled = downsample_2x(prev);
            levels.push(downsampled);
        }

        Self { levels }
    }

    pub fn get_level(&self, level: usize) -> Option<&ImageBuffer<f32>> {
        self.levels.get(level)
    }
}

fn downsample_2x(image: &ImageBuffer<f32>) -> ImageBuffer<f32> {
    let new_width = image.width / 2;
    let new_height = image.height / 2;
    let mut output = ImageBuffer::new(new_width, new_height);

    for y in 0..new_height {
        for x in 0..new_width {
            let sx = x * 2;
            let sy = y * 2;

            if let (Some(c00), Some(c10), Some(c01), Some(c11)) = (
                image.get_pixel(sx, sy),
                image.get_pixel(sx+1, sy),
                image.get_pixel(sx, sy+1),
                image.get_pixel(sx+1, sy+1),
            ) {
                let r = (c00.r + c10.r + c01.r + c11.r) / 4.0;
                let g = (c00.g + c10.g + c01.g + c11.g) / 4.0;
                let b = (c00.b + c10.b + c01.b + c11.b) / 4.0;
                output.set_pixel(x, y, lgi_math::color::Color4::new(r, g, b, 1.0));
            }
        }
    }

    output
}
