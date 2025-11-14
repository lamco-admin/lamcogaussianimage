//! Error metrics and quality assessment

use crate::ImageBuffer;

pub fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= (rendered.width * rendered.height * 3) as f32;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}

pub fn compute_region_psnr(
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
    x_start: u32,
    y_start: u32,
    region_width: u32,
    region_height: u32,
) -> f32 {
    let mut mse = 0.0;
    let mut count = 0;

    for y in y_start..(y_start + region_height).min(rendered.height) {
        for x in x_start..(x_start + region_width).min(rendered.width) {
            if let (Some(r), Some(t)) = (rendered.get_pixel(x, y), target.get_pixel(x, y)) {
                mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
                count += 3;
            }
        }
    }

    mse /= count as f32;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}

pub fn compute_mae(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mae = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mae += (r.r - t.r).abs() + (r.g - t.g).abs() + (r.b - t.b).abs();
    }
    mae / (rendered.width * rendered.height * 3) as f32
}
