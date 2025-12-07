//! Blue-Noise Parameter Sweep

use lgi_core::{ImageBuffer, blue_noise_residual};
use lgi_encoder_v2::{EncoderV2, optimizer_v2::OptimizerV2, renderer_v2::RendererV2};

fn main() {
    let path = "/home/greg/gaussian-image-projects/test_images/133784383569199567.jpg";

    let target = match ImageBuffer::load(path) {
        Ok(img) => resize_bilinear(&img, 512, 288),
        Err(e) => {
            println!("Failed: {}", e);
            return;
        }
    };

    println!("Blue-Noise Parameter Sweep\n");

    let encoder = EncoderV2::new(target.clone()).unwrap();
    let mut gaussians = encoder.initialize_gaussians_guided(20);

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 30;
    optimizer.optimize(&mut gaussians, &target);

    let rendered_base = RendererV2::render(&gaussians, target.width, target.height);
    let psnr_base = compute_psnr(&rendered_base, &target);

    println!("Baseline (no blue-noise): {:.2} dB\n", psnr_base);

    let amplitudes = vec![0.01, 0.02, 0.05];
    let frequencies = vec![1.0, 2.0, 4.0];

    println!("Amplitude | Frequency | PSNR    | Delta");
    println!("----------|-----------|---------|-------");

    for &amp in &amplitudes {
        for &freq in &frequencies {
            let blue_noise = blue_noise_residual::BlueNoiseResidual::new(
                target.width,
                target.height,
            );

            let mut rendered_with_noise = rendered_base.clone();
            blue_noise.apply_to_image(&mut rendered_with_noise);

            let psnr = compute_psnr(&rendered_with_noise, &target);
            let delta = psnr - psnr_base;

            println!(" {:.2}     | {:.1}       | {:.2} dB | {:+.2} dB {}",
                     amp, freq, psnr, delta,
                     if delta > 0.0 { "✅" } else { "❌" });
        }
    }
}

fn resize_bilinear(img: &ImageBuffer<f32>, new_width: u32, new_height: u32) -> ImageBuffer<f32> {
    let mut resized = ImageBuffer::new(new_width, new_height);
    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = x as f32 * (img.width as f32 / new_width as f32);
            let src_y = y as f32 * (img.height as f32 / new_height as f32);
            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(img.width - 1);
            let y1 = (y0 + 1).min(img.height - 1);
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;
            if let (Some(c00), Some(c10), Some(c01), Some(c11)) = (
                img.get_pixel(x0, y0), img.get_pixel(x1, y0),
                img.get_pixel(x0, y1), img.get_pixel(x1, y1),
            ) {
                let r = (1.0-fx)*(1.0-fy)*c00.r + fx*(1.0-fy)*c10.r + (1.0-fx)*fy*c01.r + fx*fy*c11.r;
                let g = (1.0-fx)*(1.0-fy)*c00.g + fx*(1.0-fy)*c10.g + (1.0-fx)*fy*c01.g + fx*fy*c11.g;
                let b = (1.0-fx)*(1.0-fy)*c00.b + fx*(1.0-fy)*c10.b + (1.0-fx)*fy*c01.b + fx*fy*c11.b;
                resized.set_pixel(x, y, lgi_math::color::Color4::new(r, g, b, 1.0));
            }
        }
    }
    resized
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= (rendered.width * rendered.height * 3) as f32;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
