//! MS-SSIM Gradient Test - EXP-4-006
//! Test if analytical MS-SSIM gradients improve quality over L2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, optimizer_v2::OptimizerV2, renderer_v2::RendererV2};

fn main() {
    let path = "/home/greg/gaussian-image-projects/test_images/133784383569199567.jpg";

    let target = match ImageBuffer::load(path) {
        Ok(img) => {
            let scale = 512.0 / img.width.max(img.height) as f32;
            let new_w = (img.width as f32 * scale) as u32;
            let new_h = (img.height as f32 * scale) as u32;
            resize_bilinear(&img, new_w, new_h)
        }
        Err(e) => {
            println!("Failed: {}", e);
            return;
        }
    };

    println!("EXP-4-006: MS-SSIM Analytical Gradients Test");
    println!("Image: {}×{}", target.width, target.height);
    println!();

    // Test N=900 (Session 4 sweet spot), 50 iterations

    // Test 1: L2 loss (baseline)
    println!("Test 1: L2 Loss (baseline)");
    let encoder1 = EncoderV2::new(target.clone()).unwrap();
    let mut gaussians_l2 = encoder1.initialize_gaussians_guided(30);  // 30×30 = 900

    let mut optimizer_l2 = OptimizerV2::default();
    optimizer_l2.max_iterations = 50;
    optimizer_l2.use_ms_ssim = false;  // L2 only

    optimizer_l2.optimize(&mut gaussians_l2, &target);

    let rendered_l2 = RendererV2::render(&gaussians_l2, target.width, target.height);
    let psnr_l2 = compute_psnr(&rendered_l2, &target);
    let ms_ssim_l2 = compute_ms_ssim(&rendered_l2, &target);

    println!("L2 PSNR: {:.2} dB, MS-SSIM: {:.4}\n", psnr_l2, ms_ssim_l2);

    // Test 2: MS-SSIM loss with analytical gradients
    println!("Test 2: MS-SSIM Loss (analytical gradients)");
    let encoder2 = EncoderV2::new(target.clone()).unwrap();
    let mut gaussians_ms = encoder2.initialize_gaussians_guided(30);

    let mut optimizer_ms = OptimizerV2::default();
    optimizer_ms.max_iterations = 50;
    optimizer_ms.use_ms_ssim = true;  // MS-SSIM with gradients

    optimizer_ms.optimize(&mut gaussians_ms, &target);

    let rendered_ms = RendererV2::render(&gaussians_ms, target.width, target.height);
    let psnr_ms = compute_psnr(&rendered_ms, &target);
    let ms_ssim_ms = compute_ms_ssim(&rendered_ms, &target);

    println!("MS-SSIM PSNR: {:.2} dB, MS-SSIM: {:.4}\n", psnr_ms, ms_ssim_ms);

    // Summary
    println!("===== SUMMARY =====");
    println!("L2:      {:.2} dB, MS-SSIM: {:.4}", psnr_l2, ms_ssim_l2);
    println!("MS-SSIM: {:.2} dB, MS-SSIM: {:.4}", psnr_ms, ms_ssim_ms);
    println!("PSNR Delta: {:+.2} dB", psnr_ms - psnr_l2);
    println!("MS-SSIM Delta: {:+.4}", ms_ssim_ms - ms_ssim_l2);

    if psnr_ms > psnr_l2 + 0.5 {
        println!("✅ MS-SSIM gradients improve quality by {:.2} dB", psnr_ms - psnr_l2);
    } else if psnr_ms > psnr_l2 {
        println!("⚠️ MS-SSIM gradients help slightly (+{:.2} dB)", psnr_ms - psnr_l2);
    } else {
        println!("❌ MS-SSIM gradients don't improve PSNR (but may improve perceptual quality)");
    }
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= (rendered.width * rendered.height * 3) as f32;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}

fn compute_ms_ssim(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    use lgi_core::ms_ssim::MSSSIM;
    let ms_ssim = MSSSIM::default();
    ms_ssim.compute(target, rendered)
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
