//! Adam vs SGD Comparison - EXP-4-004
//! Test if Adam optimizer improves quality over current SGD

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, optimizer_v2::OptimizerV2, adam_optimizer::AdamOptimizer, renderer_v2::RendererV2};

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

    println!("EXP-4-004: Adam vs SGD Optimizer Comparison");
    println!("Image: {}×{}", target.width, target.height);
    println!();

    // Test N=900 (Session 4 sweet spot), 50 iterations
    let grid_size = 30;

    // Test 1: Current SGD (OptimizerV2)
    println!("Test 1: SGD (OptimizerV2)");
    let encoder1 = EncoderV2::new(target.clone()).unwrap();
    let mut gaussians_sgd = encoder1.initialize_gaussians_guided(grid_size);

    let mut optimizer_sgd = OptimizerV2::default();
    optimizer_sgd.max_iterations = 50;
    optimizer_sgd.optimize(&mut gaussians_sgd, &target);

    let rendered_sgd = RendererV2::render(&gaussians_sgd, target.width, target.height);
    let psnr_sgd = compute_psnr(&rendered_sgd, &target);

    println!("SGD PSNR: {:.2} dB\n", psnr_sgd);

    // Test 2: Adam optimizer
    println!("Test 2: Adam Optimizer");
    let encoder2 = EncoderV2::new(target.clone()).unwrap();
    let mut gaussians_adam = encoder2.initialize_gaussians_guided(grid_size);

    let mut optimizer_adam = AdamOptimizer::default();
    optimizer_adam.max_iterations = 50;
    optimizer_adam.learning_rate = 0.01;
    let _loss_adam = optimizer_adam.optimize(&mut gaussians_adam, &target);

    let rendered_adam = RendererV2::render(&gaussians_adam, target.width, target.height);
    let psnr_adam = compute_psnr(&rendered_adam, &target);

    println!("Adam PSNR: {:.2} dB\n", psnr_adam);

    // Summary
    println!("===== SUMMARY =====");
    println!("SGD:  {:.2} dB", psnr_sgd);
    println!("Adam: {:.2} dB", psnr_adam);
    println!("Delta: {:+.2} dB", psnr_adam - psnr_sgd);

    if psnr_adam > psnr_sgd {
        println!("✅ Adam is better by {:.2} dB", psnr_adam - psnr_sgd);
    } else if psnr_sgd > psnr_adam {
        println!("⚠️ SGD is better by {:.2} dB", psnr_sgd - psnr_adam);
    } else {
        println!("= Tie");
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
