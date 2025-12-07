//! File I/O Demo - EXP-4-005
//! Demonstrates encoding, saving, and loading .lgi files

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, optimizer_v2::OptimizerV2, renderer_v2::RendererV2};
use lgi_encoder_v2::file_writer::LGIWriter;
use lgi_encoder_v2::file_reader::LGIReader;

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

    println!("EXP-4-005: File I/O Demonstration");
    println!("Image: {}×{}", target.width, target.height);
    println!();

    // Encode
    println!("Step 1: Encoding...");
    let encoder = EncoderV2::new(target.clone()).unwrap();
    let mut gaussians = encoder.initialize_gaussians_guided(30);  // N=900

    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 50;
    optimizer.optimize(&mut gaussians, &target);

    let rendered_orig = RendererV2::render(&gaussians, target.width, target.height);
    let psnr_orig = compute_psnr(&rendered_orig, &target);
    println!("Original PSNR: {:.2} dB", psnr_orig);
    println!("Gaussian count: {}", gaussians.len());
    println!();

    // Save
    println!("Step 2: Saving to demo.lgi...");
    let writer = LGIWriter::new(target.width, target.height, gaussians.len() as u32);
    match writer.write_file("/tmp/demo.lgi", &gaussians) {
        Ok(_) => {
            let file_size = std::fs::metadata("/tmp/demo.lgi").unwrap().len();
            println!("Saved successfully ({} bytes)", file_size);
            let bits_per_pixel = (file_size * 8) as f32 / (target.width * target.height) as f32;
            println!("File size: {:.3} bpp", bits_per_pixel);
        }
        Err(e) => {
            println!("Save failed: {}", e);
            return;
        }
    }
    println!();

    // Load
    println!("Step 3: Loading from demo.lgi...");
    match LGIReader::read_file("/tmp/demo.lgi") {
        Ok((header, gaussians_loaded)) => {
            println!("Loaded successfully");
            println!("Header: {}×{}, {} Gaussians", header.canvas_width, header.canvas_height, header.gaussian_count);
            println!("Version: {}.{}", header.version_major, header.version_minor);
            println!();

            // Render loaded Gaussians
            println!("Step 4: Rendering loaded Gaussians...");
            let rendered_loaded = RendererV2::render(&gaussians_loaded, target.width, target.height);
            let psnr_loaded = compute_psnr(&rendered_loaded, &target);
            println!("Loaded PSNR: {:.2} dB", psnr_loaded);
            println!();

            // Compare
            println!("===== SUMMARY =====");
            println!("Original PSNR: {:.2} dB", psnr_orig);
            println!("Loaded PSNR:   {:.2} dB", psnr_loaded);
            println!("Degradation:   {:.2} dB (quantization loss)", psnr_orig - psnr_loaded);

            let acceptable_loss = 0.5;  // Allow 0.5 dB loss from quantization
            if (psnr_orig - psnr_loaded).abs() < acceptable_loss {
                println!("✅ File I/O working correctly (quantization loss < {} dB)", acceptable_loss);
            } else {
                println!("⚠️ Significant quality loss from quantization");
            }
        }
        Err(e) => {
            println!("Load failed: {}", e);
        }
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
