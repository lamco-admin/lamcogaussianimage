//! Multi-Image Test - Sample diverse images from corpus

use lgi_core::{ImageBuffer, lod_system::LODSystem};
use lgi_encoder_v2::{EncoderV2, optimizer_v2::OptimizerV2, renderer_v2::RendererV2};

fn main() {
    let test_images = vec![
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133784383569199567.jpg",
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/new photos/20150516_153457.jpg",
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/new photos/20200708_154907_HDR.jpg",
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/new photos/20240619_184229.jpg",
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/arulj.jpg",
    ];

    println!("Testing {} diverse images...\n", test_images.len());

    for (idx, path) in test_images.iter().enumerate() {
        let filename = std::path::Path::new(path).file_name().unwrap().to_str().unwrap();
        println!("{}. {}", idx + 1, filename);

        let target = match load_and_resize(path, 512) {
            Ok(img) => img,
            Err(_) => {
                println!("   Failed to load\n");
                continue;
            }
        };

        let encoder = EncoderV2::new(target.clone()).unwrap();
        let mut gaussians = encoder.initialize_gaussians_guided(20);  // 20×20=400

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 30;
        optimizer.optimize(&mut gaussians, &target);

        let rendered = RendererV2::render(&gaussians, target.width, target.height);
        let psnr = compute_psnr(&rendered, &target);

        let lod = LODSystem::classify(&gaussians);
        let stats = lod.stats();

        println!("   {}×{}, PSNR: {:.2} dB", target.width, target.height, psnr);
        println!("   LOD: {:.0}% coarse, {:.0}% medium, {:.0}% fine\n",
                 stats.coarse_percent, stats.medium_percent, stats.fine_percent);
    }
}

fn load_and_resize(path: &str, max_size: u32) -> Result<ImageBuffer<f32>, String> {
    let img = ImageBuffer::load(path).map_err(|e| format!("{}", e))?;
    let scale = max_size as f32 / img.width.max(img.height) as f32;
    let new_w = (img.width as f32 * scale) as u32;
    let new_h = (img.height as f32 * scale) as u32;
    Ok(resize_bilinear(&img, new_w, new_h))
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
