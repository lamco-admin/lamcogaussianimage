//! Async encoding with real-time UI progress updates
//! Copyright (c) 2025 Lamco Development

use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread;
use tracing::{info, error};
use anyhow::Result;

#[derive(Debug, Clone)]
pub enum EncodingProgress {
    LoadingImage,
    ConvertingPixels { percent: f32 },
    InitializingGaussians { percent: f32 },
    OptimizingIteration { iteration: usize, max_iter: usize, loss: f32, psnr: f32 },
    CreatingFile { percent: f32 },
    Complete { gaussians: Vec<lgi_math::gaussian::Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>>, width: u32, height: u32 },
    Error(String),
}

pub fn start_encoding(
    image_path: std::path::PathBuf,
    num_gaussians: usize,
    profile: lgi_encoder::EncoderConfig,
) -> Receiver<EncodingProgress> {
    let (tx, rx) = channel();
    
    thread::spawn(move || {
        if let Err(e) = encode_with_progress(image_path, num_gaussians, profile, tx.clone()) {
            error!("Encoding failed: {}", e);
            tx.send(EncodingProgress::Error(e.to_string())).ok();
        }
    });
    
    rx
}

fn encode_with_progress(
    path: std::path::PathBuf,
    num_gaussians: usize,
    profile: lgi_encoder::EncoderConfig,
    progress: Sender<EncodingProgress>,
) -> Result<()> {
    info!("🧵 Encoding thread started");
    
    // Load image
    progress.send(EncodingProgress::LoadingImage)?;
    let img = image::open(&path)?;
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    info!("📂 Loaded {}×{}", width, height);
    
    // Convert to float
    progress.send(EncodingProgress::ConvertingPixels { percent: 10.0 })?;
    let mut float_data = lgi_core::ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = rgba.get_pixel(x, y);
            float_data.set_pixel(
                x, y,
                lgi_math::color::Color4::new(
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                    pixel[3] as f32 / 255.0,
                ),
            );
        }
    }
    
    // Initialize
    progress.send(EncodingProgress::InitializingGaussians { percent: 20.0 })?;
    let initializer = lgi_core::Initializer::new(lgi_core::InitStrategy::Gradient);
    let mut gaussians = initializer.initialize(&float_data, num_gaussians)?;
    info!("🎲 Initialized {} Gaussians", gaussians.len());
    
    // Optimize with progress updates
    progress.send(EncodingProgress::OptimizingIteration {
        iteration: 0,
        max_iter: profile.max_iterations,
        loss: 0.0,
        psnr: 0.0,
    })?;
    
    let mut optimizer = lgi_encoder::OptimizerV2::new(profile.clone());
    
    // TODO: Hook into optimizer to send progress per iteration
    // For now, optimizer logs to console
    optimizer.optimize(&mut gaussians, &float_data)?;
    
    info!("✅ Optimization complete!");
    
    progress.send(EncodingProgress::Complete {
        gaussians,
        width,
        height,
    })?;
    
    Ok(())
}
