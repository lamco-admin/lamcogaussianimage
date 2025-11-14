//! LGI Demo CLI
//!
//! Simple end-to-end demonstration of encoding and decoding

use clap::{Parser, Subcommand};
use lgi_core::{ImageBuffer, Renderer};
use lgi_encoder::{Encoder, EncoderConfig};
use lgi_math::parameterization::Euler;
use anyhow::Result;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "lgi-cli")]
#[command(about = "LGI Gaussian Image Format - Demo Tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode an image to Gaussians (and render it back)
    Encode {
        /// Input image path
        #[arg(short, long)]
        input: PathBuf,

        /// Output image path
        #[arg(short, long)]
        output: PathBuf,

        /// Number of Gaussians
        #[arg(short = 'n', long, default_value = "1000")]
        gaussians: usize,

        /// Quality preset: fast, balanced, high, ultra
        #[arg(short = 'q', long, default_value = "balanced")]
        quality: String,

        /// Output width (for resolution independence test)
        #[arg(short = 'w', long)]
        width: Option<u32>,

        /// Output height
        #[arg(short = 'h', long)]
        height: Option<u32>,
    },

    /// Create a simple test image
    Test {
        /// Output path
        #[arg(short, long)]
        output: PathBuf,

        /// Image size
        #[arg(short, long, default_value = "256")]
        size: u32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Encode {
            input,
            output,
            gaussians: num_gaussians,
            quality,
            width,
            height,
        } => {
            println!("╔══════════════════════════════════════╗");
            println!("║   LGI Gaussian Image Encoder Demo   ║");
            println!("╚══════════════════════════════════════╝\n");

            // Load target image
            println!("📷 Loading image: {}", input.display());
            let target = ImageBuffer::<f32>::load(&input)?;
            println!("   Size: {}×{}", target.width, target.height);

            // Configure encoder
            let config = match quality.as_str() {
                "fast" => EncoderConfig::fast(),
                "balanced" => EncoderConfig::balanced(),
                "high" => EncoderConfig::high_quality(),
                "ultra" => EncoderConfig::ultra(),
                _ => {
                    println!("⚠️  Unknown quality preset '{}', using 'balanced'", quality);
                    EncoderConfig::balanced()
                }
            };

            println!("\n⚙️  Configuration:");
            println!("   Gaussians: {}", num_gaussians);
            println!("   Strategy: {:?}", config.init_strategy);
            println!("   Max iterations: {}", config.max_iterations);
            println!("   Quality: {}", quality);

            // Encode
            println!("\n🔧 Encoding...");
            let start = Instant::now();

            let encoder = Encoder::with_config(config);
            let mut iteration_count = 0;
            let mut final_loss = 0.0;

            let gaussians = encoder.encode_with_progress(&target, num_gaussians, |iter, loss| {
                iteration_count = iter;
                final_loss = loss;

                if iter % 100 == 0 {
                    println!("   Iteration {}: loss = {:.6}", iter, loss);
                }
            })?;

            let encode_time = start.elapsed();
            println!("\n✅ Encoding complete!");
            println!("   Time: {:.2}s", encode_time.as_secs_f32());
            println!("   Iterations: {}", iteration_count);
            println!("   Final loss: {:.6}", final_loss);
            println!("   Gaussians: {}", gaussians.len());

            // Compute statistics
            let avg_opacity: f32 = gaussians.iter().map(|g| g.opacity).sum::<f32>() / gaussians.len() as f32;
            println!("   Avg opacity: {:.3}", avg_opacity);

            // Decode (render)
            println!("\n🎨 Rendering...");
            let render_start = Instant::now();

            let renderer = Renderer::new();
            let (render_width, render_height) = if let (Some(w), Some(h)) = (width, height) {
                println!("   Rendering at custom resolution: {}×{}", w, h);
                (w, h)
            } else {
                println!("   Rendering at original resolution: {}×{}", target.width, target.height);
                (target.width, target.height)
            };

            let rendered = renderer.render(&gaussians, render_width, render_height)?;
            let render_time = render_start.elapsed();

            println!("✅ Rendering complete!");
            println!("   Time: {:.3}s", render_time.as_secs_f32());
            println!("   FPS: {:.1}", 1.0 / render_time.as_secs_f32());

            // Save
            println!("\n💾 Saving: {}", output.display());
            rendered.save(&output)?;

            // Compute metrics if rendering at same resolution
            if render_width == target.width && render_height == target.height {
                println!("\n📊 Quality Metrics:");
                let psnr = compute_psnr(&rendered, &target);
                println!("   PSNR: {:.2} dB", psnr);

                // Estimate compression
                let uncompressed_size = target.width * target.height * 4; // RGBA bytes
                let gaussian_size = gaussians.len() * 48; // ~48 bytes per Gaussian uncompressed
                let compression_ratio = gaussian_size as f32 / uncompressed_size as f32;
                println!("\n💾 Storage (uncompressed):");
                println!("   Original: {} KB", uncompressed_size / 1024);
                println!("   Gaussians: {} KB", gaussian_size / 1024);
                println!("   Ratio: {:.1}%", compression_ratio * 100.0);
            }

            println!("\n✨ Done!");
        }

        Commands::Test { output, size } => {
            println!("🎨 Creating test image ({}×{})...", size, size);

            // Create a simple test pattern
            let mut test_img = ImageBuffer::<f32>::new(size, size);

            for y in 0..size {
                for x in 0..size {
                    let nx = x as f32 / size as f32;
                    let ny = y as f32 / size as f32;

                    // Gradient background
                    let r = nx;
                    let g = ny;
                    let b = (1.0 - nx) * (1.0 - ny);

                    // Add some shapes
                    let center_dist = ((nx - 0.5) * (nx - 0.5) + (ny - 0.5) * (ny - 0.5)).sqrt();
                    if center_dist < 0.3 {
                        // Red circle in center
                        use lgi_math::color::Color4;
                        test_img.set_pixel(x, y, Color4::rgb(1.0, 0.2, 0.2));
                    } else {
                        use lgi_math::color::Color4;
                        test_img.set_pixel(x, y, Color4::rgb(r, g, b));
                    }
                }
            }

            test_img.save(&output)?;
            println!("✅ Test image saved: {}", output.display());
        }
    }

    Ok(())
}

/// Compute PSNR between two images
fn compute_psnr(img1: &ImageBuffer<f32>, img2: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (img1.width * img1.height) as f32;

    for (p1, p2) in img1.data.iter().zip(img2.data.iter()) {
        let diff_r = p1.r - p2.r;
        let diff_g = p1.g - p2.g;
        let diff_b = p1.b - p2.b;

        mse += diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
    }

    mse /= count * 3.0; // 3 channels

    if mse < 1e-10 {
        100.0 // Essentially perfect
    } else {
        20.0 * (1.0f32 / mse.sqrt()).log10()
    }
}
