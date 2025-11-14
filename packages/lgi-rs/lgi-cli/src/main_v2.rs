//! Enhanced CLI with OptimizerV2 and comprehensive metrics
//!
//! Run with: cargo run --bin lgi-cli-v2

use clap::{Parser, Subcommand};
use lgi_core::{ImageBuffer, Renderer};
use lgi_encoder::{EncoderConfig, OptimizerV2};
use lgi_core::Initializer;
use lgi_format::{LgiFile, LgiWriter, LgiReader, LgiMetadata};
use lgi_format::metadata::{EncodingMetadata, QualityMetrics};
use anyhow::Result;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "lgi-cli-v2")]
#[command(about = "LGI with Full Optimizer - Production Quality", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode image to Gaussians (PNG → .lgi or PNG)
    Encode {
        #[arg(short, long)]
        input: PathBuf,

        #[arg(short, long)]
        output: PathBuf,

        #[arg(short = 'n', long, default_value = "1000")]
        gaussians: usize,

        #[arg(short = 'q', long, default_value = "balanced")]
        quality: String,

        /// Enable adaptive features (pruning, splitting)
        #[arg(long)]
        adaptive: bool,

        /// Enable Quantization-Aware (QA) training for better compression
        #[arg(long)]
        qa_training: bool,

        /// Save as .lgi file (VQ compressed)
        #[arg(long)]
        save_lgi: bool,

        /// Export metrics to CSV
        #[arg(long)]
        metrics_csv: Option<PathBuf>,

        /// Export metrics to JSON
        #[arg(long)]
        metrics_json: Option<PathBuf>,
    },

    /// Decode .lgi file to PNG
    Decode {
        #[arg(short, long)]
        input: PathBuf,

        #[arg(short, long)]
        output: PathBuf,
    },

    /// Display .lgi file information
    Info {
        #[arg(short, long)]
        input: PathBuf,
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
            adaptive,
            qa_training,
            save_lgi,
            metrics_csv,
            metrics_json,
        } => {
            println!("╔══════════════════════════════════════════════════╗");
            println!("║   LGI Encoder V2 - Full Optimization            ║");
            println!("╚══════════════════════════════════════════════════╝\n");

            // Load
            println!("📷 Loading: {}", input.display());
            let target = ImageBuffer::<f32>::load(&input)?;
            println!("   Size: {}×{}", target.width, target.height);

            // Configure
            let mut config = match quality.as_str() {
                "fast" => EncoderConfig::fast(),
                "balanced" => EncoderConfig::balanced(),
                "high" => EncoderConfig::high_quality(),
                "ultra" => EncoderConfig::ultra(),
                _ => EncoderConfig::balanced(),
            };

            // Enable QA training if requested
            if qa_training {
                config.enable_qa_training = true;
                config.qa_start_iteration = (config.max_iterations as f32 * 0.7) as usize;
            }

            println!("\n⚙️  Configuration:");
            println!("   Gaussians: {}", num_gaussians);
            println!("   Strategy: {:?}", config.init_strategy);
            println!("   Max iterations: {}", config.max_iterations);
            println!("   Adaptive features: {}", adaptive);
            println!("   QA training: {}", if qa_training { "✅ YES" } else { "❌ NO" });
            println!("   Full backprop: ✅ YES");

            // Save init strategy before config is moved
            let init_strategy = config.init_strategy;

            // Initialize
            println!("\n🔧 Initializing Gaussians...");
            let initializer = Initializer::new(config.init_strategy)
                .with_scale(config.initial_scale);
            let mut gaussians = initializer.initialize(&target, num_gaussians)?;
            println!("   Initialized: {} Gaussians", gaussians.len());

            // Optimize
            println!("\n🚀 Optimizing with FULL backpropagation...");
            let start = Instant::now();

            let mut optimizer = if adaptive {
                OptimizerV2::new(config).with_adaptive()
            } else {
                OptimizerV2::new(config)
            };

            let metrics = optimizer.optimize_with_metrics(&mut gaussians, &target)?;
            let encode_time = start.elapsed();

            println!("\n✅ Optimization Complete!");
            println!("   Time: {:.2}s", encode_time.as_secs_f32());
            println!("   Final Gaussians: {}", gaussians.len());

            // Get final metrics
            if let Some(last) = metrics.iterations().last() {
                println!("   Final loss: {:.6}", last.total_loss);
                if let Some(psnr) = last.psnr {
                    println!("   Final PSNR: {:.2} dB", psnr);
                }
                println!("   Avg opacity: {:.3}", last.avg_opacity);
                println!("   Active Gaussians: {}", last.num_active_gaussians);
            }

            // Export metrics if requested
            if let Some(csv_path) = metrics_csv {
                println!("\n💾 Exporting metrics to CSV: {}", csv_path.display());
                metrics.export_csv(&csv_path)?;
            }

            if let Some(json_path) = metrics_json {
                println!("💾 Exporting metrics to JSON: {}", json_path.display());
                metrics.export_json(&json_path)?;
            }

            // Render
            println!("\n🎨 Rendering final result...");
            let render_start = Instant::now();
            let renderer = Renderer::new();
            let rendered = renderer.render(&gaussians, target.width, target.height)?;
            let render_time = render_start.elapsed();

            println!("✅ Rendered in {:.3}s ({:.1} FPS)", render_time.as_secs_f32(), 1.0 / render_time.as_secs_f32());

            // Save rendered PNG
            println!("\n💾 Saving rendered PNG: {}", output.display());
            rendered.save(&output)?;

            // Save .lgi file if requested
            if save_lgi {
                let lgi_path = output.with_extension("lgi");
                println!("💾 Saving .lgi file: {}", lgi_path.display());

                // Create metadata
                let metadata = LgiMetadata::new()
                    .with_encoding(EncodingMetadata {
                        encoder_version: env!("CARGO_PKG_VERSION").to_string(),
                        iterations: metrics.iterations().len() as u32,
                        init_strategy: format!("{:?}", init_strategy),
                        qa_training,
                        encoding_time_secs: encode_time.as_secs_f32(),
                    })
                    .with_quality(QualityMetrics {
                        psnr_db: if let Some(last) = metrics.iterations().last() {
                            last.psnr.unwrap_or(0.0)
                        } else {
                            0.0
                        },
                        ssim: 0.0,  // TODO: Add SSIM computation
                        final_loss: if let Some(last) = metrics.iterations().last() {
                            last.total_loss
                        } else {
                            0.0
                        },
                    });

                // Create .lgi file with VQ compression
                let lgi_file = LgiFile::with_vq(
                    gaussians.clone(),
                    target.width,
                    target.height,
                    256  // VQ codebook size
                ).with_metadata(metadata);

                LgiWriter::write_file(&lgi_file, &lgi_path)?;

                println!("   Compression: {:.2}×", lgi_file.compression_ratio());
                println!("   File size: ~{} KB", lgi_format::writer::LgiWriter::estimated_size(&lgi_file) / 1024);
            }

            // Final quality metrics
            println!("\n📊 Final Quality Metrics:");
            let final_psnr = compute_psnr(&rendered, &target);
            println!("   PSNR: {:.2} dB", final_psnr);

            let uncompressed_size = target.width * target.height * 4;
            let gaussian_size = gaussians.len() * 48;
            println!("\n💾 Storage (uncompressed):");
            println!("   Original: {} KB", uncompressed_size / 1024);
            println!("   Gaussians: {} KB", gaussian_size / 1024);
            println!("   Ratio: {:.1}%", (gaussian_size as f32 / uncompressed_size as f32) * 100.0);

            println!("\n✨ Done!");
        }

        Commands::Decode { input, output } => {
            println!("╔══════════════════════════════════════════════════╗");
            println!("║   LGI Decoder - Load & Render                   ║");
            println!("╚══════════════════════════════════════════════════╝\n");

            // Load .lgi file
            println!("📂 Loading: {}", input.display());
            let lgi_file = LgiReader::read_file(&input)?;

            let (width, height) = lgi_file.dimensions();
            println!("   Dimensions: {}×{}", width, height);
            println!("   Gaussians: {}", lgi_file.gaussian_count());
            println!("   Compressed: {}", if lgi_file.is_compressed() { "✅ VQ" } else { "❌ No" });

            if lgi_file.is_compressed() {
                println!("   Compression: {:.2}×", lgi_file.compression_ratio());
            }

            // Display metadata if present
            if let Some(ref metadata) = lgi_file.metadata {
                println!("\n📊 Encoding Info:");
                if let Some(ref encoding) = metadata.encoding {
                    println!("   Encoder: v{}", encoding.encoder_version);
                    println!("   Iterations: {}", encoding.iterations);
                    println!("   QA Training: {}", encoding.qa_training);
                    println!("   Encode time: {:.2}s", encoding.encoding_time_secs);
                }
                if let Some(ref quality) = metadata.quality {
                    println!("\n   Quality:");
                    println!("   PSNR: {:.2} dB", quality.psnr_db);
                    println!("   Loss: {:.6}", quality.final_loss);
                }
            }

            // Reconstruct Gaussians
            println!("\n🔧 Reconstructing Gaussians...");
            let gaussians = lgi_file.gaussians();
            println!("   Loaded: {} Gaussians", gaussians.len());

            // Render
            println!("\n🎨 Rendering...");
            let render_start = Instant::now();
            let renderer = Renderer::new();
            let rendered = renderer.render(&gaussians, width, height)?;
            let render_time = render_start.elapsed();

            println!("✅ Rendered in {:.3}s ({:.1} FPS)",
                render_time.as_secs_f32(),
                1.0 / render_time.as_secs_f32()
            );

            // Save
            println!("\n💾 Saving: {}", output.display());
            rendered.save(&output)?;

            println!("\n✨ Done!");
        }

        Commands::Info { input } => {
            println!("╔══════════════════════════════════════════════════╗");
            println!("║   LGI File Info                                  ║");
            println!("╚══════════════════════════════════════════════════╝\n");

            println!("📂 File: {}", input.display());

            // Read header only (fast)
            let header = LgiReader::read_header_file(&input)?;

            println!("\n🔧 Format:");
            println!("   Version: {}", header.version);
            println!("   Dimensions: {}×{}", header.width, header.height);
            println!("   Gaussians: {}", header.gaussian_count);
            println!("   Color space: {}", header.color_space);
            println!("   Bit depth: {}", header.bit_depth);

            println!("\n💾 Compression:");
            println!("   VQ: {}", if header.compression_flags.vq_compressed {
                format!("✅ YES (codebook: {})", header.compression_flags.vq_codebook_size)
            } else {
                "❌ NO".to_string()
            });
            println!("   zstd: {}", if header.compression_flags.zstd_compressed { "✅ YES" } else { "❌ NO" });

            // Read full file to get metadata
            let lgi_file = LgiReader::read_file(&input)?;

            if let Some(ref metadata) = lgi_file.metadata {
                if let Some(ref encoding) = metadata.encoding {
                    println!("\n📊 Encoding:");
                    println!("   Encoder: v{}", encoding.encoder_version);
                    println!("   Strategy: {}", encoding.init_strategy);
                    println!("   Iterations: {}", encoding.iterations);
                    println!("   QA Training: {}", encoding.qa_training);
                    println!("   Time: {:.2}s", encoding.encoding_time_secs);
                }

                if let Some(ref quality) = metadata.quality {
                    println!("\n✨ Quality:");
                    println!("   PSNR: {:.2} dB", quality.psnr_db);
                    println!("   SSIM: {:.4}", quality.ssim);
                    println!("   Final Loss: {:.6}", quality.final_loss);
                }
            }

            // File size
            if let Ok(meta) = std::fs::metadata(&input) {
                let file_size = meta.len();
                println!("\n💾 File:");
                println!("   Size: {} KB ({} bytes)", file_size / 1024, file_size);

                // Estimate uncompressed size
                let uncompressed = header.gaussian_count as u64 * 48;
                let ratio = uncompressed as f32 / file_size as f32;
                println!("   Uncompressed: {} KB", uncompressed / 1024);
                println!("   Ratio: {:.2}×", ratio);
            }

            println!("\n✨ Done!");
        }
    }

    Ok(())
}

fn compute_psnr(img1: &ImageBuffer<f32>, img2: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (img1.width * img1.height * 3) as f32;

    for (p1, p2) in img1.data.iter().zip(img2.data.iter()) {
        mse += (p1.r - p2.r) * (p1.r - p2.r);
        mse += (p1.g - p2.g) * (p1.g - p2.g);
        mse += (p1.b - p2.b) * (p1.b - p2.b);
    }

    mse /= count;

    if mse < 1e-10 {
        100.0
    } else {
        20.0 * (1.0f32 / mse.sqrt()).log10()
    }
}
