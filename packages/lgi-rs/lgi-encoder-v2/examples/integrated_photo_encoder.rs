//! EXP-034: Complete Integrated Photo Encoder
//!
//! Commercial-grade encoder using ALL features:
//! - Guided filter colors
//! - Texture mapping
//! - Blue-noise residuals
//! - Adaptive optimization
//!
//! Goal: Reach 28-35 dB on photos

use lgi_core::{ImageBuffer, textured_gaussian::TexturedGaussian2D, blue_noise_residual::BlueNoiseResidual};
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, renderer_v3_textured::RendererV3, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  EXP-034: Integrated Commercial Photo Encoder           ║");
    println!("║  All features: Textures + Blue-noise + Optimization     ║");
    println!("║  Goal: 28-35 dB on real photos                           ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/Lamco Head.jpg";

    println!("Loading {}...", path.split('/').last().unwrap_or(path));
    let target = match ImageBuffer::load(path) {
        Ok(mut img) => {
            println!("Loaded: {}×{}", img.width, img.height);
            if img.width > 256 || img.height > 256 {
                println!("Resizing to 256×256...");
                resize_simple(&img, 256, 256)
            } else {
                img
            }
        }
        Err(e) => {
            println!("Failed: {}", e);
            return;
        }
    };

    println!("\n═══════════════════════════════════════════");
    println!("Pipeline 1: Baseline (no advanced features)");
    println!("═══════════════════════════════════════════");

    let encoder = EncoderV2::new(target.clone()).unwrap();
    let mut gaussians_base = encoder.initialize_gaussians(20);  // N=400

    let mut opt_base = OptimizerV2::default();
    opt_base.max_iterations = 100;
    opt_base.optimize(&mut gaussians_base, &target);

    let rendered_base = RendererV2::render(&gaussians_base, 256, 256);
    let psnr_base = compute_psnr(&target, &rendered_base);

    println!("  PSNR: {:.2} dB", psnr_base);

    println!("\n═══════════════════════════════════════════");
    println!("Pipeline 2: With guided filter");
    println!("═══════════════════════════════════════════");

    let mut gaussians_guided = encoder.initialize_gaussians_guided(20);

    let mut opt_guided = OptimizerV2::default();
    opt_guided.max_iterations = 100;
    opt_guided.optimize(&mut gaussians_guided, &target);

    let rendered_guided = RendererV2::render(&gaussians_guided, 256, 256);
    let psnr_guided = compute_psnr(&target, &rendered_guided);

    println!("  PSNR: {:.2} dB ({:+.2} dB vs baseline)", psnr_guided, psnr_guided - psnr_base);

    println!("\n═══════════════════════════════════════════");
    println!("Pipeline 3: Guided + Textures");
    println!("═══════════════════════════════════════════");

    let mut gaussians_tex: Vec<TexturedGaussian2D> = gaussians_guided
        .iter()
        .map(|g| TexturedGaussian2D::from_gaussian(g.clone()))
        .collect();

    // Add textures adaptively
    let mut tex_count = 0;
    for g in &mut gaussians_tex {
        if g.should_add_texture(&target, 0.005) {  // Lower threshold for photos
            g.extract_texture_from_image(&target, 16);  // 16×16 textures
            tex_count += 1;
        }
    }

    println!("  Added textures to {}/{} Gaussians", tex_count, gaussians_tex.len());

    let rendered_tex = RendererV3::render(&gaussians_tex, 256, 256);
    let psnr_tex = compute_psnr(&target, &rendered_tex);

    println!("  PSNR: {:.2} dB ({:+.2} dB vs guided)", psnr_tex, psnr_tex - psnr_guided);

    println!("\n═══════════════════════════════════════════");
    println!("Pipeline 4: Guided + Textures + Blue-Noise");
    println!("═══════════════════════════════════════════");

    // Detect residuals
    let residual = BlueNoiseResidual::detect_residual_regions(&target, &rendered_tex, 0.05);

    // Count masked pixels
    let masked_pixels = residual.mask.iter().filter(|&&m| m > 0.1).count();
    println!("  Residual mask covers {} pixels", masked_pixels);

    // Apply blue-noise
    let mut rendered_final = rendered_tex.clone();
    residual.apply_to_image(&mut rendered_final);

    let psnr_final = compute_psnr(&target, &rendered_final);

    println!("  PSNR: {:.2} dB ({:+.2} dB vs textured)", psnr_final, psnr_final - psnr_tex);

    println!("\n═══════════════════════════════════════════");
    println!("Summary");
    println!("═══════════════════════════════════════════");
    println!("  Baseline:                 {:.2} dB", psnr_base);
    println!("  + Guided filter:          {:.2} dB ({:+.2} dB)", psnr_guided, psnr_guided - psnr_base);
    println!("  + Textures:               {:.2} dB ({:+.2} dB)", psnr_tex, psnr_tex - psnr_guided);
    println!("  + Blue-noise residuals:   {:.2} dB ({:+.2} dB)", psnr_final, psnr_final - psnr_tex);
    println!("  TOTAL IMPROVEMENT:        {:+.2} dB", psnr_final - psnr_base);

    if psnr_final >= 28.0 {
        println!("\n✅ SUCCESS: Reached production quality target!");
    } else if psnr_final >= 25.0 {
        println!("\n✓ GOOD: Close to target, may need more iterations or N");
    } else {
        println!("\n⚠️  MARGINAL: Need additional features or higher N");
    }

    // Save final output
    if let Err(e) = rendered_final.save("/tmp/photo_integrated_final.png") {
        println!("\nWarning: Could not save: {}", e);
    } else {
        println!("\nSaved: /tmp/photo_integrated_final.png");
    }
}

fn resize_simple(img: &ImageBuffer<f32>, w: u32, h: u32) -> ImageBuffer<f32> {
    let mut out = ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sx = (x as f32 / w as f32 * img.width as f32) as u32;
            let sy = (y as f32 / h as f32 * img.height as f32) as u32;
            if let Some(c) = img.get_pixel(sx.min(img.width-1), sy.min(img.height-1)) {
                out.set_pixel(x, y, c);
            }
        }
    }
    out
}

fn compute_psnr(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (original.width * original.height * 3) as f32;

    for (p1, p2) in original.data.iter().zip(rendered.data.iter()) {
        mse += (p1.r - p2.r).powi(2);
        mse += (p1.g - p2.g).powi(2);
        mse += (p1.b - p2.b).powi(2);
    }

    mse /= count;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
