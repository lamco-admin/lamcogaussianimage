//! Texture Modulation Method Test - EXP-3-008
//! Test multiplicative vs additive vs blend modulation
//! Find which method works best for photo quality

use lgi_core::{ImageBuffer, texture_map::TextureMap, textured_gaussian::TexturedGaussian2D};
use lgi_encoder_v2::{EncoderV2, optimizer_v2::OptimizerV2};
use lgi_math::{color::Color4, vec::Vector2};

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Texture Modulation Method Test - EXP-3-008             ║");
    println!("║  Multiplicative vs Additive vs Blend                     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Load test photo
    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133784383569199567.jpg";

    let target = match ImageBuffer::load(path) {
        Ok(img) => {
            let scale = 768.0 / img.width.max(img.height) as f32;
            let new_w = (img.width as f32 * scale) as u32;
            let new_h = (img.height as f32 * scale) as u32;
            resize_bilinear(&img, new_w, new_h)
        }
        Err(e) => {
            println!("Failed to load: {}", e);
            return;
        }
    };

    println!("Target: {}×{}\n", target.width, target.height);

    // Initialize base Gaussians
    let encoder = EncoderV2::new(target.clone()).unwrap();
    let mut gaussians_base = encoder.initialize_gaussians_guided(40);  // N=1600

    println!("Initializing {} Gaussians...\n", gaussians_base.len());

    // Optimize base Gaussians
    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 100;

    println!("Optimizing base Gaussians (100 iterations, CPU only)...");
    optimizer.optimize(&mut gaussians_base, &target);

    // Create textured versions
    let mut textured_gaussians = Vec::new();

    for gaussian in &gaussians_base {
        let mut tg = TexturedGaussian2D::from_gaussian(gaussian.clone());

        // Extract texture (16×16)
        tg.extract_texture_from_image(&target, 16);

        textured_gaussians.push(tg);
    }

    println!("Extracted textures for {} Gaussians\n", textured_gaussians.len());

    // Test 1: Baseline (no textures)
    println!("═══════════════════════════════════════════");
    println!("Test 1: BASELINE (No Textures)");
    println!("═══════════════════════════════════════════");

    let rendered_base = lgi_encoder_v2::renderer_v2::RendererV2::render(&gaussians_base, target.width, target.height);
    let psnr_base = compute_psnr(&rendered_base, &target);
    println!("  PSNR: {:.2} dB (baseline)\n", psnr_base);

    // Test 2: Multiplicative modulation (current implementation)
    println!("═══════════════════════════════════════════");
    println!("Test 2: MULTIPLICATIVE Modulation");
    println!("═══════════════════════════════════════════");
    println!("  Formula: color = base_color * texture_value");

    let rendered_mult = render_with_multiplicative(&textured_gaussians, target.width, target.height);
    let psnr_mult = compute_psnr(&rendered_mult, &target);
    let delta_mult = psnr_mult - psnr_base;

    println!("  PSNR: {:.2} dB", psnr_mult);
    println!("  Δ: {:.2} dB {}", delta_mult, if delta_mult > 0.0 { "✅" } else { "❌" });
    println!();

    // Test 3: Additive modulation
    println!("═══════════════════════════════════════════");
    println!("Test 3: ADDITIVE Modulation");
    println!("═══════════════════════════════════════════");
    println!("  Formula: color = base_color + (texture_value - 0.5) * scale");

    let rendered_add = render_with_additive(&textured_gaussians, target.width, target.height, 0.5);
    let psnr_add = compute_psnr(&rendered_add, &target);
    let delta_add = psnr_add - psnr_base;

    println!("  PSNR: {:.2} dB", psnr_add);
    println!("  Δ: {:.2} dB {}", delta_add, if delta_add > 0.0 { "✅" } else { "❌" });
    println!();

    // Test 4: Blend modulation
    println!("═══════════════════════════════════════════");
    println!("Test 4: BLEND Modulation");
    println!("═══════════════════════════════════════════");
    println!("  Formula: color = lerp(base_color, texture_color, alpha)");

    let rendered_blend = render_with_blend(&textured_gaussians, target.width, target.height, 0.3);
    let psnr_blend = compute_psnr(&rendered_blend, &target);
    let delta_blend = psnr_blend - psnr_base;

    println!("  PSNR: {:.2} dB", psnr_blend);
    println!("  Δ: {:.2} dB {}", delta_blend, if delta_blend > 0.0 { "✅" } else { "❌" });
    println!();

    // Results summary
    println!("═══════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════");
    println!("  Baseline:       {:.2} dB", psnr_base);
    println!("  Multiplicative: {:.2} dB ({:+.2} dB)", psnr_mult, delta_mult);
    println!("  Additive:       {:.2} dB ({:+.2} dB)", psnr_add, delta_add);
    println!("  Blend:          {:.2} dB ({:+.2} dB)", psnr_blend, delta_blend);
    println!();

    let best = psnr_mult.max(psnr_add).max(psnr_blend);
    let best_method = if best == psnr_mult {
        "Multiplicative"
    } else if best == psnr_add {
        "Additive"
    } else {
        "Blend"
    };

    println!("  Best method: {} ({:.2} dB, {:+.2} dB improvement)", best_method, best, best - psnr_base);

    if best < psnr_base {
        println!("\n  ❌ ALL METHODS WORSE THAN BASELINE!");
        println!("     Textures hurting quality regardless of modulation");
        println!("     Issue may be in SELECTION (applying to wrong Gaussians)");
    }
}

// Rendering functions with different modulation methods

fn render_with_multiplicative(gaussians: &[TexturedGaussian2D], width: u32, height: u32) -> ImageBuffer<f32> {
    use lgi_encoder_v2::renderer_v3_textured::RendererV3;
    RendererV3::render(gaussians, width, height)
}

fn render_with_additive(gaussians: &[TexturedGaussian2D], width: u32, height: u32, scale: f32) -> ImageBuffer<f32> {
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let px = x as f32 / width as f32;
            let py = y as f32 / height as f32;
            let world_pos = Vector2::new(px, py);

            let mut weight_sum = 0.0;
            let mut color_sum = Color4::new(0.0, 0.0, 0.0, 0.0);

            for tg in gaussians {
                let gaussian = &tg.gaussian;
                let dx = px - gaussian.position.x;
                let dy = py - gaussian.position.y;

                let sx = gaussian.shape.scale_x;
                let sy = gaussian.shape.scale_y;
                let theta = gaussian.shape.rotation;

                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let dx_rot = dx * cos_t + dy * sin_t;
                let dy_rot = -dx * sin_t + dy * cos_t;

                let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);
                if dist_sq > 12.25 { continue; }

                let gaussian_val = (-0.5 * dist_sq).exp();
                let weight = gaussian.opacity * gaussian_val;

                // ADDITIVE modulation
                let mut color = gaussian.color;
                if let Some(ref texture) = tg.texture {
                    let local = tg.world_to_local(world_pos);
                    let tex = texture.sample(local.x, local.y);
                    // Add centered texture detail
                    color.r += (tex.r - 0.5) * scale;
                    color.g += (tex.g - 0.5) * scale;
                    color.b += (tex.b - 0.5) * scale;
                    color.r = color.r.clamp(0.0, 1.0);
                    color.g = color.g.clamp(0.0, 1.0);
                    color.b = color.b.clamp(0.0, 1.0);
                }

                weight_sum += weight;
                color_sum.r += weight * color.r;
                color_sum.g += weight * color.g;
                color_sum.b += weight * color.b;
            }

            let final_color = if weight_sum > 1e-10 {
                Color4::new(
                    color_sum.r / weight_sum,
                    color_sum.g / weight_sum,
                    color_sum.b / weight_sum,
                    1.0,
                )
            } else {
                Color4::new(0.0, 0.0, 0.0, 1.0)
            };

            output.set_pixel(x, y, final_color);
        }
    }

    output
}

fn render_with_blend(gaussians: &[TexturedGaussian2D], width: u32, height: u32, alpha: f32) -> ImageBuffer<f32> {
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let px = x as f32 / width as f32;
            let py = y as f32 / height as f32;
            let world_pos = Vector2::new(px, py);

            let mut weight_sum = 0.0;
            let mut color_sum = Color4::new(0.0, 0.0, 0.0, 0.0);

            for tg in gaussians {
                let gaussian = &tg.gaussian;
                let dx = px - gaussian.position.x;
                let dy = py - gaussian.position.y;

                let sx = gaussian.shape.scale_x;
                let sy = gaussian.shape.scale_y;
                let theta = gaussian.shape.rotation;

                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let dx_rot = dx * cos_t + dy * sin_t;
                let dy_rot = -dx * sin_t + dy * cos_t;

                let dist_sq = (dx_rot / sx).powi(2) + (dy_rot / sy).powi(2);
                if dist_sq > 12.25 { continue; }

                let gaussian_val = (-0.5 * dist_sq).exp();
                let weight = gaussian.opacity * gaussian_val;

                // BLEND modulation
                let mut color = gaussian.color;
                if let Some(ref texture) = tg.texture {
                    let local = tg.world_to_local(world_pos);
                    let tex = texture.sample(local.x, local.y);
                    // Alpha blend between base and texture
                    color.r = color.r * (1.0 - alpha) + tex.r * alpha;
                    color.g = color.g * (1.0 - alpha) + tex.g * alpha;
                    color.b = color.b * (1.0 - alpha) + tex.b * alpha;
                }

                weight_sum += weight;
                color_sum.r += weight * color.r;
                color_sum.g += weight * color.g;
                color_sum.b += weight * color.b;
            }

            let final_color = if weight_sum > 1e-10 {
                Color4::new(
                    color_sum.r / weight_sum,
                    color_sum.g / weight_sum,
                    color_sum.b / weight_sum,
                    1.0,
                )
            } else {
                Color4::new(0.0, 0.0, 0.0, 1.0)
            };

            output.set_pixel(x, y, final_color);
        }
    }

    output
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
                img.get_pixel(x0, y0),
                img.get_pixel(x1, y0),
                img.get_pixel(x0, y1),
                img.get_pixel(x1, y1),
            ) {
                let r = (1.0 - fx) * (1.0 - fy) * c00.r + fx * (1.0 - fy) * c10.r +
                        (1.0 - fx) * fy * c01.r + fx * fy * c11.r;
                let g = (1.0 - fx) * (1.0 - fy) * c00.g + fx * (1.0 - fy) * c10.g +
                        (1.0 - fx) * fy * c01.g + fx * fy * c11.g;
                let b = (1.0 - fx) * (1.0 - fy) * c00.b + fx * (1.0 - fy) * c10.b +
                        (1.0 - fx) * fy * c01.b + fx * fy * c11.b;
                resized.set_pixel(x, y, Color4::new(r, g, b, 1.0));
            }
        }
    }
    resized
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2);
        mse += (r.g - t.g).powi(2);
        mse += (r.b - t.b).powi(2);
    }
    mse /= (rendered.width * rendered.height * 3) as f32;
    if mse < 1e-10 {
        100.0
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}
