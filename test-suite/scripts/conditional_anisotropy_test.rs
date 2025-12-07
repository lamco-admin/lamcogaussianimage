//! Test conditional anisotropy based on structure tensor coherence
//! Validates proper application of isotropic (flat) vs anisotropic (edges)

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_encoder_v2::renderer_v2::RendererV2;

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Conditional Anisotropy Test                             ║");
    println!("║  Coherence < 0.2 → Isotropic                             ║");
    println!("║  Coherence ≥ 0.2 → Anisotropic                           ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Create test image: vertical edge (black | white at x=128)
    let width = 256;
    let height = 256;
    let mut target = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let color = if x < 128 {
                Color4::new(0.0, 0.0, 0.0, 1.0)  // Black
            } else {
                Color4::new(1.0, 1.0, 1.0, 1.0)  // White
            };
            target.set_pixel(x, y, color);
        }
    }

    println!("Target: Vertical edge at x=128 ({}×{})\n", width, height);

    // Compute structure tensor
    let tensor_field = StructureTensorField::compute(&target, 1.2, 1.0).unwrap();

    // Initialize Gaussians with CONDITIONAL anisotropy
    let n = 400;  // 20×20 grid
    let grid_size = (n as f32).sqrt() as u32;

    println!("Initializing {} Gaussians with conditional anisotropy...\n", n);

    let mut gaussians = Vec::new();
    let sigma_base = 0.6 * ((width * height) as f32 / n as f32).sqrt();

    let mut isotropic_count = 0;
    let mut anisotropic_count = 0;
    let mut isotropic_psnr_sum = 0.0;
    let mut anisotropic_psnr_sum = 0.0;

    for i in 0..grid_size {
        for j in 0..grid_size {
            let px = (i as f32 + 0.5) / grid_size as f32;
            let py = (j as f32 + 0.5) / grid_size as f32;

            let x = (px * width as f32) as u32;
            let y = (py * height as f32) as u32;

            // Get structure tensor at this position
            let tensor = tensor_field.get(x.min(width-1), y.min(height-1));
            let coherence = tensor.coherence;

            // CONDITIONAL APPLICATION (from Debug Plan)
            let (scale_x, scale_y, rotation) = if coherence < 0.2 {
                // FLAT REGION → ISOTROPIC
                isotropic_count += 1;
                (sigma_base, sigma_base, 0.0)
            } else {
                // EDGE REGION → ANISOTROPIC
                anisotropic_count += 1;

                // Thin perpendicular to edge, long along edge
                let sigma_perp = sigma_base / (1.0 + 3.0 * coherence);
                let sigma_para = 4.0 * sigma_perp;  // β=4 anisotropy ratio

                // Orientation from structure tensor
                let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);

                (sigma_para, sigma_perp, angle)
            };

            // Sample color from target
            let color = target.get_pixel(x, y).copied().unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            let gaussian = Gaussian2D::new(
                Vector2::new(px, py),
                Euler::new(scale_x, scale_y, rotation),
                Color4::new(color_sample.r, color_sample.g, color_sample.b, 1.0),
                1.0,  // opacity
            );

            gaussians.push(gaussian);
        }
    }

    println!("Initialization Statistics:");
    println!("  Isotropic:    {} Gaussians ({:.1}%)", isotropic_count, 100.0 * isotropic_count as f32 / n as f32);
    println!("  Anisotropic:  {} Gaussians ({:.1}%)", anisotropic_count, 100.0 * anisotropic_count as f32 / n as f32);
    println!("  Base σ:       {:.2} pixels", sigma_base);
    println!();

    // Render and measure quality
    let rendered = RendererV2::render(&gaussians, width, height);
    let psnr = compute_psnr(&rendered, &target);

    println!("═══════════════════════════════════════════");
    println!("RESULTS");
    println!("═══════════════════════════════════════════");
    println!("  PSNR: {:.2} dB", psnr);
    println!();

    // Measure PSNR in edge region (x=100-156)
    let edge_psnr = compute_region_psnr(&rendered, &target, 100, 0, 56, height);
    println!("  Edge region PSNR: {:.2} dB", edge_psnr);
    println!();

    // Compare to all-isotropic
    println!("Comparison to all-isotropic (β=1.0):");
    let mut gaussians_iso = gaussians.clone();
    for g in &mut gaussians_iso {
        g.shape.scale_y = g.shape.scale_x;  // Make isotropic
        g.shape.rotation = 0.0;
    }
    let rendered_iso = RendererV2::render(&gaussians_iso, width, height);
    let psnr_iso = compute_psnr(&rendered_iso, &target);
    let edge_psnr_iso = compute_region_psnr(&rendered_iso, &target, 100, 0, 56, height);

    println!("  All-isotropic PSNR:       {:.2} dB (Δ {:.2} dB)", psnr_iso, psnr_iso - psnr);
    println!("  All-isotropic edge PSNR:  {:.2} dB (Δ {:.2} dB)", edge_psnr_iso, edge_psnr_iso - edge_psnr);
    println!();

    // Compare to all-anisotropic
    println!("Comparison to all-anisotropic (β=4.0):");
    let mut gaussians_aniso = Vec::new();
    for i in 0..grid_size {
        for j in 0..grid_size {
            let px = (i as f32 + 0.5) / grid_size as f32;
            let py = (j as f32 + 0.5) / grid_size as f32;
            let x = (px * width as f32) as u32;
            let y = (py * height as f32) as u32;

            let tensor = tensor_field.get(x.min(width-1), y.min(height-1));
            let sigma_perp = sigma_base / (1.0 + 3.0 * tensor.coherence);
            let sigma_para = 4.0 * sigma_perp;
            let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);

            let color = target.get_pixel(x, y).copied().unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            gaussians_aniso.push(Gaussian2D::new(
                Vector2::new(px, py),
                Euler::new(sigma_para, sigma_perp, angle),
                Color4::new(color.r, color.g, color.b, 1.0),
                1.0,  // opacity
            ));
        }
    }

    let rendered_aniso = RendererV2::render(&gaussians_aniso, width, height);
    let psnr_aniso = compute_psnr(&rendered_aniso, &target);
    let edge_psnr_aniso = compute_region_psnr(&rendered_aniso, &target, 100, 0, 56, height);

    println!("  All-anisotropic PSNR:       {:.2} dB (Δ {:.2} dB)", psnr_aniso, psnr_aniso - psnr);
    println!("  All-anisotropic edge PSNR:  {:.2} dB (Δ {:.2} dB)", edge_psnr_aniso, edge_psnr_aniso - edge_psnr);
    println!();

    println!("═══════════════════════════════════════════");
    println!("CONCLUSION");
    println!("═══════════════════════════════════════════");

    if psnr > psnr_iso && psnr > psnr_aniso {
        println!("  ✅ CONDITIONAL is BEST!");
        println!("  Beats isotropic by {:.2} dB", psnr - psnr_iso);
        println!("  Beats anisotropic by {:.2} dB", psnr - psnr_aniso);
    } else if psnr_iso > psnr {
        println!("  ⚠️  All-isotropic performed better (+{:.2} dB)", psnr_iso - psnr);
    } else {
        println!("  ⚠️  All-anisotropic performed better (+{:.2} dB)", psnr_aniso - psnr);
    }
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

fn compute_region_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>, x_start: u32, y_start: u32, region_width: u32, region_height: u32) -> f32 {
    let mut mse = 0.0;
    let mut count = 0;

    for y in y_start..(y_start + region_height).min(rendered.height) {
        for x in x_start..(x_start + region_width).min(rendered.width) {
            if let (Some(r), Some(t)) = (rendered.get_pixel(x, y), target.get_pixel(x, y)) {
                mse += (r.r - t.r).powi(2);
                mse += (r.g - t.g).powi(2);
                mse += (r.b - t.b).powi(2);
                count += 3;
            }
        }
    }

    mse /= count as f32;

    if mse < 1e-10 {
        100.0
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}
