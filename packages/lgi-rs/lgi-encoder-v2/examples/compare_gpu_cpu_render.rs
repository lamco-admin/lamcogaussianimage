//! Compare GPU and CPU rendering to identify differences
//!
//! This test verifies that GPU and CPU renderers produce identical output.

use lgi_core::{ImageBuffer, RenderMode};
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_gpu::GpuRenderer;

fn create_test_gaussians() -> Vec<Gaussian2D<f32, Euler<f32>>> {
    // Simple test case: a few Gaussians with known positions
    vec![
        Gaussian2D {
            position: Vector2 { x: 0.25, y: 0.25 },
            shape: Euler { scale_x: 0.05, scale_y: 0.05, rotation: 0.0 },
            color: Color4::new(1.0, 0.0, 0.0, 1.0),  // Red
            opacity: 1.0,
            weight: None,
        },
        Gaussian2D {
            position: Vector2 { x: 0.75, y: 0.25 },
            shape: Euler { scale_x: 0.05, scale_y: 0.03, rotation: 0.5 },  // Anisotropic with rotation
            color: Color4::new(0.0, 1.0, 0.0, 1.0),  // Green
            opacity: 1.0,
            weight: None,
        },
        Gaussian2D {
            position: Vector2 { x: 0.5, y: 0.75 },
            shape: Euler { scale_x: 0.03, scale_y: 0.03, rotation: 0.0 },
            color: Color4::new(0.0, 0.0, 1.0, 1.0),  // Blue
            opacity: 0.5,  // Semi-transparent
            weight: None,
        },
    ]
}

fn compute_image_diff(a: &ImageBuffer<f32>, b: &ImageBuffer<f32>) -> (f32, f32, f32) {
    let mut mse = 0.0;
    let mut max_diff = 0.0f32;
    let mut num_pixels = 0;

    for y in 0..a.height {
        for x in 0..a.width {
            let pa = a.get_pixel(x, y).unwrap();
            let pb = b.get_pixel(x, y).unwrap();

            let dr = (pa.r - pb.r).abs();
            let dg = (pa.g - pb.g).abs();
            let db = (pa.b - pb.b).abs();

            mse += dr * dr + dg * dg + db * db;
            max_diff = max_diff.max(dr).max(dg).max(db);
            num_pixels += 1;
        }
    }

    let avg_mse = mse / (num_pixels * 3) as f32;
    let psnr = if avg_mse > 0.0 { 10.0 * (1.0 / avg_mse).log10() } else { 100.0 };

    (avg_mse, max_diff, psnr)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("===== GPU vs CPU Render Comparison =====\n");

    let gaussians = create_test_gaussians();
    let width = 256;
    let height = 256;

    // CPU render
    println!("[CPU Render]");
    let cpu_start = std::time::Instant::now();
    let cpu_image = RendererV2::render(&gaussians, width, height);
    let cpu_time = cpu_start.elapsed();
    println!("  Time: {:.2}ms", cpu_time.as_secs_f32() * 1000.0);

    // Sample a few pixels
    println!("  Sample pixels:");
    for (x, y) in [(64, 64), (192, 64), (128, 192)] {
        let p = cpu_image.get_pixel(x, y).unwrap();
        println!("    ({}, {}): [{:.3}, {:.3}, {:.3}]", x, y, p.r, p.g, p.b);
    }

    // GPU render
    println!("\n[GPU Render]");

    match pollster::block_on(async {
        let mut renderer = GpuRenderer::new()
            .await
            .map_err(|e| format!("Renderer creation failed: {:?}", e))?;

        let gpu_start = std::time::Instant::now();
        let gpu_image = renderer.render(&gaussians, width, height, RenderMode::AccumulatedSum)
            .map_err(|e| format!("Render failed: {:?}", e))?;
        let gpu_time = gpu_start.elapsed();

        Ok::<(ImageBuffer<f32>, std::time::Duration), String>((gpu_image, gpu_time))
    }) {
        Ok((gpu_image, gpu_time)) => {
            println!("  Time: {:.2}ms", gpu_time.as_secs_f32() * 1000.0);

            // Sample same pixels
            println!("  Sample pixels:");
            for (x, y) in [(64, 64), (192, 64), (128, 192)] {
                let p = gpu_image.get_pixel(x, y).unwrap();
                println!("    ({}, {}): [{:.3}, {:.3}, {:.3}]", x, y, p.r, p.g, p.b);
            }

            // Compare
            println!("\n[Comparison]");
            let (mse, max_diff, psnr) = compute_image_diff(&cpu_image, &gpu_image);
            println!("  MSE: {:.8}", mse);
            println!("  Max pixel diff: {:.6}", max_diff);
            println!("  PSNR (CPU vs GPU): {:.2} dB", psnr);

            if max_diff < 0.001 {
                println!("\n✓ GPU and CPU render IDENTICAL (diff < 0.001)");
            } else if max_diff < 0.01 {
                println!("\n⚠ GPU and CPU render SIMILAR (diff < 0.01)");
            } else {
                println!("\n✗ GPU and CPU render DIFFERENT (diff = {:.4})", max_diff);
                println!("  This could explain gradient differences!");
            }
        }
        Err(e) => {
            println!("  GPU test failed: {}", e);
        }
    }

    Ok(())
}
