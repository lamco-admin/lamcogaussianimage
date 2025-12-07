//! Compare GPU and CPU gradient values to identify the exact difference
//!
//! This diagnostic test computes gradients for the SAME Gaussians on both CPU and GPU
//! and compares them value-by-value to find discrepancies.

use lgi_core::ImageBuffer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use lgi_gpu::gradient::GpuGradientComputer;
use wgpu;

fn create_test_gaussians() -> Vec<Gaussian2D<f32, Euler<f32>>> {
    vec![
        Gaussian2D {
            position: Vector2 { x: 0.3, y: 0.3 },
            shape: Euler { scale_x: 0.05, scale_y: 0.05, rotation: 0.0 },
            color: Color4::new(1.0, 0.0, 0.0, 1.0),
            opacity: 1.0,
            weight: None,
        },
        Gaussian2D {
            position: Vector2 { x: 0.7, y: 0.5 },
            shape: Euler { scale_x: 0.08, scale_y: 0.04, rotation: 0.3 },
            color: Color4::new(0.0, 1.0, 0.5, 1.0),
            opacity: 0.8,
            weight: None,
        },
        Gaussian2D {
            position: Vector2 { x: 0.5, y: 0.8 },
            shape: Euler { scale_x: 0.03, scale_y: 0.06, rotation: -0.2 },
            color: Color4::new(0.2, 0.3, 1.0, 1.0),
            opacity: 1.0,
            weight: None,
        },
    ]
}

fn create_target_image(width: u32, height: u32) -> ImageBuffer<f32> {
    // Create a simple gradient target
    let mut data = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = x as f32 / width as f32;
            let g = y as f32 / height as f32;
            let b = 0.5;
            data.push(Color4::new(r, g, b, 1.0));
        }
    }
    ImageBuffer { width, height, data }
}

/// CPU gradient computation (matching lgi-encoder-v2/src/adam_optimizer.rs)
fn compute_cpu_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
) -> Vec<CpuGradient> {
    let width = target.width;
    let height = target.height;
    let mut gradients = vec![CpuGradient::zero(); gaussians.len()];

    for y in 0..height {
        for x in 0..width {
            let px = x as f32 / width as f32;
            let py = y as f32 / height as f32;
            let rendered_color = rendered.get_pixel(x, y).unwrap();
            let target_color = target.get_pixel(x, y).unwrap();

            let error_r = 2.0 * (rendered_color.r - target_color.r);
            let error_g = 2.0 * (rendered_color.g - target_color.g);
            let error_b = 2.0 * (rendered_color.b - target_color.b);

            for (i, gaussian) in gaussians.iter().enumerate() {
                let scale_x = gaussian.shape.scale_x.max(0.001);
                let scale_y = gaussian.shape.scale_y.max(0.001);

                let dx = px - gaussian.position.x;
                let dy = py - gaussian.position.y;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq > 0.1 { continue; }

                let scale_product = (scale_x * scale_y).max(1e-6);
                let weight = (-0.5 * dist_sq / scale_product).exp();

                if weight < 1e-6 || weight.is_nan() || weight.is_infinite() {
                    continue;
                }

                gradients[i].d_color_r += error_r * weight;
                gradients[i].d_color_g += error_g * weight;
                gradients[i].d_color_b += error_b * weight;

                let error_weighted = error_r * gaussian.color.r
                    + error_g * gaussian.color.g
                    + error_b * gaussian.color.b;
                gradients[i].d_position_x += error_weighted * weight * dx;
                gradients[i].d_position_y += error_weighted * weight * dy;

                let scale_x_sq = scale_x.powi(2).max(1e-6);
                let scale_y_sq = scale_y.powi(2).max(1e-6);

                gradients[i].d_scale_x += error_weighted * weight * dist_sq / scale_x_sq;
                gradients[i].d_scale_y += error_weighted * weight * dist_sq / scale_y_sq;
            }
        }
    }

    let pixel_count = (width * height) as f32;
    for grad in &mut gradients {
        grad.d_color_r /= pixel_count;
        grad.d_color_g /= pixel_count;
        grad.d_color_b /= pixel_count;
        grad.d_position_x /= pixel_count;
        grad.d_position_y /= pixel_count;
        grad.d_scale_x /= pixel_count;
        grad.d_scale_y /= pixel_count;
    }

    gradients
}

#[derive(Clone, Debug)]
struct CpuGradient {
    d_position_x: f32,
    d_position_y: f32,
    d_color_r: f32,
    d_color_g: f32,
    d_color_b: f32,
    d_scale_x: f32,
    d_scale_y: f32,
}

impl CpuGradient {
    fn zero() -> Self {
        Self {
            d_position_x: 0.0,
            d_position_y: 0.0,
            d_color_r: 0.0,
            d_color_g: 0.0,
            d_color_b: 0.0,
            d_scale_x: 0.0,
            d_scale_y: 0.0,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("===== GPU vs CPU Gradient Comparison =====\n");

    let gaussians = create_test_gaussians();
    let width = 128;
    let height = 128;
    let target = create_target_image(width, height);

    // Render with CPU RendererV2
    println!("[1] Rendering with CPU RendererV2...");
    let rendered_cpu = RendererV2::render(&gaussians, width, height);

    // Compute CPU gradients
    println!("[2] Computing CPU gradients...");
    let cpu_grads = compute_cpu_gradients(&gaussians, &rendered_cpu, &target);

    // GPU gradients
    println!("[3] Computing GPU gradients...");

    let gpu_result = pollster::block_on(async {
        // Initialize wgpu
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;

        let adapter = match adapter {
            Ok(a) => a,
            Err(e) => return Err(format!("No GPU adapter found: {:?}", e)),
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| format!("Device request failed: {:?}", e))?;

        let computer = GpuGradientComputer::new(&device)
            .map_err(|e| format!("Gradient computer creation failed: {:?}", e))?;

        let gpu_grads = computer.compute_gradients(&device, &queue, &gaussians, &rendered_cpu, &target)
            .map_err(|e| format!("Gradient computation failed: {:?}", e))?;

        Ok::<_, String>(gpu_grads)
    });

    match gpu_result {
        Ok(gpu_grads) => {
            println!("\n[4] Comparing gradients...\n");

            println!("{:>8} {:>12} {:>12} {:>12} {:>12}", "Param", "CPU", "GPU", "Diff", "Rel%");
            println!("{}", "-".repeat(60));

            for (i, (cpu, gpu)) in cpu_grads.iter().zip(gpu_grads.iter()).enumerate() {
                println!("\nGaussian {}:", i);

                compare_value("pos.x", cpu.d_position_x, gpu.d_position[0]);
                compare_value("pos.y", cpu.d_position_y, gpu.d_position[1]);
                compare_value("color.r", cpu.d_color_r, gpu.d_color[0]);
                compare_value("color.g", cpu.d_color_g, gpu.d_color[1]);
                compare_value("color.b", cpu.d_color_b, gpu.d_color[2]);
                compare_value("scale_x", cpu.d_scale_x, gpu.d_scale_x);
                compare_value("scale_y", cpu.d_scale_y, gpu.d_scale_y);
            }

            // Check if any significant differences
            let mut max_diff = 0.0f32;
            for (cpu, gpu) in cpu_grads.iter().zip(gpu_grads.iter()) {
                max_diff = max_diff
                    .max((cpu.d_position_x - gpu.d_position[0]).abs())
                    .max((cpu.d_position_y - gpu.d_position[1]).abs())
                    .max((cpu.d_color_r - gpu.d_color[0]).abs())
                    .max((cpu.d_color_g - gpu.d_color[1]).abs())
                    .max((cpu.d_color_b - gpu.d_color[2]).abs())
                    .max((cpu.d_scale_x - gpu.d_scale_x).abs())
                    .max((cpu.d_scale_y - gpu.d_scale_y).abs());
            }

            println!("\n{}", "=".repeat(60));
            if max_diff < 1e-5 {
                println!("✓ Gradients MATCH (max diff: {:.2e})", max_diff);
            } else if max_diff < 1e-3 {
                println!("⚠ Gradients SIMILAR (max diff: {:.2e})", max_diff);
            } else {
                println!("✗ Gradients DIFFER SIGNIFICANTLY (max diff: {:.2e})", max_diff);
                println!("  This explains the optimization quality gap!");
            }
        }
        Err(e) => {
            println!("GPU gradient computation failed: {:?}", e);
            println!("\nFalling back to manual comparison...");

            // Print CPU gradients for reference
            println!("\nCPU gradients:");
            for (i, g) in cpu_grads.iter().enumerate() {
                println!("  Gaussian {}: pos=({:.6}, {:.6}) color=({:.6}, {:.6}, {:.6}) scale=({:.6}, {:.6})",
                    i, g.d_position_x, g.d_position_y,
                    g.d_color_r, g.d_color_g, g.d_color_b,
                    g.d_scale_x, g.d_scale_y);
            }
        }
    }

    Ok(())
}

fn compare_value(name: &str, cpu: f32, gpu: f32) {
    let diff = (cpu - gpu).abs();
    let rel_pct = if cpu.abs() > 1e-8 {
        (diff / cpu.abs()) * 100.0
    } else if gpu.abs() > 1e-8 {
        (diff / gpu.abs()) * 100.0
    } else {
        0.0
    };

    let status = if diff < 1e-6 { "✓" } else if diff < 1e-4 { "~" } else { "✗" };

    println!("{} {:>8} {:>12.6} {:>12.6} {:>12.2e} {:>10.1}%",
             status, name, cpu, gpu, diff, rel_pct);
}
