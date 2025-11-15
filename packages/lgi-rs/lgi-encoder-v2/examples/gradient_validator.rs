//! Gradient Validation Tool
//!
//! Validates analytical gradients against numerical gradients computed
//! via finite differences. This helps identify bugs in gradient computation.

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_encoder_v2::{renderer_v2::RendererV2, loss_functions};

/// Compute numerical gradient via finite differences
fn compute_numerical_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    epsilon: f32,
) -> Vec<NumericalGradient> {
    let mut gradients = vec![NumericalGradient::zero(); gaussians.len()];

    for i in 0..gaussians.len() {
        let mut test_gaussians = gaussians.to_vec();

        // Position X gradient
        test_gaussians[i].position.x += epsilon;
        let rendered_plus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_plus = loss_functions::compute_l2_loss(&rendered_plus, target);

        test_gaussians[i].position.x -= 2.0 * epsilon;
        let rendered_minus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_minus = loss_functions::compute_l2_loss(&rendered_minus, target);

        gradients[i].position.x = (loss_plus - loss_minus) / (2.0 * epsilon);

        // Reset
        test_gaussians[i].position.x = gaussians[i].position.x;

        // Position Y gradient
        test_gaussians[i].position.y += epsilon;
        let rendered_plus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_plus = loss_functions::compute_l2_loss(&rendered_plus, target);

        test_gaussians[i].position.y -= 2.0 * epsilon;
        let rendered_minus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_minus = loss_functions::compute_l2_loss(&rendered_minus, target);

        gradients[i].position.y = (loss_plus - loss_minus) / (2.0 * epsilon);

        test_gaussians[i].position.y = gaussians[i].position.y;

        // Color R gradient
        test_gaussians[i].color.r += epsilon;
        let rendered_plus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_plus = loss_functions::compute_l2_loss(&rendered_plus, target);

        test_gaussians[i].color.r -= 2.0 * epsilon;
        let rendered_minus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_minus = loss_functions::compute_l2_loss(&rendered_minus, target);

        gradients[i].color.r = (loss_plus - loss_minus) / (2.0 * epsilon);

        test_gaussians[i].color.r = gaussians[i].color.r;

        // Color G gradient
        test_gaussians[i].color.g += epsilon;
        let rendered_plus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_plus = loss_functions::compute_l2_loss(&rendered_plus, target);

        test_gaussians[i].color.g -= 2.0 * epsilon;
        let rendered_minus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_minus = loss_functions::compute_l2_loss(&rendered_minus, target);

        gradients[i].color.g = (loss_plus - loss_minus) / (2.0 * epsilon);

        test_gaussians[i].color.g = gaussians[i].color.g;

        // Color B gradient
        test_gaussians[i].color.b += epsilon;
        let rendered_plus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_plus = loss_functions::compute_l2_loss(&rendered_plus, target);

        test_gaussians[i].color.b -= 2.0 * epsilon;
        let rendered_minus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_minus = loss_functions::compute_l2_loss(&rendered_minus, target);

        gradients[i].color.b = (loss_plus - loss_minus) / (2.0 * epsilon);

        test_gaussians[i].color.b = gaussians[i].color.b;

        // Scale X gradient
        test_gaussians[i].shape.scale_x += epsilon;
        let rendered_plus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_plus = loss_functions::compute_l2_loss(&rendered_plus, target);

        test_gaussians[i].shape.scale_x -= 2.0 * epsilon;
        let rendered_minus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_minus = loss_functions::compute_l2_loss(&rendered_minus, target);

        gradients[i].scale_x = (loss_plus - loss_minus) / (2.0 * epsilon);

        test_gaussians[i].shape.scale_x = gaussians[i].shape.scale_x;

        // Scale Y gradient
        test_gaussians[i].shape.scale_y += epsilon;
        let rendered_plus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_plus = loss_functions::compute_l2_loss(&rendered_plus, target);

        test_gaussians[i].shape.scale_y -= 2.0 * epsilon;
        let rendered_minus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_minus = loss_functions::compute_l2_loss(&rendered_minus, target);

        gradients[i].scale_y = (loss_plus - loss_minus) / (2.0 * epsilon);

        test_gaussians[i].shape.scale_y = gaussians[i].shape.scale_y;

        // Rotation gradient
        test_gaussians[i].shape.rotation += epsilon;
        let rendered_plus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_plus = loss_functions::compute_l2_loss(&rendered_plus, target);

        test_gaussians[i].shape.rotation -= 2.0 * epsilon;
        let rendered_minus = RendererV2::render(&test_gaussians, target.width, target.height);
        let loss_minus = loss_functions::compute_l2_loss(&rendered_minus, target);

        gradients[i].rotation = (loss_plus - loss_minus) / (2.0 * epsilon);
    }

    gradients
}

/// Compute analytical gradients (current implementation)
fn compute_analytical_gradients(
    gaussians: &[Gaussian2D<f32, Euler<f32>>],
    rendered: &ImageBuffer<f32>,
    target: &ImageBuffer<f32>,
) -> Vec<NumericalGradient> {
    let width = target.width;
    let height = target.height;
    let mut gradients = vec![NumericalGradient::zero(); gaussians.len()];

    for y in 0..height {
        for x in 0..width {
            let px = x as f32 / width as f32;
            let py = y as f32 / height as f32;
            let rendered_color = rendered.get_pixel(x, y).unwrap();
            let target_color = target.get_pixel(x, y).unwrap();

            // L2 gradient: ∂L2/∂rendered = 2(rendered - target)
            let error_r = 2.0 * (rendered_color.r - target_color.r);
            let error_g = 2.0 * (rendered_color.g - target_color.g);
            let error_b = 2.0 * (rendered_color.b - target_color.b);

            for (i, gaussian) in gaussians.iter().enumerate() {
                let dx = px - gaussian.position.x;
                let dy = py - gaussian.position.y;

                // Rotation
                let cos_t = gaussian.shape.rotation.cos();
                let sin_t = gaussian.shape.rotation.sin();
                let dx_rot = dx * cos_t + dy * sin_t;
                let dy_rot = -dx * sin_t + dy * cos_t;

                let dist_sq = (dx_rot / gaussian.shape.scale_x).powi(2) + (dy_rot / gaussian.shape.scale_y).powi(2);

                if dist_sq > 12.25 {
                    continue;
                }

                let gaussian_val = (-0.5 * dist_sq).exp();
                let weight = gaussian.opacity * gaussian_val;

                // Color gradients
                gradients[i].color.r += error_r * weight;
                gradients[i].color.g += error_g * weight;
                gradients[i].color.b += error_b * weight;

                // Position gradients (current simplified version)
                let error_weighted = error_r * gaussian.color.r + error_g * gaussian.color.g + error_b * gaussian.color.b;
                gradients[i].position.x += error_weighted * weight * dx_rot / gaussian.shape.scale_x.powi(2);
                gradients[i].position.y += error_weighted * weight * dy_rot / gaussian.shape.scale_y.powi(2);

                // Scale gradients
                gradients[i].scale_x += error_weighted * weight * dx_rot.powi(2) / gaussian.shape.scale_x.powi(3);
                gradients[i].scale_y += error_weighted * weight * dy_rot.powi(2) / gaussian.shape.scale_y.powi(3);
            }
        }
    }

    let pixel_count = (width * height) as f32;
    for grad in &mut gradients {
        grad.color.r /= pixel_count;
        grad.color.g /= pixel_count;
        grad.color.b /= pixel_count;
        grad.position.x /= pixel_count;
        grad.position.y /= pixel_count;
        grad.scale_x /= pixel_count;
        grad.scale_y /= pixel_count;
        grad.rotation /= pixel_count;
    }

    gradients
}

#[derive(Clone, Debug)]
struct NumericalGradient {
    position: Vector2<f32>,
    color: Color4<f32>,
    scale_x: f32,
    scale_y: f32,
    rotation: f32,
}

impl NumericalGradient {
    fn zero() -> Self {
        Self {
            position: Vector2::zero(),
            color: Color4::new(0.0, 0.0, 0.0, 0.0),
            scale_x: 0.0,
            scale_y: 0.0,
            rotation: 0.0,
        }
    }

    fn relative_error(&self, other: &Self) -> GradientError {
        let pos_x_err = relative_error(self.position.x, other.position.x);
        let pos_y_err = relative_error(self.position.y, other.position.y);
        let color_r_err = relative_error(self.color.r, other.color.r);
        let color_g_err = relative_error(self.color.g, other.color.g);
        let color_b_err = relative_error(self.color.b, other.color.b);
        let scale_x_err = relative_error(self.scale_x, other.scale_x);
        let scale_y_err = relative_error(self.scale_y, other.scale_y);
        let rotation_err = relative_error(self.rotation, other.rotation);

        GradientError {
            position_x: pos_x_err,
            position_y: pos_y_err,
            color_r: color_r_err,
            color_g: color_g_err,
            color_b: color_b_err,
            scale_x: scale_x_err,
            scale_y: scale_y_err,
            rotation: rotation_err,
        }
    }
}

#[derive(Debug)]
struct GradientError {
    position_x: f32,
    position_y: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    scale_x: f32,
    scale_y: f32,
    rotation: f32,
}

impl GradientError {
    fn max_error(&self) -> f32 {
        self.position_x.max(self.position_y)
            .max(self.color_r)
            .max(self.color_g)
            .max(self.color_b)
            .max(self.scale_x)
            .max(self.scale_y)
            .max(self.rotation)
    }

    fn average_error(&self) -> f32 {
        (self.position_x + self.position_y + self.color_r + self.color_g +
         self.color_b + self.scale_x + self.scale_y + self.rotation) / 8.0
    }
}

fn relative_error(analytical: f32, numerical: f32) -> f32 {
    if numerical.abs() < 1e-10 && analytical.abs() < 1e-10 {
        return 0.0; // Both near zero
    }
    if numerical.abs() < 1e-10 {
        return analytical.abs(); // Numerical is zero, return absolute error
    }
    ((analytical - numerical) / numerical).abs()
}

fn create_test_image(size: u32) -> ImageBuffer<f32> {
    let mut img = ImageBuffer::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let gradient = x as f32 / size as f32;
            let checker = if (x / 8 + y / 8) % 2 == 0 { 0.2 } else { 0.0 };
            let value = gradient * 0.7 + checker;
            img.set_pixel(x, y, Color4::new(value, value, value, 1.0));
        }
    }
    img
}

fn main() {
    let sep = "=".repeat(80);
    println!("{}", sep);
    println!("GRADIENT VALIDATION: Analytical vs Numerical");
    println!("{}", sep);

    // Create small test case
    let target = create_test_image(32);

    // Create a few test Gaussians
    let gaussians = vec![
        Gaussian2D::new(
            Vector2::new(0.25, 0.25),
            Euler::new(0.05, 0.03, 0.5),
            Color4::new(0.8, 0.2, 0.1, 1.0),
            1.0,
        ),
        Gaussian2D::new(
            Vector2::new(0.75, 0.75),
            Euler::new(0.04, 0.04, 0.0),
            Color4::new(0.2, 0.7, 0.3, 1.0),
            1.0,
        ),
        Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::new(0.06, 0.02, 1.2),
            Color4::new(0.5, 0.5, 0.5, 1.0),
            1.0,
        ),
    ];

    println!("\n📊 Test Setup:");
    println!("   Image size: {}×{}", target.width, target.height);
    println!("   N Gaussians: {}", gaussians.len());
    println!("   Finite difference epsilon: 1e-5");

    // Render current state
    let rendered = RendererV2::render(&gaussians, target.width, target.height);
    let loss = loss_functions::compute_l2_loss(&rendered, &target);
    println!("   Current loss: {:.6}", loss);

    println!("\n⏳ Computing numerical gradients (this may take a moment)...");
    let numerical_grads = compute_numerical_gradients(&gaussians, &target, 1e-5);

    println!("⏳ Computing analytical gradients...");
    let analytical_grads = compute_analytical_gradients(&gaussians, &rendered, &target);

    println!("\n{}", sep);
    println!("GRADIENT COMPARISON");
    println!("{}", sep);

    for i in 0..gaussians.len() {
        println!("\nGaussian {}:", i);
        println!("  Position: ({:.4}, {:.4}), Scale: ({:.4}, {:.4}), Rotation: {:.4}",
            gaussians[i].position.x, gaussians[i].position.y,
            gaussians[i].shape.scale_x, gaussians[i].shape.scale_y,
            gaussians[i].shape.rotation);

        let error = numerical_grads[i].relative_error(&analytical_grads[i]);

        println!("\n  Position X:");
        println!("    Numerical:  {:12.6e}", numerical_grads[i].position.x);
        println!("    Analytical: {:12.6e}", analytical_grads[i].position.x);
        println!("    Rel Error:  {:12.2}%", error.position_x * 100.0);

        println!("\n  Position Y:");
        println!("    Numerical:  {:12.6e}", numerical_grads[i].position.y);
        println!("    Analytical: {:12.6e}", analytical_grads[i].position.y);
        println!("    Rel Error:  {:12.2}%", error.position_y * 100.0);

        println!("\n  Color R:");
        println!("    Numerical:  {:12.6e}", numerical_grads[i].color.r);
        println!("    Analytical: {:12.6e}", analytical_grads[i].color.r);
        println!("    Rel Error:  {:12.2}%", error.color_r * 100.0);

        println!("\n  Color G:");
        println!("    Numerical:  {:12.6e}", numerical_grads[i].color.g);
        println!("    Analytical: {:12.6e}", analytical_grads[i].color.g);
        println!("    Rel Error:  {:12.2}%", error.color_g * 100.0);

        println!("\n  Color B:");
        println!("    Numerical:  {:12.6e}", numerical_grads[i].color.b);
        println!("    Analytical: {:12.6e}", analytical_grads[i].color.b);
        println!("    Rel Error:  {:12.2}%", error.color_b * 100.0);

        println!("\n  Scale X:");
        println!("    Numerical:  {:12.6e}", numerical_grads[i].scale_x);
        println!("    Analytical: {:12.6e}", analytical_grads[i].scale_x);
        println!("    Rel Error:  {:12.2}%", error.scale_x * 100.0);

        println!("\n  Scale Y:");
        println!("    Numerical:  {:12.6e}", numerical_grads[i].scale_y);
        println!("    Analytical: {:12.6e}", analytical_grads[i].scale_y);
        println!("    Rel Error:  {:12.2}%", error.scale_y * 100.0);

        println!("\n  Rotation:");
        println!("    Numerical:  {:12.6e}", numerical_grads[i].rotation);
        println!("    Analytical: {:12.6e}", analytical_grads[i].rotation);
        println!("    Rel Error:  {:12.2}%", error.rotation * 100.0);

        println!("\n  Summary:");
        println!("    Max Error:  {:6.2}%", error.max_error() * 100.0);
        println!("    Avg Error:  {:6.2}%", error.average_error() * 100.0);
    }

    println!("\n{}", sep);
    println!("OVERALL ASSESSMENT");
    println!("{}", sep);

    let total_errors: Vec<GradientError> = numerical_grads.iter()
        .zip(analytical_grads.iter())
        .map(|(num, ana)| num.relative_error(ana))
        .collect();

    let avg_max_error: f32 = total_errors.iter().map(|e| e.max_error()).sum::<f32>() / total_errors.len() as f32;
    let avg_avg_error: f32 = total_errors.iter().map(|e| e.average_error()).sum::<f32>() / total_errors.len() as f32;

    println!("Average Maximum Error: {:.2}%", avg_max_error * 100.0);
    println!("Average Overall Error: {:.2}%", avg_avg_error * 100.0);

    if avg_max_error < 0.01 {
        println!("\n✅ EXCELLENT: Gradients are highly accurate (<1% error)");
    } else if avg_max_error < 0.05 {
        println!("\n✅ GOOD: Gradients are reasonably accurate (<5% error)");
    } else if avg_max_error < 0.20 {
        println!("\n⚠️  WARNING: Gradients have moderate errors (5-20%)");
    } else {
        println!("\n❌ FAIL: Gradients are significantly wrong (>20% error)");
        println!("   → Gradient computation needs to be fixed!");
    }

    println!("\n{}", sep);
}
