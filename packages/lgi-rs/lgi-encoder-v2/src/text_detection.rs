//! Text and stroke detection for specialized handling
//! From Hair Toolkit: coherence > 0.8, high gradient, thin width

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_math::{vec::Vector2, color::Color4};

#[derive(Debug, Clone)]
pub struct StrokeRegion {
    pub center: Vector2<f32>,  // Normalized [0,1]
    pub tangent: Vector2<f32>, // Orientation
    pub width: f32,            // Estimated width in pixels
    pub length: f32,           // Estimated length
    pub color: Color4<f32>,
    pub contrast: f32,         // Gradient magnitude
}

/// Detect text/stroke regions using structure tensor
pub fn detect_strokes(
    image: &ImageBuffer<f32>,
    structure_tensor: &StructureTensorField,
) -> Vec<StrokeRegion> {
    let mut strokes = Vec::new();

    // Scan image for stroke characteristics
    // Use denser sampling for text (every 2 pixels)
    for y in (5..image.height-5).step_by(2) {
        for x in (5..image.width-5).step_by(2) {
            let tensor = structure_tensor.get(x, y);

            // Criteria from Hair Toolkit:
            // 1. High coherence (linear structure)
            if tensor.coherence < 0.8 {
                continue;
            }

            // 2. High gradient magnitude (high contrast)
            let grad_mag = tensor.eigenvalue_major.sqrt();
            if grad_mag < 0.01 {  // Lowered threshold (was 0.2, too high)
                continue;
            }

            // 3. Estimate width perpendicular to stroke
            let width = estimate_stroke_width(image, x, y, &tensor);
            if width > 8.0 || width < 1.0 {  // Too wide or too narrow
                continue;
            }

            // This is likely a stroke!
            let pixel = image.get_pixel(x, y).unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            strokes.push(StrokeRegion {
                center: Vector2::new(x as f32 / image.width as f32, y as f32 / image.height as f32),
                tangent: tensor.eigenvector_major,  // Along stroke
                width,
                length: 10.0,  // Estimated, could refine
                color: pixel,
                contrast: grad_mag,
            });
        }
    }

    // Merge nearby strokes
    merge_nearby_strokes(&mut strokes, image);

    strokes
}

/// Estimate stroke width by sampling perpendicular to edge
fn estimate_stroke_width(
    image: &ImageBuffer<f32>,
    x: u32,
    y: u32,
    tensor: &lgi_core::StructureTensor,
) -> f32 {
    // Sample along perpendicular direction (eigenvector_minor)
    let perp = tensor.eigenvector_minor;

    let center_color = image.get_pixel(x, y)
        .map(|c| (c.r + c.g + c.b) / 3.0)
        .unwrap_or(0.5);

    let mut width = 0.0;

    // Sample in both directions
    for &direction in &[1.0, -1.0] {
        for step in 1..10 {
            let sample_x = x as i32 + (direction * perp.x * step as f32) as i32;
            let sample_y = y as i32 + (direction * perp.y * step as f32) as i32;

            if sample_x < 0 || sample_x >= image.width as i32 ||
               sample_y < 0 || sample_y >= image.height as i32 {
                break;
            }

            let sample_color = image.get_pixel(sample_x as u32, sample_y as u32)
                .map(|c| (c.r + c.g + c.b) / 3.0)
                .unwrap_or(0.5);

            // Check if color changed significantly (crossed stroke boundary)
            if (sample_color - center_color).abs() > 0.3 {
                width += step as f32;
                break;
            }
        }
    }

    width
}

/// Merge nearby strokes (simple spatial clustering)
fn merge_nearby_strokes(strokes: &mut Vec<StrokeRegion>, image: &ImageBuffer<f32>) {
    // Sort by position for efficient clustering
    strokes.sort_by(|a, b| {
        let dist_a = a.center.x + a.center.y;
        let dist_b = b.center.x + b.center.y;
        dist_a.partial_cmp(&dist_b).unwrap()
    });

    // Simple merging: if two strokes within 5 pixels, keep one
    let mut i = 0;
    while i < strokes.len() {
        let mut j = i + 1;
        while j < strokes.len() {
            let dx = (strokes[i].center.x - strokes[j].center.x) * image.width as f32;
            let dy = (strokes[i].center.y - strokes[j].center.y) * image.height as f32;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < 5.0 {
                // Remove duplicate
                strokes.remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }
}

/// Create specialized Gaussians for text strokes
/// From Hair Toolkit: σ⊥≈0.3 px, σ∥≈5 px for thin lines
pub fn create_stroke_gaussians(strokes: &[StrokeRegion]) -> Vec<lgi_math::gaussian::Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>> {
    let mut gaussians = Vec::new();

    for stroke in strokes {
        // Thin Gaussian perpendicular to stroke, long along stroke
        let sigma_perp = (stroke.width * 0.4).clamp(0.5, 2.0);  // Adapt to actual width
        let sigma_parallel = (stroke.length * 0.8).clamp(3.0, 10.0);

        // Convert to normalized (assuming 256×256)
        let sigma_x_norm = sigma_parallel / 256.0;
        let sigma_y_norm = sigma_perp / 256.0;

        // Rotation from tangent vector
        let rotation = stroke.tangent.y.atan2(stroke.tangent.x);

        gaussians.push(lgi_math::gaussian::Gaussian2D::new(
            stroke.center,
            lgi_math::parameterization::Euler::new(sigma_x_norm, sigma_y_norm, rotation),
            stroke.color,
            1.0,
        ));
    }

    gaussians
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stroke_detection() {
        // Create simple vertical line
        let mut img = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let val = if x >= 30 && x <= 34 { 1.0 } else { 0.0 };
                img.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let tensor = StructureTensorField::compute(&img, 1.2, 1.0).unwrap();
        let strokes = detect_strokes(&img, &tensor);

        assert!(strokes.len() > 0, "Should detect at least one stroke");
        assert!(strokes[0].width < 8.0, "Stroke should be thin");
        assert!(strokes[0].contrast > 0.0, "Stroke should have contrast");
    }
}
