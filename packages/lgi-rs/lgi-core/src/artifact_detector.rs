//! Artifact Detection for Gaussian Image Encoding
//!
//! Detects common Gaussian encoding artifacts to guide refinement:
//!
//! # Artifact Types
//!
//! | Artifact | Cause | Detection | Fix |
//! |----------|-------|-----------|-----|
//! | Color Bleeding | Gaussians crossing edges | Cross-edge color correlation | Geodesic clamping |
//! | Blur | Insufficient Gaussians | High-freq loss vs target | Add smaller Gaussians |
//! | Ringing | Oscillating overlaps | Sign-alternating error | Adjust positions/scales |
//! | Banding | Quantization in gradients | Step-like transitions | Add gradient Gaussians |
//! | Ghosting | Misaligned overlaps | Faint duplicates | Prune/merge similar |
//!
//! # Usage
//!
//! ```ignore
//! use lgi_core::artifact_detector::{ArtifactDetector, ArtifactReport};
//!
//! let detector = ArtifactDetector::default();
//! let report = detector.analyze(&rendered, &target)?;
//!
//! println!("Bleeding score: {:.2}", report.bleeding_score);
//! println!("Blur regions: {}", report.blur_regions.len());
//! ```

use crate::{ImageBuffer, structure_tensor::StructureTensorField, laplacian::compute_laplacian};

/// Detection thresholds
#[derive(Debug, Clone)]
pub struct ArtifactThresholds {
    /// Minimum edge strength for bleeding detection (default: 0.1)
    pub edge_strength_min: f32,

    /// Cross-edge color difference threshold (default: 0.05)
    pub bleeding_threshold: f32,

    /// High-frequency loss ratio for blur (default: 0.5)
    pub blur_hf_ratio: f32,

    /// Sign alternation count for ringing (default: 3)
    pub ringing_alternations: usize,

    /// Gradient step size for banding (default: 0.03)
    pub banding_step_size: f32,

    /// Similarity threshold for ghosting (default: 0.9)
    pub ghost_similarity: f32,

    /// Laplacian sigma for HF analysis (default: 1.0)
    pub laplacian_sigma: f32,

    /// Window size for local analysis (default: 7)
    pub window_size: usize,
}

impl Default for ArtifactThresholds {
    fn default() -> Self {
        Self {
            edge_strength_min: 0.1,
            bleeding_threshold: 0.05,
            blur_hf_ratio: 0.5,
            ringing_alternations: 3,
            banding_step_size: 0.03,
            ghost_similarity: 0.9,
            laplacian_sigma: 1.0,
            window_size: 7,
        }
    }
}

/// Comprehensive artifact analysis result
#[derive(Debug, Clone)]
pub struct ArtifactReport {
    /// Overall artifact severity (0-1, higher = worse)
    pub overall_score: f32,

    /// Color bleeding score (0-1)
    pub bleeding_score: f32,

    /// Blur score (0-1)
    pub blur_score: f32,

    /// Ringing score (0-1)
    pub ringing_score: f32,

    /// Banding score (0-1)
    pub banding_score: f32,

    /// Ghosting score (0-1)
    pub ghosting_score: f32,

    /// Pixel-level bleeding intensity map
    pub bleeding_map: Vec<f32>,

    /// Pixel-level blur intensity map
    pub blur_map: Vec<f32>,

    /// Regions with high error (x, y, error)
    pub high_error_regions: Vec<(u32, u32, f32)>,

    /// Image dimensions
    pub width: u32,
    pub height: u32,
}

impl ArtifactReport {
    /// Get top N error regions
    pub fn top_error_regions(&self, n: usize) -> Vec<(u32, u32, f32)> {
        let mut sorted = self.high_error_regions.clone();
        sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        sorted.into_iter().take(n).collect()
    }

    /// Create visualization showing artifact locations
    pub fn visualize(&self) -> ImageBuffer<f32> {
        use lgi_math::color::Color4;

        let mut vis = ImageBuffer::new(self.width, self.height);

        // Blend bleeding (red) and blur (blue) maps
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) as usize;
                let bleeding = self.bleeding_map.get(idx).copied().unwrap_or(0.0);
                let blur = self.blur_map.get(idx).copied().unwrap_or(0.0);

                // Color: red = bleeding, blue = blur, green = OK
                let ok = 1.0 - bleeding.max(blur);
                let color = Color4::new(bleeding, ok * 0.5, blur, 1.0);
                vis.set_pixel(x, y, color);
            }
        }

        vis
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("Artifact Analysis Report:");
        println!("  Overall severity: {:.1}%", self.overall_score * 100.0);
        println!("  Breakdown:");
        println!("    Bleeding: {:.1}%", self.bleeding_score * 100.0);
        println!("    Blur:     {:.1}%", self.blur_score * 100.0);
        println!("    Ringing:  {:.1}%", self.ringing_score * 100.0);
        println!("    Banding:  {:.1}%", self.banding_score * 100.0);
        println!("    Ghosting: {:.1}%", self.ghosting_score * 100.0);
        println!("  High error regions: {}", self.high_error_regions.len());
    }
}

/// Artifact detector for Gaussian image encoding
pub struct ArtifactDetector {
    /// Detection thresholds
    pub thresholds: ArtifactThresholds,
}

impl Default for ArtifactDetector {
    fn default() -> Self {
        Self {
            thresholds: ArtifactThresholds::default(),
        }
    }
}

impl ArtifactDetector {
    /// Create with custom thresholds
    pub fn with_thresholds(thresholds: ArtifactThresholds) -> Self {
        Self { thresholds }
    }

    /// Analyze rendered image for artifacts compared to target
    pub fn analyze(
        &self,
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> crate::Result<ArtifactReport> {
        let width = target.width;
        let height = target.height;

        // Compute structure tensor for edge detection
        let tensor_field = StructureTensorField::compute(target, 1.2, 2.5)?;

        // Detect individual artifact types
        let (bleeding_score, bleeding_map) = self.detect_bleeding(rendered, target, &tensor_field);
        let (blur_score, blur_map) = self.detect_blur(rendered, target);
        let ringing_score = self.detect_ringing(rendered, target, &tensor_field);
        let banding_score = self.detect_banding(rendered, target);
        let ghosting_score = self.detect_ghosting(rendered);

        // Find high-error regions
        let high_error_regions = self.find_high_error_regions(rendered, target);

        // Compute overall score (weighted average)
        let overall_score = (bleeding_score * 0.3 +
                           blur_score * 0.25 +
                           ringing_score * 0.2 +
                           banding_score * 0.15 +
                           ghosting_score * 0.1).min(1.0);

        Ok(ArtifactReport {
            overall_score,
            bleeding_score,
            blur_score,
            ringing_score,
            banding_score,
            ghosting_score,
            bleeding_map,
            blur_map,
            high_error_regions,
            width,
            height,
        })
    }

    /// Detect color bleeding across edges
    ///
    /// Bleeding occurs when Gaussian colors "leak" across edges that should be sharp.
    /// Detection: For edge pixels, compare color difference across edge in target vs rendered.
    /// If rendered has smaller difference (colors mixed), that's bleeding.
    fn detect_bleeding(
        &self,
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
        tensor_field: &StructureTensorField,
    ) -> (f32, Vec<f32>) {
        let width = target.width;
        let height = target.height;
        let mut bleeding_map = vec![0.0; (width * height) as usize];
        let mut total_bleeding = 0.0;
        let mut edge_count = 0;

        for y in 2..(height - 2) {
            for x in 2..(width - 2) {
                let tensor = tensor_field.get(x, y);
                let energy = tensor.eigenvalue_major + tensor.eigenvalue_minor;

                // Only check at edges
                if energy < self.thresholds.edge_strength_min {
                    continue;
                }

                edge_count += 1;

                // Get edge normal direction
                let normal = tensor.eigenvector_major;
                let dx = (normal.x * 2.0).round() as i32;
                let dy = (normal.y * 2.0).round() as i32;

                // Sample colors on each side of edge
                let x1 = (x as i32 - dx).clamp(0, width as i32 - 1) as u32;
                let y1 = (y as i32 - dy).clamp(0, height as i32 - 1) as u32;
                let x2 = (x as i32 + dx).clamp(0, width as i32 - 1) as u32;
                let y2 = (y as i32 + dy).clamp(0, height as i32 - 1) as u32;

                let target_side1 = target.get_pixel(x1, y1).unwrap();
                let target_side2 = target.get_pixel(x2, y2).unwrap();
                let rendered_side1 = rendered.get_pixel(x1, y1).unwrap();
                let rendered_side2 = rendered.get_pixel(x2, y2).unwrap();

                // Color difference across edge
                let target_diff = color_distance(&target_side1, &target_side2);
                let rendered_diff = color_distance(&rendered_side1, &rendered_side2);

                // Bleeding = target has sharp edge but rendered doesn't
                if target_diff > self.thresholds.edge_strength_min {
                    let bleeding = ((target_diff - rendered_diff) / target_diff).max(0.0);
                    bleeding_map[(y * width + x) as usize] = bleeding;
                    total_bleeding += bleeding;
                }
            }
        }

        let bleeding_score = if edge_count > 0 {
            (total_bleeding / edge_count as f32).min(1.0)
        } else {
            0.0
        };

        (bleeding_score, bleeding_map)
    }

    /// Detect blur (insufficient high-frequency detail)
    ///
    /// Blur occurs when the rendered image lacks fine detail present in target.
    /// Detection: Compare Laplacian (high-frequency content) of target vs rendered.
    fn detect_blur(
        &self,
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> (f32, Vec<f32>) {
        let width = target.width;
        let height = target.height;

        // Compute Laplacian for both images
        let target_lap = compute_laplacian(target);
        let rendered_lap = compute_laplacian(rendered);

        let mut blur_map = vec![0.0; (width * height) as usize];
        let mut total_blur = 0.0;
        let mut valid_count = 0;

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let idx = (y * width + x) as usize;
                let target_hf = target_lap[idx];
                let rendered_hf = rendered_lap[idx];

                // Only check where target has HF content
                if target_hf > 0.01 {
                    valid_count += 1;

                    // Blur = rendered has less HF than target
                    let hf_ratio = rendered_hf / target_hf;
                    let blur = (1.0 - hf_ratio).max(0.0).min(1.0);
                    blur_map[idx] = blur;
                    total_blur += blur;
                }
            }
        }

        let blur_score = if valid_count > 0 {
            (total_blur / valid_count as f32).min(1.0)
        } else {
            0.0
        };

        (blur_score, blur_map)
    }

    /// Detect ringing (oscillating errors near edges)
    ///
    /// Ringing occurs when overlapping Gaussians create alternating over/under-estimation.
    /// Detection: Near edges, look for sign-alternating error patterns.
    fn detect_ringing(
        &self,
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
        tensor_field: &StructureTensorField,
    ) -> f32 {
        let width = target.width;
        let height = target.height;
        let mut ringing_count = 0;
        let mut edge_count = 0;

        let half_window = self.thresholds.window_size / 2;

        for y in (half_window as u32)..(height - half_window as u32) {
            for x in (half_window as u32)..(width - half_window as u32) {
                let tensor = tensor_field.get(x, y);
                let energy = tensor.eigenvalue_major + tensor.eigenvalue_minor;

                // Only check near edges
                if energy < self.thresholds.edge_strength_min {
                    continue;
                }

                edge_count += 1;

                // Sample errors along edge normal
                let normal = tensor.eigenvector_major;
                let mut sign_changes = 0;
                let mut prev_error: Option<f32> = None;

                for d in -(half_window as i32)..=(half_window as i32) {
                    let sx = (x as i32 + (normal.x * d as f32).round() as i32)
                        .clamp(0, width as i32 - 1) as u32;
                    let sy = (y as i32 + (normal.y * d as f32).round() as i32)
                        .clamp(0, height as i32 - 1) as u32;

                    let t = target.get_pixel(sx, sy).unwrap();
                    let r = rendered.get_pixel(sx, sy).unwrap();
                    let error = (r.r - t.r) + (r.g - t.g) + (r.b - t.b); // Signed error

                    if let Some(pe) = prev_error {
                        if error.signum() != pe.signum() && (error.abs() > 0.02 || pe.abs() > 0.02) {
                            sign_changes += 1;
                        }
                    }
                    prev_error = Some(error);
                }

                // Multiple sign changes = ringing
                if sign_changes >= self.thresholds.ringing_alternations {
                    ringing_count += 1;
                }
            }
        }

        if edge_count > 0 {
            (ringing_count as f32 / edge_count as f32).min(1.0)
        } else {
            0.0
        }
    }

    /// Detect banding (quantization artifacts in gradients)
    ///
    /// Banding occurs when smooth gradients become stepped.
    /// Detection: In smooth gradient regions, look for discrete steps.
    fn detect_banding(
        &self,
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> f32 {
        let width = target.width;
        let height = target.height;
        let mut banding_count = 0;
        let mut gradient_count = 0;

        // Simple horizontal banding detection
        for y in 0..height {
            for x in 1..(width - 1) {
                let t_left = target.get_pixel(x - 1, y).unwrap();
                let t_center = target.get_pixel(x, y).unwrap();
                let t_right = target.get_pixel(x + 1, y).unwrap();

                // Check if target has smooth gradient
                let t_grad_l = luma(&t_center) - luma(&t_left);
                let t_grad_r = luma(&t_right) - luma(&t_center);

                if (t_grad_l - t_grad_r).abs() < 0.01 && t_grad_l.abs() > 0.001 {
                    // This is a smooth gradient in target
                    gradient_count += 1;

                    let r_left = rendered.get_pixel(x - 1, y).unwrap();
                    let r_center = rendered.get_pixel(x, y).unwrap();
                    let r_right = rendered.get_pixel(x + 1, y).unwrap();

                    let r_grad_l = luma(&r_center) - luma(&r_left);
                    let r_grad_r = luma(&r_right) - luma(&r_center);

                    // Banding = rendered has discontinuity where target was smooth
                    if (r_grad_l - r_grad_r).abs() > self.thresholds.banding_step_size {
                        banding_count += 1;
                    }
                }
            }
        }

        if gradient_count > 0 {
            (banding_count as f32 / gradient_count as f32).min(1.0)
        } else {
            0.0
        }
    }

    /// Detect ghosting (faint duplicate features)
    ///
    /// Ghosting occurs when multiple similar Gaussians create faint duplicates.
    /// Detection: Look for repeated low-contrast patterns.
    fn detect_ghosting(&self, rendered: &ImageBuffer<f32>) -> f32 {
        // Simplified ghosting detection: look for low-contrast duplicates
        // Full implementation would require Gaussian parameter analysis
        // For now, return 0 as this requires access to Gaussian data
        0.0
    }

    /// Find regions with highest error
    fn find_high_error_regions(
        &self,
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> Vec<(u32, u32, f32)> {
        let width = target.width;
        let height = target.height;
        let window = self.thresholds.window_size as u32;
        let mut regions = Vec::new();

        // Sample every window_size pixels
        for y in (0..height).step_by(window as usize) {
            for x in (0..width).step_by(window as usize) {
                let mut total_error = 0.0;
                let mut count = 0;

                // Compute average error in window
                for wy in y..(y + window).min(height) {
                    for wx in x..(x + window).min(width) {
                        let t = target.get_pixel(wx, wy).unwrap();
                        let r = rendered.get_pixel(wx, wy).unwrap();
                        let error = (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
                        total_error += error;
                        count += 1;
                    }
                }

                let avg_error = total_error / count as f32;
                if avg_error > 0.01 {  // Only include significant errors
                    regions.push((x + window / 2, y + window / 2, avg_error));
                }
            }
        }

        // Sort by error (descending)
        regions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        regions
    }
}

/// Compute color distance between two pixels
fn color_distance(a: &lgi_math::color::Color4<f32>, b: &lgi_math::color::Color4<f32>) -> f32 {
    let dr = a.r - b.r;
    let dg = a.g - b.g;
    let db = a.b - b.b;
    (dr * dr + dg * dg + db * db).sqrt()
}

/// Compute luminance of a pixel
fn luma(p: &lgi_math::color::Color4<f32>) -> f32 {
    0.2126 * p.r + 0.7152 * p.g + 0.0722 * p.b
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgi_math::color::Color4;

    #[test]
    fn test_detect_on_identical() {
        let image = ImageBuffer::with_background(64, 64, Color4::new(0.5, 0.5, 0.5, 1.0));
        let detector = ArtifactDetector::default();
        let report = detector.analyze(&image, &image).unwrap();

        // Identical images should have low artifact scores
        assert!(report.overall_score < 0.1, "Identical images should have low artifacts");
    }

    #[test]
    fn test_detect_blur() {
        // Target with edge
        let mut target = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let val = if x < 32 { 0.0 } else { 1.0 };
                target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        // Rendered with blurred edge
        let mut rendered = ImageBuffer::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let val = ((x as f32 - 32.0) / 10.0).clamp(0.0, 1.0);
                rendered.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let detector = ArtifactDetector::default();
        let report = detector.analyze(&rendered, &target).unwrap();

        // Blurred edge should be detected
        assert!(report.blur_score > 0.0 || report.bleeding_score > 0.0,
            "Blurred edge should be detected");
    }
}
