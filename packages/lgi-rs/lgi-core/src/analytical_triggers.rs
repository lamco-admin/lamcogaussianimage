//! Analytical Trigger Framework
//!
//! Commercial-grade adaptive processing framework
//! All 7 metrics from Hair Toolkit + research
//! Automatic detection of when to apply advanced techniques

use crate::{ImageBuffer, StructureTensorField};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, color::Color4};

/// Complete analytical trigger system
///
/// Computes all 7 metrics and provides refinement recommendations
pub struct AnalyticalTriggers {
    /// Anisotropy Gradient Divergence
    pub agd: f32,

    /// Laplacian Consistency Check
    pub lcc: f32,

    /// Jacobian Condition Index
    pub jci: f32,

    /// Entropy-Residual Ratio
    pub err: f32,

    /// Rate-Curvature Heuristic
    pub rch: f32,

    /// Structure-Perceptual Error Correlation
    pub spec: f32,

    /// Spectral Energy Drop
    pub sed: f32,

    /// Per-region trigger maps
    pub refinement_mask: Vec<f32>,  // Where to refine
    pub texture_mask: Vec<f32>,     // Where to add textures
    pub residual_mask: Vec<f32>,    // Where to add blue-noise
}

impl AnalyticalTriggers {
    /// Analyze image and Gaussian representation
    ///
    /// Computes all metrics and generates refinement recommendations
    pub fn analyze(
        original: &ImageBuffer<f32>,
        rendered: &ImageBuffer<f32>,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        structure_tensor: &StructureTensorField,
    ) -> Self {
        let width = original.width;
        let height = original.height;

        // Initialize triggers
        let mut triggers = Self {
            agd: 0.0,
            lcc: 0.0,
            jci: 0.0,
            err: 0.0,
            rch: 0.0,
            spec: 0.0,
            sed: 0.0,
            refinement_mask: vec![0.0; (width * height) as usize],
            texture_mask: vec![0.0; (width * height) as usize],
            residual_mask: vec![0.0; (width * height) as usize],
        };

        // Compute each metric
        triggers.compute_sed(original, rendered);
        triggers.compute_err(original, rendered);
        triggers.compute_lcc(original, rendered);
        triggers.compute_agd(structure_tensor);
        triggers.compute_jci(original);
        triggers.compute_rch(gaussians, original);
        triggers.compute_spec(original, rendered, structure_tensor);

        // Generate masks from metrics
        triggers.generate_masks(original, rendered);

        triggers
    }

    /// SED: Spectral Energy Drop
    ///
    /// Detects high-frequency detail loss
    /// Formula: SED = 1 - E_high(rendered) / E_high(original)
    fn compute_sed(&mut self, original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) {
        let original_hf = compute_high_frequency_energy(original);
        let rendered_hf = compute_high_frequency_energy(rendered);

        self.sed = if original_hf > 1e-6 {
            1.0 - (rendered_hf / original_hf)
        } else {
            0.0
        };
    }

    /// ERR: Entropy-Residual Ratio
    ///
    /// Detects regions where model capacity exceeded
    /// Formula: ERR = H / (E_L2 + ε)
    fn compute_err(&mut self, original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) {
        let entropy = compute_local_entropy(original);
        let l2_error = compute_l2_error(original, rendered);

        self.err = entropy / (l2_error + 1e-6);
    }

    /// LCC: Laplacian Consistency Check
    ///
    /// Detects edge sharpness loss
    /// Formula: LCC = |∇²I - ∇²Î|
    fn compute_lcc(&mut self, original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) {
        let lap_orig = compute_laplacian(original);
        let lap_rend = compute_laplacian(rendered);

        let mut diff_sum = 0.0;
        for i in 0..lap_orig.len() {
            diff_sum += (lap_orig[i] - lap_rend[i]).abs();
        }

        self.lcc = diff_sum / lap_orig.len() as f32;
    }

    /// AGD: Anisotropy Gradient Divergence
    ///
    /// Detects abrupt anisotropy changes (region boundaries)
    /// Formula: AGD = ∇·(Σ11 - Σ22)
    fn compute_agd(&mut self, structure_tensor: &StructureTensorField) {
        // Simplified: compute variance of coherence field
        let mut coherence_values = Vec::new();

        for y in 0..structure_tensor.height {
            for x in 0..structure_tensor.width {
                coherence_values.push(structure_tensor.get(x, y).coherence);
            }
        }

        let mean = coherence_values.iter().sum::<f32>() / coherence_values.len() as f32;
        let variance = coherence_values.iter()
            .map(|&c| (c - mean).powi(2))
            .sum::<f32>() / coherence_values.len() as f32;

        self.agd = variance.sqrt();  // Standard deviation as proxy
    }

    /// JCI: Jacobian Condition Index
    ///
    /// Detects chromatic instability
    /// Formula: cond(∇RGB) = σ_max(J) / σ_min(J)
    fn compute_jci(&mut self, image: &ImageBuffer<f32>) {
        // Simplified: compute color gradient magnitude variance
        let mut gradient_mags = Vec::new();

        for y in 1..image.height-1 {
            for x in 1..image.width-1 {
                if let (Some(c), Some(cx), Some(cy)) = (
                    image.get_pixel(x, y),
                    image.get_pixel(x+1, y),
                    image.get_pixel(x, y+1),
                ) {
                    let grad_r = ((cx.r - c.r).powi(2) + (cy.r - c.r).powi(2)).sqrt();
                    let grad_g = ((cx.g - c.g).powi(2) + (cy.g - c.g).powi(2)).sqrt();
                    let grad_b = ((cx.b - c.b).powi(2) + (cy.b - c.b).powi(2)).sqrt();

                    let grad_mag = (grad_r + grad_g + grad_b) / 3.0;
                    gradient_mags.push(grad_mag);
                }
            }
        }

        if !gradient_mags.is_empty() {
            gradient_mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p95 = gradient_mags[(gradient_mags.len() as f32 * 0.95) as usize];
            let p05 = gradient_mags[(gradient_mags.len() as f32 * 0.05) as usize];

            self.jci = if p05 > 1e-6 { p95 / p05 } else { 1.0 };
        }
    }

    /// RCH: Rate-Curvature Heuristic
    ///
    /// Detects over-allocation in flat regions
    /// Formula: RCH = R / (1 + |κ|)
    fn compute_rch(&mut self, gaussians: &[Gaussian2D<f32, Euler<f32>>], image: &ImageBuffer<f32>) {
        // Rate = number of primitives
        let rate = gaussians.len() as f32;

        // Curvature proxy: variance of Laplacian
        let laplacian = compute_laplacian(image);
        let curvature = laplacian.iter().map(|&l| l.abs()).sum::<f32>() / laplacian.len() as f32;

        self.rch = rate / (1.0 + curvature);
    }

    /// SPEC: Structure-Perceptual Error Correlation
    ///
    /// Detects perceptual error at edges
    /// Formula: ρ(MS-SSIM, |∇I|)
    fn compute_spec(
        &mut self,
        original: &ImageBuffer<f32>,
        rendered: &ImageBuffer<f32>,
        structure_tensor: &StructureTensorField,
    ) {
        // Simplified: correlation between gradient and error
        let mut gradient_values = Vec::new();
        let mut error_values = Vec::new();

        for y in 0..original.height {
            for x in 0..original.width {
                let tensor = structure_tensor.get(x, y);
                let gradient_mag = tensor.eigenvalue_major.sqrt();

                if let (Some(orig), Some(rend)) = (original.get_pixel(x, y), rendered.get_pixel(x, y)) {
                    let error = ((orig.r - rend.r).powi(2) +
                                (orig.g - rend.g).powi(2) +
                                (orig.b - rend.b).powi(2)).sqrt();

                    gradient_values.push(gradient_mag);
                    error_values.push(error);
                }
            }
        }

        // Compute correlation
        self.spec = compute_correlation(&gradient_values, &error_values);
    }

    /// Generate refinement masks from metrics
    fn generate_masks(&mut self, original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) {
        let width = original.width;
        let height = original.height;

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;

                // Texture mask: high SED regions
                self.texture_mask[idx] = if self.sed > 0.3 { 1.0 } else { 0.0 };

                // Residual mask: high ERR regions
                self.residual_mask[idx] = if self.err > 2.0 { 1.0 } else { 0.0 };

                // Refinement mask: high LCC or JCI
                self.refinement_mask[idx] = if self.lcc > 0.1 || self.jci > 10.0 { 1.0 } else { 0.0 };
            }
        }
    }

    /// Get refinement recommendations
    pub fn should_add_textures(&self) -> bool {
        self.sed > 0.3  // Spectral energy drop significant
    }

    pub fn should_add_residuals(&self) -> bool {
        self.err > 2.0  // Entropy-residual ratio high
    }

    pub fn should_refine_gaussians(&self) -> bool {
        self.lcc > 0.1 || self.agd > 0.2  // Laplacian mismatch or anisotropy changes
    }

    pub fn should_split_gaussians(&self) -> bool {
        self.rch > 100.0  // Too many primitives in flat regions
    }
}

// Helper functions

fn compute_high_frequency_energy(image: &ImageBuffer<f32>) -> f32 {
    // High-pass filter: Laplacian energy
    let laplacian = compute_laplacian(image);
    laplacian.iter().map(|&l| l * l).sum::<f32>()
}

fn compute_laplacian(image: &ImageBuffer<f32>) -> Vec<f32> {
    let mut result = vec![0.0; (image.width * image.height) as usize];

    for y in 1..image.height-1 {
        for x in 1..image.width-1 {
            if let (Some(c), Some(cx1), Some(cx2), Some(cy1), Some(cy2)) = (
                image.get_pixel(x, y),
                image.get_pixel(x+1, y),
                image.get_pixel(x-1, y),
                image.get_pixel(x, y+1),
                image.get_pixel(x, y-1),
            ) {
                // Laplacian: ∇²I = Ixx + Iyy
                let gray_c = (c.r + c.g + c.b) / 3.0;
                let gray_x1 = (cx1.r + cx1.g + cx1.b) / 3.0;
                let gray_x2 = (cx2.r + cx2.g + cx2.b) / 3.0;
                let gray_y1 = (cy1.r + cy1.g + cy1.b) / 3.0;
                let gray_y2 = (cy2.r + cy2.g + cy2.b) / 3.0;

                let laplacian = (gray_x1 + gray_x2 + gray_y1 + gray_y2) - 4.0 * gray_c;
                result[(y * image.width + x) as usize] = laplacian;
            }
        }
    }

    result
}

fn compute_local_entropy(image: &ImageBuffer<f32>) -> f32 {
    // Shannon entropy of intensity histogram
    let mut histogram = vec![0u32; 256];

    for pixel in &image.data {
        let intensity = ((pixel.r + pixel.g + pixel.b) / 3.0 * 255.0) as usize;
        let bin = intensity.min(255);
        histogram[bin] += 1;
    }

    let total = image.data.len() as f32;
    let mut entropy = 0.0;

    for &count in &histogram {
        if count > 0 {
            let p = count as f32 / total;
            entropy -= p * p.log2();
        }
    }

    entropy
}

fn compute_l2_error(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut error = 0.0;

    for (o, r) in original.data.iter().zip(rendered.data.iter()) {
        error += (o.r - r.r).powi(2) + (o.g - r.g).powi(2) + (o.b - r.b).powi(2);
    }

    error / (original.data.len() * 3) as f32
}

fn compute_correlation(values1: &[f32], values2: &[f32]) -> f32 {
    if values1.len() != values2.len() || values1.is_empty() {
        return 0.0;
    }

    let mean1 = values1.iter().sum::<f32>() / values1.len() as f32;
    let mean2 = values2.iter().sum::<f32>() / values2.len() as f32;

    let mut covariance = 0.0;
    let mut var1 = 0.0;
    let mut var2 = 0.0;

    for i in 0..values1.len() {
        let diff1 = values1[i] - mean1;
        let diff2 = values2[i] - mean2;

        covariance += diff1 * diff2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
    }

    if var1 > 1e-6 && var2 > 1e-6 {
        covariance / (var1.sqrt() * var2.sqrt())
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_computation() {
        let mut img = ImageBuffer::new(64, 64);

        // Create edge
        for y in 0..64 {
            for x in 0..64 {
                let val = if x < 32 { 0.0 } else { 1.0 };
                img.set_pixel(x, y, Color4::new(val, val, val, 1.0));
            }
        }

        let laplacian = compute_laplacian(&img);

        // Laplacian should be strong at edge (x=32)
        let edge_idx = (32 * 64 + 32) as usize;
        assert!(laplacian[edge_idx].abs() > 0.5);
    }

    #[test]
    fn test_entropy_computation() {
        let mut img = ImageBuffer::new(64, 64);

        // Uniform image = low entropy
        for y in 0..64 {
            for x in 0..64 {
                img.set_pixel(x, y, Color4::new(0.5, 0.5, 0.5, 1.0));
            }
        }

        let entropy = compute_local_entropy(&img);
        assert!(entropy < 1.0, "Uniform should have low entropy");
    }

    #[test]
    fn test_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];  // Perfect correlation

        let corr = compute_correlation(&a, &b);
        assert!((corr - 1.0).abs() < 0.01, "Perfect correlation should be ~1.0");
    }
}
