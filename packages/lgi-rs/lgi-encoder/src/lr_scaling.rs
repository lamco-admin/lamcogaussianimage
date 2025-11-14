//! Learning rate scaling strategies
//!
//! CRITICAL FIX: Scale LR based on Gaussian count
//! This fixes the 1500G failure (12.65 dB)

/// Compute scaled learning rate based on parameter count
pub fn scale_lr_by_count(base_lr: f32, num_gaussians: usize) -> f32 {
    // Standard scaling: lr ∝ 1/sqrt(N)
    // This prevents gradient explosion with more parameters
    base_lr / (num_gaussians as f32).sqrt()
}

/// Compute scaled learning rate with minimum bound
pub fn scale_lr_with_min(base_lr: f32, num_gaussians: usize, min_lr: f32) -> f32 {
    let scaled = scale_lr_by_count(base_lr, num_gaussians);
    scaled.max(min_lr)
}

/// Per-parameter learning rate scaling
pub struct AdaptiveLRScaling {
    base_lrs: ParameterLRs,
    num_gaussians: usize,
}

pub struct ParameterLRs {
    pub position: f32,
    pub scale: f32,
    pub rotation: f32,
    pub color: f32,
    pub opacity: f32,
}

impl AdaptiveLRScaling {
    pub fn new(base_lrs: ParameterLRs, num_gaussians: usize) -> Self {
        Self { base_lrs, num_gaussians }
    }

    /// Get scaled learning rates for current Gaussian count
    pub fn get_scaled_lrs(&self) -> ParameterLRs {
        let scale_factor = 1.0 / (self.num_gaussians as f32).sqrt();

        ParameterLRs {
            position: self.base_lrs.position * scale_factor,
            scale: self.base_lrs.scale * scale_factor,
            rotation: self.base_lrs.rotation * scale_factor,
            color: self.base_lrs.color * scale_factor,
            opacity: self.base_lrs.opacity * scale_factor,
        }
    }

    /// Adaptive per-Gaussian learning rates based on local density
    pub fn get_adaptive_lrs(&self, gaussian_densities: &[f32]) -> Vec<ParameterLRs> {
        gaussian_densities.iter().map(|&density| {
            // Higher density → lower LR (more crowded)
            let density_factor = 1.0 / (1.0 + density);

            let base_scaled = self.get_scaled_lrs();

            ParameterLRs {
                position: base_scaled.position * density_factor,
                scale: base_scaled.scale * density_factor,
                rotation: base_scaled.rotation * density_factor,
                color: base_scaled.color * density_factor,
                opacity: base_scaled.opacity * density_factor,
            }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr_scaling() {
        let base_lr = 0.01;

        // 500 Gaussians
        let lr_500 = scale_lr_by_count(base_lr, 500);
        assert!((lr_500 - 0.000447).abs() < 0.0001);  // 0.01 / sqrt(500) ≈ 0.000447

        // 1500 Gaussians
        let lr_1500 = scale_lr_by_count(base_lr, 1500);
        assert!((lr_1500 - 0.000258).abs() < 0.0001);  // 0.01 / sqrt(1500) ≈ 0.000258

        // Ratio should be sqrt(500/1500) = 0.577
        assert!((lr_1500 / lr_500 - 0.577).abs() < 0.01);
    }
}
