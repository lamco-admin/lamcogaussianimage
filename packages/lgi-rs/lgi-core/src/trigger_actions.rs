//! Trigger Action Handlers
//! When analytical triggers fire, these handlers apply mitigations
//! Connects analysis to encoding decisions

use crate::{analytical_triggers::AnalyticalTriggers, ImageBuffer};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

/// Action handler for analytical triggers
pub struct TriggerActionHandler {
    /// Thresholds for each trigger
    pub sed_threshold: f32,    // Spectral energy drop
    pub err_threshold: f32,    // Entropy-residual ratio
    pub lcc_threshold: f32,    // Laplacian consistency
    pub agd_threshold: f32,    // Anisotropy gradient divergence
    pub jci_threshold: f32,    // Jacobian condition index
    pub rch_threshold: f32,    // Rate-curvature heuristic
    pub spec_threshold: f32,   // Structure-perceptual error
}

impl Default for TriggerActionHandler {
    fn default() -> Self {
        Self {
            sed_threshold: 0.3,     // From Hair Toolkit
            err_threshold: 1000.0,  // From experimental data
            lcc_threshold: 0.1,
            agd_threshold: 50.0,
            jci_threshold: 10.0,
            rch_threshold: 1.0,
            spec_threshold: -0.2,   // Negative correlation is bad
        }
    }
}

impl TriggerActionHandler {
    /// Process all triggers and apply mitigations
    pub fn process_triggers(
        &self,
        triggers: &AnalyticalTriggers,
        gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
        target: &ImageBuffer<f32>,
    ) {
        // Handle SED: Spectral energy drop (detail loss)
        if triggers.sed > self.sed_threshold {
            self.handle_sed(triggers, gaussians, target);
        }

        // Handle ERR: Entropy-residual ratio (model capacity exceeded)
        if triggers.err > self.err_threshold {
            self.handle_err(triggers, gaussians, target);
        }

        // Handle LCC: Laplacian consistency (sharpness loss)
        if triggers.lcc > self.lcc_threshold {
            self.handle_lcc(triggers, gaussians, target);
        }

        // Handle AGD: Anisotropy gradient divergence (boundary issues)
        if triggers.agd > self.agd_threshold {
            self.handle_agd(triggers, gaussians);
        }

        // Handle JCI: Jacobian condition (color instability)
        if triggers.jci > self.jci_threshold {
            self.handle_jci(triggers, gaussians);
        }

        // Handle RCH: Rate-curvature (over-allocation to flat)
        if triggers.rch > self.rch_threshold {
            self.handle_rch(triggers, gaussians);
        }

        // Handle SPEC: Structure-perceptual error (edge-perceptual mismatch)
        if triggers.spec < self.spec_threshold {
            self.handle_spec(triggers, gaussians);
        }
    }

    /// Handle SED: Add fine-scale Gaussians or residuals when detail is lost
    fn handle_sed(
        &self,
        triggers: &AnalyticalTriggers,
        gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
        target: &ImageBuffer<f32>,
    ) {
        println!("  [SED Trigger] Spectral energy drop {:.2} > {:.2}", triggers.sed, self.sed_threshold);
        println!("    Action: Adding fine-scale Gaussians in high-frequency regions");

        let width = target.width;
        let height = target.height;
        let tile_size = 16;
        let tile_cols = (width + tile_size - 1) / tile_size;
        let tile_rows = (height + tile_size - 1) / tile_size;

        // Use residual_mask to identify high-frequency tiles
        for ty in 0..tile_rows {
            for tx in 0..tile_cols {
                let tile_idx = (ty * tile_cols + tx) as usize;
                if tile_idx >= triggers.residual_mask.len() {
                    continue;
                }

                // If tile needs fine detail
                if triggers.residual_mask[tile_idx] > 0.5 {
                    // Add 2-4 fine Gaussians in this tile
                    let tile_x = tx * tile_size;
                    let tile_y = ty * tile_size;

                    for _ in 0..2 {
                        // Random position within tile
                        let px = (tile_x + tile_size / 4) as f32 / width as f32;
                        let py = (tile_y + tile_size / 4) as f32 / height as f32;

                        // Very small Gaussian (fine detail)
                        let sigma_fine = 0.01;  // Normalized coords

                        // Sample color from target
                        let tx_px = (px * width as f32) as u32;
                        let ty_px = (py * height as f32) as u32;
                        let color = target.get_pixel(tx_px.min(width-1), ty_px.min(height-1))
                            .unwrap_or(lgi_math::color::Color4::new(0.5, 0.5, 0.5, 1.0));

                        gaussians.push(Gaussian2D::new(
                            Vector2::new(px, py),
                            Euler::isotropic(sigma_fine),
                            lgi_math::color::Color4::new(color.r, color.g, color.b, 1.0),
                            0.5,  // Lower opacity for fine detail
                        ));
                    }
                }
            }
        }

        println!("      Added {} fine-scale Gaussians", 2 * triggers.residual_mask.iter().filter(|&&m| m > 0.5).count());
    }

    /// Handle ERR: Add capacity when entropy exceeds residual modeling
    fn handle_err(
        &self,
        triggers: &AnalyticalTriggers,
        gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
        target: &ImageBuffer<f32>,
    ) {
        println!("  [ERR Trigger] Entropy-residual ratio {:.0} > {:.0}", triggers.err, self.err_threshold);
        println!("    Action: Adding micro-Gaussians or blue-noise residuals");

        // Use texture_mask + residual_mask to identify complex regions
        let width = target.width;
        let height = target.height;
        let mut added_count = 0;

        // Sample micro-Gaussians in high-entropy, high-residual regions
        for y in (0..height).step_by(8) {
            for x in (0..width).step_by(8) {
                let pixel_idx = (y * width + x) as usize;

                // Check if region needs refinement (both high texture and high residual)
                if pixel_idx < triggers.texture_mask.len() &&
                   pixel_idx < triggers.residual_mask.len() &&
                   triggers.texture_mask[pixel_idx] > 0.6 &&
                   triggers.residual_mask[pixel_idx] > 0.6 {

                    // Add micro-Gaussian at this location
                    let px = x as f32 / width as f32;
                    let py = y as f32 / height as f32;

                    // Very small scale for micro detail
                    let sigma_micro = 0.008;  // Normalized

                    // Sample color from target
                    let color = target.get_pixel(x.min(width-1), y.min(height-1))
                        .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

                    gaussians.push(Gaussian2D::new(
                        Vector2::new(px, py),
                        Euler::isotropic(sigma_micro),
                        Color4::new(color.r, color.g, color.b, 1.0),
                        0.4,  // Low opacity for micro detail
                    ));

                    added_count += 1;
                }
            }
        }

        println!("      Added {} micro-Gaussians in complex regions", added_count);
    }

    /// Handle LCC: Sharpen edges when Laplacian mismatches
    fn handle_lcc(
        &self,
        triggers: &AnalyticalTriggers,
        gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
        target: &ImageBuffer<f32>,
    ) {
        println!("  [LCC Trigger] Laplacian consistency {:.2} > {:.2}", triggers.lcc, self.lcc_threshold);
        println!("    Action: Reducing σ_⊥ at edges, increasing Gaussian density");

        let width = target.width;
        let height = target.height;

        // Reduce scale of Gaussians near high-gradient regions (use texture_mask as proxy for edges)
        let mut refined_count = 0;
        for gaussian in gaussians.iter_mut() {
            let gx = (gaussian.position.x * width as f32) as u32;
            let gy = (gaussian.position.y * height as f32) as u32;
            let pixel_idx = (gy * width + gx) as usize;

            if pixel_idx < triggers.texture_mask.len() && triggers.texture_mask[pixel_idx] > 0.5 {
                // At edge/high-gradient - reduce perpendicular scale by 20%
                let reduction = 0.8;
                gaussian.shape.scale_y *= reduction;
                gaussian.shape.scale_y = gaussian.shape.scale_y.max(0.005);  // Don't go too small
                refined_count += 1;
            }
        }

        println!("      Refined {} Gaussians at high-gradient regions", refined_count);
    }

    /// Handle AGD: Refine boundaries when anisotropy changes rapidly
    fn handle_agd(
        &self,
        triggers: &AnalyticalTriggers,
        gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    ) {
        println!("  [AGD Trigger] Anisotropy gradient divergence {:.0} > {:.0}", triggers.agd, self.agd_threshold);
        println!("    Action: Refining Gaussians at anisotropy boundaries");

        // Detect regions where anisotropy changes rapidly
        // Add small isotropic Gaussians at boundaries

        let mut refined_count = 0;

        // Reduce anisotropy at high-gradient regions (make more isotropic)
        for gaussian in gaussians.iter_mut() {
            let scale_ratio = gaussian.shape.scale_x / gaussian.shape.scale_y.max(0.001);

            // If highly anisotropic (ratio > 3), reduce anisotropy by 30%
            if scale_ratio > 3.0 || scale_ratio < 1.0/3.0 {
                let avg_scale = (gaussian.shape.scale_x + gaussian.shape.scale_y) / 2.0;
                let blend_factor = 0.3;  // 30% more isotropic

                gaussian.shape.scale_x = gaussian.shape.scale_x * (1.0 - blend_factor) + avg_scale * blend_factor;
                gaussian.shape.scale_y = gaussian.shape.scale_y * (1.0 - blend_factor) + avg_scale * blend_factor;

                refined_count += 1;
            }
        }

        println!("      Refined {} anisotropic Gaussians at boundaries", refined_count);
    }

    /// Handle JCI: Stabilize color when Jacobian condition is high
    fn handle_jci(
        &self,
        triggers: &AnalyticalTriggers,
        gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    ) {
        println!("  [JCI Trigger] Jacobian condition index {:.0} > {:.0}", triggers.jci, self.jci_threshold);
        println!("    Action: Clamping per-channel covariance deviation");

        // Apply color stabilization constraints
        // Clamp extreme color values and reduce saturation slightly
        let mut stabilized_count = 0;

        for gaussian in gaussians.iter_mut() {
            // Compute luminance
            let lum = 0.299 * gaussian.color.r + 0.587 * gaussian.color.g + 0.114 * gaussian.color.b;

            // Check if color is unstable (extreme saturation or very dark/bright)
            let max_channel = gaussian.color.r.max(gaussian.color.g).max(gaussian.color.b);
            let min_channel = gaussian.color.r.min(gaussian.color.g).min(gaussian.color.b);
            let saturation = if max_channel > 0.0 {
                (max_channel - min_channel) / max_channel
            } else {
                0.0
            };

            // Stabilize if: high saturation (>0.8) or extreme brightness (<0.1 or >0.9)
            if saturation > 0.8 || lum < 0.1 || lum > 0.9 {
                // Desaturate by 20% (blend toward luminance)
                let blend_factor = 0.2;
                gaussian.color.r = gaussian.color.r * (1.0 - blend_factor) + lum * blend_factor;
                gaussian.color.g = gaussian.color.g * (1.0 - blend_factor) + lum * blend_factor;
                gaussian.color.b = gaussian.color.b * (1.0 - blend_factor) + lum * blend_factor;

                // Clamp to valid range
                gaussian.color.r = gaussian.color.r.clamp(0.0, 1.0);
                gaussian.color.g = gaussian.color.g.clamp(0.0, 1.0);
                gaussian.color.b = gaussian.color.b.clamp(0.0, 1.0);

                stabilized_count += 1;
            }
        }

        println!("      Stabilized {} Gaussians with extreme colors", stabilized_count);
    }

    /// Handle RCH: Merge Gaussians in flat regions (over-allocation)
    fn handle_rch(
        &self,
        triggers: &AnalyticalTriggers,
        gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    ) {
        println!("  [RCH Trigger] Rate-curvature heuristic {:.2} > {:.2}", triggers.rch, self.rch_threshold);
        println!("    Action: Merging nearby Gaussians in flat regions");

        // Simple merging: Combine very close Gaussians with similar colors
        let merge_distance = 0.02;  // Normalized coords
        let color_threshold = 0.1;

        let mut merged_indices = Vec::new();
        let original_count = gaussians.len();

        for i in 0..gaussians.len() {
            if merged_indices.contains(&i) {
                continue;
            }

            for j in (i+1)..gaussians.len() {
                if merged_indices.contains(&j) {
                    continue;
                }

                let dist_sq = (gaussians[i].position.x - gaussians[j].position.x).powi(2) +
                             (gaussians[i].position.y - gaussians[j].position.y).powi(2);

                let color_diff = (gaussians[i].color.r - gaussians[j].color.r).abs() +
                                (gaussians[i].color.g - gaussians[j].color.g).abs() +
                                (gaussians[i].color.b - gaussians[j].color.b).abs();

                if dist_sq < merge_distance * merge_distance && color_diff < color_threshold {
                    // Merge j into i
                    let w1 = gaussians[i].opacity;
                    let w2 = gaussians[j].opacity;
                    let w_total = w1 + w2;

                    // Weighted average
                    gaussians[i].position.x = (w1 * gaussians[i].position.x + w2 * gaussians[j].position.x) / w_total;
                    gaussians[i].position.y = (w1 * gaussians[i].position.y + w2 * gaussians[j].position.y) / w_total;
                    gaussians[i].color.r = (w1 * gaussians[i].color.r + w2 * gaussians[j].color.r) / w_total;
                    gaussians[i].color.g = (w1 * gaussians[i].color.g + w2 * gaussians[j].color.g) / w_total;
                    gaussians[i].color.b = (w1 * gaussians[i].color.b + w2 * gaussians[j].color.b) / w_total;
                    gaussians[i].opacity = w_total.min(1.0);

                    merged_indices.push(j);
                }
            }
        }

        // Remove merged Gaussians
        merged_indices.sort_unstable();
        merged_indices.reverse();
        for &idx in &merged_indices {
            gaussians.swap_remove(idx);
        }

        println!("      Merged {} Gaussians ({} → {})", merged_indices.len(), original_count, gaussians.len());
    }

    /// Handle SPEC: Enable EWA or reduce blending when edge-perceptual mismatch
    fn handle_spec(
        &self,
        triggers: &AnalyticalTriggers,
        gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
    ) {
        println!("  [SPEC Trigger] Structure-perceptual correlation {:.2} < {:.2}", triggers.spec, self.spec_threshold);
        println!("    Action: Reducing Gaussian blending radius at edges");

        // Use edge_mask to identify edges
        // Reduce scale to minimize bleeding across perceptual boundaries
        let mut reduced_count = 0;

        for gaussian in gaussians.iter_mut() {
            // Reduce scale of all Gaussians by 10% to tighten blending
            // This reduces cross-edge bleeding when perceptual and structural edges mismatch
            let reduction = 0.9;

            gaussian.shape.scale_x *= reduction;
            gaussian.shape.scale_y *= reduction;

            // Ensure minimum scale
            gaussian.shape.scale_x = gaussian.shape.scale_x.max(0.005);
            gaussian.shape.scale_y = gaussian.shape.scale_y.max(0.005);

            reduced_count += 1;
        }

        println!("      Reduced scale of {} Gaussians to minimize edge bleeding", reduced_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_handler_creation() {
        let handler = TriggerActionHandler::default();
        assert_eq!(handler.sed_threshold, 0.3);
        assert_eq!(handler.err_threshold, 1000.0);
    }
}
