//! Rate-distortion optimization framework
//!
//! Implements R-D optimization for Gaussian image encoding:
//! - Rate estimation (entropy-based, profile-aware)
//! - Distortion metrics (MSE, PSNR, contribution analysis)
//! - R-D cost computation (J = D + λR)
//! - Pruning and selection based on R-D curves
//! - Adaptive λ tuning for target bitrate/quality

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};
use crate::quantization::LGIQProfile;
use crate::ImageBuffer;
use crate::ms_ssim_loss::MsssimLoss;

/// Quantization profile with associated rate
#[derive(Debug, Clone, Copy)]
pub struct ProfileRate {
    pub profile: LGIQProfile,
    pub bits_per_gaussian: f32,
}

impl ProfileRate {
    /// Bytes per Gaussian for different profiles (without compression)
    pub fn bytes_per_gaussian(&self) -> f32 {
        match self.profile {
            LGIQProfile::Baseline => 11.0,      // 2+2+2+2+2+1+1+1 bytes
            LGIQProfile::Standard => 13.0,      // 10-bit color adds 2 bytes
            LGIQProfile::HighFidelity => 18.0,  // 14-bit geometry adds ~5 bytes
            LGIQProfile::Extended => 36.0,      // f32 × 9 params (lossless)
        }
    }

    /// Bits per Gaussian
    pub fn bits_per_gaussian(&self) -> f32 {
        self.bytes_per_gaussian() * 8.0
    }

    /// Estimated compressed rate (with typical compression ratio)
    pub fn compressed_bits_per_gaussian(&self) -> f32 {
        let compression_ratio = match self.profile {
            LGIQProfile::Extended => 1.0,  // Lossless/extended doesn't compress well
            _ => 5.0,  // Typical zstd + delta ratio from EXP-6-002
        };
        self.bits_per_gaussian() / compression_ratio
    }
}

/// Distortion metric for R-D optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionMetric {
    /// Mean Squared Error (L2 loss)
    MSE,
    /// Multi-Scale Structural Similarity (perceptual)
    MSSSIM,
}

/// Rate-distortion optimizer
pub struct RateDistortionOptimizer {
    /// Lagrange multiplier (λ) for R-D tradeoff
    /// Higher λ → prefer lower rate (smaller file)
    /// Lower λ → prefer lower distortion (higher quality)
    pub lambda: f32,

    /// Target profile for rate estimation
    pub profile: LGIQProfile,

    /// Whether to use compression in rate estimation
    pub use_compression: bool,

    /// Distortion metric to use
    pub distortion_metric: DistortionMetric,

    /// MS-SSIM loss computer (cached)
    ms_ssim_loss: MsssimLoss,
}

impl Default for RateDistortionOptimizer {
    fn default() -> Self {
        Self {
            lambda: 0.01,
            profile: LGIQProfile::Baseline,
            use_compression: true,
            distortion_metric: DistortionMetric::MSE,
            ms_ssim_loss: MsssimLoss::default(),
        }
    }
}

impl RateDistortionOptimizer {
    /// Create optimizer with specific λ
    pub fn new(lambda: f32) -> Self {
        Self {
            lambda,
            ..Default::default()
        }
    }

    /// Create optimizer with profile and compression settings
    pub fn with_profile(lambda: f32, profile: LGIQProfile, use_compression: bool) -> Self {
        Self {
            lambda,
            profile,
            use_compression,
            ..Default::default()
        }
    }

    /// Create optimizer with MS-SSIM perceptual distortion metric
    pub fn with_msssim(lambda: f32, profile: LGIQProfile, use_compression: bool) -> Self {
        Self {
            lambda,
            profile,
            use_compression,
            distortion_metric: DistortionMetric::MSSSIM,
            ms_ssim_loss: MsssimLoss::default(),
        }
    }

    /// Compute R-D cost: J = D + λR
    pub fn compute_cost(&self, distortion: f32, rate: f32) -> f32 {
        distortion + self.lambda * rate
    }

    /// Estimate rate (bits) for a single Gaussian
    pub fn estimate_rate(&self, _gaussian: &Gaussian2D<f32, Euler<f32>>) -> f32 {
        let profile_rate = ProfileRate { profile: self.profile, bits_per_gaussian: 0.0 };

        if self.use_compression {
            profile_rate.compressed_bits_per_gaussian()
        } else {
            profile_rate.bits_per_gaussian()
        }
    }

    /// Estimate total rate (bits) for all Gaussians
    pub fn estimate_total_rate(&self, num_gaussians: usize) -> f32 {
        // Per-Gaussian rate
        let per_gaussian_rate = if self.use_compression {
            ProfileRate { profile: self.profile, bits_per_gaussian: 0.0 }.compressed_bits_per_gaussian()
        } else {
            ProfileRate { profile: self.profile, bits_per_gaussian: 0.0 }.bits_per_gaussian()
        };

        // Header overhead (64 bytes = 512 bits)
        let header_bits = 512.0;

        // Chunk overhead (8 bytes per chunk = 64 bits)
        let chunk_overhead = 64.0;

        header_bits + chunk_overhead + (num_gaussians as f32 * per_gaussian_rate)
    }

    /// Compute distortion between images using configured metric
    pub fn compute_distortion(&self, original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
        match self.distortion_metric {
            DistortionMetric::MSE => self.compute_mse(original, rendered),
            DistortionMetric::MSSSIM => {
                // MS-SSIM returns similarity in [0, 1], convert to dissimilarity
                let ms_ssim = self.ms_ssim_loss.compute_loss(original, rendered);
                1.0 - ms_ssim  // Convert to dissimilarity (0 = identical, 1 = very different)
            }
        }
    }

    /// Compute MSE distortion between images
    pub fn compute_mse(&self, original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
        if original.width != rendered.width || original.height != rendered.height {
            return f32::INFINITY;
        }

        let mut sum_squared_error = 0.0f32;
        let pixel_count = (original.width * original.height) as f32;

        for i in 0..original.data.len() {
            let orig = original.data[i];
            let rend = rendered.data[i];

            // Compute squared differences for RGB channels
            let diff_r = orig.r - rend.r;
            let diff_g = orig.g - rend.g;
            let diff_b = orig.b - rend.b;

            sum_squared_error += diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
        }

        sum_squared_error / (pixel_count * 3.0)  // Divide by 3 for RGB
    }

    /// Compute PSNR from MSE
    pub fn mse_to_psnr(&self, mse: f32) -> f32 {
        if mse < 1e-10 {
            100.0  // Cap at 100 dB
        } else {
            10.0 * (1.0f32 / mse).log10()
        }
    }

    /// Compute PSNR between images
    pub fn compute_psnr(&self, original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
        let mse = self.compute_mse(original, rendered);
        self.mse_to_psnr(mse)
    }

    /// Estimate per-Gaussian contribution to total error
    /// Uses opacity and coverage area as proxy for importance
    pub fn estimate_contributions(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        error_image: &ImageBuffer<f32>,
    ) -> Vec<f32> {
        let mut contributions = vec![0.0f32; gaussians.len()];

        for (i, gaussian) in gaussians.iter().enumerate() {
            // Simple contribution estimate: opacity × coverage area × local error
            let coverage = gaussian.shape.scale_x * gaussian.shape.scale_y * std::f32::consts::PI;
            let opacity = gaussian.opacity;

            // Sample error at Gaussian center
            let cx = (gaussian.position.x * error_image.width as f32).clamp(0.0, (error_image.width - 1) as f32) as u32;
            let cy = (gaussian.position.y * error_image.height as f32).clamp(0.0, (error_image.height - 1) as f32) as u32;

            let local_error = if let Some(error_pixel) = error_image.get_pixel(cx, cy) {
                // Average RGB error
                (error_pixel.r + error_pixel.g + error_pixel.b) / 3.0
            } else {
                0.0
            };

            contributions[i] = coverage * opacity * local_error;
        }

        contributions
    }

    /// Prune Gaussians based on R-D cost
    /// Returns indices of Gaussians to keep
    pub fn prune_by_rd(
        &self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        contributions: &[f32],
    ) -> Vec<usize> {
        gaussians
            .iter()
            .enumerate()
            .filter_map(|(i, g)| {
                let distortion_reduction = contributions[i];
                let rate = self.estimate_rate(g);
                let rd_cost = self.compute_cost(-distortion_reduction, rate);

                // Keep if removing would increase R-D cost
                // (i.e., Gaussian reduces distortion more than it costs in rate)
                if distortion_reduction > self.lambda * rate {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Select optimal Gaussian count for target PSNR
    /// Returns recommended count
    pub fn select_gaussian_count_for_psnr(
        &self,
        image_width: u32,
        image_height: u32,
        target_psnr: f32,
    ) -> usize {
        // Empirical formula: N ≈ (width × height) / (scale² × quality_factor)
        // Higher target PSNR → more Gaussians needed

        let pixel_count = (image_width * image_height) as f32;

        // Quality factor based on target PSNR
        let quality_factor = if target_psnr < 25.0 {
            2000.0  // Low quality: ~1 Gaussian per 2000 pixels
        } else if target_psnr < 30.0 {
            1000.0  // Medium quality: ~1 Gaussian per 1000 pixels
        } else if target_psnr < 35.0 {
            500.0   // High quality: ~1 Gaussian per 500 pixels
        } else {
            250.0   // Very high quality: ~1 Gaussian per 250 pixels
        };

        (pixel_count / quality_factor).max(100.0) as usize
    }

    /// Compute R-D curve point (rate in bits, distortion as MSE)
    pub fn compute_rd_point(
        &self,
        original: &ImageBuffer<f32>,
        rendered: &ImageBuffer<f32>,
        num_gaussians: usize,
    ) -> (f32, f32) {
        let rate = self.estimate_total_rate(num_gaussians);
        let mse = self.compute_mse(original, rendered);
        (rate, mse)
    }

    /// Compute optimal λ for target bitrate (bits)
    /// Returns recommended λ value
    pub fn compute_lambda_for_bitrate(&self, target_bitrate: f32, current_rate: f32) -> f32 {
        // Simple scaling: if rate is too high, increase λ
        let rate_ratio = current_rate / target_bitrate;
        self.lambda * rate_ratio.powf(0.5)  // Square root for stability
    }

    /// Compute optimal λ for target PSNR
    /// Returns recommended λ value
    pub fn compute_lambda_for_psnr(&self, target_psnr: f32, current_psnr: f32) -> f32 {
        // If PSNR is too low, decrease λ (prioritize quality)
        // If PSNR is too high, increase λ (prioritize rate)
        let psnr_ratio = current_psnr / target_psnr;
        self.lambda / psnr_ratio.powf(0.5)
    }
}
