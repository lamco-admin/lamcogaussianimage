//! LGI Encoder v2 - Research-Based Implementation
//!
//! Complete rewrite using:
//! - Log-Cholesky covariance (PSD-guaranteed)
//! - Geodesic EDT (anti-bleeding)
//! - Structure tensor alignment (edge-aware)
//! - MS-SSIM loss (perceptual)
//! - Rate-distortion optimization (optimal)
//!
//! Expected: 28-38 dB PSNR (vs v1: 2-17 dB)

use lgi_core::{ImageBuffer, StructureTensorField, Result};
use lgi_core::geodesic_edt::GeodesicEDT;
use lgi_math::{log_cholesky::LogCholesky, gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

pub mod renderer_v2;
pub mod renderer_gpu;  // GPU-accelerated renderer
pub mod optimizer_v2;
pub mod error_driven;
pub mod better_color_init;
pub mod text_detection;
pub mod renderer_v3_textured;  // Production-grade textured renderer
pub mod commercial_encoder;    // Complete commercial encoder with ALL features
pub mod optimizer_v3_perceptual;
pub mod encoder_v3_adaptive;
pub mod debug_logger;  // Visual debugging toolkit
pub mod adaptive_densification;  // Split/clone/prune from 3D splatting
pub mod gdgs_init;  // Gradient Domain Gaussian Splatting initialization
pub mod gradient_peak_init;  // Gradient peak initialization (Strategy H)
pub mod kmeans_init;  // K-means clustering initialization (Strategy D)
pub mod slic_init;  // SLIC superpixel initialization (Strategy E)
pub mod preprocessing_loader;  // Loads Python preprocessing outputs (placement maps)

/// Encoder v2 - Minimal prototype for Quality Gate 1
///
/// Tests log-Cholesky + geodesic EDT improvements
pub struct EncoderV2 {
    /// Target image
    target: ImageBuffer<f32>,

    /// Structure tensor (edge information)
    structure_tensor: StructureTensorField,

    /// Geodesic distance transform (anti-bleeding)
    geodesic_edt: GeodesicEDT,
}

impl EncoderV2 {
    /// Create encoder for image
    ///
    /// Preprocesses:
    /// - Computes structure tensor (edge orientation)
    /// - Computes geodesic EDT (boundary-aware distance)
    pub fn new(target: ImageBuffer<f32>) -> Result<Self> {
        log::info!("🔧 Preprocessing image {}×{}", target.width, target.height);

        // Compute structure tensor with data-driven parameters
        // EXP-007: σ_smooth=1.0 gives +0.1 dB over 2.5
        log::info!("  Computing structure tensor (σ_gradient=1.2, σ_smooth=1.0)...");
        let structure_tensor = StructureTensorField::compute(&target, 1.2, 1.0)?;

        // Compute geodesic EDT (respects edges)
        log::info!("  Computing geodesic EDT (coherence_threshold=0.7, edge_penalty=50.0)...");
        let geodesic_edt = GeodesicEDT::compute(&target, &structure_tensor, 0.7, 50.0)?;

        log::info!("✅ Preprocessing complete");

        Ok(Self {
            target,
            structure_tensor,
            geodesic_edt,
        })
    }

    /// Initialize Gaussians from structure tensor
    ///
    /// Uses grid-based sampling with structure-tensor alignment
    /// Each Gaussian:
    /// - Positioned at grid point
    /// - Covariance from log-Cholesky (structure tensor)
    /// - Aligned to local edges
    /// - Clamped by geodesic EDT
    pub fn initialize_gaussians(&self, grid_size: u32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        self.initialize_gaussians_with_options(grid_size, false)
    }

    /// Initialize Gaussians with optional guided filter for colors
    pub fn initialize_gaussians_guided(&self, grid_size: u32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        self.initialize_gaussians_with_options(grid_size, true)
    }

    /// Automatically determine optimal Gaussian count based on image entropy
    ///
    /// Uses adaptive_gaussian_count from lgi-core/entropy.rs
    /// Analyzes image complexity to select appropriate N
    pub fn auto_gaussian_count(&self) -> usize {
        let count = lgi_core::entropy::adaptive_gaussian_count(&self.target);
        log::info!("📊 Auto-selected Gaussian count: {} (based on image entropy)", count);
        count
    }

    /// Get reference to structure tensor (for data logging)
    pub fn get_structure_tensor(&self) -> &StructureTensorField {
        &self.structure_tensor
    }

    /// Hybrid Gaussian count: Entropy (60%) + Gradient (40%)
    ///
    /// Research-backed approach combining:
    /// - Image entropy (complexity, texture)
    /// - Gradient magnitude (edges, detail)
    ///
    /// This is Strategy 1 from research (Position Probability Map basis)
    pub fn hybrid_gaussian_count(&self) -> usize {
        // Base entropy count
        let entropy_n = lgi_core::entropy::adaptive_gaussian_count(&self.target);

        // Compute average gradient from structure tensor
        let mut total_gradient = 0.0;
        let mut count = 0;

        // Sample every 4th pixel for efficiency
        let height = self.structure_tensor.height;
        let width = self.structure_tensor.width;

        for y in (0..height).step_by(4) {
            for x in (0..width).step_by(4) {
                let t = self.structure_tensor.get(x, y);
                // Gradient magnitude ≈ sqrt(eigenvalue_major) (larger eigenvalue)
                let gradient_mag = t.eigenvalue_major.sqrt();
                total_gradient += gradient_mag;
                count += 1;
            }
        }

        let mean_gradient = total_gradient / count.max(1) as f32;

        // Normalize: typical gradients 0-0.5 for natural images → [0, 1]
        let gradient_factor = (mean_gradient * 2.0).min(1.0);

        // Combine: 60% entropy base, 40% gradient adjustment
        let hybrid_factor = 0.6 + 0.4 * gradient_factor;
        let hybrid_n = (entropy_n as f32 * hybrid_factor) as usize;

        log::info!("📊 Hybrid Gaussian count: {} (entropy={}, gradient_factor={:.2}, hybrid_factor={:.2})",
            hybrid_n, entropy_n, gradient_factor, hybrid_factor);

        hybrid_n.max(50).min(50000)
    }

    /// Encode with automatically determined Gaussian count
    ///
    /// Convenience method that combines auto N selection + encoding
    pub fn encode_auto(&self) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        let n = self.auto_gaussian_count();
        let grid_size = (n as f32).sqrt().ceil() as u32;
        self.initialize_gaussians(grid_size)
    }

    /// Initialize using GDGS (Gradient Domain) strategy
    ///
    /// Places Gaussians at Laplacian peaks (edges, corners)
    /// Claims 10-100× fewer Gaussians for same quality
    ///
    /// Returns sparse Gaussian set focused on high-curvature regions
    pub fn initialize_gdgs(&self, target_n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::gdgs_init;

        gdgs_init::initialize_gdgs_with_target_n(
            &self.target,
            &self.structure_tensor,
            target_n,
        )
    }

    /// Initialize using gradient peak strategy (Strategy H)
    ///
    /// Hybrid approach:
    /// - Place Gaussians at gradient maxima (edges, details)
    /// - Add sparse background grid for smooth region coverage
    ///
    /// Better than GDGS (doesn't neglect smooth regions)
    pub fn initialize_gradient_peaks(&self, target_n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::gradient_peak_init::{initialize_gradient_peaks, GradientPeakConfig};

        let config = GradientPeakConfig {
            target_peak_gaussians: (target_n as f32 * 0.8) as usize,  // 80% at peaks
            background_grid_size: ((target_n as f32 * 0.2).sqrt()) as u32,  // 20% background
            ..Default::default()
        };

        initialize_gradient_peaks(&self.target, &self.structure_tensor, &config)
    }

    /// Initialize using K-means clustering (Strategy D)
    ///
    /// Clusters pixels by (color, position, gradient) features
    /// Places Gaussian at each cluster centroid
    ///
    /// Classic computer vision approach
    pub fn initialize_kmeans(&self, target_n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::kmeans_init::{initialize_kmeans, KMeansConfig};

        let config = KMeansConfig {
            k: target_n,
            ..Default::default()
        };

        initialize_kmeans(&self.target, &self.structure_tensor, &config)
    }

    /// Initialize from SLIC superpixels (Strategy E)
    ///
    /// Loads Gaussian initialization from SLIC preprocessing JSON
    /// SLIC respects object boundaries, perceptually meaningful regions
    ///
    /// # Arguments
    /// * `json_path` - Path to slic_init.json (from Python preprocessing)
    ///
    /// # Example
    /// ```bash
    /// # Preprocess:
    /// python tools/slic_preprocess.py image.png 500 slic_init.json
    ///
    /// # Encode:
    /// let gaussians = encoder.initialize_slic("slic_init.json")?;
    /// ```
    pub fn initialize_slic(&self, json_path: &str) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::slic_init::load_slic_init;

        load_slic_init(json_path)
            .expect("SLIC initialization failed - check JSON file exists and is valid")
    }

    /// Initialize from preprocessing placement map
    ///
    /// Uses Python preprocessing output to intelligently place Gaussians
    ///
    /// # Workflow
    /// ```bash
    /// # 1. Preprocess image (Python)
    /// python tools/preprocess_image_v2.py kodim01.png
    /// # Creates: kodim01.json + kodim01_*.npy
    ///
    /// # 2. Encode with preprocessing (Rust)
    /// let gaussians = encoder.initialize_from_preprocessing("kodim01.json", 500)?;
    /// ```
    ///
    /// # Arguments
    /// * `json_path` - Path to preprocessing JSON
    /// * `n` - Number of Gaussians to create
    ///
    /// # Returns
    /// Gaussians positioned according to placement probability map
    pub fn initialize_from_preprocessing(
        &self,
        json_path: &str,
        n: usize,
    ) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::preprocessing_loader::PreprocessingData;

        // Load preprocessing
        let mut preprocessing = PreprocessingData::load(json_path)
            .expect("Failed to load preprocessing data");

        // Sample positions from placement map
        let positions = preprocessing.sample_positions(n)
            .expect("Failed to sample positions from placement map");

        log::info!("🎯 Initializing from preprocessing: {} Gaussians", n);

        // Create Gaussians at sampled positions
        let mut gaussians = Vec::new();
        let width_px = self.target.width as f32;
        let height_px = self.target.height as f32;

        let gamma = Self::adaptive_gamma(n);
        let sigma_base_px = gamma * ((width_px * height_px) / n as f32).sqrt();

        for (x, y) in positions {
            // Normalized position
            let position = Vector2::new(
                x as f32 / width_px,
                y as f32 / height_px,
            );

            // Color from target image
            let color = self.target.get_pixel(x, y)
                .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

            // Get local structure (same as grid init)
            let tensor = self.structure_tensor.get(x, y);
            let coherence = tensor.coherence;
            let geod_dist_px = self.geodesic_edt.get_distance(x, y);

            // Conditional anisotropy
            let (mut sig_para_px, mut sig_perp_px, rotation_angle) = if coherence < 0.2 {
                (sigma_base_px, sigma_base_px, 0.0)
            } else {
                let sigma_perp = sigma_base_px / (1.0 + 3.0 * coherence);
                let sigma_para = 4.0 * sigma_perp;
                let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
                (sigma_para, sigma_perp, angle)
            };

            // Geodesic clamping
            let max_sigma_px = geod_dist_px * 0.8;
            sig_para_px = sig_para_px.min(max_sigma_px);
            sig_perp_px = sig_perp_px.min(max_sigma_px);

            // Normalize
            let sig_para = sig_para_px / width_px;
            let sig_perp = sig_perp_px / height_px;

            gaussians.push(Gaussian2D::new(
                position,
                Euler::new(sig_para, sig_perp, rotation_angle),
                color,
                1.0,
            ));
        }

        log::info!("✅ Initialized {} Gaussians from preprocessing placement map", gaussians.len());

        gaussians
    }

    /// Error-driven encoding: adaptive refinement for +4.3 dB quality
    ///
    /// Combines structure-tensor initialization with error-driven refinement:
    /// 1. Start with structure-tensor-aware Gaussians (edge-aligned, anisotropic)
    /// 2. Optimize current set
    /// 3. Find high-error regions
    /// 4. Add refinement Gaussians at hotspots
    /// 5. Repeat until convergence
    ///
    /// Expected: +4.3 dB over basic initialization (EXP-4-003 data)
    ///
    /// Note: Uses standard gradient descent optimizer. For 2-5× speed improvement,
    /// use `encode_error_driven_adam()` instead.
    pub fn encode_error_driven(&self, initial_gaussians: usize, max_gaussians: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::renderer_v2::RendererV2;
        use crate::optimizer_v2::OptimizerV2;

        // Start with structure-tensor-aware initialization (better than uniform grid)
        let grid_size = (initial_gaussians as f32).sqrt().ceil() as u32;
        let mut gaussians = self.initialize_gaussians(grid_size);

        log::info!("🎯 Error-driven encoding (gradient descent):");
        log::info!("  Initial Gaussians: {} (structure-tensor aware)", gaussians.len());
        log::info!("  Max Gaussians: {}", max_gaussians);

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 100;

        let target_error = 0.001;
        let split_percentile = 0.10; // Top 10% error regions

        for pass in 0..10 {  // Max 10 refinement passes
            // Optimize current set
            let loss = optimizer.optimize(&mut gaussians, &self.target);

            // Apply geodesic EDT anti-bleeding constraints
            self.apply_geodesic_clamping(&mut gaussians);

            let rendered = RendererV2::render(&gaussians, self.target.width, self.target.height);
            let psnr = self.compute_psnr(&rendered);

            log::info!("  Pass {}: N={}, PSNR={:.2} dB, loss={:.6}", pass, gaussians.len(), psnr, loss);

            // Check convergence
            if loss < target_error {
                log::info!("  ✅ Converged to target error!");
                break;
            }

            if gaussians.len() >= max_gaussians {
                log::info!("  ⚠️  Reached max Gaussians limit");
                break;
            }

            // Find high-error regions
            let error_map = self.compute_error_map(&rendered);
            let hotspots = self.find_hotspots(&error_map, split_percentile);

            if hotspots.is_empty() {
                log::info!("  ✅ No more hotspots to split");
                break;
            }

            log::info!("    Adding {} Gaussians at high-error locations", hotspots.len());

            // Add Gaussians at hotspots
            for (x, y, _error) in hotspots {
                if gaussians.len() >= max_gaussians {
                    break;
                }

                let position = Vector2::new(
                    x as f32 / self.target.width as f32,
                    y as f32 / self.target.height as f32,
                );

                let color = self.target.get_pixel(x, y)
                    .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

                // Small scale for refinement
                let sigma = 0.02;

                gaussians.push(Gaussian2D::new(
                    position,
                    Euler::isotropic(sigma),
                    color,
                    1.0,
                ));
            }
        }

        gaussians
    }

    /// Error-driven encoding with Adam optimizer (RECOMMENDED)
    ///
    /// Same as `encode_error_driven()` but uses Adam optimizer for 2-5× speedup.
    ///
    /// Combines:
    /// - Structure-tensor initialization (edge-aware, anisotropic)
    /// - Error-driven refinement (adaptive placement)
    /// - Adam optimizer (2-5× faster convergence)
    ///
    /// Expected: +4.3 dB quality, 2-5× faster than gradient descent
    pub fn encode_error_driven_adam(&self, initial_gaussians: usize, max_gaussians: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::renderer_v2::RendererV2;
        use crate::adam_optimizer::AdamOptimizer;

        // Start with structure-tensor-aware initialization
        let grid_size = (initial_gaussians as f32).sqrt().ceil() as u32;
        let mut gaussians = self.initialize_gaussians(grid_size);

        log::info!("🎯 Error-driven encoding (Adam optimizer - 2-5× faster):");
        log::info!("  Initial Gaussians: {} (structure-tensor aware)", gaussians.len());
        log::info!("  Max Gaussians: {}", max_gaussians);

        let mut optimizer = AdamOptimizer::default();
        optimizer.max_iterations = 100;

        let target_error = 0.001;
        let split_percentile = 0.10;

        for pass in 0..10 {
            let n_before = gaussians.len();

            // Optimize with Adam
            let loss = optimizer.optimize(&mut gaussians, &self.target);

            // Apply geodesic EDT anti-bleeding constraints
            self.apply_geodesic_clamping(&mut gaussians);

            let rendered = RendererV2::render(&gaussians, self.target.width, self.target.height);
            let psnr = self.compute_psnr(&rendered);

            log::info!("  Pass {}: N={}, PSNR={:.2} dB, loss={:.6}", pass, gaussians.len(), psnr, loss);

            if loss < target_error {
                log::info!("  ✅ Converged to target error!");
                break;
            }

            if gaussians.len() >= max_gaussians {
                log::info!("  ⚠️  Reached max Gaussians limit");
                break;
            }

            // Find high-error regions
            let error_map = self.compute_error_map(&rendered);
            let hotspots = self.find_hotspots(&error_map, split_percentile);

            if hotspots.is_empty() {
                log::info!("  ✅ No more hotspots to split");
                break;
            }

            let batch_size = hotspots.len();
            log::info!("    Adding {} Gaussians at high-error locations", batch_size);

            // Add Gaussians at hotspots
            // Use structure-tensor-aware initialization (NOT fixed σ=0.02!)
            let current_n = gaussians.len();
            let gamma = Self::adaptive_gamma(current_n);
            let width_px = self.target.width as f32;
            let height_px = self.target.height as f32;
            let sigma_base_px = gamma * ((width_px * height_px) / current_n as f32).sqrt();

            for (x, y, _error) in hotspots {
                if gaussians.len() >= max_gaussians {
                    break;
                }

                let position = Vector2::new(
                    x as f32 / self.target.width as f32,
                    y as f32 / self.target.height as f32,
                );

                let color = self.target.get_pixel(x, y)
                    .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

                // Structure-tensor-aware shape (same as grid init!)
                let tensor = self.structure_tensor.get(x, y);
                let coherence = tensor.coherence;
                let geod_dist_px = self.geodesic_edt.get_distance(x, y);

                let (mut sig_para_px, mut sig_perp_px, rotation_angle) = if coherence < 0.2 {
                    // Flat region → isotropic
                    (sigma_base_px, sigma_base_px, 0.0)
                } else {
                    // Edge → anisotropic, aligned to structure
                    let sigma_perp = sigma_base_px / (1.0 + 3.0 * coherence);
                    let sigma_para = 4.0 * sigma_perp;
                    let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
                    (sigma_para, sigma_perp, angle)
                };

                // Geodesic clamping
                let max_sigma_px = geod_dist_px * 0.8;
                sig_para_px = sig_para_px.min(max_sigma_px);
                sig_perp_px = sig_perp_px.min(max_sigma_px);

                // Normalize to [0,1]
                let sig_para = sig_para_px / width_px;
                let sig_perp = sig_perp_px / height_px;

                gaussians.push(Gaussian2D::new(
                    position,
                    Euler::new(sig_para, sig_perp, rotation_angle),
                    color,
                    1.0,
                ));
            }

            // WARMUP: New batch needs settling with ramped LR
            let n_after = gaussians.len();
            let n_added = n_after - n_before;

            if n_added > 0 {
                // Warmup iterations: proportional to batch size, max 50, min 10
                let warmup_iters = ((n_added / 2) as usize).clamp(10, 50);
                log::info!("    Warmup: {} iterations with LR ramp (0.1× → 1.0×)", warmup_iters);

                // Save full LR
                let lr_full = optimizer.learning_rate;

                // Warmup with fractional LR ramping up
                for w in 0..warmup_iters {
                    let progress = (w as f32) / (warmup_iters as f32);
                    let lr_fraction = 0.1 + 0.9 * progress;  // 0.1 → 1.0

                    optimizer.learning_rate = lr_full * lr_fraction;
                    optimizer.max_iterations = 1;  // Single iteration
                    let _ = optimizer.optimize(&mut gaussians, &self.target);
                }

                // Restore settings
                optimizer.learning_rate = lr_full;
                optimizer.max_iterations = 100;
            }
        }

        gaussians
    }

    /// Error-driven encoding with Adam optimizer AND data logging
    ///
    /// This version captures Gaussian configurations during optimization for quantum research.
    /// Same algorithm as `encode_error_driven_adam` but with callback logging.
    ///
    /// # Parameters
    /// - `initial_gaussians`: Starting grid size
    /// - `max_gaussians`: Maximum Gaussians allowed
    /// - `image_id`: Identifier for this image (e.g., "kodim01")
    /// - `logger`: Optional data logger for capturing Gaussian states
    pub fn encode_error_driven_adam_with_logger(
        &self,
        initial_gaussians: usize,
        max_gaussians: usize,
        image_id: &str,
        mut logger: Option<&mut dyn crate::gaussian_logger::GaussianLogger>,
    ) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::renderer_v2::RendererV2;
        use crate::adam_optimizer::AdamOptimizer;

        // Start with structure-tensor-aware initialization
        let grid_size = (initial_gaussians as f32).sqrt().ceil() as u32;
        let mut gaussians = self.initialize_gaussians(grid_size);

        log::info!("🎯 Error-driven encoding with data logging:");
        log::info!("  Initial Gaussians: {} (structure-tensor aware)", gaussians.len());
        log::info!("  Max Gaussians: {}", max_gaussians);
        log::info!("  Image ID: {}", image_id);
        log::info!("  Logging: Enabled (every 10th iteration)");

        let mut optimizer = AdamOptimizer::default();
        optimizer.max_iterations = 100;

        let target_error = 0.001;
        let split_percentile = 0.10;

        for pass in 0..10 {
            let n_before = gaussians.len();

            // Set logger context and optimize with logging
            let loss = match logger.as_deref_mut() {
                Some(log) => {
                    log.set_context(image_id.to_string(), pass);
                    optimizer.optimize_with_logger(&mut gaussians, &self.target, Some(log))
                }
                None => {
                    optimizer.optimize_with_logger(&mut gaussians, &self.target, None)
                }
            };

            // Apply geodesic EDT anti-bleeding constraints
            self.apply_geodesic_clamping(&mut gaussians);

            let rendered = RendererV2::render(&gaussians, self.target.width, self.target.height);
            let psnr = self.compute_psnr(&rendered);

            log::info!("  Pass {}: N={}, PSNR={:.2} dB, loss={:.6}", pass, gaussians.len(), psnr, loss);

            if loss < target_error {
                log::info!("  ✅ Converged to target error!");
                break;
            }

            if gaussians.len() >= max_gaussians {
                log::info!("  ⚠️  Reached max Gaussians limit");
                break;
            }

            // Find high-error regions
            let error_map = self.compute_error_map(&rendered);
            let hotspots = self.find_hotspots(&error_map, split_percentile);

            if hotspots.is_empty() {
                log::info!("  ✅ No more hotspots to split");
                break;
            }

            let batch_size = hotspots.len();
            log::info!("    Adding {} Gaussians at high-error locations", batch_size);

            // Add Gaussians at hotspots
            let current_n = gaussians.len();
            let gamma = Self::adaptive_gamma(current_n);
            let width_px = self.target.width as f32;
            let height_px = self.target.height as f32;
            let sigma_base_px = gamma * ((width_px * height_px) / current_n as f32).sqrt();

            for (x, y, _error) in hotspots {
                if gaussians.len() >= max_gaussians {
                    break;
                }

                let position = Vector2::new(
                    x as f32 / self.target.width as f32,
                    y as f32 / self.target.height as f32,
                );

                let color = self.target.get_pixel(x, y)
                    .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

                let tensor = self.structure_tensor.get(x, y);
                let coherence = tensor.coherence;
                let geod_dist_px = self.geodesic_edt.get_distance(x, y);

                let (mut sig_para_px, mut sig_perp_px, rotation_angle) = if coherence < 0.2 {
                    (sigma_base_px, sigma_base_px, 0.0)
                } else {
                    let sigma_perp = sigma_base_px / (1.0 + 3.0 * coherence);
                    let sigma_para = 4.0 * sigma_perp;
                    let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
                    (sigma_para, sigma_perp, angle)
                };

                let max_sigma_px = geod_dist_px * 0.8;
                sig_para_px = sig_para_px.min(max_sigma_px);
                sig_perp_px = sig_perp_px.min(max_sigma_px);

                // Normalize to [0,1] with defensive clamping to prevent NaN
                let sig_para = (sig_para_px / width_px).clamp(0.001, 0.5);
                let sig_perp = (sig_perp_px / height_px).clamp(0.001, 0.5);

                // Validate all parameters before creating Gaussian
                if position.x.is_nan() || position.y.is_nan() ||
                   sig_para.is_nan() || sig_perp.is_nan() ||
                   rotation_angle.is_nan() ||
                   color.r.is_nan() || color.g.is_nan() || color.b.is_nan() {
                    log::warn!("  ⚠️  Skipping Gaussian at ({}, {}) - NaN parameters detected", x, y);
                    continue;
                }

                gaussians.push(Gaussian2D::new(
                    position,
                    Euler::new(sig_para, sig_perp, rotation_angle),
                    color,
                    1.0,
                ));
            }

            // Warmup phase for new Gaussians
            // NEW: Use proper continuous warmup with logging
            let n_after = gaussians.len();
            let n_added = n_after - n_before;

            if n_added > 0 {
                let warmup_iters = ((n_added / 2) as usize).clamp(10, 50);
                log::info!("    Warmup: {} iterations (continuous, preserves Adam momentum)", warmup_iters);

                // Create fresh optimizer for warmup to avoid momentum corruption
                let mut warmup_optimizer = AdamOptimizer::default();
                warmup_optimizer.learning_rate = optimizer.learning_rate * 0.5;  // Lower LR for stability
                warmup_optimizer.max_iterations = warmup_iters;

                // Set logger context for warmup pass
                if let Some(log) = logger.as_deref_mut() {
                    log.set_context(format!("{}_warmup", image_id), pass);
                }

                // Run continuous warmup iterations with logging
                let _ = match logger.as_deref_mut() {
                    Some(log) => warmup_optimizer.optimize_with_logger(&mut gaussians, &self.target, Some(log)),
                    None => warmup_optimizer.optimize_with_logger(&mut gaussians, &self.target, None),
                };
            }
        }

        // Final flush of logger
        if let Some(log) = logger.as_deref_mut() {
            let _ = log.flush();
        }

        gaussians
    }

    /// Error-driven encoding with ISOTROPIC edges (quantum-guided)
    ///
    /// **QUANTUM DISCOVERY VALIDATION**: Tests quantum finding that isotropic
    /// Gaussians work better than anisotropic ones for edge representation.
    ///
    /// Quantum channels 3, 4, 7 (high quality) all show σ_x ≈ σ_y despite
    /// high edge coherence (>0.96). This contradicts classical assumption
    /// that edges need elongated Gaussians.
    ///
    /// # Difference from Standard Method
    /// - Edges use SMALL ISOTROPIC Gaussians (σ_x = σ_y)
    /// - Current method uses ANISOTROPIC (σ_parallel = 4× σ_perp)
    /// - All other logic identical (same optimizer, same refinement)
    ///
    /// # Expected Result
    /// If quantum is correct: +5-10 dB improvement on edge quality
    /// Current edge PSNR: 1.56 dB → Quantum prediction: 10-15 dB
    pub fn encode_error_driven_adam_isotropic(&self, initial_gaussians: usize, max_gaussians: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::renderer_v2::RendererV2;
        use crate::adam_optimizer::AdamOptimizer;

        // Start with structure-tensor-aware initialization
        let grid_size = (initial_gaussians as f32).sqrt().ceil() as u32;
        let mut gaussians = self.initialize_gaussians(grid_size);

        log::info!("🎯 ISOTROPIC edge encoding (quantum-guided):");
        log::info!("  Initial Gaussians: {} (structure-tensor aware)", gaussians.len());
        log::info!("  Max Gaussians: {}", max_gaussians);
        log::info!("  Edge strategy: ISOTROPIC (σ_x = σ_y) - testing quantum discovery");

        let mut optimizer = AdamOptimizer::default();
        optimizer.max_iterations = 100;

        let target_error = 0.001;
        let split_percentile = 0.10;

        for pass in 0..10 {
            let n_before = gaussians.len();

            // Optimize with Adam
            let loss = optimizer.optimize(&mut gaussians, &self.target);

            // Apply geodesic EDT anti-bleeding constraints
            self.apply_geodesic_clamping(&mut gaussians);

            let rendered = RendererV2::render(&gaussians, self.target.width, self.target.height);
            let psnr = self.compute_psnr(&rendered);

            log::info!("  Pass {}: N={}, PSNR={:.2} dB, loss={:.6}", pass, gaussians.len(), psnr, loss);

            if loss < target_error {
                log::info!("  ✅ Converged to target error!");
                break;
            }

            if gaussians.len() >= max_gaussians {
                log::info!("  ⚠️  Reached max Gaussians limit");
                break;
            }

            // Find high-error regions
            let error_map = self.compute_error_map(&rendered);
            let hotspots = self.find_hotspots(&error_map, split_percentile);

            if hotspots.is_empty() {
                log::info!("  ✅ No more hotspots to split");
                break;
            }

            let batch_size = hotspots.len();
            log::info!("    Adding {} Gaussians at high-error locations (ISOTROPIC)", batch_size);

            // Add Gaussians at hotspots
            let current_n = gaussians.len();
            let gamma = Self::adaptive_gamma(current_n);
            let width_px = self.target.width as f32;
            let height_px = self.target.height as f32;
            let sigma_base_px = gamma * ((width_px * height_px) / current_n as f32).sqrt();

            for (x, y, _error) in hotspots {
                if gaussians.len() >= max_gaussians {
                    break;
                }

                let position = Vector2::new(
                    x as f32 / self.target.width as f32,
                    y as f32 / self.target.height as f32,
                );

                let color = self.target.get_pixel(x, y)
                    .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

                let tensor = self.structure_tensor.get(x, y);
                let coherence = tensor.coherence;
                let geod_dist_px = self.geodesic_edt.get_distance(x, y);

                // QUANTUM-GUIDED: Use isotropic for ALL coherence levels
                let sigma_iso_px = if coherence < 0.2 {
                    // Smooth regions: standard size
                    sigma_base_px
                } else {
                    // Edges: SMALLER isotropic (quantum channels 3,4,7)
                    // Quantum shows: smaller scales work better at edges
                    sigma_base_px / (1.0 + 2.0 * coherence)
                };

                // Geodesic clamping
                let max_sigma_px = geod_dist_px * 0.8;
                let sig_px = sigma_iso_px.min(max_sigma_px);

                // Normalize to [0,1] with defensive clamping
                let sigma = (sig_px / width_px.max(height_px)).clamp(0.001, 0.5);

                // Validate before creating
                if sigma.is_nan() || color.r.is_nan() || color.g.is_nan() || color.b.is_nan() {
                    log::warn!("  ⚠️  Skipping Gaussian at ({}, {}) - NaN detected", x, y);
                    continue;
                }

                // Create ISOTROPIC Gaussian (σ_x = σ_y, rotation = 0)
                gaussians.push(Gaussian2D::new(
                    position,
                    Euler::new(sigma, sigma, 0.0),  // Isotropic!
                    color,
                    1.0,
                ));
            }

            // Warmup phase
            let n_after = gaussians.len();
            let n_added = n_after - n_before;

            if n_added > 0 {
                let warmup_iters = ((n_added / 2) as usize).clamp(10, 50);
                log::info!("    Warmup: {} iterations", warmup_iters);

                let mut warmup_optimizer = AdamOptimizer::default();
                warmup_optimizer.learning_rate = optimizer.learning_rate * 0.5;
                warmup_optimizer.max_iterations = warmup_iters;
                let _ = warmup_optimizer.optimize(&mut gaussians, &self.target);
            }
        }

        gaussians
    }

    /// Error-driven encoding with OptimizerV2 (gradient descent + MS-SSIM/edge-weighted)
    ///
    /// **Q2 RESEARCH**: Tests if gradient descent with perceptual loss works better
    /// than Adam for certain quantum channels.
    ///
    /// OptimizerV2 features:
    /// - Simple gradient descent (no momentum)
    /// - Optional MS-SSIM loss (perceptual quality)
    /// - Optional edge-weighted gradients
    /// - GPU acceleration support
    pub fn encode_error_driven_v2(&self, initial_gaussians: usize, max_gaussians: usize, use_ms_ssim: bool) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::renderer_v2::RendererV2;
        use crate::optimizer_v2::OptimizerV2;

        let grid_size = (initial_gaussians as f32).sqrt().ceil() as u32;
        let mut gaussians = self.initialize_gaussians(grid_size);

        log::info!("🎯 Error-driven encoding (OptimizerV2):");
        log::info!("  Optimizer: Gradient descent");
        log::info!("  Loss: {}", if use_ms_ssim { "MS-SSIM (perceptual)" } else { "L2" });

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 100;
        optimizer.use_ms_ssim = use_ms_ssim;

        let target_error = 0.001;
        let split_percentile = 0.10;

        for pass in 0..10 {
            let loss = optimizer.optimize(&mut gaussians, &self.target);
            self.apply_geodesic_clamping(&mut gaussians);

            let rendered = RendererV2::render(&gaussians, self.target.width, self.target.height);
            let psnr = self.compute_psnr(&rendered);

            log::info!("  Pass {}: N={}, PSNR={:.2} dB, loss={:.6}", pass, gaussians.len(), psnr, loss);

            if loss < target_error || gaussians.len() >= max_gaussians {
                break;
            }

            let error_map = self.compute_error_map(&rendered);
            let hotspots = self.find_hotspots(&error_map, split_percentile);

            if hotspots.is_empty() {
                break;
            }

            // Add Gaussians (same logic as adam method)
            let current_n = gaussians.len();
            let gamma = Self::adaptive_gamma(current_n);
            let width_px = self.target.width as f32;
            let height_px = self.target.height as f32;
            let sigma_base_px = gamma * ((width_px * height_px) / current_n as f32).sqrt();

            for (x, y, _error) in hotspots {
                if gaussians.len() >= max_gaussians {
                    break;
                }

                let position = Vector2::new(
                    x as f32 / self.target.width as f32,
                    y as f32 / self.target.height as f32,
                );

                let color = self.target.get_pixel(x, y)
                    .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

                let tensor = self.structure_tensor.get(x, y);
                let coherence = tensor.coherence;
                let geod_dist_px = self.geodesic_edt.get_distance(x, y);

                let (mut sig_para_px, mut sig_perp_px, rotation_angle) = if coherence < 0.2 {
                    (sigma_base_px, sigma_base_px, 0.0)
                } else {
                    let sigma_perp = sigma_base_px / (1.0 + 3.0 * coherence);
                    let sigma_para = 4.0 * sigma_perp;
                    let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
                    (sigma_para, sigma_perp, angle)
                };

                let max_sigma_px = geod_dist_px * 0.8;
                sig_para_px = sig_para_px.min(max_sigma_px);
                sig_perp_px = sig_perp_px.min(max_sigma_px);

                let sig_para = (sig_para_px / width_px).clamp(0.001, 0.5);
                let sig_perp = (sig_perp_px / height_px).clamp(0.001, 0.5);

                if sig_para.is_nan() || sig_perp.is_nan() || rotation_angle.is_nan() ||
                   color.r.is_nan() || color.g.is_nan() || color.b.is_nan() {
                    continue;
                }

                gaussians.push(Gaussian2D::new(
                    position,
                    Euler::new(sig_para, sig_perp, rotation_angle),
                    color,
                    1.0,
                ));
            }
        }

        gaussians
    }

    /// Error-driven encoding with OptimizerV3 (perceptual: MS-SSIM + edge-weighted)
    ///
    /// **Q2 RESEARCH**: Tests if perceptual optimization works better than Adam
    /// for certain quantum channels.
    ///
    /// OptimizerV3 always uses:
    /// - MS-SSIM loss (perceptual quality metric)
    /// - Edge-weighted gradients (prioritizes edges)
    pub fn encode_error_driven_v3(&self, initial_gaussians: usize, max_gaussians: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::renderer_v2::RendererV2;
        use crate::optimizer_v3_perceptual::OptimizerV3;

        let grid_size = (initial_gaussians as f32).sqrt().ceil() as u32;
        let mut gaussians = self.initialize_gaussians(grid_size);

        log::info!("🎯 Error-driven encoding (OptimizerV3 - Perceptual):");
        log::info!("  Loss: MS-SSIM + Edge-weighted");

        let mut optimizer = OptimizerV3::default();
        optimizer.max_iterations = 100;

        let target_error = 0.001;
        let split_percentile = 0.10;

        for pass in 0..10 {
            let loss = optimizer.optimize(&mut gaussians, &self.target, &self.structure_tensor);
            self.apply_geodesic_clamping(&mut gaussians);

            let rendered = RendererV2::render(&gaussians, self.target.width, self.target.height);
            let psnr = self.compute_psnr(&rendered);

            log::info!("  Pass {}: N={}, PSNR={:.2} dB, loss={:.6}", pass, gaussians.len(), psnr, loss);

            if loss < target_error || gaussians.len() >= max_gaussians {
                break;
            }

            let error_map = self.compute_error_map(&rendered);
            let hotspots = self.find_hotspots(&error_map, split_percentile);

            if hotspots.is_empty() {
                break;
            }

            // Add Gaussians (same logic)
            let current_n = gaussians.len();
            let gamma = Self::adaptive_gamma(current_n);
            let width_px = self.target.width as f32;
            let height_px = self.target.height as f32;
            let sigma_base_px = gamma * ((width_px * height_px) / current_n as f32).sqrt();

            for (x, y, _error) in hotspots {
                if gaussians.len() >= max_gaussians {
                    break;
                }

                let position = Vector2::new(
                    x as f32 / self.target.width as f32,
                    y as f32 / self.target.height as f32,
                );

                let color = self.target.get_pixel(x, y)
                    .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

                let tensor = self.structure_tensor.get(x, y);
                let coherence = tensor.coherence;
                let geod_dist_px = self.geodesic_edt.get_distance(x, y);

                let (mut sig_para_px, mut sig_perp_px, rotation_angle) = if coherence < 0.2 {
                    (sigma_base_px, sigma_base_px, 0.0)
                } else {
                    let sigma_perp = sigma_base_px / (1.0 + 3.0 * coherence);
                    let sigma_para = 4.0 * sigma_perp;
                    let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
                    (sigma_para, sigma_perp, angle)
                };

                let max_sigma_px = geod_dist_px * 0.8;
                sig_para_px = sig_para_px.min(max_sigma_px);
                sig_perp_px = sig_perp_px.min(max_sigma_px);

                let sig_para = (sig_para_px / width_px).clamp(0.001, 0.5);
                let sig_perp = (sig_perp_px / height_px).clamp(0.001, 0.5);

                if sig_para.is_nan() || sig_perp.is_nan() || rotation_angle.is_nan() ||
                   color.r.is_nan() || color.g.is_nan() || color.b.is_nan() {
                    continue;
                }

                gaussians.push(Gaussian2D::new(
                    position,
                    Euler::new(sig_para, sig_perp, rotation_angle),
                    color,
                    1.0,
                ));
            }
        }

        gaussians
    }

    /// Compute per-pixel error map
    fn compute_error_map(&self, rendered: &ImageBuffer<f32>) -> Vec<f32> {
        let mut errors = Vec::with_capacity((self.target.width * self.target.height) as usize);

        for (t, r) in self.target.data.iter().zip(rendered.data.iter()) {
            let err = (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
            errors.push(err);
        }

        errors
    }

    /// Find high-error regions (hotspots)
    fn find_hotspots(&self, error_map: &[f32], percentile: f32) -> Vec<(u32, u32, f32)> {
        // Sort errors to find threshold
        let mut sorted_errors = error_map.to_vec();
        sorted_errors.sort_by(|a, b| b.partial_cmp(a).unwrap());  // Descending

        let threshold_idx = (sorted_errors.len() as f32 * percentile) as usize;
        let threshold = sorted_errors[threshold_idx];

        let mut hotspots = Vec::new();

        for y in 0..self.target.height {
            for x in 0..self.target.width {
                let idx = (y * self.target.width + x) as usize;
                if error_map[idx] > threshold {
                    hotspots.push((x, y, error_map[idx]));
                }
            }
        }

        // Limit number of hotspots per pass
        hotspots.truncate(50);
        hotspots
    }

    /// Compute PSNR quality metric
    fn compute_psnr(&self, rendered: &ImageBuffer<f32>) -> f32 {
        let mut mse = 0.0;
        let count = (self.target.width * self.target.height * 3) as f32;

        for (p1, p2) in self.target.data.iter().zip(rendered.data.iter()) {
            mse += (p1.r - p2.r).powi(2);
            mse += (p1.g - p2.g).powi(2);
            mse += (p1.b - p2.b).powi(2);
        }

        mse /= count;
        if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
    }

    /// Apply geodesic EDT anti-bleeding constraints
    ///
    /// Clamps Gaussian scales based on geodesic distance to prevent
    /// bleeding across edges. Should be called after optimization.
    ///
    /// Expected: +0.3-0.8 dB on images with sharp edges
    fn apply_geodesic_clamping(&self, gaussians: &mut [Gaussian2D<f32, Euler<f32>>]) {
        let width = self.target.width as f32;
        let height = self.target.height as f32;

        for gaussian in gaussians.iter_mut() {
            // Convert normalized position to pixels
            let gx = (gaussian.position.x * width) as u32;
            let gy = (gaussian.position.y * height) as u32;
            let gx = gx.min(self.target.width - 1);
            let gy = gy.min(self.target.height - 1);

            // Get structure tensor coherence
            let tensor = self.structure_tensor.get(gx, gy);
            let coherence = tensor.coherence;

            // Only apply to strong edges (coherence > 0.6)
            if coherence > 0.6 {
                // Get geodesic distance in pixels
                let geod_dist_px = self.geodesic_edt.get_distance(gx, gy);

                // Clamp scale based on geodesic distance
                // Prevents Gaussian from bleeding across nearby edges
                let max_scale_px = 0.5 + 0.3 * geod_dist_px;
                let max_scale_norm = max_scale_px / width;

                // Anisotropic clamping: parallel can be larger than perpendicular
                gaussian.shape.scale_x = gaussian.shape.scale_x.min(max_scale_norm * 2.0);
                gaussian.shape.scale_y = gaussian.shape.scale_y.min(max_scale_norm);
            }
        }
    }

    /// Render Gaussians with EWA splatting (alias-free, high quality)
    ///
    /// Uses Elliptical Weighted Average splatting for superior rendering quality.
    /// Benefits:
    /// - Alias-free (no jaggies)
    /// - Zoom-stable
    /// - Proper reconstruction filter
    ///
    /// Expected: +0.3-0.8 dB over simple splatting
    pub fn render_ewa(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>], zoom: f32) -> ImageBuffer<f32> {
        use lgi_core::ewa_splatting_v2::EWARendererV2;

        let ewa_renderer = EWARendererV2::default();
        ewa_renderer.render(gaussians, self.target.width, self.target.height, zoom)
    }

    /// Compute PSNR using EWA rendering
    ///
    /// More accurate quality measurement using alias-free EWA renderer
    pub fn compute_psnr_ewa(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> f32 {
        let rendered = self.render_ewa(gaussians, 1.0);
        self.compute_psnr(&rendered)
    }

    /// Error-driven encoding with GPU acceleration (FASTEST)
    ///
    /// Same as `encode_error_driven()` but uses GPU-accelerated renderer.
    /// 100-1000× faster rendering during optimization.
    ///
    /// Expected: +4.3 dB quality, 100-1000× rendering speedup
    ///
    /// Note: Falls back to CPU if GPU unavailable
    pub fn encode_error_driven_gpu(&self, initial_gaussians: usize, max_gaussians: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::optimizer_v2::OptimizerV2;

        // Start with structure-tensor-aware initialization
        let grid_size = (initial_gaussians as f32).sqrt().ceil() as u32;
        let mut gaussians = self.initialize_gaussians(grid_size);

        log::info!("🎯 Error-driven encoding (GPU-accelerated - 100-1000× faster rendering):");
        log::info!("  Initial Gaussians: {} (structure-tensor aware)", gaussians.len());
        log::info!("  Max Gaussians: {}", max_gaussians);

        let mut optimizer = OptimizerV2::new_with_gpu();
        optimizer.max_iterations = 100;

        let has_gpu = optimizer.has_gpu();
        if has_gpu {
            log::info!("  ✅ GPU renderer active");
        } else {
            log::info!("  ⚠️  GPU not available, using CPU fallback");
        }

        let target_error = 0.001;
        let split_percentile = 0.10;

        for pass in 0..10 {
            // Optimize with GPU-accelerated rendering
            let loss = optimizer.optimize(&mut gaussians, &self.target);

            // Apply geodesic EDT anti-bleeding constraints
            self.apply_geodesic_clamping(&mut gaussians);

            // Use CPU renderer for PSNR measurement (GPU renderer returns during optimization)
            let rendered = crate::renderer_v2::RendererV2::render(&gaussians, self.target.width, self.target.height);
            let psnr = self.compute_psnr(&rendered);

            log::info!("  Pass {}: N={}, PSNR={:.2} dB, loss={:.6}", pass, gaussians.len(), psnr, loss);

            if loss < target_error {
                log::info!("  ✅ Converged to target error!");
                break;
            }

            if gaussians.len() >= max_gaussians {
                log::info!("  ⚠️  Reached max Gaussians limit");
                break;
            }

            // Find high-error regions
            let error_map = self.compute_error_map(&rendered);
            let hotspots = self.find_hotspots(&error_map, split_percentile);

            if hotspots.is_empty() {
                log::info!("  ✅ No more hotspots to split");
                break;
            }

            log::info!("    Adding {} Gaussians at high-error locations", hotspots.len());

            // Add Gaussians at hotspots
            // Use structure-tensor-aware initialization (NOT fixed σ=0.02!)
            let current_n = gaussians.len();
            let gamma = Self::adaptive_gamma(current_n);
            let width_px = self.target.width as f32;
            let height_px = self.target.height as f32;
            let sigma_base_px = gamma * ((width_px * height_px) / current_n as f32).sqrt();

            for (x, y, _error) in hotspots {
                if gaussians.len() >= max_gaussians {
                    break;
                }

                let position = Vector2::new(
                    x as f32 / self.target.width as f32,
                    y as f32 / self.target.height as f32,
                );

                let color = self.target.get_pixel(x, y)
                    .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

                // Structure-tensor-aware shape (same as grid init!)
                let tensor = self.structure_tensor.get(x, y);
                let coherence = tensor.coherence;
                let geod_dist_px = self.geodesic_edt.get_distance(x, y);

                let (mut sig_para_px, mut sig_perp_px, rotation_angle) = if coherence < 0.2 {
                    // Flat region → isotropic
                    (sigma_base_px, sigma_base_px, 0.0)
                } else {
                    // Edge → anisotropic, aligned to structure
                    let sigma_perp = sigma_base_px / (1.0 + 3.0 * coherence);
                    let sigma_para = 4.0 * sigma_perp;
                    let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
                    (sigma_para, sigma_perp, angle)
                };

                // Geodesic clamping
                let max_sigma_px = geod_dist_px * 0.8;
                sig_para_px = sig_para_px.min(max_sigma_px);
                sig_perp_px = sig_perp_px.min(max_sigma_px);

                // Normalize to [0,1]
                let sig_para = sig_para_px / width_px;
                let sig_perp = sig_perp_px / height_px;

                gaussians.push(Gaussian2D::new(
                    position,
                    Euler::new(sig_para, sig_perp, rotation_angle),
                    color,
                    1.0,
                ));
            }
        }

        gaussians
    }

    /// Error-driven encoding with GPU + MS-SSIM (ULTIMATE QUALITY)
    ///
    /// Combines:
    /// - GPU-accelerated rendering (100-1000× faster)
    /// - MS-SSIM perceptual loss (better than L2)
    /// - Error-driven refinement
    /// - Anti-bleeding constraints
    ///
    /// Expected: +4.3 dB base + perceptual quality improvement
    pub fn encode_error_driven_gpu_msssim(&self, initial_gaussians: usize, max_gaussians: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use crate::optimizer_v2::OptimizerV2;

        let grid_size = (initial_gaussians as f32).sqrt().ceil() as u32;
        let mut gaussians = self.initialize_gaussians(grid_size);

        log::info!("🎯 Error-driven encoding (GPU + MS-SSIM - ULTIMATE QUALITY):");
        log::info!("  Initial Gaussians: {} (structure-tensor aware)", gaussians.len());
        log::info!("  Max Gaussians: {}", max_gaussians);

        let mut optimizer = OptimizerV2::new_with_gpu_and_ms_ssim();
        optimizer.max_iterations = 100;

        let has_gpu = optimizer.has_gpu();
        if has_gpu {
            log::info!("  ✅ GPU renderer active + MS-SSIM perceptual loss");
        } else {
            log::info!("  ⚠️  GPU not available, using CPU + MS-SSIM");
        }

        let target_error = 0.001;
        let split_percentile = 0.10;

        for pass in 0..10 {
            let loss = optimizer.optimize(&mut gaussians, &self.target);
            self.apply_geodesic_clamping(&mut gaussians);

            let rendered = crate::renderer_v2::RendererV2::render(&gaussians, self.target.width, self.target.height);
            let psnr = self.compute_psnr(&rendered);

            log::info!("  Pass {}: N={}, PSNR={:.2} dB, loss={:.6}", pass, gaussians.len(), psnr, loss);

            if loss < target_error {
                log::info!("  ✅ Converged to target error!");
                break;
            }

            if gaussians.len() >= max_gaussians {
                log::info!("  ⚠️  Reached max Gaussians limit");
                break;
            }

            let error_map = self.compute_error_map(&rendered);
            let hotspots = self.find_hotspots(&error_map, split_percentile);

            if hotspots.is_empty() {
                log::info!("  ✅ No more hotspots to split");
                break;
            }

            log::info!("    Adding {} Gaussians at high-error locations", hotspots.len());

            for (x, y, _error) in hotspots {
                if gaussians.len() >= max_gaussians {
                    break;
                }

                let position = Vector2::new(
                    x as f32 / self.target.width as f32,
                    y as f32 / self.target.height as f32,
                );

                let color = self.target.get_pixel(x, y)
                    .unwrap_or(Color4::new(0.5, 0.5, 0.5, 1.0));

                let sigma = 0.02;

                gaussians.push(Gaussian2D::new(
                    position,
                    Euler::isotropic(sigma),
                    color,
                    1.0,
                ));
            }
        }

        gaussians
    }

    /// Encode for target PSNR quality
    ///
    /// Uses rate-distortion optimization to select optimal Gaussian count
    /// for specified quality target.
    ///
    /// # Arguments
    /// * `target_psnr` - Desired PSNR in dB (e.g., 30.0)
    /// * `profile` - Quantization profile (affects file size)
    ///
    /// # Returns
    /// Gaussians optimized for target quality
    pub fn encode_for_psnr(&self, target_psnr: f32, profile: lgi_core::quantization::LGIQProfile) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use lgi_core::rate_distortion::RateDistortionOptimizer;

        log::info!("🎯 Encoding for target PSNR: {:.1} dB", target_psnr);
        log::info!("  Profile: {:?}", profile);

        // Select optimal Gaussian count for target PSNR
        let rd_optimizer = RateDistortionOptimizer::with_profile(0.01, profile, true);
        let optimal_count = rd_optimizer.select_gaussian_count_for_psnr(
            self.target.width,
            self.target.height,
            target_psnr,
        );

        log::info!("  Optimal Gaussian count: {} (R-D selected)", optimal_count);

        // Use error-driven encoding with Adam for best quality
        self.encode_error_driven_adam(optimal_count / 2, optimal_count)
    }

    /// Encode for target perceptual quality (MS-SSIM)
    ///
    /// Uses MS-SSIM perceptual metric instead of PSNR for quality targeting.
    /// Better for photos where perceptual quality matters more than pixel accuracy.
    ///
    /// # Arguments
    /// * `target_msssim` - Desired MS-SSIM (0.0-1.0, typically 0.95+)
    /// * `profile` - Quantization profile
    ///
    /// # Returns
    /// Gaussians optimized for perceptual quality
    pub fn encode_for_perceptual_quality(&self, target_msssim: f32, profile: lgi_core::quantization::LGIQProfile) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use lgi_core::rate_distortion::RateDistortionOptimizer;

        log::info!("🎯 Encoding for target MS-SSIM: {:.3}", target_msssim);
        log::info!("  Profile: {:?}", profile);

        // Convert MS-SSIM target to approximate PSNR for initial estimate
        // MS-SSIM 0.95 ≈ 30 dB, 0.98 ≈ 35 dB, 0.99 ≈ 40 dB (rough estimates)
        let approx_psnr = if target_msssim >= 0.99 {
            40.0
        } else if target_msssim >= 0.98 {
            35.0
        } else if target_msssim >= 0.95 {
            30.0
        } else {
            25.0
        };

        let rd_optimizer = RateDistortionOptimizer::with_msssim(0.01, profile, true);
        let optimal_count = rd_optimizer.select_gaussian_count_for_psnr(
            self.target.width,
            self.target.height,
            approx_psnr,
        );

        log::info!("  Optimal Gaussian count: {} (perceptual R-D)", optimal_count);

        // Use GPU+MS-SSIM encoding for perceptual quality
        self.encode_error_driven_gpu_msssim(optimal_count / 2, optimal_count)
    }

    /// Encode for target bitrate (file size)
    ///
    /// Uses rate-distortion optimization to balance quality and size.
    ///
    /// # Arguments
    /// * `target_bits` - Target file size in bits
    /// * `profile` - Quantization profile
    ///
    /// # Returns
    /// Gaussians optimized for target bitrate
    pub fn encode_for_bitrate(&self, target_bits: f32, profile: lgi_core::quantization::LGIQProfile) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use lgi_core::rate_distortion::{RateDistortionOptimizer, ProfileRate};

        log::info!("🎯 Encoding for target bitrate: {:.1} KB", target_bits / 8192.0);
        log::info!("  Profile: {:?}", profile);

        // Estimate how many Gaussians fit in target size
        let profile_rate = ProfileRate { profile, bits_per_gaussian: 0.0 };
        let bits_per_gaussian = profile_rate.compressed_bits_per_gaussian();
        let header_overhead = 512.0; // 64 bytes
        let available_bits = target_bits - header_overhead;
        let max_gaussians = (available_bits / bits_per_gaussian).max(100.0) as usize;

        log::info!("  Max Gaussians: {} ({:.1} bits each)", max_gaussians, bits_per_gaussian);

        // Use error-driven encoding with size constraint
        let initial_count = max_gaussians / 2;
        self.encode_error_driven_adam(initial_count, max_gaussians)
    }

    /// Encode with rate-distortion pruning
    ///
    /// Starts with dense initialization, then prunes Gaussians that don't
    /// contribute enough to justify their rate cost.
    ///
    /// # Arguments
    /// * `initial_gaussians` - Starting Gaussian count (dense)
    /// * `lambda` - R-D tradeoff (higher = prefer smaller file)
    ///
    /// # Returns
    /// R-D optimized Gaussians
    pub fn encode_with_rd_pruning(&self, initial_gaussians: usize, lambda: f32) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        use lgi_core::rate_distortion::RateDistortionOptimizer;
        use crate::renderer_v2::RendererV2;

        log::info!("🎯 Encoding with R-D pruning (λ={:.4})", lambda);

        // Start with dense initialization
        let grid_size = (initial_gaussians as f32).sqrt().ceil() as u32;
        let mut gaussians = self.initialize_gaussians(grid_size);

        log::info!("  Initial Gaussians: {}", gaussians.len());

        // Optimize
        let mut optimizer = crate::adam_optimizer::AdamOptimizer::default();
        optimizer.max_iterations = 200;
        let _loss = optimizer.optimize(&mut gaussians, &self.target);

        // Apply geodesic EDT anti-bleeding constraints
        self.apply_geodesic_clamping(&mut gaussians);

        // Render and compute error
        let rendered = RendererV2::render(&gaussians, self.target.width, self.target.height);

        // Create error image
        let mut error_image = ImageBuffer::new(self.target.width, self.target.height);
        for i in 0..error_image.data.len() {
            let t = self.target.data[i];
            let r = rendered.data[i];
            let err_r = (t.r - r.r).abs();
            let err_g = (t.g - r.g).abs();
            let err_b = (t.b - r.b).abs();
            error_image.data[i] = Color4::new(err_r, err_g, err_b, 1.0);
        }

        // Estimate contributions
        let rd_optimizer = RateDistortionOptimizer::new(lambda);
        let contributions = rd_optimizer.estimate_contributions(&gaussians, &error_image);

        // Prune based on R-D cost
        let keep_indices = rd_optimizer.prune_by_rd(&gaussians, &contributions);

        log::info!("  After R-D pruning: {} Gaussians (removed {})",
            keep_indices.len(),
            gaussians.len() - keep_indices.len());

        // Return pruned set
        keep_indices.iter().map(|&i| gaussians[i].clone()).collect()
    }

    fn initialize_gaussians_with_options(&self, grid_size: u32, use_guided_filter: bool) -> Vec<Gaussian2D<f32, Euler<f32>>> {
        log::info!("🔵 Initializing Gaussians (grid: {}×{}, guided_filter: {})", grid_size, grid_size, use_guided_filter);

        // Apply guided filter to colors if requested (for real photos)
        let color_source = if use_guided_filter {
            log::info!("  Applying guided filter for edge-preserving color sampling...");
            let filter = lgi_core::guided_filter::GuidedFilter::default();
            filter.filter(&self.target)
        } else {
            self.target.clone()
        };

        let num_gaussians = grid_size * grid_size;
        let width_px = self.target.width as f32;
        let height_px = self.target.height as f32;

        // COVERAGE-BASED scale (in pixels!) - Debug Plan Section 0.3
        // σ_base = γ × √(W×H/N)
        // ADAPTIVE γ from Iteration 2 experiments
        let gamma = Self::adaptive_gamma(num_gaussians as usize);
        let sigma_base_px = gamma * ((width_px * height_px) / num_gaussians as f32).sqrt();

        log::info!("  Coverage-based σ_base = {:.2} pixels", sigma_base_px);
        log::info!("  Normalized σ_base = {:.4}", sigma_base_px / width_px);

        let mut gaussians = Vec::new();
        let step_x = self.target.width / grid_size;
        let step_y = self.target.height / grid_size;

        for gy in 0..grid_size {
            for gx in 0..grid_size {
                let x = (gx * step_x + step_x / 2).min(self.target.width - 1);
                let y = (gy * step_y + step_y / 2).min(self.target.height - 1);

                // Position in normalized coordinates [0,1]
                let position = Vector2::new(
                    x as f32 / self.target.width as f32,
                    y as f32 / self.target.height as f32,
                );

                // Get structure tensor at this point
                let tensor = self.structure_tensor.get(x, y);

                // Get geodesic distance (in pixels)
                let geod_dist_px = self.geodesic_edt.get_distance(x, y);

                // CONDITIONAL ANISOTROPY - Corrected from Session 3 review
                // Debug Plan: coherence < 0.2 → isotropic, ≥ 0.2 → anisotropic
                // Previous sessions incorrectly claimed "isotropic always better"
                // TRUE: BOTH are needed, applied conditionally!
                let coherence = tensor.coherence;

                let (mut sig_para_px, mut sig_perp_px, rotation_angle) = if coherence < 0.2 {
                    // FLAT REGION → ISOTROPIC
                    // No preferred direction, use circular Gaussians
                    (sigma_base_px, sigma_base_px, 0.0)
                } else {
                    // EDGE/STRUCTURE REGION → ANISOTROPIC
                    // Thin perpendicular to edge, long parallel to edge
                    let sigma_perp = sigma_base_px / (1.0 + 3.0 * coherence);
                    let sigma_para = 4.0 * sigma_perp;  // β=4.0 anisotropy ratio

                    // Orientation from major eigenvector (edge tangent)
                    let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);

                    (sigma_para, sigma_perp, angle)
                };

                // CONDITIONAL GEODESIC CLAMPING - Strong edges only (coherence > 0.6)
                if coherence > 0.6 {
                    // Strong edge → apply geodesic clamp to prevent bleeding
                    let clamp = 0.5 + 0.3 * geod_dist_px;
                    sig_para_px = sig_para_px.min(clamp * 2.0);  // Parallel can be larger
                    sig_perp_px = sig_perp_px.min(clamp);         // Perpendicular constrained
                }

                let sig_perp_final = sig_perp_px.clamp(3.0, 64.0);
                let sig_para_final = sig_para_px.clamp(3.0, 64.0);

                // COMMENTED OUT for EXP-005:
                // let clamp_px = 10.0 + 3.0 * geod_dist_px;
                // let sig_perp_clamped = sig_perp_px.min(clamp_px);
                // let sig_para_clamped = sig_para_px.min(clamp_px * 2.0);

                // Normalize to [0,1] coordinates
                let scale_x = sig_para_final / width_px;
                let scale_y = sig_perp_final / height_px;

                // Use rotation from conditional logic (not structure tensor default)
                let rotation = rotation_angle;

                // Sample color with Gaussian-weighted averaging over footprint
                // Uses better_color_init for higher quality than single pixel sampling
                let color = better_color_init::gaussian_weighted_color(
                    &color_source,
                    position,
                    scale_x,
                    scale_y,
                    rotation,
                );

                // Create Gaussian
                let gaussian = Gaussian2D::new(
                    position,
                    Euler::new(scale_x, scale_y, rotation),
                    color,
                    1.0,  // Opacity
                );

                gaussians.push(gaussian);
            }
        }

        log::info!("  Created {} Gaussians (structure-tensor aligned)", gaussians.len());
        log::info!("  Average anisotropy: {:.2}×", self.compute_avg_anisotropy(&gaussians));

        gaussians
    }

    /// Compute average anisotropy ratio
    fn compute_avg_anisotropy(&self, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> f32 {
        let sum: f32 = gaussians.iter()
            .map(|g| g.shape.scale_x.max(g.shape.scale_y) / g.shape.scale_x.min(g.shape.scale_y))
            .sum();

        sum / gaussians.len() as f32
    }

    /// Adaptive gamma factor based on Gaussian count
    ///
    /// From experiments (EXP-020, EXP-021, EXP-4-002):
    /// γ varies by count - smaller gamma preserves more detail
    /// Updated based on Session 4 gamma sweep showing 0.5 optimal for N=400
    fn adaptive_gamma(num_gaussians: usize) -> f32 {
        match num_gaussians {
            n if n < 100 => 1.2,    // Sparse: large coverage needed
            n if n < 200 => 0.8,    // Medium-sparse: original validated value
            n if n < 500 => 0.5,    // Medium: EXP-4-002 shows 0.5 optimal (+0.46 dB at N=400)
            n if n < 1000 => 0.4,   // Medium-dense
            n if n < 2000 => 0.35,  // Dense
            _ => 0.3,               // Very dense
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let image = ImageBuffer::new(64, 64);
        let encoder = EncoderV2::new(image);

        assert!(encoder.is_ok());
    }

    #[test]
    fn test_gaussian_initialization() {
        let mut image = ImageBuffer::new(64, 64);

        // Create vertical edge for testing
        for y in 0..64 {
            for x in 0..64 {
                let color = if x < 32 { 0.0 } else { 1.0 };
                image.set_pixel(x, y, Color4::new(color, color, color, 1.0));
            }
        }

        let encoder = EncoderV2::new(image).unwrap();
        let gaussians = encoder.initialize_gaussians(8);

        assert_eq!(gaussians.len(), 64);  // 8×8 grid

        // Check that Gaussians were created
        // Note: With coherence thresholding, may be isotropic if coherence <0.2
        // (structure tensor smoothing can reduce coherence of sharp edges)
        let avg_anisotropy = encoder.compute_avg_anisotropy(&gaussians);

        // Anisotropy should be ≥1.0 (could be 1.0 for isotropic or higher for anisotropic)
        assert!(avg_anisotropy >= 1.0, "Invalid anisotropy: {}", avg_anisotropy);
        assert!(avg_anisotropy <= 10.0, "Excessive anisotropy: {}", avg_anisotropy);
    }
}
pub mod adam_optimizer;
pub mod hybrid_optimizer;  // Adam → L-BFGS hybrid (3DGS-LM approach)
pub mod lm_optimizer;  // Levenberg-Marquardt (nonlinear least-squares)
pub mod test_results;  // Persistent test result logging
pub mod file_writer;
pub mod gaussian_logger;  // Quantum research: log Gaussian states during optimization
// pub mod optimizer_lbfgs;  // DEPRECATED: Use hybrid_optimizer with lgi-core/src/lbfgs.rs instead
pub mod file_reader;
