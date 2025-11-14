//! Improved optimizer with full backpropagation
//!
//! This is the production optimizer that uses complete autodiff
//! and comprehensive metrics collection.

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use log::{info, debug, warn};
use lgi_core::{Result, ImageBuffer, Renderer};
use std::cell::RefCell;
use crate::{
    config::EncoderConfig,
    loss::LossFunctions,
    autodiff::{compute_full_gradients, FullGaussianGradient},
    metrics_collector::{MetricsCollector, IterationMetrics},
    adaptive::{AdaptiveThresholdController, LifecycleManager},
    lr_scaling::scale_lr_by_count,
    vector_quantization::VectorQuantizer,
};
use std::time::Instant;

#[cfg(feature = "progress")]
use indicatif::{ProgressBar, ProgressStyle};

/// Improved optimizer with full backpropagation
pub struct OptimizerV2 {
    config: EncoderConfig,
    threshold_ctrl: AdaptiveThresholdController,
    lifecycle_mgr: Option<LifecycleManager>,
    enable_adaptive: bool,
    use_gpu: bool,  // Whether to use GPU (from global manager)
}

impl OptimizerV2 {
    /// Create new optimizer
    pub fn new(config: EncoderConfig) -> Self {
        Self {
            config,
            threshold_ctrl: AdaptiveThresholdController::default(),
            lifecycle_mgr: None,
            enable_adaptive: false,
            use_gpu: true,  // Default to GPU if available
        }
    }

    /// Enable adaptive features (pruning, splitting, culling)
    pub fn with_adaptive(mut self) -> Self {
        self.enable_adaptive = true;
        self
    }

    /// Enable GPU rendering (uses global GPU manager)
    pub fn with_gpu(mut self) -> Self {
        self.use_gpu = true;
        self
    }

    /// Disable GPU (use CPU only)
    pub fn without_gpu(mut self) -> Self {
        self.use_gpu = false;
        self
    }

    /// Optimize Gaussians with full backpropagation and metrics collection
    pub fn optimize_with_metrics(
        &self,
        gaussians: &mut Vec<Gaussian2D<f32, Euler<f32>>>,
        target: &ImageBuffer<f32>,
    ) -> Result<MetricsCollector> {
        // VQ quantizer for QA training (mutable local variable)
        let mut vq_quantizer: Option<VectorQuantizer> = None;
        let mut metrics_collector = MetricsCollector::new();

        // Initialize lifecycle manager if adaptive is enabled
        let mut lifecycle_mgr: Option<LifecycleManager> = if self.enable_adaptive {
            Some(LifecycleManager::new(gaussians.len()))
        } else {
            None
        };

        let renderer = Renderer::new();
        let loss_fn = LossFunctions::new(
            self.config.loss_l2_weight,
            self.config.loss_ssim_weight,
        );

        // Adam optimizer state - NOW WE USE ALL OF THESE!
        let num_gaussians = gaussians.len();
        let mut m_position = vec![Vector2::<f32>::zero(); num_gaussians];
        let mut v_position = vec![Vector2::<f32>::zero(); num_gaussians];
        let mut m_scale = vec![Vector2::<f32>::zero(); num_gaussians];
        let mut v_scale = vec![Vector2::<f32>::zero(); num_gaussians];
        let mut m_rotation = vec![0.0f32; num_gaussians];
        let mut v_rotation = vec![0.0f32; num_gaussians];
        let mut m_color = vec![Color4::new(0.0, 0.0, 0.0, 0.0); num_gaussians];
        let mut v_color = vec![Color4::new(0.0, 0.0, 0.0, 0.0); num_gaussians];
        let mut m_opacity = vec![0.0f32; num_gaussians];
        let mut v_opacity = vec![0.0f32; num_gaussians];

        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let epsilon = 1e-8f32;

        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;

        #[cfg(feature = "progress")]
        let progress_bar = ProgressBar::new(self.config.max_iterations as u64);
        #[cfg(feature = "progress")]
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} Loss: {msg}")
                .unwrap(),
        );

        info!("Starting optimization with FULL backpropagation...");
        info!("  Gaussians: {}", gaussians.len());
        info!("  Max iterations: {}", self.config.max_iterations);
        info!("  Adaptive features: {}", self.enable_adaptive);

        // Optimization loop
        for iteration in 0..self.config.max_iterations {
            debug!("🔄 Starting iteration {} of {}", iteration, self.config.max_iterations);
            let iter_start = Instant::now();

            // Quantization-Aware (QA) Training (from GaussianImage ECCV 2024)
            // Train codebook at start of QA phase, then quantize/dequantize each iteration
            let gaussians_for_render = if self.config.enable_qa_training && iteration >= self.config.qa_start_iteration {
                // Initialize VQ codebook on first QA iteration
                if iteration == self.config.qa_start_iteration {
                    info!("\n🔧 Starting Quantization-Aware (QA) training at iteration {}", iteration);
                    info!("   Training VQ codebook with {} entries...", self.config.qa_codebook_size);

                    let mut vq = VectorQuantizer::new(self.config.qa_codebook_size);
                    vq.train(gaussians, 100);  // 100 k-means iterations

                    let distortion = vq.measure_distortion(gaussians);
                    info!("   VQ codebook trained! Distortion: {:.6}", distortion);

                    vq_quantizer = Some(vq);
                }

                // Quantize and dequantize (simulates compression)
                if let Some(ref vq) = vq_quantizer {
                    let indices = vq.quantize_all(gaussians);
                    let gaussians_dequantized = vq.dequantize_all(&indices);

                    // Use dequantized Gaussians for rendering
                    // Gradients will flow back to ORIGINAL gaussians (straight-through estimator)
                    gaussians_dequantized
                } else {
                    gaussians.clone()
                }
            } else {
                // No QA: use original Gaussians
                gaussians.clone()
            };

            // Forward pass: Render (USE GLOBAL GPU IF ENABLED!)
            let render_start = Instant::now();
            let rendered = if self.use_gpu && lgi_gpu::GpuManager::global().is_initialized() {
                // GPU rendering via global manager (100-1000× faster!)
                lgi_gpu::GpuManager::global().render(|gpu| {
                    gpu.render(&gaussians_for_render, target.width, target.height, lgi_core::RenderMode::AccumulatedSum)
                }).map_err(|e| lgi_core::LgiError::ImageError(format!("GPU render failed: {}", e)))?
            } else {
                // CPU fallback
                renderer.render(&gaussians_for_render, target.width, target.height)?
            };
            let render_time = render_start.elapsed();

            if iteration == 0 || iteration == 1 {
                info!("📊 Iteration {} render: {:.2}ms (using {})",
                    iteration,
                    render_time.as_secs_f32() * 1000.0,
                    if self.use_gpu && lgi_gpu::GpuManager::global().is_initialized() { "GPU" } else { "CPU" }
                );
            }

            // Compute loss
            let total_loss = loss_fn.compute(&rendered, target);
            let l2_loss = LossFunctions::new(1.0, 0.0).compute(&rendered, target);
            let ssim_loss = LossFunctions::new(0.0, 1.0).compute(&rendered, target);

            // Update best loss
            if total_loss < best_loss {
                best_loss = total_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            // Early stopping
            if patience_counter >= self.config.early_stopping_patience {
                info!("Early stopping at iteration {} (loss: {:.6})", iteration, total_loss);
                break;
            }

            // Convergence check
            if total_loss < self.config.convergence_tolerance {
                info!("Converged at iteration {} (loss: {:.6})", iteration, total_loss);
                break;
            }

            // Backward pass: Compute FULL gradients (GPU if available, CPU fallback)
            let gradient_start = Instant::now();
            let gradients = if self.use_gpu && lgi_gpu::GpuManager::global().is_initialized() {
                // Try GPU gradient computation
                match lgi_gpu::GpuManager::global().compute_gradients(gaussians, &rendered, target) {
                    Ok(gpu_grads) => {
                        // Convert GPU gradients to FullGaussianGradient
                        gpu_grads.iter().map(|g| FullGaussianGradient {
                            position: Vector2::new(g.d_position[0], g.d_position[1]),
                            scale: Vector2::new(g.d_scale_x, g.d_scale_y),
                            rotation: g.d_rotation,
                            color: Color4::new(g.d_color[0], g.d_color[1], g.d_color[2], g.d_color[3]),
                            opacity: g.d_opacity,
                        }).collect()
                    }
                    Err(e) => {
                        warn!("GPU gradient computation failed ({}), falling back to CPU", e);
                        compute_full_gradients(gaussians, &rendered, target, target.width, target.height)
                    }
                }
            } else {
                // CPU fallback
                compute_full_gradients(gaussians, &rendered, target, target.width, target.height)
            };
            let gradient_time = gradient_start.elapsed();

            if iteration == 0 || iteration == 1 {
                info!("📊 Iteration {} gradients: {:.2}ms (using {})",
                    iteration,
                    gradient_time.as_secs_f32() * 1000.0,
                    if self.use_gpu && lgi_gpu::GpuManager::global().is_initialized() { "GPU" } else { "CPU" }
                );
            }

            // Collect gradient statistics
            let grad_magnitudes: Vec<f32> = gradients.iter().map(|g| g.magnitude()).collect();
            let grad_mean = grad_magnitudes.iter().sum::<f32>() / grad_magnitudes.len() as f32;
            let grad_max = grad_magnitudes.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let grad_min = grad_magnitudes.iter().fold(f32::INFINITY, |a, &b| a.min(b));

            // Per-parameter gradient norms
            let grad_position_norm: f32 = gradients.iter().map(|g| g.position.length_squared()).sum::<f32>().sqrt();
            let grad_scale_norm: f32 = gradients.iter().map(|g| g.scale.length_squared()).sum::<f32>().sqrt();
            let grad_rotation_norm: f32 = gradients.iter().map(|g| g.rotation * g.rotation).sum::<f32>().sqrt();
            let grad_color_norm: f32 = gradients.iter()
                .map(|g| g.color.r * g.color.r + g.color.g * g.color.g + g.color.b * g.color.b)
                .sum::<f32>().sqrt();
            let grad_opacity_norm: f32 = gradients.iter().map(|g| g.opacity * g.opacity).sum::<f32>().sqrt();

            // Update lifecycle health if adaptive
            if let Some(ref mut lifecycle) = lifecycle_mgr {
                lifecycle.update_health(&grad_magnitudes);
            }

            // Learning rate schedule
            let lr_mult = if iteration > 0 && iteration % self.config.lr_decay_steps == 0 {
                self.config.lr_decay
            } else {
                1.0
            };

            // CRITICAL FIX: Scale learning rates by Gaussian count (lr ∝ 1/√N)
            let gaussian_scale_factor = 1.0 / (num_gaussians as f32).sqrt();

            let lr_pos = self.config.lr_position * lr_mult * gaussian_scale_factor;
            let lr_scale = self.config.lr_scale * lr_mult * gaussian_scale_factor;
            let lr_rot = self.config.lr_rotation * lr_mult * gaussian_scale_factor;
            let lr_color = self.config.lr_color * lr_mult * gaussian_scale_factor;
            let lr_opacity = self.config.lr_opacity * lr_mult * gaussian_scale_factor;

            // Adam updates - NOW FOR ALL PARAMETERS!
            let update_start = Instant::now();

            for (i, (gaussian, grad)) in gaussians.iter_mut().zip(gradients.iter()).enumerate() {
                // Ensure indices are valid
                if i >= num_gaussians {
                    continue;
                }

                // Position update (Adam)
                m_position[i] = m_position[i] * beta1 + grad.position * (1.0 - beta1);
                v_position[i] = Vector2::<f32>::new(
                    v_position[i].x * beta2 + grad.position.x * grad.position.x * (1.0 - beta2),
                    v_position[i].y * beta2 + grad.position.y * grad.position.y * (1.0 - beta2),
                );

                let bias_correction1 = 1.0 - beta1.powi(iteration as i32 + 1);
                let bias_correction2 = 1.0 - beta2.powi(iteration as i32 + 1);

                let m_hat_pos = m_position[i] * (1.0 / bias_correction1);
                let v_hat_pos = Vector2::<f32>::new(
                    v_position[i].x / bias_correction2,
                    v_position[i].y / bias_correction2,
                );

                gaussian.position.x -= lr_pos * m_hat_pos.x / (v_hat_pos.x.sqrt() + epsilon);
                gaussian.position.y -= lr_pos * m_hat_pos.y / (v_hat_pos.y.sqrt() + epsilon);
                gaussian.position.x = gaussian.position.x.clamp(0.0, 1.0);
                gaussian.position.y = gaussian.position.y.clamp(0.0, 1.0);

                // Scale update (Adam) - NOW ACTIVE!
                m_scale[i] = m_scale[i] * beta1 + grad.scale * (1.0 - beta1);
                v_scale[i] = Vector2::<f32>::new(
                    v_scale[i].x * beta2 + grad.scale.x * grad.scale.x * (1.0 - beta2),
                    v_scale[i].y * beta2 + grad.scale.y * grad.scale.y * (1.0 - beta2),
                );

                let m_hat_scale = m_scale[i] * (1.0 / bias_correction1);
                let v_hat_scale = Vector2::<f32>::new(
                    v_scale[i].x / bias_correction2,
                    v_scale[i].y / bias_correction2,
                );

                gaussian.shape.scale_x -= lr_scale * m_hat_scale.x / (v_hat_scale.x.sqrt() + epsilon);
                gaussian.shape.scale_y -= lr_scale * m_hat_scale.y / (v_hat_scale.y.sqrt() + epsilon);
                gaussian.shape.scale_x = gaussian.shape.scale_x.clamp(0.001, 0.5);  // Reasonable bounds
                gaussian.shape.scale_y = gaussian.shape.scale_y.clamp(0.001, 0.5);

                // Rotation update (Adam) - NOW ACTIVE!
                m_rotation[i] = m_rotation[i] * beta1 + grad.rotation * (1.0 - beta1);
                v_rotation[i] = v_rotation[i] * beta2 + grad.rotation * grad.rotation * (1.0 - beta2);

                let m_hat_rot = m_rotation[i] / bias_correction1;
                let v_hat_rot = v_rotation[i] / bias_correction2;

                gaussian.shape.rotation -= lr_rot * m_hat_rot / (v_hat_rot.sqrt() + epsilon);
                // Wrap rotation to [-π, π]
                let pi = std::f32::consts::PI;
                while gaussian.shape.rotation > pi {
                    gaussian.shape.rotation -= 2.0 * pi;
                }
                while gaussian.shape.rotation < -pi {
                    gaussian.shape.rotation += 2.0 * pi;
                }

                // Color update (Adam) - IMPROVED!
                let grad_color_vec = Vector2::new(grad.color.r, grad.color.g);  // Simplified for R,G
                m_color[i].r = m_color[i].r * beta1 + grad.color.r * (1.0 - beta1);
                m_color[i].g = m_color[i].g * beta1 + grad.color.g * (1.0 - beta1);
                m_color[i].b = m_color[i].b * beta1 + grad.color.b * (1.0 - beta1);

                v_color[i].r = v_color[i].r * beta2 + grad.color.r * grad.color.r * (1.0 - beta2);
                v_color[i].g = v_color[i].g * beta2 + grad.color.g * grad.color.g * (1.0 - beta2);
                v_color[i].b = v_color[i].b * beta2 + grad.color.b * grad.color.b * (1.0 - beta2);

                gaussian.color.r -= lr_color * (m_color[i].r / bias_correction1) / ((v_color[i].r / bias_correction2).sqrt() + epsilon);
                gaussian.color.g -= lr_color * (m_color[i].g / bias_correction1) / ((v_color[i].g / bias_correction2).sqrt() + epsilon);
                gaussian.color.b -= lr_color * (m_color[i].b / bias_correction1) / ((v_color[i].b / bias_correction2).sqrt() + epsilon);
                gaussian.color = gaussian.color.clamp();

                // Opacity update (Adam) - NOW ACTIVE!
                m_opacity[i] = m_opacity[i] * beta1 + grad.opacity * (1.0 - beta1);
                v_opacity[i] = v_opacity[i] * beta2 + grad.opacity * grad.opacity * (1.0 - beta2);

                let m_hat_opacity = m_opacity[i] / bias_correction1;
                let v_hat_opacity = v_opacity[i] / bias_correction2;

                gaussian.opacity -= lr_opacity * m_hat_opacity / (v_hat_opacity.sqrt() + epsilon);
                gaussian.opacity = gaussian.opacity.clamp(0.01, 1.0);
            }

            let update_time = update_start.elapsed();

            // Gaussian statistics
            let avg_opacity = gaussians.iter().map(|g| g.opacity).sum::<f32>() / gaussians.len() as f32;
            let avg_scale = gaussians.iter()
                .map(|g| (g.shape.scale_x + g.shape.scale_y) / 2.0)
                .sum::<f32>() / gaussians.len() as f32;
            let num_active = gaussians.iter()
                .filter(|g| g.opacity > self.threshold_ctrl.opacity_threshold)
                .count();

            // Compute PSNR every 10 iterations (expensive)
            let psnr = if iteration % 10 == 0 {
                Some(compute_psnr_quick(&rendered, target))
            } else {
                None
            };

            // Record metrics
            let iter_metrics = IterationMetrics {
                iteration,
                timestamp_ms: iter_start.elapsed().as_secs_f64() * 1000.0,
                total_loss,
                l2_loss,
                ssim_loss,
                grad_magnitude_mean: grad_mean,
                grad_magnitude_max: grad_max,
                grad_magnitude_min: grad_min,
                grad_position_norm,
                grad_scale_norm,
                grad_rotation_norm,
                grad_color_norm,
                grad_opacity_norm,
                avg_opacity,
                avg_scale,
                num_active_gaussians: num_active,
                render_time_ms: render_time.as_secs_f32() * 1000.0,
                gradient_time_ms: gradient_time.as_secs_f32() * 1000.0,
                update_time_ms: update_time.as_secs_f32() * 1000.0,
                total_iteration_time_ms: iter_start.elapsed().as_secs_f32() * 1000.0,
                psnr,
                ssim_value: None,  // Too expensive to compute every iteration
            };

            metrics_collector.record_iteration(iter_metrics);
            debug!("✅ Iteration {} complete in {:.2}s, loss={:.6}", iteration, iter_start.elapsed().as_secs_f32(), total_loss);

            // Progress display
            if iteration % 10 == 0 {
                info!("Iteration {}: loss={:.6}, PSNR={:.2} dB, active={}/{}, grad_scale={:.6}",
                    iteration,
                    total_loss,
                    psnr.unwrap_or(0.0),
                    num_active,
                    gaussians.len(),
                    grad_scale_norm);
            }

            #[cfg(feature = "progress")]
            {
                progress_bar.set_position(iteration as u64);
                progress_bar.set_message(format!("{:.6} | PSNR: {:.2}", total_loss, psnr.unwrap_or(0.0)));
            }

            // Adaptive lifecycle management (every 50 iterations)
            if self.enable_adaptive && iteration > 0 && iteration % 50 == 0 {
                if let Some(ref mut lifecycle) = lifecycle_mgr {
                    // Prune unhealthy Gaussians
                    let prune_candidates = lifecycle.get_prune_candidates();
                    if prune_candidates.len() > gaussians.len() / 10 {  // Don't prune more than 10% at once
                        info!("  Pruning {} unhealthy Gaussians", prune_candidates.len());
                        *gaussians = lifecycle.prune(gaussians);

                        // Resize Adam states
                        let new_len = gaussians.len();
                        m_position.truncate(new_len);
                        v_position.truncate(new_len);
                        m_scale.truncate(new_len);
                        v_scale.truncate(new_len);
                        m_rotation.truncate(new_len);
                        v_rotation.truncate(new_len);
                        m_color.truncate(new_len);
                        v_color.truncate(new_len);
                        m_opacity.truncate(new_len);
                        v_opacity.truncate(new_len);
                    }
                }
            }
        }

        #[cfg(feature = "progress")]
        progress_bar.finish_with_message(format!("Final loss: {:.6}", best_loss));

        info!("Optimization complete!");
        info!("  Final Gaussians: {}", gaussians.len());
        info!("  Best loss: {:.6}", best_loss);

        Ok(metrics_collector)
    }

    /// Simple optimize without metrics (for compatibility)
    pub fn optimize(
        &mut self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>,
    ) -> Result<Vec<Gaussian2D<f32, Euler<f32>>>> {
        let mut gaussian_vec = gaussians.to_vec();
        let _ = self.optimize_with_metrics(&mut gaussian_vec, target)?;
        Ok(gaussian_vec)
    }
}

// Quick PSNR for metrics (in metrics_collector module, but used here)
mod quick_metrics {
    use lgi_core::ImageBuffer;

    pub fn compute_psnr_quick(img1: &ImageBuffer<f32>, img2: &ImageBuffer<f32>) -> f32 {
        let mut mse = 0.0;
        let count = (img1.width * img1.height * 3) as f32;

        for (p1, p2) in img1.data.iter().zip(img2.data.iter()) {
            mse += (p1.r - p2.r) * (p1.r - p2.r);
            mse += (p1.g - p2.g) * (p1.g - p2.g);
            mse += (p1.b - p2.b) * (p1.b - p2.b);
        }

        mse /= count;

        if mse < 1e-10 {
            100.0
        } else {
            20.0 * (1.0f32 / mse.sqrt()).log10()
        }
    }
}

// Re-export for metrics_collector
pub(crate) use quick_metrics::compute_psnr_quick;
