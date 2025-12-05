//! Gaussian data logging for quantum research
//!
//! Captures Gaussian configurations during optimization for analysis.
//! Logs: parameters, quality metrics, image context for quantum clustering.

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};
use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;

/// Snapshot of a single Gaussian at a specific optimization iteration
#[derive(Clone, Debug)]
pub struct GaussianSnapshot {
    // Gaussian parameters
    pub position_x: f32,
    pub position_y: f32,
    pub sigma_x: f32,        // scale_x (perpendicular)
    pub sigma_y: f32,        // scale_y (parallel)
    pub rotation: f32,       // radians
    pub alpha: f32,          // opacity (always 1.0 in current impl)
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,

    // Quality metrics
    pub iteration: u32,
    pub refinement_pass: u32,
    pub loss: f32,

    // Image context (computed from structure tensor if available)
    pub edge_coherence: f32,
    pub local_gradient: f32,

    // Metadata
    pub gaussian_id: usize,
    pub image_id: String,
}

/// Callback trait for logging Gaussian states during optimization
pub trait GaussianLogger {
    fn log_iteration(
        &mut self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        iteration: u32,
        loss: f32,
    );

    fn set_context(&mut self, image_id: String, refinement_pass: u32);

    fn flush(&mut self) -> std::io::Result<()>;
}

/// CSV-based Gaussian logger
pub struct CsvGaussianLogger {
    writer: Option<BufWriter<File>>,
    image_id: String,
    refinement_pass: u32,
    structure_tensor: Option<StructureTensorField>,
    target: Option<ImageBuffer<f32>>,
}

impl CsvGaussianLogger {
    /// Create new CSV logger
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write CSV header
        writeln!(writer, "image_id,refinement_pass,iteration,gaussian_id,\
                         position_x,position_y,sigma_x,sigma_y,rotation,alpha,\
                         color_r,color_g,color_b,loss,edge_coherence,local_gradient")?;

        Ok(Self {
            writer: Some(writer),
            image_id: String::from("unknown"),
            refinement_pass: 0,
            structure_tensor: None,
            target: None,
        })
    }

    /// Set structure tensor for context extraction
    pub fn set_structure_tensor(&mut self, tensor: StructureTensorField) {
        self.structure_tensor = Some(tensor);
    }

    /// Set target image for context extraction
    pub fn set_target(&mut self, target: ImageBuffer<f32>) {
        self.target = Some(target);
    }

    /// Extract image context at Gaussian position
    fn get_context(&self, gaussian: &Gaussian2D<f32, Euler<f32>>) -> (f32, f32) {
        if let (Some(ref tensor), Some(ref target)) = (&self.structure_tensor, &self.target) {
            // Convert normalized position to pixel coordinates
            let x = (gaussian.position.x * target.width as f32) as u32;
            let y = (gaussian.position.y * target.height as f32) as u32;

            // Clamp to image bounds
            let x = x.min(target.width - 1);
            let y = y.min(target.height - 1);

            // Get structure tensor at this location
            let st = tensor.get(x, y);
            let coherence = st.coherence;

            // Compute local gradient magnitude
            let gradient = (st.eigenvalue_major - st.eigenvalue_minor).abs();

            (coherence, gradient)
        } else {
            // No context available
            (0.0, 0.0)
        }
    }
}

impl GaussianLogger for CsvGaussianLogger {
    fn log_iteration(
        &mut self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        iteration: u32,
        loss: f32,
    ) {
        if let Some(ref mut writer) = self.writer {
            for (idx, gaussian) in gaussians.iter().enumerate() {
                // Extract context before borrowing writer mutably
                let (coherence, gradient) = if let (Some(ref tensor), Some(ref target)) = (&self.structure_tensor, &self.target) {
                    let x = (gaussian.position.x * target.width as f32) as u32;
                    let y = (gaussian.position.y * target.height as f32) as u32;
                    let x = x.min(target.width - 1);
                    let y = y.min(target.height - 1);
                    let st = tensor.get(x, y);
                    let coherence = st.coherence;
                    let gradient = (st.eigenvalue_major - st.eigenvalue_minor).abs();
                    (coherence, gradient)
                } else {
                    (0.0, 0.0)
                };

                let _ = writeln!(
                    writer,
                    "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.8},{:.6},{:.6}",
                    self.image_id,
                    self.refinement_pass,
                    iteration,
                    idx,
                    gaussian.position.x,
                    gaussian.position.y,
                    gaussian.shape.scale_x,
                    gaussian.shape.scale_y,
                    gaussian.shape.rotation,
                    gaussian.opacity,
                    gaussian.color.r,
                    gaussian.color.g,
                    gaussian.color.b,
                    loss,
                    coherence,
                    gradient,
                );
            }
        }
    }

    fn set_context(&mut self, image_id: String, refinement_pass: u32) {
        self.image_id = image_id;
        self.refinement_pass = refinement_pass;
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(ref mut writer) = self.writer {
            writer.flush()?;
        }
        Ok(())
    }
}

impl Drop for CsvGaussianLogger {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// In-memory Gaussian logger (for testing/analysis)
pub struct MemoryGaussianLogger {
    snapshots: Vec<GaussianSnapshot>,
    image_id: String,
    refinement_pass: u32,
}

impl MemoryGaussianLogger {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            image_id: String::from("unknown"),
            refinement_pass: 0,
        }
    }

    pub fn get_snapshots(&self) -> &[GaussianSnapshot] {
        &self.snapshots
    }

    pub fn into_snapshots(self) -> Vec<GaussianSnapshot> {
        self.snapshots
    }
}

impl GaussianLogger for MemoryGaussianLogger {
    fn log_iteration(
        &mut self,
        gaussians: &[Gaussian2D<f32, Euler<f32>>],
        iteration: u32,
        loss: f32,
    ) {
        for (idx, gaussian) in gaussians.iter().enumerate() {
            self.snapshots.push(GaussianSnapshot {
                position_x: gaussian.position.x,
                position_y: gaussian.position.y,
                sigma_x: gaussian.shape.scale_x,
                sigma_y: gaussian.shape.scale_y,
                rotation: gaussian.shape.rotation,
                alpha: gaussian.opacity,
                color_r: gaussian.color.r,
                color_g: gaussian.color.g,
                color_b: gaussian.color.b,
                iteration,
                refinement_pass: self.refinement_pass,
                loss,
                edge_coherence: 0.0,  // Not computed for memory logger
                local_gradient: 0.0,
                gaussian_id: idx,
                image_id: self.image_id.clone(),
            });
        }
    }

    fn set_context(&mut self, image_id: String, refinement_pass: u32) {
        self.image_id = image_id;
        self.refinement_pass = refinement_pass;
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())  // Nothing to flush for memory logger
    }
}

impl Default for MemoryGaussianLogger {
    fn default() -> Self {
        Self::new()
    }
}
