//! Comprehensive profiling and debugging for LGI Viewer
//! Copyright (c) 2025 Lamco Development

use std::time::{Duration, Instant};
use std::collections::HashMap;
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct OperationTiming {
    pub name: String,
    pub duration: Duration,
    pub timestamp: Instant,
    pub memory_before: usize,
    pub memory_after: usize,
    pub gpu_memory: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct GaussianDebugInfo {
    pub index: usize,
    pub position: (f32, f32),
    pub scale: (f32, f32),
    pub rotation: f32,
    pub color: (f32, f32, f32, f32),
    pub opacity: f32,
    pub contribution: f32,  // How much this Gaussian contributes to image
}

pub struct Profiler {
    timings: Arc<Mutex<Vec<OperationTiming>>>,
    gaussian_stats: Arc<Mutex<HashMap<usize, GaussianDebugInfo>>>,
    system: sysinfo::System,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            timings: Arc::new(Mutex::new(Vec::new())),
            gaussian_stats: Arc::new(Mutex::new(HashMap::new())),
            system: sysinfo::System::new_all(),
        }
    }

    /// Start timing an operation
    pub fn start_operation(&self, name: &str) -> OperationTimer {
        tracing::info!(operation = name, "Starting operation");
        
        let memory_before = self.get_memory_usage();
        
        OperationTimer {
            name: name.to_string(),
            start: Instant::now(),
            memory_before,
            profiler: self.timings.clone(),
        }
    }

    /// Get current memory usage (approximate, no system refresh)
    pub fn get_memory_usage(&self) -> usize {
        // For now, return 0 - proper memory tracking requires mut
        // TODO: Use Arc<Mutex<System>> for interior mutability
        0
    }

    /// Get all timings
    pub fn get_timings(&self) -> Vec<OperationTiming> {
        self.timings.lock().clone()
    }

    /// Export profiling report
    pub fn export_report(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        use std::io::Write;
        
        let mut file = std::fs::File::create(path)?;
        
        writeln!(file, "LGI Viewer Profiling Report")?;
        writeln!(file, "============================")?;
        writeln!(file)?;
        
        writeln!(file, "Operation Timings:")?;
        writeln!(file, "-----------------")?;
        
        for timing in self.timings.lock().iter() {
            writeln!(file, "{:30} | {:8.2}ms | Mem: {} → {} ({:+} KB)",
                timing.name,
                timing.duration.as_secs_f32() * 1000.0,
                timing.memory_before / 1024,
                timing.memory_after / 1024,
                (timing.memory_after as i64 - timing.memory_before as i64) / 1024
            )?;
        }
        
        writeln!(file)?;
        writeln!(file, "Gaussian Statistics:")?;
        writeln!(file, "-------------------")?;
        writeln!(file, "Total Gaussians: {}", self.gaussian_stats.lock().len())?;
        
        Ok(())
    }

    /// Clear all profiling data
    pub fn clear(&self) {
        self.timings.lock().clear();
        self.gaussian_stats.lock().clear();
    }
}

pub struct OperationTimer {
    name: String,
    start: Instant,
    memory_before: usize,
    profiler: Arc<Mutex<Vec<OperationTiming>>>,
}

impl Drop for OperationTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        
        tracing::info!(
            operation = %self.name,
            duration_ms = duration.as_secs_f32() * 1000.0,
            "Operation complete"
        );

        // Store timing
        let timing = OperationTiming {
            name: self.name.clone(),
            duration,
            timestamp: self.start,
            memory_before: self.memory_before,
            memory_after: self.memory_before, // TODO: measure actual
            gpu_memory: None,
        };

        self.profiler.lock().push(timing);
    }
}

#[macro_export]
macro_rules! trace_operation {
    ($profiler:expr, $name:expr, $block:block) => {{
        #[cfg(feature = "tracing")]
        {
            let _timer = $profiler.start_operation($name);
            let result = $block;
            result
        }
        #[cfg(not(feature = "tracing"))]
        {
            $block
        }
    }};
}
