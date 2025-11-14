//! Progressive Loading System
//! 3-phase loading: preview → usable → full quality

use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};
use crate::{lod_system::LODSystem, ImageBuffer};

/// Progressive loading phases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadPhase {
    Preview = 0,   // Band 0 only (~60% quality, fast)
    Usable = 1,    // Bands 0+1 (~85% quality)
    Full = 2,      // All bands (100% quality)
}

/// Progressive loader
pub struct ProgressiveLoader {
    lod: LODSystem,
}

impl ProgressiveLoader {
    pub fn new(gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> Self {
        Self {
            lod: LODSystem::classify(gaussians),
        }
    }

    /// Get Gaussians for loading phase
    pub fn get_gaussians_for_phase(&self, phase: LoadPhase) -> Vec<&Gaussian2D<f32, Euler<f32>>> {
        use crate::lod_system::LODBand;

        match phase {
            LoadPhase::Preview => self.lod.get_band(LODBand::Coarse).iter().collect(),
            LoadPhase::Usable => {
                let mut gs: Vec<&Gaussian2D<f32, Euler<f32>>> = Vec::new();
                gs.extend(self.lod.get_band(LODBand::Coarse).iter());
                gs.extend(self.lod.get_band(LODBand::Medium).iter());
                gs
            }
            LoadPhase::Full => {
                let mut gs: Vec<&Gaussian2D<f32, Euler<f32>>> = Vec::new();
                gs.extend(self.lod.get_band(LODBand::Coarse).iter());
                gs.extend(self.lod.get_band(LODBand::Medium).iter());
                gs.extend(self.lod.get_band(LODBand::Fine).iter());
                gs
            }
        }
    }

    /// Get expected quality for phase
    pub fn expected_quality(&self, phase: LoadPhase) -> f32 {
        match phase {
            LoadPhase::Preview => 0.6,
            LoadPhase::Usable => 0.85,
            LoadPhase::Full => 1.0,
        }
    }
}
