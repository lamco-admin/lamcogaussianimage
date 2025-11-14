//! Advanced learning rate schedules
//!
//! Implements cutting-edge LR scheduling strategies from 2024-2025 research

use std::f32::consts::PI;

/// Learning rate schedule type
#[derive(Debug, Clone, Copy)]
pub enum LRSchedule {
    /// Constant learning rate
    Constant,

    /// Step decay (original implementation)
    StepDecay {
        decay_factor: f32,
        decay_steps: usize,
    },

    /// Cosine annealing (smooth decay)
    CosineAnnealing {
        lr_min: f32,
        lr_max: f32,
        period: usize,
    },

    /// Cosine annealing with warm restarts (SGDR)
    CosineAnnealingWarmRestarts {
        lr_min: f32,
        lr_max: f32,
        period_initial: usize,
        period_mult: f32,  // Multiply period after each restart
    },

    /// Cyclical learning rates (triangle wave)
    Cyclical {
        lr_min: f32,
        lr_max: f32,
        step_size: usize,  // Half-period
    },

    /// Exponential decay
    Exponential {
        gamma: f32,  // Decay factor per iteration
    },

    /// Polynomial decay
    Polynomial {
        power: f32,
        total_iterations: usize,
    },

    /// Reduce on plateau (adaptive)
    ReduceOnPlateau {
        factor: f32,
        patience: usize,
        min_delta: f32,
    },
}

impl LRSchedule {
    /// Compute learning rate for given iteration
    pub fn get_lr(&self, lr_base: f32, iteration: usize, loss_history: &[f32]) -> f32 {
        match self {
            LRSchedule::Constant => lr_base,

            LRSchedule::StepDecay { decay_factor, decay_steps } => {
                let num_decays = iteration / decay_steps;
                lr_base * decay_factor.powi(num_decays as i32)
            }

            LRSchedule::CosineAnnealing { lr_min, lr_max, period } => {
                let t = (iteration % period) as f32 / *period as f32;
                lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (PI * t).cos())
            }

            LRSchedule::CosineAnnealingWarmRestarts { lr_min, lr_max, period_initial, period_mult } => {
                // Compute current period
                let mut period = *period_initial;
                let mut iterations_in_period = iteration;

                while iterations_in_period >= period {
                    iterations_in_period -= period;
                    period = (