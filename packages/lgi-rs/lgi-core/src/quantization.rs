//! LGIQ quantization profiles

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LGIQProfile {
    Baseline,   // 11 bytes/Gaussian
    Standard,   // 13 bytes/Gaussian
    HighFidelity, // 18 bytes/Gaussian
    Extended,   // 36 bytes/Gaussian
}

pub struct QuantizedGaussian {
    pub position: [u16; 2],     // 16-bit
    pub scale: [u16; 2],        // Log-scale, 12 or 14-bit
    pub rotation: u16,          // 12 or 14-bit
    pub color: [u16; 3],        // 8 or 10-bit per channel
    pub opacity: u16,           // 8 or 10-bit
    pub profile: LGIQProfile,   // Profile type for correct dequantization
}

impl QuantizedGaussian {
    /// LGIQ-B: Baseline profile (11 bytes/Gaussian)
    /// - 16-bit position (2 bytes × 2)
    /// - 12-bit scales (2 bytes × 2, top 12 bits used)
    /// - 12-bit rotation (2 bytes, top 12 bits used)
    /// - 8-bit color (1 byte × 3)
    /// - 8-bit opacity (1 byte)
    pub fn quantize_baseline(position: (f32, f32), scale: (f32, f32), rotation: f32, color: (f32, f32, f32), opacity: f32) -> Self {
        // Log-scale quantization for scales (better precision at small values)
        // Scale range: [0.001, 0.5] → log range: [-6.9, -0.69]
        let scale_x_log = (scale.0.max(0.001).min(0.5)).ln();
        let scale_y_log = (scale.1.max(0.001).min(0.5)).ln();
        let scale_x_norm = (scale_x_log + 7.0) / 7.0;  // Normalize to [0,1]
        let scale_y_norm = (scale_y_log + 7.0) / 7.0;

        Self {
            position: [(position.0 * 65535.0) as u16, (position.1 * 65535.0) as u16],
            scale: [(scale_x_norm * 4095.0) as u16, (scale_y_norm * 4095.0) as u16],  // 12-bit log-scale
            rotation: ((rotation + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * 4095.0) as u16,
            color: [(color.0 * 255.0) as u16, (color.1 * 255.0) as u16, (color.2 * 255.0) as u16],  // 8-bit
            opacity: (opacity * 255.0) as u16,
            profile: LGIQProfile::Baseline,
        }
    }

    /// LGIQ-S: Standard profile (13 bytes/Gaussian)
    /// - 16-bit position (2 bytes × 2)
    /// - 12-bit scales (2 bytes × 2)
    /// - 12-bit rotation (2 bytes)
    /// - 10-bit color (2 bytes × 3)
    /// - 10-bit opacity (2 bytes)
    pub fn quantize_standard(position: (f32, f32), scale: (f32, f32), rotation: f32, color: (f32, f32, f32), opacity: f32) -> Self {
        let scale_x_log = (scale.0.max(0.001).min(0.5)).ln();
        let scale_y_log = (scale.1.max(0.001).min(0.5)).ln();
        let scale_x_norm = (scale_x_log + 7.0) / 7.0;
        let scale_y_norm = (scale_y_log + 7.0) / 7.0;

        Self {
            position: [(position.0 * 65535.0) as u16, (position.1 * 65535.0) as u16],
            scale: [(scale_x_norm * 4095.0) as u16, (scale_y_norm * 4095.0) as u16],  // 12-bit
            rotation: ((rotation + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * 4095.0) as u16,
            color: [(color.0 * 1023.0) as u16, (color.1 * 1023.0) as u16, (color.2 * 1023.0) as u16],  // 10-bit
            opacity: (opacity * 1023.0) as u16,  // 10-bit
            profile: LGIQProfile::Standard,
        }
    }

    /// LGIQ-H: High Fidelity profile (18 bytes/Gaussian)
    /// - 16-bit position (2 bytes × 2)
    /// - 14-bit scales (2 bytes × 2)
    /// - 14-bit rotation (2 bytes)
    /// - 10-bit color (2 bytes × 3)
    /// - 10-bit opacity (2 bytes)
    pub fn quantize_high_fidelity(position: (f32, f32), scale: (f32, f32), rotation: f32, color: (f32, f32, f32), opacity: f32) -> Self {
        let scale_x_log = (scale.0.max(0.001).min(0.5)).ln();
        let scale_y_log = (scale.1.max(0.001).min(0.5)).ln();
        let scale_x_norm = (scale_x_log + 7.0) / 7.0;
        let scale_y_norm = (scale_y_log + 7.0) / 7.0;

        Self {
            position: [(position.0 * 65535.0) as u16, (position.1 * 65535.0) as u16],
            scale: [(scale_x_norm * 16383.0) as u16, (scale_y_norm * 16383.0) as u16],  // 14-bit
            rotation: ((rotation + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * 16383.0) as u16,  // 14-bit
            color: [(color.0 * 1023.0) as u16, (color.1 * 1023.0) as u16, (color.2 * 1023.0) as u16],  // 10-bit
            opacity: (opacity * 1023.0) as u16,  // 10-bit
            profile: LGIQProfile::HighFidelity,
        }
    }

    /// Dequantize based on the profile type
    pub fn dequantize(&self) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32) {
        let px = self.position[0] as f32 / 65535.0;
        let py = self.position[1] as f32 / 65535.0;

        // Use correct bit depths based on profile
        let (scale_max, rotation_max, color_max, opacity_max) = match self.profile {
            LGIQProfile::Baseline => (4095.0f32, 4095.0, 255.0, 255.0),
            LGIQProfile::Standard => (4095.0, 4095.0, 1023.0, 1023.0),
            LGIQProfile::HighFidelity => (16383.0, 16383.0, 1023.0, 1023.0),
            LGIQProfile::Extended => (1.0, 1.0, 1.0, 1.0), // Not used for quantized
        };

        // Log-scale dequantization
        let scale_x_norm = self.scale[0] as f32 / scale_max;
        let scale_y_norm = self.scale[1] as f32 / scale_max;
        let sx = (scale_x_norm * 7.0 - 7.0).exp();
        let sy = (scale_y_norm * 7.0 - 7.0).exp();

        let rot = (self.rotation as f32 / rotation_max) * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
        let r = self.color[0] as f32 / color_max;
        let g = self.color[1] as f32 / color_max;
        let b = self.color[2] as f32 / color_max;
        let a = self.opacity as f32 / opacity_max;

        (px, py, sx, sy, rot, r, g, b, a)
    }
}
