//! Predictive coding for scales/rotations

pub struct PredictiveCoder;

impl PredictiveCoder {
    pub fn predict_scale(neighbors: &[(f32, f32)]) -> (f32, f32) {
        if neighbors.is_empty() {
            return (0.01, 0.01);
        }

        let avg_x: f32 = neighbors.iter().map(|(x, _)| x).sum::<f32>() / neighbors.len() as f32;
        let avg_y: f32 = neighbors.iter().map(|(_, y)| y).sum::<f32>() / neighbors.len() as f32;

        (avg_x, avg_y)
    }

    pub fn encode_residual(actual: (f32, f32), predicted: (f32, f32)) -> (f32, f32) {
        (actual.0 - predicted.0, actual.1 - predicted.1)
    }

    pub fn decode_residual(predicted: (f32, f32), residual: (f32, f32)) -> (f32, f32) {
        (predicted.0 + residual.0, predicted.1 + residual.1)
    }
}
