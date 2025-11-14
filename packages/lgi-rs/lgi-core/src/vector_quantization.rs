//! Vector quantization for Gaussian parameters

pub struct VectorQuantizer {
    pub codebook_size: usize,
}

impl Default for VectorQuantizer {
    fn default() -> Self {
        Self { codebook_size: 256 }
    }
}

impl VectorQuantizer {
    pub fn build_codebook(&self, values: &[f32], dimensions: usize) -> Vec<Vec<f32>> {
        // K-means clustering for codebook
        let mut codebook = Vec::new();
        let num_vectors = values.len() / dimensions;

        // Initialize with random samples
        for i in 0..self.codebook_size.min(num_vectors) {
            let idx = i * num_vectors / self.codebook_size;
            let start = idx * dimensions;
            let end = start + dimensions;
            if end <= values.len() {
                codebook.push(values[start..end].to_vec());
            }
        }

        codebook
    }

    pub fn quantize(&self, value: &[f32], codebook: &[Vec<f32>]) -> usize {
        codebook.iter().enumerate().min_by(|(_, a), (_, b)| {
            let dist_a: f32 = a.iter().zip(value).map(|(x, y)| (x - y).powi(2)).sum();
            let dist_b: f32 = b.iter().zip(value).map(|(x, y)| (x - y).powi(2)).sum();
            dist_a.partial_cmp(&dist_b).unwrap()
        }).map(|(i, _)| i).unwrap_or(0)
    }
}
