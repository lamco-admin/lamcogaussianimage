//! Compression utilities (zstd, delta coding)

pub fn delta_encode_positions(positions: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut deltas = Vec::with_capacity(positions.len());
    if positions.is_empty() { return deltas; }

    deltas.push(positions[0]);
    for i in 1..positions.len() {
        deltas.push((
            positions[i].0 - positions[i-1].0,
            positions[i].1 - positions[i-1].1,
        ));
    }
    deltas
}

pub fn delta_decode_positions(deltas: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut positions = Vec::with_capacity(deltas.len());
    if deltas.is_empty() { return positions; }

    positions.push(deltas[0]);
    for i in 1..deltas.len() {
        positions.push((
            positions[i-1].0 + deltas[i].0,
            positions[i-1].1 + deltas[i].1,
        ));
    }
    positions
}
