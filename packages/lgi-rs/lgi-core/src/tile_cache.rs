//! Tile caching for panning performance

use std::collections::HashMap;
use crate::ImageBuffer;

pub struct TileCache {
    tiles: HashMap<(u32, u32), ImageBuffer<f32>>,
    tile_width: u32,
    tile_height: u32,
    max_tiles: usize,
}

impl TileCache {
    pub fn new(tile_width: u32, tile_height: u32, max_tiles: usize) -> Self {
        Self {
            tiles: HashMap::new(),
            tile_width,
            tile_height,
            max_tiles,
        }
    }

    pub fn get(&self, tile_x: u32, tile_y: u32) -> Option<&ImageBuffer<f32>> {
        self.tiles.get(&(tile_x, tile_y))
    }

    pub fn insert(&mut self, tile_x: u32, tile_y: u32, tile: ImageBuffer<f32>) {
        if self.tiles.len() >= self.max_tiles {
            // Simple LRU: remove first
            if let Some(&key) = self.tiles.keys().next() {
                self.tiles.remove(&key);
            }
        }
        self.tiles.insert((tile_x, tile_y), tile);
    }

    pub fn clear(&mut self) {
        self.tiles.clear();
    }
}
