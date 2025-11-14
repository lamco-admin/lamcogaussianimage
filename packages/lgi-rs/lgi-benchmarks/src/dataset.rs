//! Standard dataset integration

use lgi_core::ImageBuffer;
use std::path::{Path, PathBuf};

/// Standard test dataset
#[derive(Debug, Clone, Copy)]
pub enum Dataset {
    /// Kodak PhotoCD dataset (24 images, 768×512)
    Kodak,
    /// DIV2K dataset (800 2K images)
    DIV2K,
    /// Custom directory
    Custom,
}

/// Dataset manager
pub struct DatasetManager {
    dataset_dir: PathBuf,
}

impl DatasetManager {
    /// Create new dataset manager
    pub fn new<P: AsRef<Path>>(dataset_dir: P) -> Self {
        Self {
            dataset_dir: dataset_dir.as_ref().to_path_buf(),
        }
    }

    /// Load all images from dataset
    pub fn load_images(&self, dataset: Dataset) -> Vec<(PathBuf, ImageBuffer<f32>)> {
        let dir = match dataset {
            Dataset::Kodak => self.dataset_dir.join("kodak"),
            Dataset::DIV2K => self.dataset_dir.join("div2k"),
            Dataset::Custom => self.dataset_dir.clone(),
        };

        let mut images = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();

                if let Some(ext) = path.extension() {
                    if ext == "png" || ext == "jpg" || ext == "jpeg" {
                        if let Ok(img) = ImageBuffer::load(&path) {
                            images.push((path, img));
                        }
                    }
                }
            }
        }

        images
    }

    /// Get number of images in dataset
    pub fn count(&self, dataset: Dataset) -> usize {
        self.load_images(dataset).len()
    }
}

/// Download Kodak dataset
pub fn download_kodak_dataset<P: AsRef<Path>>(output_dir: P) -> std::io::Result<()> {
    let _urls = vec![
        "http://r0k.us/graphics/kodak/kodak/kodim01.png",
        "http://r0k.us/graphics/kodak/kodak/kodim02.png",
        "http://r0k.us/graphics/kodak/kodak/kodim03.png",
        "http://r0k.us/graphics/kodak/kodak/kodim04.png",
        "http://r0k.us/graphics/kodak/kodak/kodim05.png",
        "http://r0k.us/graphics/kodak/kodak/kodim06.png",
        "http://r0k.us/graphics/kodak/kodak/kodim07.png",
        "http://r0k.us/graphics/kodak/kodak/kodim08.png",
        "http://r0k.us/graphics/kodak/kodak/kodim09.png",
        "http://r0k.us/graphics/kodak/kodak/kodim10.png",
        // ... (would include all 24)
    ];

    let output_path = output_dir.as_ref();
    std::fs::create_dir_all(output_path)?;

    println!("Note: Kodak dataset download URLs listed.");
    println!("For actual download, use a tool like wget or curl.");
    println!("Images available at: http://r0k.us/graphics/kodak/");

    Ok(())
}
