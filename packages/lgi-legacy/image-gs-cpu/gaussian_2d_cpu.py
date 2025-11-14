#!/usr/bin/env python3
"""
Simplified CPU-compatible implementation of Image-GS: Content-Adaptive Image Representation via 2D Gaussians
Based on the SIGGRAPH 2025 paper by Zhang, Li et al.

This implementation focuses on the core concepts without requiring CUDA or complex dependencies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Tuple, Optional
from tqdm import tqdm


class Gaussian2D:
    """Represents a single 2D Gaussian with position, scale, rotation, and color."""
    
    def __init__(self, position: torch.Tensor, scale: torch.Tensor, 
                 rotation: float, color: torch.Tensor, opacity: float = 1.0):
        self.position = position  # (x, y)
        self.scale = scale  # (sx, sy)
        self.rotation = rotation  # angle in radians
        self.color = color  # (r, g, b)
        self.opacity = opacity


class ImageGS(nn.Module):
    """Simplified Image-GS model for CPU implementation."""
    
    def __init__(self, num_gaussians: int = 1000, image_size: Tuple[int, int] = (512, 512),
                 device: str = 'cpu'):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.image_size = image_size
        self.device = device
        
        # Initialize Gaussian parameters
        self.positions = nn.Parameter(torch.rand(num_gaussians, 2, device=device))
        self.scales = nn.Parameter(torch.ones(num_gaussians, 2, device=device) * 0.01)
        self.rotations = nn.Parameter(torch.zeros(num_gaussians, device=device))
        self.colors = nn.Parameter(torch.rand(num_gaussians, 3, device=device))
        self.opacities = nn.Parameter(torch.ones(num_gaussians, device=device))
        
    def compute_gaussian_2d(self, x: torch.Tensor, y: torch.Tensor, 
                            pos: torch.Tensor, scale: torch.Tensor, 
                            rotation: float) -> torch.Tensor:
        """Compute 2D Gaussian value at position (x, y)."""
        # Rotation matrix
        cos_r = torch.cos(rotation)
        sin_r = torch.sin(rotation)
        
        # Transform to local coordinates
        dx = x - pos[0]
        dy = y - pos[1]
        
        # Apply rotation
        x_rot = cos_r * dx + sin_r * dy
        y_rot = -sin_r * dx + cos_r * dy
        
        # Compute Gaussian
        gaussian = torch.exp(-0.5 * ((x_rot / scale[0])**2 + (y_rot / scale[1])**2))
        return gaussian
    
    def render(self, resolution: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Render the image from Gaussians."""
        if resolution is None:
            resolution = self.image_size
            
        h, w = resolution
        
        # Create coordinate grids
        y_coords = torch.linspace(0, 1, h, device=self.device)
        x_coords = torch.linspace(0, 1, w, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Initialize output image
        image = torch.zeros(3, h, w, device=self.device)
        
        # Render each Gaussian
        for i in range(self.num_gaussians):
            # Compute Gaussian values
            gaussian = self.compute_gaussian_2d(
                grid_x, grid_y,
                self.positions[i],
                torch.abs(self.scales[i]) + 1e-6,  # Ensure positive scale
                self.rotations[i]
            )
            
            # Apply opacity
            alpha = gaussian * torch.sigmoid(self.opacities[i])
            
            # Add colored Gaussian to image
            for c in range(3):
                image[c] += alpha * torch.sigmoid(self.colors[i, c])
        
        # Clamp to valid range
        image = torch.clamp(image, 0, 1)
        
        return image
    
    def forward(self) -> torch.Tensor:
        """Forward pass renders the image."""
        return self.render()


class ImageGSTrainer:
    """Trainer for the Image-GS model."""
    
    def __init__(self, model: ImageGS, target_image: torch.Tensor, 
                 learning_rate: float = 0.01):
        self.model = model
        self.target_image = target_image
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def compute_loss(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        # L2 loss
        l2_loss = F.mse_loss(rendered, target)
        
        # Simple SSIM-like loss (structural similarity)
        mean_rendered = rendered.mean(dim=(1, 2), keepdim=True)
        mean_target = target.mean(dim=(1, 2), keepdim=True)
        
        var_rendered = ((rendered - mean_rendered) ** 2).mean(dim=(1, 2))
        var_target = ((target - mean_target) ** 2).mean(dim=(1, 2))
        
        covar = ((rendered - mean_rendered) * (target - mean_target)).mean(dim=(1, 2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mean_rendered * mean_target + c1) * (2 * covar + c2)) / \
               ((mean_rendered ** 2 + mean_target ** 2 + c1) * (var_rendered + var_target + c2))
        
        ssim_loss = 1 - ssim.mean()
        
        # Combined loss
        total_loss = 0.8 * l2_loss + 0.2 * ssim_loss
        
        return total_loss
    
    def train_step(self) -> float:
        """Perform one training step."""
        self.optimizer.zero_grad()
        
        # Render image
        rendered = self.model()
        
        # Compute loss
        loss = self.compute_loss(rendered, self.target_image)
        
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_iterations: int = 1000, log_interval: int = 100):
        """Train the model."""
        losses = []
        
        pbar = tqdm(range(num_iterations), desc="Training")
        for iteration in pbar:
            loss = self.train_step()
            losses.append(loss)
            
            if iteration % log_interval == 0:
                pbar.set_postfix({'Loss': f'{loss:.4f}'})
        
        return losses


def load_image(image_path: str, size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size, Image.Resampling.LANCZOS)
    image = np.array(image) / 255.0
    image = torch.from_numpy(image).float().permute(2, 0, 1)
    return image


def save_image(tensor: torch.Tensor, path: str):
    """Save a tensor as an image."""
    # Convert to numpy and transpose
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image * 255).astype(np.uint8)
    
    # Save using PIL
    Image.fromarray(image).save(path)


def visualize_gaussians(model: ImageGS, save_path: Optional[str] = None):
    """Visualize the Gaussian positions and scales."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Render the image
    rendered = model.render().detach().cpu().numpy()
    rendered = np.transpose(rendered, (1, 2, 0))
    
    ax1.imshow(rendered)
    ax1.set_title('Rendered Image')
    ax1.axis('off')
    
    # Plot Gaussian centers
    positions = model.positions.detach().cpu().numpy()
    scales = model.scales.detach().cpu().numpy()
    
    ax2.scatter(positions[:, 0], positions[:, 1], 
                s=scales.mean(axis=1) * 1000, alpha=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(1, 0)  # Flip y-axis to match image coordinates
    ax2.set_title(f'Gaussian Centers ({model.num_gaussians} Gaussians)')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Image-GS CPU Implementation')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--num_gaussians', type=int, default=1000,
                       help='Number of Gaussians')
    parser.add_argument('--iterations', type=int, default=2000,
                       help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--size', type=int, default=256,
                       help='Image size (will be resized to size x size)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.image:
        # Load target image
        print(f"Loading image: {args.image}")
        target_image = load_image(args.image, size=(args.size, args.size)).to(device)
        
        # Create model
        print(f"Creating model with {args.num_gaussians} Gaussians")
        model = ImageGS(num_gaussians=args.num_gaussians, 
                       image_size=(args.size, args.size),
                       device=device)
        
        # Create trainer
        trainer = ImageGSTrainer(model, target_image, learning_rate=args.lr)
        
        # Train
        print(f"Training for {args.iterations} iterations...")
        losses = trainer.train(num_iterations=args.iterations, log_interval=100)
        
        # Save results
        print("Saving results...")
        rendered = model.render()
        save_image(rendered, output_dir / 'rendered.png')
        save_image(target_image, output_dir / 'target.png')
        
        # Visualize Gaussians
        visualize_gaussians(model, output_dir / 'gaussians.png')
        
        # Plot loss curve
        plt.figure(figsize=(8, 6))
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.savefig(output_dir / 'loss.png')
        plt.close()
        
        print(f"Results saved to {output_dir}")
    else:
        # Demo mode
        print("Running demo mode (no input image provided)")
        print("Creating synthetic target...")
        
        # Create a simple synthetic target
        size = args.size
        target = torch.zeros(3, size, size, device=device)
        
        # Add some colored regions
        target[0, size//4:size//2, size//4:size//2] = 1.0  # Red square
        target[1, size//2:3*size//4, size//2:3*size//4] = 1.0  # Green square
        target[2, size//4:3*size//4, size//4:3*size//4] = 0.5  # Blue overlay
        
        # Create and train model
        model = ImageGS(num_gaussians=args.num_gaussians, 
                       image_size=(size, size),
                       device=device)
        trainer = ImageGSTrainer(model, target, learning_rate=args.lr)
        
        print(f"Training for {args.iterations} iterations...")
        losses = trainer.train(num_iterations=args.iterations, log_interval=100)
        
        # Save results
        rendered = model.render()
        save_image(rendered, output_dir / 'demo_rendered.png')
        save_image(target, output_dir / 'demo_target.png')
        visualize_gaussians(model, output_dir / 'demo_gaussians.png')
        
        print(f"Demo results saved to {output_dir}")


if __name__ == '__main__':
    main()