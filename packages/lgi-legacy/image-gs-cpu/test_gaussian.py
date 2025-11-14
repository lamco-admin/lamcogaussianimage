#!/usr/bin/env python3
"""
Test script for the simplified Image-GS implementation.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from gaussian_2d_cpu import ImageGS, ImageGSTrainer, load_image, save_image, visualize_gaussians


def create_test_image():
    """Create a simple test image with geometric patterns."""
    size = 256
    image = np.zeros((size, size, 3))
    
    # Create a gradient background
    for i in range(size):
        for j in range(size):
            image[i, j, 0] = i / size  # Red gradient
            image[i, j, 1] = j / size  # Green gradient
            image[i, j, 2] = 0.5  # Constant blue
    
    # Add a white circle
    center = size // 2
    radius = size // 4
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 < radius**2:
                image[i, j] = [1, 1, 1]
    
    # Add colored squares
    square_size = size // 8
    # Red square
    image[20:20+square_size, 20:20+square_size] = [1, 0, 0]
    # Green square
    image[20:20+square_size, size-20-square_size:size-20] = [0, 1, 0]
    # Blue square
    image[size-20-square_size:size-20, 20:20+square_size] = [0, 0, 1]
    # Yellow square
    image[size-20-square_size:size-20, size-20-square_size:size-20] = [1, 1, 0]
    
    return torch.from_numpy(image).float().permute(2, 0, 1)


def test_basic_reconstruction():
    """Test basic image reconstruction with different numbers of Gaussians."""
    print("Testing basic reconstruction...")
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create test image
    target = create_test_image()
    save_image(target, output_dir / "test_target.png")
    
    # Test with different numbers of Gaussians
    gaussian_counts = [100, 500, 1000, 2000]
    
    fig, axes = plt.subplots(2, len(gaussian_counts), figsize=(16, 8))
    
    for idx, num_gaussians in enumerate(gaussian_counts):
        print(f"\nTesting with {num_gaussians} Gaussians...")
        
        # Create model
        model = ImageGS(num_gaussians=num_gaussians, image_size=(256, 256))
        trainer = ImageGSTrainer(model, target, learning_rate=0.01)
        
        # Train
        losses = trainer.train(num_iterations=1000, log_interval=250)
        
        # Get result
        rendered = model.render().detach().cpu().numpy()
        rendered = np.transpose(rendered, (1, 2, 0))
        
        # Plot rendered image
        axes[0, idx].imshow(rendered)
        axes[0, idx].set_title(f'{num_gaussians} Gaussians')
        axes[0, idx].axis('off')
        
        # Plot loss curve
        axes[1, idx].plot(losses)
        axes[1, idx].set_xlabel('Iteration')
        axes[1, idx].set_ylabel('Loss')
        axes[1, idx].set_title(f'Final Loss: {losses[-1]:.4f}')
        axes[1, idx].grid(True)
        
        # Save individual result
        save_image(torch.from_numpy(rendered.transpose(2, 0, 1)), 
                  output_dir / f"rendered_{num_gaussians}.png")
    
    plt.suptitle('Image-GS Reconstruction with Different Numbers of Gaussians')
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png")
    plt.show()
    
    print(f"\nResults saved to {output_dir}")


def test_progressive_optimization():
    """Test progressive optimization (adding Gaussians gradually)."""
    print("\nTesting progressive optimization...")
    
    output_dir = Path("test_output_progressive")
    output_dir.mkdir(exist_ok=True)
    
    # Create test image
    target = create_test_image()
    
    # Start with few Gaussians
    initial_gaussians = 50
    model = ImageGS(num_gaussians=initial_gaussians, image_size=(256, 256))
    trainer = ImageGSTrainer(model, target, learning_rate=0.02)
    
    # Progressive training
    stages = [
        (50, 500),   # 50 Gaussians, 500 iterations
        (100, 500),  # Add 50 more, train 500 iterations
        (200, 500),  # Add 100 more, train 500 iterations
        (500, 1000), # Add 300 more, train 1000 iterations
    ]
    
    fig, axes = plt.subplots(2, len(stages), figsize=(16, 8))
    
    for stage_idx, (total_gaussians, iterations) in enumerate(stages):
        print(f"\nStage {stage_idx + 1}: {total_gaussians} Gaussians, {iterations} iterations")
        
        if stage_idx > 0:
            # Add more Gaussians
            num_to_add = total_gaussians - model.num_gaussians
            
            # Save current parameters
            old_positions = model.positions.data.clone()
            old_scales = model.scales.data.clone()
            old_rotations = model.rotations.data.clone()
            old_colors = model.colors.data.clone()
            old_opacities = model.opacities.data.clone()
            
            # Create new model with more Gaussians
            model = ImageGS(num_gaussians=total_gaussians, image_size=(256, 256))
            
            # Copy old parameters
            model.positions.data[:len(old_positions)] = old_positions
            model.scales.data[:len(old_scales)] = old_scales
            model.rotations.data[:len(old_rotations)] = old_rotations
            model.colors.data[:len(old_colors)] = old_colors
            model.opacities.data[:len(old_opacities)] = old_opacities
            
            # Reinitialize trainer
            trainer = ImageGSTrainer(model, target, learning_rate=0.01)
        
        # Train
        losses = trainer.train(num_iterations=iterations, log_interval=iterations//4)
        
        # Get result
        rendered = model.render().detach().cpu().numpy()
        rendered = np.transpose(rendered, (1, 2, 0))
        
        # Plot
        axes[0, stage_idx].imshow(rendered)
        axes[0, stage_idx].set_title(f'{total_gaussians} Gaussians')
        axes[0, stage_idx].axis('off')
        
        axes[1, stage_idx].plot(losses)
        axes[1, stage_idx].set_xlabel('Iteration')
        axes[1, stage_idx].set_ylabel('Loss')
        axes[1, stage_idx].set_title(f'Final Loss: {losses[-1]:.4f}')
        axes[1, stage_idx].grid(True)
    
    plt.suptitle('Progressive Gaussian Optimization')
    plt.tight_layout()
    plt.savefig(output_dir / "progressive_optimization.png")
    plt.show()
    
    # Save final result
    save_image(torch.from_numpy(rendered.transpose(2, 0, 1)), 
              output_dir / "final_progressive.png")
    visualize_gaussians(model, output_dir / "final_gaussians.png")
    
    print(f"\nProgressive optimization results saved to {output_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("Image-GS CPU Implementation Test")
    print("=" * 60)
    
    # Run tests
    test_basic_reconstruction()
    test_progressive_optimization()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)