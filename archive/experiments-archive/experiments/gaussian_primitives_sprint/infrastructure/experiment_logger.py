"""
Experiment Logging and Management System
Handles: configuration, metrics, images, checkpoints, reports
"""

import numpy as np
import json
import csv
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import time

from gaussian_primitives import Gaussian2D, compute_metrics


class ExperimentLogger:
    """Comprehensive experiment logging system"""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.experiment_dir / "images").mkdir(exist_ok=True)
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)

        self.config = {}
        self.metrics = {}
        self.notes = []

        self.start_time = time.time()

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        self.config = config
        self.config['timestamp'] = datetime.now().isoformat()

        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def log_iteration(self, iteration: int, loss: float, extra_metrics: Optional[Dict] = None):
        """Log metrics for an iteration"""

        # Append to loss curve
        loss_curve_path = self.experiment_dir / "loss_curve.csv"
        file_exists = loss_curve_path.exists()

        with open(loss_curve_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                headers = ['iteration', 'loss', 'timestamp']
                if extra_metrics:
                    headers.extend(extra_metrics.keys())
                writer.writerow(headers)

            row = [iteration, loss, time.time() - self.start_time]
            if extra_metrics:
                row.extend(extra_metrics.values())
            writer.writerow(row)

    def save_checkpoint(self, iteration: int, gaussians: List[Gaussian2D]):
        """Save Gaussian parameters at checkpoint"""

        checkpoint_path = self.experiment_dir / "checkpoints" / f"iter_{iteration:04d}.json"

        checkpoint_data = {
            'iteration': iteration,
            'timestamp': time.time() - self.start_time,
            'gaussians': [g.to_dict() for g in gaussians]
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def save_image(self, name: str, image: np.ndarray):
        """Save image as PNG"""

        image_path = self.experiment_dir / "images" / f"{name}.png"

        # Normalize to [0, 255]
        if image.dtype != np.uint8:
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image

        # Save using matplotlib
        plt.imsave(image_path, image_uint8, cmap='gray' if len(image.shape) == 2 else None)

    def save_gaussians_init(self, gaussians: List[Gaussian2D]):
        """Save initial Gaussian parameters"""
        init_path = self.experiment_dir / "gaussians_init.json"
        with open(init_path, 'w') as f:
            json.dump([g.to_dict() for g in gaussians], f, indent=2)

    def save_gaussians_final(self, gaussians: List[Gaussian2D]):
        """Save final Gaussian parameters"""
        final_path = self.experiment_dir / "gaussians_final.json"
        with open(final_path, 'w') as f:
            json.dump([g.to_dict() for g in gaussians], f, indent=2)

    def log_metrics(self, metrics: Dict[str, float]):
        """Log final metrics"""
        self.metrics = metrics
        self.metrics['total_time'] = time.time() - self.start_time

        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def add_note(self, note: str):
        """Add observation or note"""
        self.notes.append(f"[{time.time() - self.start_time:.1f}s] {note}")

        notes_path = self.experiment_dir / "notes.txt"
        with open(notes_path, 'a') as f:
            f.write(f"{self.notes[-1]}\n")

    def visualize_gaussians(self, gaussians: List[Gaussian2D],
                           target_image: np.ndarray,
                           name: str = "gaussians_overlay"):
        """Create visualization of Gaussian placement over target"""

        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(8, 8))

        # Show target image
        ax.imshow(target_image, cmap='gray', alpha=0.5)

        # Draw Gaussian ellipses
        for g in gaussians:
            # Convert to matplotlib ellipse
            width = 2 * g.sigma_parallel  # 2*sigma = ~95% of mass
            height = 2 * g.sigma_perp
            angle_deg = np.degrees(g.theta)

            ellipse = patches.Ellipse(
                (g.x, g.y), width, height, angle=angle_deg,
                fill=False, edgecolor='red', linewidth=1, alpha=g.alpha
            )
            ax.add_patch(ellipse)

            # Mark center
            ax.plot(g.x, g.y, 'r+', markersize=5)

        ax.set_xlim(0, target_image.shape[1])
        ax.set_ylim(target_image.shape[0], 0)
        ax.set_aspect('equal')
        ax.set_title(f'Gaussian Placement (N={len(gaussians)})')

        output_path = self.experiment_dir / "images" / f"{name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_loss_curve(self):
        """Generate loss curve plot"""

        loss_curve_path = self.experiment_dir / "loss_curve.csv"
        if not loss_curve_path.exists():
            return

        # Read loss curve
        iterations = []
        losses = []

        with open(loss_curve_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                iterations.append(int(row['iteration']))
                losses.append(float(row['loss']))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations, losses, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Convergence Curve')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        output_path = self.experiment_dir / "loss_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_report(self) -> str:
        """Generate markdown report"""

        report_lines = [
            f"# Experiment Report",
            f"",
            f"**Directory:** `{self.experiment_dir.name}`",
            f"**Timestamp:** {self.config.get('timestamp', 'N/A')}",
            f"**Total Time:** {self.metrics.get('total_time', 0):.2f}s",
            f"",
            f"## Configuration",
            f"```json"
        ]

        # Add config (excluding timestamp)
        config_clean = {k: v for k, v in self.config.items() if k != 'timestamp'}
        report_lines.append(json.dumps(config_clean, indent=2))
        report_lines.append("```")
        report_lines.append("")

        report_lines.extend([
            f"## Final Metrics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|"
        ])

        for key, value in self.metrics.items():
            if isinstance(value, float):
                report_lines.append(f"| {key} | {value:.6f} |")
            else:
                report_lines.append(f"| {key} | {value} |")

        report_lines.append("")

        # Add notes if any
        if self.notes:
            report_lines.extend([
                f"## Notes",
                f""
            ])
            for note in self.notes:
                report_lines.append(f"- {note}")

        report_lines.append("")

        # Save report
        report_path = self.experiment_dir / "report.md"
        report_text = "\n".join(report_lines)
        with open(report_path, 'w') as f:
            f.write(report_text)

        return report_text


class ExperimentBatch:
    """Manage a batch of related experiments"""

    def __init__(self, batch_dir: Path, batch_name: str):
        self.batch_dir = Path(batch_dir)
        self.batch_name = batch_name
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        self.experiments = []

    def create_experiment(self, name: str) -> ExperimentLogger:
        """Create a new experiment in this batch"""

        exp_dir = self.batch_dir / name
        logger = ExperimentLogger(exp_dir)
        self.experiments.append(name)

        return logger

    def generate_batch_report(self) -> str:
        """Generate comparative report across all experiments"""

        report_lines = [
            f"# Batch Report: {self.batch_name}",
            f"",
            f"**Total Experiments:** {len(self.experiments)}",
            f"**Timestamp:** {datetime.now().isoformat()}",
            f"",
            f"## Summary Table",
            f"",
            f"| Experiment | PSNR (dB) | MSE | Time (s) | Notes |",
            f"|------------|-----------|-----|----------|-------|"
        ]

        # Collect metrics from all experiments
        for exp_name in self.experiments:
            exp_dir = self.batch_dir / exp_name
            metrics_path = exp_dir / "metrics.json"

            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)

                psnr = metrics.get('psnr', 0)
                mse = metrics.get('mse', 0)
                total_time = metrics.get('total_time', 0)

                report_lines.append(
                    f"| {exp_name} | {psnr:.2f} | {mse:.6f} | {total_time:.2f} | |"
                )

        report_lines.append("")

        # Save batch report
        report_path = self.batch_dir / "batch_report.md"
        report_text = "\n".join(report_lines)
        with open(report_path, 'w') as f:
            f.write(report_text)

        return report_text


if __name__ == "__main__":
    print("Testing Experiment Logger...")

    # Create test experiment
    from gaussian_primitives import SyntheticDataGenerator, GaussianRenderer
    from initializers import EdgeInitializer

    test_dir = Path("test_experiment")
    logger = ExperimentLogger(test_dir)

    # Log config
    logger.log_config({
        'test_case': 'straight_edge',
        'strategy': 'E1_uniform',
        'N': 10,
        'optimizer': 'adam',
        'lr': 0.01
    })
    print("✓ Config logged")

    # Generate test data
    target, desc = SyntheticDataGenerator.generate_edge(
        edge_type='straight',
        blur_sigma=1.0,
        contrast=0.8
    )

    # Initialize Gaussians
    gaussians = EdgeInitializer.uniform(desc, N=10)

    # Log initial state
    logger.save_gaussians_init(gaussians)
    logger.save_image("target", target)

    # Render initial
    renderer = GaussianRenderer()
    rendered_init = renderer.render(gaussians, 100, 100, channels=1)
    logger.save_image("init", rendered_init)

    # Simulate optimization
    for i in range(5):
        logger.log_iteration(i, 0.1 / (i + 1))

    # Final state
    logger.save_gaussians_final(gaussians)
    rendered_final = renderer.render(gaussians, 100, 100, channels=1)
    logger.save_image("final", rendered_final)

    # Compute metrics
    metrics = compute_metrics(target, rendered_final)
    logger.log_metrics(metrics)

    # Visualizations
    logger.visualize_gaussians(gaussians, target)
    logger.plot_loss_curve()

    # Add notes
    logger.add_note("Test experiment completed successfully")

    # Generate report
    report = logger.generate_report()
    print("✓ Report generated")
    print(f"\nReport preview:\n{report[:500]}...")

    # Clean up test
    import shutil
    shutil.rmtree(test_dir)
    print("\n✓ Experiment logger tests passed!")
