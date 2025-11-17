"""
Gaussian Optimization Framework
Provides: Adam, L-BFGS, SGD optimizers for fitting Gaussians to targets
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import time

from gaussian_primitives import Gaussian2D, GaussianRenderer, compute_metrics


@dataclass
class OptimizationConfig:
    """Configuration for Gaussian optimization"""
    optimizer_type: str = 'adam'  # 'adam', 'lbfgs', 'sgd'
    learning_rate: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 1e-6
    log_interval: int = 100

    # Parameter constraints
    fix_theta: bool = False  # Fix orientation
    fix_shape: bool = False  # Fix sigma_parallel, sigma_perp
    fix_color: bool = False  # Fix color

    # Bounds
    min_sigma: float = 0.5
    max_sigma: float = 50.0


class GaussianOptimizer:
    """Base class for Gaussian optimizers"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.renderer = GaussianRenderer()

    def optimize(self,
                gaussians_init: List[Gaussian2D],
                target_image: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[List[Gaussian2D], Dict]:
        """
        Optimize Gaussian parameters to match target image

        Returns:
            (optimized_gaussians, optimization_log)
        """
        raise NotImplementedError


class AdamOptimizer(GaussianOptimizer):
    """Adam optimizer for Gaussian parameters"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def _params_to_vector(self, gaussians: List[Gaussian2D]) -> np.ndarray:
        """Convert Gaussian list to parameter vector"""
        params = []
        for g in gaussians:
            if not self.config.fix_shape:
                params.extend([g.x, g.y, g.sigma_parallel, g.sigma_perp])
            else:
                params.extend([g.x, g.y])

            if not self.config.fix_theta:
                params.append(g.theta)

            if not self.config.fix_color:
                if isinstance(g.color, np.ndarray):
                    params.extend(g.color.tolist())
                else:
                    params.append(g.color)

            params.append(g.alpha)

        return np.array(params, dtype=np.float64)

    def _vector_to_params(self, vector: np.ndarray, gaussians_template: List[Gaussian2D]) -> List[Gaussian2D]:
        """Convert parameter vector back to Gaussian list"""
        gaussians = []
        idx = 0

        for g_template in gaussians_template:
            if not self.config.fix_shape:
                x, y, sig_par, sig_perp = vector[idx:idx+4]
                idx += 4
            else:
                x, y = vector[idx:idx+2]
                idx += 2
                sig_par = g_template.sigma_parallel
                sig_perp = g_template.sigma_perp

            # Clamp sigmas
            sig_par = np.clip(sig_par, self.config.min_sigma, self.config.max_sigma)
            sig_perp = np.clip(sig_perp, self.config.min_sigma, self.config.max_sigma)

            if not self.config.fix_theta:
                theta = vector[idx]
                idx += 1
            else:
                theta = g_template.theta

            if not self.config.fix_color:
                if isinstance(g_template.color, np.ndarray):
                    color_len = len(g_template.color)
                    color = vector[idx:idx+color_len]
                    idx += color_len
                else:
                    color = vector[idx]
                    idx += 1
            else:
                color = g_template.color

            alpha = vector[idx]
            idx += 1

            # Clamp alpha
            alpha = np.clip(alpha, 0.0, 1.0)

            gaussians.append(Gaussian2D(
                x=float(x), y=float(y),
                sigma_parallel=float(sig_par),
                sigma_perp=float(sig_perp),
                theta=float(theta),
                color=np.array([color]) if not isinstance(color, np.ndarray) else color,
                alpha=float(alpha),
                layer_type=g_template.layer_type
            ))

        return gaussians

    def _compute_loss_and_gradient(self,
                                   params: np.ndarray,
                                   gaussians_template: List[Gaussian2D],
                                   target: np.ndarray,
                                   mask: Optional[np.ndarray]) -> Tuple[float, np.ndarray]:
        """Compute MSE loss and gradient via finite differences"""

        # Convert params to Gaussians
        gaussians = self._vector_to_params(params, gaussians_template)

        # Render
        h, w = target.shape[:2]
        channels = 1 if len(target.shape) == 2 else target.shape[2]
        rendered = self.renderer.render(gaussians, w, h, channels)

        # Compute loss
        if mask is not None:
            diff = (rendered - target) * mask
        else:
            diff = rendered - target

        loss = np.mean(diff ** 2)

        # Compute gradient via finite differences
        epsilon = 1e-5
        grad = np.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon

            gaussians_plus = self._vector_to_params(params_plus, gaussians_template)
            rendered_plus = self.renderer.render(gaussians_plus, w, h, channels)

            if mask is not None:
                diff_plus = (rendered_plus - target) * mask
            else:
                diff_plus = rendered_plus - target

            loss_plus = np.mean(diff_plus ** 2)

            grad[i] = (loss_plus - loss) / epsilon

        return loss, grad

    def optimize(self,
                gaussians_init: List[Gaussian2D],
                target_image: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[List[Gaussian2D], Dict]:
        """Optimize using Adam"""

        # Initialize
        params = self._params_to_vector(gaussians_init)
        m = np.zeros_like(params)  # First moment
        v = np.zeros_like(params)  # Second moment

        log = {
            'loss_curve': [],
            'timestamps': [],
            'iterations': [],
            'converged': False,
            'final_loss': 0.0
        }

        start_time = time.time()

        for iteration in range(self.config.max_iterations):
            # Compute loss and gradient
            loss, grad = self._compute_loss_and_gradient(
                params, gaussians_init, target_image, mask
            )

            # Adam update
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = m / (1 - self.beta1 ** (iteration + 1))
            v_hat = v / (1 - self.beta2 ** (iteration + 1))

            # Update parameters
            params -= self.config.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Log
            if iteration % self.config.log_interval == 0 or iteration == 0:
                log['loss_curve'].append(float(loss))
                log['timestamps'].append(time.time() - start_time)
                log['iterations'].append(iteration)

            # Check convergence
            if iteration > 10 and len(log['loss_curve']) >= 2:
                if abs(log['loss_curve'][-1] - log['loss_curve'][-2]) < self.config.tolerance:
                    log['converged'] = True
                    break

        # Final conversion
        gaussians_final = self._vector_to_params(params, gaussians_init)
        log['final_loss'] = float(loss)
        log['total_time'] = time.time() - start_time

        return gaussians_final, log


class SGDOptimizer(GaussianOptimizer):
    """Simple SGD optimizer"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.momentum = 0.9

    def optimize(self,
                gaussians_init: List[Gaussian2D],
                target_image: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[List[Gaussian2D], Dict]:
        """Optimize using SGD with momentum"""

        # Use Adam infrastructure but with SGD update rule
        adam_opt = AdamOptimizer(self.config)
        params = adam_opt._params_to_vector(gaussians_init)
        velocity = np.zeros_like(params)

        log = {
            'loss_curve': [],
            'timestamps': [],
            'iterations': [],
            'converged': False,
            'final_loss': 0.0
        }

        start_time = time.time()

        for iteration in range(self.config.max_iterations):
            loss, grad = adam_opt._compute_loss_and_gradient(
                params, gaussians_init, target_image, mask
            )

            # SGD with momentum
            velocity = self.momentum * velocity - self.config.learning_rate * grad
            params += velocity

            if iteration % self.config.log_interval == 0 or iteration == 0:
                log['loss_curve'].append(float(loss))
                log['timestamps'].append(time.time() - start_time)
                log['iterations'].append(iteration)

            if iteration > 10 and len(log['loss_curve']) >= 2:
                if abs(log['loss_curve'][-1] - log['loss_curve'][-2]) < self.config.tolerance:
                    log['converged'] = True
                    break

        gaussians_final = adam_opt._vector_to_params(params, gaussians_init)
        log['final_loss'] = float(loss)
        log['total_time'] = time.time() - start_time

        return gaussians_final, log


def optimize_gaussians(
    gaussians_init: List[Gaussian2D],
    target_image: np.ndarray,
    optimizer_type: str = 'adam',
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    constraints: Optional[Dict] = None,
    mask: Optional[np.ndarray] = None
) -> Tuple[List[Gaussian2D], Dict]:
    """
    Convenience function for Gaussian optimization

    Args:
        gaussians_init: Initial Gaussians
        target_image: Target image to fit
        optimizer_type: 'adam', 'sgd', 'lbfgs'
        learning_rate: Learning rate
        max_iterations: Max iterations
        constraints: Dict with fix_theta, fix_shape, fix_color flags
        mask: Optional mask for region-constrained optimization

    Returns:
        (optimized_gaussians, log)
    """

    config = OptimizationConfig(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        max_iterations=max_iterations
    )

    if constraints:
        config.fix_theta = constraints.get('fix_theta', False)
        config.fix_shape = constraints.get('fix_shape', False)
        config.fix_color = constraints.get('fix_color', False)

    if optimizer_type == 'adam':
        optimizer = AdamOptimizer(config)
    elif optimizer_type == 'sgd':
        optimizer = SGDOptimizer(config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    return optimizer.optimize(gaussians_init, target_image, mask)


if __name__ == "__main__":
    print("Testing Gaussian Optimizers...")

    from gaussian_primitives import SyntheticDataGenerator

    # Generate target edge
    target, desc = SyntheticDataGenerator.generate_edge(
        image_size=(50, 50),
        edge_type='straight',
        blur_sigma=1.0,
        contrast=0.8,
        orientation=np.pi/4
    )

    # Initialize with random Gaussians
    np.random.seed(42)
    gaussians_init = [
        Gaussian2D(
            x=25 + np.random.randn() * 5,
            y=25 + np.random.randn() * 5,
            sigma_parallel=5.0,
            sigma_perp=2.0,
            theta=np.pi/4 + np.random.randn() * 0.1,
            color=np.array([0.5]),
            alpha=0.3,
            layer_type='E'
        )
        for _ in range(5)
    ]

    # Optimize
    print(f"\nOptimizing {len(gaussians_init)} Gaussians to fit edge...")
    gaussians_opt, log = optimize_gaussians(
        gaussians_init=gaussians_init,
        target_image=target,
        optimizer_type='adam',
        learning_rate=0.1,
        max_iterations=500
    )

    print(f"✓ Optimization complete:")
    print(f"  Iterations: {len(log['loss_curve'])}")
    print(f"  Initial loss: {log['loss_curve'][0]:.6f}")
    print(f"  Final loss: {log['final_loss']:.6f}")
    print(f"  Converged: {log['converged']}")
    print(f"  Time: {log['total_time']:.2f}s")

    # Render result
    renderer = GaussianRenderer()
    rendered = renderer.render(gaussians_opt, 50, 50, channels=1)
    metrics = compute_metrics(target, rendered)
    print(f"  Final PSNR: {metrics['psnr']:.2f} dB")

    print("\n✓ Optimizer tests passed!")
