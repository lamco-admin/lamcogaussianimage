//! L-BFGS (Limited-memory BFGS) optimizer
//!
//! Quasi-Newton optimization method that approximates the inverse Hessian
//! using a limited history of gradient vectors. Memory-efficient alternative
//! to full BFGS for high-dimensional optimization problems.

use std::collections::VecDeque;

/// L-BFGS optimizer
pub struct LBFGS {
    /// History size (m parameter, typically 5-20)
    history_size: usize,
    /// History of position differences (s_k = x_{k+1} - x_k)
    s_history: VecDeque<Vec<f32>>,
    /// History of gradient differences (y_k = g_{k+1} - g_k)
    y_history: VecDeque<Vec<f32>>,
    /// Line search parameters
    c1: f32,  // Armijo condition constant (typically 1e-4)
    c2: f32,  // Wolfe condition constant (typically 0.9)
    max_line_search_iters: usize,
}

impl Default for LBFGS {
    fn default() -> Self {
        Self::new(10)
    }
}

impl LBFGS {
    /// Create new L-BFGS optimizer with specified history size
    pub fn new(history_size: usize) -> Self {
        Self {
            history_size,
            s_history: VecDeque::with_capacity(history_size),
            y_history: VecDeque::with_capacity(history_size),
            c1: 1e-4,
            c2: 0.9,
            max_line_search_iters: 20,
        }
    }

    /// Reset optimizer state (clear history)
    pub fn reset(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
    }

    /// Compute search direction using L-BFGS two-loop recursion
    ///
    /// Given current gradient, returns search direction p
    pub fn compute_direction(&self, gradient: &[f32]) -> Vec<f32> {
        let n = gradient.len();
        let m = self.s_history.len();

        if m == 0 {
            // First iteration: use steepest descent
            return gradient.iter().map(|&g| -g).collect();
        }

        // Two-loop recursion algorithm
        let mut q = gradient.to_vec();
        let mut alpha = vec![0.0f32; m];

        // First loop (backward)
        for i in (0..m).rev() {
            let rho_i = 1.0 / dot_product(&self.y_history[i], &self.s_history[i]);
            alpha[i] = rho_i * dot_product(&self.s_history[i], &q);
            for j in 0..n {
                q[j] -= alpha[i] * self.y_history[i][j];
            }
        }

        // Initial Hessian approximation: H_0 = γI
        // γ = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
        let gamma = {
            let last_s = &self.s_history[m - 1];
            let last_y = &self.y_history[m - 1];
            dot_product(last_s, last_y) / dot_product(last_y, last_y)
        };

        let mut r: Vec<f32> = q.iter().map(|&qi| gamma * qi).collect();

        // Second loop (forward)
        for i in 0..m {
            let rho_i = 1.0 / dot_product(&self.y_history[i], &self.s_history[i]);
            let beta = rho_i * dot_product(&self.y_history[i], &r);
            for j in 0..n {
                r[j] += self.s_history[i][j] * (alpha[i] - beta);
            }
        }

        // Return search direction (negative because we're minimizing)
        r.iter().map(|&ri| -ri).collect()
    }

    /// Perform line search to find step size
    ///
    /// Uses backtracking line search with Armijo condition
    pub fn line_search<F>(
        &self,
        x: &[f32],
        direction: &[f32],
        gradient: &[f32],
        f: &mut F,
        f_x: f32,
    ) -> f32
    where
        F: FnMut(&[f32]) -> f32,
    {
        let mut alpha = 1.0;
        let directional_derivative = dot_product(gradient, direction);

        // If direction is not a descent direction, return small step
        if directional_derivative >= 0.0 {
            return 1e-8;
        }

        let mut x_new = vec![0.0f32; x.len()];

        for _ in 0..self.max_line_search_iters {
            // x_new = x + alpha * direction
            for i in 0..x.len() {
                x_new[i] = x[i] + alpha * direction[i];
            }

            let f_new = f(&x_new);

            // Armijo condition: f(x + αp) ≤ f(x) + c1·α·∇f^T·p
            if f_new <= f_x + self.c1 * alpha * directional_derivative {
                return alpha;
            }

            // Reduce step size
            alpha *= 0.5;
        }

        // If line search fails, return small step
        alpha
    }

    /// Update optimizer with new position and gradient
    ///
    /// Call after taking a step to update history
    pub fn update(&mut self, x_prev: &[f32], x_new: &[f32], grad_prev: &[f32], grad_new: &[f32]) {
        let n = x_prev.len();

        // Compute s_k = x_{k+1} - x_k
        let s: Vec<f32> = (0..n).map(|i| x_new[i] - x_prev[i]).collect();

        // Compute y_k = g_{k+1} - g_k
        let y: Vec<f32> = (0..n).map(|i| grad_new[i] - grad_prev[i]).collect();

        // Check curvature condition: s^T y > 0
        let sy = dot_product(&s, &y);
        if sy > 1e-10 {
            // Add to history
            if self.s_history.len() >= self.history_size {
                self.s_history.pop_front();
                self.y_history.pop_front();
            }
            self.s_history.push_back(s);
            self.y_history.push_back(y);
        }
    }

    /// Optimize a function starting from initial point
    ///
    /// Returns (optimized_x, final_loss, iterations)
    pub fn optimize<F, G>(
        &mut self,
        x0: Vec<f32>,
        mut f: F,
        mut grad_f: G,
        max_iterations: usize,
        tolerance: f32,
    ) -> (Vec<f32>, f32, usize)
    where
        F: FnMut(&[f32]) -> f32,
        G: FnMut(&[f32]) -> Vec<f32>,
    {
        let mut x = x0;
        let mut gradient = grad_f(&x);
        let mut f_x = f(&x);

        for iter in 0..max_iterations {
            // Check convergence (gradient norm)
            let grad_norm = gradient.iter().map(|&g| g * g).sum::<f32>().sqrt();
            if grad_norm < tolerance {
                return (x, f_x, iter);
            }

            // Compute search direction
            let direction = self.compute_direction(&gradient);

            // Line search for step size
            let alpha = self.line_search(&x, &direction, &gradient, &mut f, f_x);

            // Update position
            let x_prev = x.clone();
            let grad_prev = gradient.clone();

            for i in 0..x.len() {
                x[i] += alpha * direction[i];
            }

            // Compute new gradient and loss
            gradient = grad_f(&x);
            f_x = f(&x);

            // Update L-BFGS history
            self.update(&x_prev, &x, &grad_prev, &gradient);
        }

        (x, f_x, max_iterations)
    }
}

/// Compute dot product of two vectors
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rosenbrock() {
        // Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        // Minimum at (1, 1) with f(1,1) = 0

        let mut optimizer = LBFGS::new(10);

        let f = |x: &[f32]| -> f32 {
            let x0 = x[0];
            let x1 = x[1];
            (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
        };

        let grad_f = |x: &[f32]| -> Vec<f32> {
            let x0 = x[0];
            let x1 = x[1];
            vec![
                -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0),
                200.0 * (x1 - x0 * x0),
            ]
        };

        let x0 = vec![-1.0, 1.0];
        let (x_opt, f_opt, iters) = optimizer.optimize(x0, f, grad_f, 100, 1e-5);

        println!("Rosenbrock optimization:");
        println!("  Iterations: {}", iters);
        println!("  Solution: x = [{:.6}, {:.6}]", x_opt[0], x_opt[1]);
        println!("  f(x) = {:.10}", f_opt);

        // Check convergence to (1, 1)
        assert!((x_opt[0] - 1.0).abs() < 0.01);
        assert!((x_opt[1] - 1.0).abs() < 0.01);
        assert!(f_opt < 0.01);
    }

    #[test]
    fn test_quadratic() {
        // Simple quadratic: f(x) = (x-2)^2 + (y-3)^2
        // Minimum at (2, 3) with f(2,3) = 0

        let mut optimizer = LBFGS::new(5);

        let f = |x: &[f32]| -> f32 {
            (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2)
        };

        let grad_f = |x: &[f32]| -> Vec<f32> {
            vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 3.0)]
        };

        let x0 = vec![0.0, 0.0];
        let (x_opt, f_opt, iters) = optimizer.optimize(x0, f, grad_f, 50, 1e-6);

        println!("Quadratic optimization:");
        println!("  Iterations: {}", iters);
        println!("  Solution: x = [{:.6}, {:.6}]", x_opt[0], x_opt[1]);
        println!("  f(x) = {:.10}", f_opt);

        // Check convergence
        assert!((x_opt[0] - 2.0).abs() < 0.001);
        assert!((x_opt[1] - 3.0).abs() < 0.001);
        assert!(f_opt < 0.001);
    }
}
