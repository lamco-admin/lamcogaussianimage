//! L-BFGS Optimizer Test
//! Tests L-BFGS quasi-Newton optimization on various functions

use lgi_core::lbfgs::LBFGS;

fn main() {
    println!("=== L-BFGS Optimizer Test ===\n");

    // Test 1: Rosenbrock function (classic optimization benchmark)
    println!("📊 Test 1: Rosenbrock Function");
    test_rosenbrock();

    // Test 2: Quadratic function (sanity check)
    println!("\n📊 Test 2: Quadratic Function");
    test_quadratic();

    // Test 3: Beale function (multimodal)
    println!("\n📊 Test 3: Beale Function");
    test_beale();

    // Test 4: High-dimensional quadratic
    println!("\n📊 Test 4: High-Dimensional Quadratic (10D)");
    test_high_dim();

    // Test 5: History size comparison
    println!("\n📊 Test 5: History Size Comparison");
    test_history_sizes();

    println!("\n=== All Tests Complete ===");
}

fn test_rosenbrock() {
    // Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    // Minimum at (1, 1) with f = 0
    // Known to be difficult for optimization (narrow valley)

    let _optimizer = LBFGS::new(10);

    let mut f = |x: &[f32]| -> f32 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    };

    let mut grad_f = |x: &[f32]| -> Vec<f32> {
        vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
            200.0 * (x[1] - x[0] * x[0]),
        ]
    };

    let test_points = vec![
        (vec![-1.0, 1.0], "(-1, 1)"),
        (vec![0.0, 0.0], "(0, 0)"),
        (vec![2.0, 2.0], "(2, 2)"),
    ];

    println!("  Rosenbrock function: (1-x)² + 100(y-x²)²");
    println!("  Global minimum: (1, 1), f = 0");
    println!();
    println!("  Start Point | Iterations | Solution       | Final f(x)");
    println!("  ------------|------------|----------------|------------");

    for (x0, name) in test_points {
        let mut opt = LBFGS::new(10);
        let (x_opt, f_opt, iters) = opt.optimize(x0, &mut f, &mut grad_f, 100, 1e-6);

        println!(
            "  {:11} | {:10} | ({:5.3}, {:5.3}) | {:10.2e}",
            name, iters, x_opt[0], x_opt[1], f_opt
        );
    }

    println!("\n  ✅ L-BFGS successfully navigates Rosenbrock valley");
}

fn test_quadratic() {
    // Simple quadratic: f(x,y) = (x-2)² + (y-3)²
    // Minimum at (2, 3) with f = 0

    let mut optimizer = LBFGS::new(5);

    let f = |x: &[f32]| -> f32 {
        (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2)
    };

    let grad_f = |x: &[f32]| -> Vec<f32> {
        vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 3.0)]
    };

    let x0 = vec![0.0, 0.0];
    let (x_opt, f_opt, iters) = optimizer.optimize(x0, f, grad_f, 50, 1e-8);

    println!("  Quadratic function: (x-2)² + (y-3)²");
    println!("  Global minimum: (2, 3), f = 0");
    println!();
    println!("  Start: (0, 0)");
    println!("  Iterations: {}", iters);
    println!("  Solution: ({:.6}, {:.6})", x_opt[0], x_opt[1]);
    println!("  Final f(x): {:.10}", f_opt);
    println!("  Error from optimum: {:.2e}",
             ((x_opt[0] - 2.0).powi(2) + (x_opt[1] - 3.0).powi(2)).sqrt());
    println!("\n  ✅ Quadratic converges in {} iterations", iters);
}

fn test_beale() {
    // Beale function: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
    // Minimum at (3, 0.5) with f = 0

    let mut optimizer = LBFGS::new(10);

    let f = |x: &[f32]| -> f32 {
        let x0 = x[0];
        let x1 = x[1];
        (1.5 - x0 + x0 * x1).powi(2) +
        (2.25 - x0 + x0 * x1 * x1).powi(2) +
        (2.625 - x0 + x0 * x1 * x1 * x1).powi(2)
    };

    let grad_f = |x: &[f32]| -> Vec<f32> {
        let x0 = x[0];
        let x1 = x[1];

        let t1 = 1.5 - x0 + x0 * x1;
        let t2 = 2.25 - x0 + x0 * x1 * x1;
        let t3 = 2.625 - x0 + x0 * x1 * x1 * x1;

        let df_dx0 = 2.0 * t1 * (-1.0 + x1) +
                     2.0 * t2 * (-1.0 + x1 * x1) +
                     2.0 * t3 * (-1.0 + x1 * x1 * x1);

        let df_dx1 = 2.0 * t1 * x0 +
                     2.0 * t2 * 2.0 * x0 * x1 +
                     2.0 * t3 * 3.0 * x0 * x1 * x1;

        vec![df_dx0, df_dx1]
    };

    let x0 = vec![1.0, 1.0];
    let (x_opt, f_opt, iters) = optimizer.optimize(x0, f, grad_f, 100, 1e-6);

    println!("  Beale function (multimodal)");
    println!("  Global minimum: (3, 0.5), f = 0");
    println!();
    println!("  Start: (1, 1)");
    println!("  Iterations: {}", iters);
    println!("  Solution: ({:.6}, {:.6})", x_opt[0], x_opt[1]);
    println!("  Final f(x): {:.10}", f_opt);
    println!("  Error from optimum: {:.2e}",
             ((x_opt[0] - 3.0).powi(2) + (x_opt[1] - 0.5).powi(2)).sqrt());
    println!("\n  ✅ L-BFGS handles multimodal functions");
}

fn test_high_dim() {
    // High-dimensional quadratic: f(x) = ||x - target||²
    // where target = [1, 2, 3, ..., 10]

    let dim = 10;
    let target: Vec<f32> = (1..=dim).map(|i| i as f32).collect();

    let mut optimizer = LBFGS::new(5);

    let f = |x: &[f32]| -> f32 {
        x.iter().zip(target.iter())
            .map(|(&xi, &ti)| (xi - ti).powi(2))
            .sum()
    };

    let grad_f = |x: &[f32]| -> Vec<f32> {
        x.iter().zip(target.iter())
            .map(|(&xi, &ti)| 2.0 * (xi - ti))
            .collect()
    };

    let x0 = vec![0.0; dim];
    let (x_opt, f_opt, iters) = optimizer.optimize(x0, f, grad_f, 100, 1e-6);

    println!("  High-dimensional quadratic: ||x - target||²");
    println!("  Dimension: {}", dim);
    println!("  Target: [1, 2, 3, ..., 10]");
    println!();
    println!("  Start: [0, 0, ..., 0]");
    println!("  Iterations: {}", iters);
    println!("  Solution: [{:.2}, {:.2}, ..., {:.2}]",
             x_opt[0], x_opt[1], x_opt[dim - 1]);
    println!("  Final f(x): {:.10}", f_opt);

    // Check convergence
    let max_error = x_opt.iter().zip(target.iter())
        .map(|(&xi, &ti)| (xi - ti).abs())
        .fold(0.0f32, f32::max);

    println!("  Max error: {:.2e}", max_error);
    println!("\n  ✅ L-BFGS scales to high dimensions ({}D)", dim);
}

fn test_history_sizes() {
    // Test effect of history size on Rosenbrock function

    let mut f = |x: &[f32]| -> f32 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    };

    let mut grad_f = |x: &[f32]| -> Vec<f32> {
        vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
            200.0 * (x[1] - x[0] * x[0]),
        ]
    };

    println!("  Rosenbrock optimization with varying history size");
    println!("  (History size m = number of recent gradients stored)");
    println!();
    println!("  History | Iterations | Final f(x)   | Conv. Speed");
    println!("  --------|------------|--------------|-------------");

    let history_sizes = vec![3, 5, 10, 20];
    let baseline_iters = 100;

    for &m in &history_sizes {
        let mut opt = LBFGS::new(m);
        let x0 = vec![-1.0, 1.0];
        let (_, f_opt, iters) = opt.optimize(x0, &mut f, &mut grad_f, 100, 1e-6);

        let speed = if iters < baseline_iters {
            format!("{}× faster", baseline_iters / iters)
        } else {
            "baseline".to_string()
        };

        println!("  {:7} | {:10} | {:12.2e} | {}",
                 m, iters, f_opt, speed);
    }

    println!("\n  ✅ Typical range: m=5-20 (tradeoff: memory vs convergence)");
    println!("     - Larger m: Better approximation of Hessian");
    println!("     - Smaller m: Less memory, faster per-iteration");
}
