use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lgi_math::prelude::*;

fn bench_gaussian_evaluation(c: &mut Criterion) {
    let gaussian = Gaussian2D::new(
        Vector2::new(0.5, 0.5),
        Euler::new(0.1, 0.1, 0.0),
        Color4::white(),
        1.0f32,
    );

    let evaluator = GaussianEvaluator::default();
    let point = Vector2::new(0.55, 0.55);

    c.bench_function("evaluate_single", |b| {
        b.iter(|| {
            black_box(evaluator.evaluate(&gaussian, point))
        })
    });
}

fn bench_batch_evaluation(c: &mut Criterion) {
    let gaussian = Gaussian2D::new(
        Vector2::new(0.5, 0.5),
        Euler::new(0.1, 0.1, 0.0),
        Color4::white(),
        1.0f32,
    );

    let evaluator = GaussianEvaluator::default();

    let mut group = c.benchmark_group("batch_evaluation");
    for size in [16, 64, 256, 1024].iter() {
        let points: Vec<Vector2<f32>> = (0..*size)
            .map(|i| Vector2::new((i as f32) / (*size as f32), 0.5))
            .collect();
        let mut weights = vec![0.0f32; *size];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                evaluator.evaluate_batch(&gaussian, &points, &mut weights);
                black_box(&weights);
            })
        });
    }
    group.finish();
}

fn bench_parameterization_conversion(c: &mut Criterion) {
    let euler = Euler::new(0.1f32, 0.05, 0.3);

    c.bench_function("euler_to_cholesky", |b| {
        b.iter(|| {
            let chol: Cholesky<f32> = black_box(euler).into();
            black_box(chol)
        })
    });

    c.bench_function("euler_inverse_cov", |b| {
        b.iter(|| {
            black_box(euler.inverse_covariance())
        })
    });
}

criterion_group!(benches, bench_gaussian_evaluation, bench_batch_evaluation, bench_parameterization_conversion);
criterion_main!(benches);
