//! Scaling analysis benchmarks (multi-threading, memory, etc.)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lgi_benchmarks::{TestImageGenerator, test_images::TestPattern};
use lgi_core::{Renderer, Initializer, InitStrategy, RenderConfig};
use lgi_math::compositing::AlphaMode;

fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");

    let gen = TestImageGenerator::new(256);
    let test_image = gen.generate(TestPattern::NaturalScene);

    let initializer = Initializer::new(InitStrategy::Grid);
    let gaussians = initializer.initialize(&test_image, 500).unwrap();

    // Sequential
    group.bench_function("sequential", |b| {
        let mut config = RenderConfig::default();
        config.parallel = false;
        let renderer = Renderer::with_config(config);

        b.iter(|| {
            let _rendered = renderer.render_basic(black_box(&gaussians), black_box(256), black_box(256)).unwrap();
        });
    });

    // Parallel
    group.bench_function("parallel", |b| {
        let mut config = RenderConfig::default();
        config.parallel = true;
        let renderer = Renderer::with_config(config);

        b.iter(|| {
            let _rendered = renderer.render(black_box(&gaussians), black_box(256), black_box(256)).unwrap();
        });
    });

    group.finish();
}

fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");

    let gen = TestImageGenerator::new(256);
    let test_image = gen.generate(TestPattern::FrequencySweep);

    for count in [500, 1000, 2000, 5000].iter() {
        let initializer = Initializer::new(InitStrategy::Random).with_seed(42);
        let gaussians = initializer.initialize(&test_image, *count).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, _| {
            let renderer = Renderer::new();

            b.iter(|| {
                let _rendered = renderer.render(black_box(&gaussians), black_box(256), black_box(256)).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_parallel_vs_sequential, bench_memory_scaling);
criterion_main!(benches);
