//! Rendering performance benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use lgi_benchmarks::TestImageGenerator;
use lgi_benchmarks::test_images::TestPattern;
use lgi_core::{Renderer, Initializer, InitStrategy};

fn bench_rendering_by_gaussian_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("rendering_by_gaussian_count");

    let gen = TestImageGenerator::new(256);
    let test_image = gen.generate(TestPattern::NaturalScene);

    for count in [100, 200, 500, 1000].iter() {
        // Initialize Gaussians
        let initializer = Initializer::new(InitStrategy::Grid);
        let gaussians = initializer.initialize(&test_image, *count).unwrap();

        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, _| {
            let renderer = Renderer::new();

            b.iter(|| {
                let _rendered = renderer.render(black_box(&gaussians), black_box(256), black_box(256)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_rendering_by_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("rendering_by_resolution");

    let gen = TestImageGenerator::new(256);
    let test_image = gen.generate(TestPattern::RadialGradient);

    let initializer = Initializer::new(InitStrategy::Grid);
    let gaussians = initializer.initialize(&test_image, 500).unwrap();

    for size in [128, 256, 512].iter() {
        group.throughput(Throughput::Elements((size * size) as u64)); // Pixels

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let renderer = Renderer::new();

            b.iter(|| {
                let _rendered = renderer.render(black_box(&gaussians), black_box(size), black_box(size)).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_rendering_by_gaussian_count, bench_rendering_by_resolution);
criterion_main!(benches);
