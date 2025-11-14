//! Encoding performance benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lgi_benchmarks::TestImageGenerator;
use lgi_benchmarks::test_images::TestPattern;
use lgi_encoder::{Encoder, EncoderConfig};

fn bench_encoding_by_gaussian_count(c: &mut Criterion) {
    let gen = TestImageGenerator::new(256);
    let test_image = gen.generate(TestPattern::NaturalScene);

    let mut group = c.benchmark_group("encoding_by_gaussian_count");
    group.sample_size(10); // Fewer samples for slow benchmarks

    for count in [100, 200, 500].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            let encoder = Encoder::with_config(EncoderConfig::fast());

            b.iter(|| {
                let _gaussians = encoder.encode(black_box(&test_image), black_box(count)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_encoding_by_image_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoding_by_image_size");
    group.sample_size(10);

    for size in [128, 256].iter() {
        let gen = TestImageGenerator::new(*size);
        let test_image = gen.generate(TestPattern::LinearGradient);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            let encoder = Encoder::with_config(EncoderConfig::fast());

            b.iter(|| {
                let _gaussians = encoder.encode(black_box(&test_image), black_box(200)).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_encoding_by_gaussian_count, bench_encoding_by_image_size);
criterion_main!(benches);
