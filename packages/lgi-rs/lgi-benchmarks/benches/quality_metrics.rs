//! Quality metric computation benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lgi_benchmarks::{TestImageGenerator, test_images::TestPattern, metrics};

fn bench_psnr_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("psnr_computation");

    for size in [128, 256, 512].iter() {
        let gen = TestImageGenerator::new(*size);
        let img1 = gen.generate(TestPattern::NaturalScene);
        let img2 = gen.generate(TestPattern::RadialGradient);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let _psnr = metrics::compute_psnr(black_box(&img1), black_box(&img2));
            });
        });
    }

    group.finish();
}

fn bench_ssim_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssim_computation");
    group.sample_size(10); // SSIM is slower

    for size in [128, 256].iter() {
        let gen = TestImageGenerator::new(*size);
        let img1 = gen.generate(TestPattern::NaturalScene);
        let img2 = gen.generate(TestPattern::RadialGradient);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let _ssim = metrics::compute_ssim(black_box(&img1), black_box(&img2));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_psnr_computation, bench_ssim_computation);
criterion_main!(benches);
