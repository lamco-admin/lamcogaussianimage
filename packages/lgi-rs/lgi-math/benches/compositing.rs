use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lgi_math::prelude::*;

fn bench_compositing(c: &mut Criterion) {
    let compositor = Compositor::default();
    let mut accum_color = Color4::black();
    let mut accum_alpha = 0.0f32;
    let src_color = Color4::white();
    let src_alpha = 0.5;
    let weight = 1.0;

    c.bench_function("composite_single", |b| {
        b.iter(|| {
            compositor.composite_over(
                black_box(&mut accum_color),
                black_box(&mut accum_alpha),
                black_box(src_color),
                black_box(src_alpha),
                black_box(weight),
            )
        })
    });
}

fn bench_batch_compositing(c: &mut Criterion) {
    let compositor = BatchCompositor::new(AlphaMode::Straight);

    let mut group = c.benchmark_group("batch_compositing");
    for size in [16, 64, 256, 1024].iter() {
        let mut accum_colors = vec![Color4::black(); *size];
        let mut accum_alphas = vec![0.0f32; *size];
        let src_colors = vec![Color4::white(); *size];
        let src_alphas = vec![0.5f32; *size];
        let weights = vec![1.0f32; *size];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                compositor.composite_batch(
                    black_box(&mut accum_colors),
                    black_box(&mut accum_alphas),
                    black_box(&src_colors),
                    black_box(&src_alphas),
                    black_box(&weights),
                );
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_compositing, bench_batch_compositing);
criterion_main!(benches);
