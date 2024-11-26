use std::time::Duration;

use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, Criterion};

use ark_bls12_381::Bls12_381 as E;
use ark_ec::pairing::Pairing;
use ark_std::UniformRand;
use rcca::Params;

type G1 = <E as Pairing>::G1Affine;

criterion_group! {
    name = pke1;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(2));
    targets = bench_pke1_encrypt, bench_pke1_decrypt
}

criterion_group! {
    name = pke2;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(4));
    targets = bench_pke2_encrypt, bench_pke2_decrypt
}

criterion_main!(pke1, pke2);

fn bench_pke1_encrypt(c: &mut Criterion) {
    let rng = &mut test_rng();
    let k = 1;

    let pp = Params::<E>::rand(rng);
    let (_dk, ek) = rcca::pke1(rng, &pp, k);

    let m = G1::rand(rng);

    c.bench_function("pke1_encrypt", |b| b.iter(|| ek.encrypt(rng, m)));
}

fn bench_pke1_decrypt(c: &mut Criterion) {
    let rng = &mut test_rng();
    let k = 1;

    let pp = Params::<E>::rand(rng);
    let (dk, ek) = rcca::pke1(rng, &pp, k);

    let m = G1::rand(rng);

    let ciphertext = ek.encrypt(rng, m);

    c.bench_function("pke1_decrypt", |b| b.iter(|| dk.decrypt(&pp, &ciphertext)));
}

fn bench_pke2_encrypt(c: &mut Criterion) {
    let rng = &mut test_rng();
    let k = 1;

    let pp = Params::<E>::rand(rng);
    let (_dk, ek) = rcca::pke2(rng, &pp, k);

    let m = G1::rand(rng);

    c.bench_function("pke2_encrypt", |b| b.iter(|| ek.encrypt(rng, &pp, m)));
}

fn bench_pke2_decrypt(c: &mut Criterion) {
    let rng = &mut test_rng();
    let k = 1;

    let pp = Params::<E>::rand(rng);
    let (dk, ek) = rcca::pke2(rng, &pp, k);

    let m = G1::rand(rng);

    let ciphertext = ek.encrypt(rng, &pp, m);

    c.bench_function("pke2_decrypt", |b| b.iter(|| dk.decrypt(&ciphertext)));
}
