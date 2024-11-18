use ark_ec::pairing::Pairing;
use ark_std::{rand::Rng, UniformRand};

#[derive(Clone, Copy)]
pub struct Params<E: Pairing> {
    pub(crate) p1: E::G1Affine,
    pub(crate) p2: E::G2Affine,
}

impl<E: Pairing> Params<E> {
    pub fn rand<R: Rng>(rng: &mut R) -> Self {
        Self {
            p1: E::G1Affine::rand(rng),
            p2: E::G2Affine::rand(rng),
        }
    }
}
