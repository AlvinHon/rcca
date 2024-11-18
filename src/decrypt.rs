use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr, CurveGroup, PrimeGroup,
};
use ark_std::{rand::Rng, UniformRand, Zero};
use ndarray::{Array, Array2, Axis};
use std::ops::{Add, Mul};

use crate::{arith::dot_s1, ciphertext::Ciphertext, Params};

pub struct DecryptKey<E: Pairing> {
    // dim = (k+1, 1)
    pub(crate) a: Array2<E::ScalarField>,
    // dim = (k+1, 1)
    pub(crate) f: Array2<E::ScalarField>,
    // dim = (k+1, 1)
    pub(crate) g: Array2<E::ScalarField>,
    // dim = (k+1, k+1)
    pub(crate) big_f: Array2<E::ScalarField>,
    // dim = (k+1, k+2)
    pub(crate) big_g: Array2<E::ScalarField>,
}

impl<E: Pairing> DecryptKey<E> {
    pub fn decrypt(&self, c: &Ciphertext<E>) -> E::G1Affine {
        let k = self.a.len() - 1;

        // parse x, [x^T]_1 = ([u^T], p)
        let (u_t, p) = {
            let mut x_t = c.x.clone().reversed_axes(); // (1, k+2)
            let p = x_t[(0, k + 1)];
            x_t.remove_index(Axis(1), k + 1);
            let u_t = x_t; // (1, k+1)
            (u_t, p)
        };

        // m = p - [a^T u]_1
        let a_t = self.a.clone().reversed_axes();
        let m = (p - dot_s1::<E>(&a_t, &u_t.reversed_axes())[[0, 0]]).into();

        m
    }
}
