use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr, CurveGroup, PrimeGroup,
};
use ark_std::{rand::Rng, UniformRand, Zero};
use ndarray::{Array, Array2, Axis};
use std::ops::{Add, Mul};

use crate::{
    arith::{dot_1s, dot_2s},
    ciphertext::Ciphertext,
};

#[derive(Debug)]
pub struct EncryptKey<E: Pairing> {
    // dim = (k+1, k)
    pub(crate) big_d: Array2<E::G1Affine>,
    // dim = (k+1, k)
    pub(crate) big_e: Array2<E::G2Affine>,
    // dim = (1, k)
    pub(crate) at_d: Array2<E::G1Affine>,
    // dim = (1, k)
    pub(crate) ft_d: Array2<PairingOutput<E>>,
    // dim = (k+1, k)
    pub(crate) big_ft_d: Array2<E::G1Affine>,
    // dim = (1, k)
    pub(crate) gt_e: Array2<PairingOutput<E>>,
    // dim = (k+2, k)
    pub(crate) big_gt_e: Array2<E::G2Affine>,
    // dim = (k+1, k)
    pub(crate) big_g_d: Array2<E::G1Affine>,
    // dim = (k+1, k)
    pub(crate) big_f_e: Array2<E::G2Affine>,
}

impl<E: Pairing> EncryptKey<E> {
    pub fn encrypt<R: Rng>(&self, rng: &mut R, m: E::G1Affine) -> Ciphertext<E> {
        let k = self.big_d.dim().1;

        let r = Array2::from_shape_fn((k, 1), |_| E::ScalarField::rand(rng));
        let s = Array2::from_shape_fn((k, 1), |_| E::ScalarField::rand(rng));
        let m = Array::from_elem((1, 1), m.into_group());

        // [u]_1 = [D]_1 * r
        let u = dot_1s::<E>(&self.big_d, &r); // (k+1, 1)

        // [p]_1 = [a^T D]_1 * r + [m]_1
        let p = dot_1s::<E>(&self.at_d, &r).mapv(|x| x.into_group()) + m; // (1, 1)

        // [x] = ([u^T], p)^T
        let x = {
            let mut tmp = u.reversed_axes(); // (1, k+1)
            tmp.append(Axis(1), p.mapv(|x| x.into()).view()).unwrap(); // (1, k+2)
            tmp.reversed_axes()
        };

        // [v]_2 = [E]_2 * s
        let v = dot_2s::<E>(&self.big_e, &s); // (k+1, 1)

        Ciphertext { x, v }
    }
}
