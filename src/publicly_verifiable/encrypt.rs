use ark_ec::{pairing::Pairing, AffineRepr};
use ark_std::{rand::Rng, UniformRand};
use ndarray::{Array, Array2, Axis};

use crate::{
    arith::{dot_1s, dot_2s, dot_e},
    publicly_verifiable::nizk,
    Params,
};

use super::{Ciphertext, Crs};

#[derive(Clone, Debug)]
pub struct EncryptKey<E: Pairing> {
    // ... same as pk in PKE1, except for ft_d and gt_e should be changed to G1 and G2 (see Appendix B.1) ...
    // dim = (k+1, k)
    pub(crate) big_d: Array2<E::G1>,
    // dim = (k+1, k)
    pub(crate) big_e: Array2<E::G2>,
    // dim = (1, k)
    pub(crate) at_d: Array2<E::G1>,
    // dim = (1, k)
    pub(crate) ft_d: Array2<E::G1>,
    // dim = (k+1, k)
    pub(crate) big_ft_d: Array2<E::G1>,
    // dim = (1, k)
    pub(crate) gt_e: Array2<E::G2>,
    // dim = (k+2, k)
    pub(crate) big_gt_e: Array2<E::G2>,
    // dim = (k+1, k)
    pub(crate) big_g_d: Array2<E::G1>,
    // dim = (k+1, k)
    pub(crate) big_f_e: Array2<E::G2>,

    // Common reference string
    pub(crate) crs: Crs<E>,
}

impl<E: Pairing> EncryptKey<E> {
    pub fn encrypt<R: Rng>(&self, rng: &mut R, pp: &Params<E>, m: E::G1Affine) -> Ciphertext<E> {
        let k = self.big_d.dim().1;

        let r = Array2::from_shape_fn((k, 1), |_| E::ScalarField::rand(rng));
        let s = Array2::from_shape_fn((k, 1), |_| E::ScalarField::rand(rng));
        let m = Array::from_elem((1, 1), m.into_group());

        // [u]_1 = [D]_1 * r
        let u = dot_1s::<E>(&self.big_d, &r); // (k+1, 1)

        // [p]_1 = [a^T D]_1 * r + [m]_1
        let p = dot_1s::<E>(&self.at_d, &r) + m; // (1, 1)

        // [x] = ([u^T], p)^T
        let x = {
            let mut tmp = u.reversed_axes(); // (1, k+1)
            tmp.append(Axis(1), p.view()).unwrap(); // (1, k+2)
            tmp.reversed_axes()
        }; // (k+2, 1)

        // [v]_2 = [E]_2 * s
        let v = dot_2s::<E>(&self.big_e, &s); // (k+1, 1)

        // [pi1]_T = [f^T D]_T * r + [[F^T D]_T * r v]_T
        let ft_d_r = dot_1s::<E>(&self.ft_d, &r);
        let big_ft_d_r = dot_1s::<E>(&self.big_ft_d, &r);
        let pi1 = {
            let ones = Array::from_elem((ft_d_r.dim().0, 1), pp.p2);
            dot_e::<E>(&ft_d_r.clone().reversed_axes(), &ones)
                + dot_e(&big_ft_d_r.clone().reversed_axes(), &v)
        };

        // [pi2]_T = [g^T E]_T * s + [[x]_1  [G^T E] * s]_T
        let gt_e_s = dot_2s::<E>(&self.gt_e, &s);
        let big_gt_e_s = dot_2s::<E>(&self.big_gt_e, &s);
        let pi2 = {
            let ones = Array::from_elem((1, gt_e_s.dim().0), pp.p1);
            dot_e(&ones, &gt_e_s.clone()) + dot_e(&x.clone().reversed_axes(), &big_gt_e_s.clone())
        };

        let pi = pi1 + pi2;

        let proof = nizk::prove(
            rng,
            &self.crs,
            &ft_d_r,
            &big_ft_d_r,
            &v,
            &gt_e_s,
            &x,
            &big_gt_e_s,
            &pi,
        );

        Ciphertext { x, v, proof }
    }

    pub fn verify(&self, c: &Ciphertext<E>) -> bool {
        let Ciphertext { x, v, proof } = c;

        nizk::verify(&self.crs, proof, v, x)
    }
}
