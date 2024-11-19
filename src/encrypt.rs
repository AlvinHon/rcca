use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr,
};
use ark_std::{rand::Rng, UniformRand};
use ndarray::{Array, Array2, Axis};

use crate::{
    arith::{dot_1s, dot_2s, dot_e, dot_es},
    ciphertext::Ciphertext,
};

#[derive(Clone, Debug)]
pub struct EncryptKey<E: Pairing> {
    // dim = (k+1, k)
    pub(crate) big_d: Array2<E::G1>,
    // dim = (k+1, k)
    pub(crate) big_e: Array2<E::G2>,
    // dim = (1, k)
    pub(crate) at_d: Array2<E::G1>,
    // dim = (1, k)
    pub(crate) ft_d: Array2<PairingOutput<E>>,
    // dim = (k+1, k)
    pub(crate) big_ft_d: Array2<E::G1>,
    // dim = (1, k)
    pub(crate) gt_e: Array2<PairingOutput<E>>,
    // dim = (k+2, k)
    pub(crate) big_gt_e: Array2<E::G2>,
    // dim = (k+1, k)
    pub(crate) big_g_d: Array2<E::G1>,
    // dim = (k+1, k)
    pub(crate) big_f_e: Array2<E::G2>,
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
        let pi1 = dot_es(&self.ft_d, &r)
            + dot_e::<E>(&dot_1s::<E>(&self.big_ft_d, &r).reversed_axes(), &v);

        // [pi2]_T = [g^T E]_T * s + [[x]_1  [G^T E] * s]_T
        let pi2 = dot_es(&self.gt_e, &s)
            + dot_e::<E>(&x.clone().reversed_axes(), &dot_2s::<E>(&self.big_gt_e, &s));

        let pi = pi1 + pi2;

        Ciphertext { x, v, pi }
    }

    pub fn randomize<R: Rng>(&self, rng: &mut R, c: &Ciphertext<E>) -> Ciphertext<E> {
        let k = self.big_d.dim().1;

        let r = Array2::from_shape_fn((k, 1), |_| E::ScalarField::rand(rng));
        let s = Array2::from_shape_fn((k, 1), |_| E::ScalarField::rand(rng));

        // parse x, [x^T]_1 = ([u^T], p)
        let u_t = {
            let mut x_t = c.x.clone().reversed_axes(); // (1, k+2)
            x_t.remove_index(Axis(1), k + 1);
            x_t // (1, k+1)
        };

        // D* = (D^T, (a^T D)^T)^T
        let big_d_star = {
            let mut res = self.big_d.clone().reversed_axes(); // (k, k+1)
            let at_d_t = self.at_d.clone().reversed_axes(); // (k, 1)
            res.append(Axis(1), at_d_t.view()).unwrap(); // (k, k+2)
            res.reversed_axes() // (k+2, k)
        };

        // [x_cap]_1 = [x]_1 + [D*]_1 * r
        let x_cap = &c.x + dot_1s::<E>(&big_d_star, &r); // (k+2, 1)

        // [v_cap]_2 = [v]_2 + [E]_2 * s
        let v_cap = &c.v + dot_2s::<E>(&self.big_e, &s); // (k+1, 1)

        // [pi1_cap]_T = [f^T D]_T * r + [[F^T D]_T * r v_cap]_T + [u^T [FE]_2 s]_T
        let pi1_cap = {
            let part1 = dot_es(&self.ft_d, &r);
            let part2 = dot_e::<E>(&dot_1s::<E>(&self.big_ft_d, &r).reversed_axes(), &v_cap);
            let part3 = dot_e::<E>(&u_t, &dot_2s::<E>(&self.big_f_e, &s));

            part1 + part2 + part3
        };

        // [pi2_cap]_T = [g^T E]_T * s + [[x_cap]_1^T gt_e * s]_T + [([big_g_d]] * r)^T v]_T
        let pi2_cap = {
            let part1 = dot_es(&self.gt_e, &s);
            let part2 = dot_e::<E>(
                &x_cap.clone().reversed_axes(),
                &dot_2s::<E>(&self.big_gt_e, &s),
            );
            let part3 = dot_e::<E>(&dot_1s::<E>(&self.big_g_d, &r).reversed_axes(), &c.v);

            part1 + part2 + part3
        };

        let pi_t_cap = c.pi.clone() + pi1_cap + pi2_cap;
        Ciphertext {
            x: x_cap,
            v: v_cap,
            pi: pi_t_cap,
        }
    }
}
