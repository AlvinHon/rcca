//! Defines the `EncryptKey` struct and its methods, for the publicly verifiable PKE scheme, PKE2.

use ark_ec::{pairing::Pairing, AffineRepr};
use ark_std::{rand::Rng, UniformRand};
use ndarray::{Array, Array2, Axis};

use crate::{
    arith::{dot_1s, dot_2s, dot_e},
    publicly_verifiable::nizk,
    Params,
};

use super::{Ciphertext, Crs};

/// The encryption key used in the PKE2 scheme.
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
    /// Encrypt the message `m` using the public key.
    pub fn encrypt<R: Rng>(&self, rng: &mut R, pp: &Params<E>, m: E::G1Affine) -> Ciphertext<E> {
        // Implements the encryption algorithm in the PKE2 scheme in the section 4, aka.
        // the algorithm `Enc` in the figure 5 of the paper.

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
            &x,
            &gt_e_s,
            &big_gt_e_s,
            &ft_d_r,
            &big_ft_d_r,
            &v,
            &pi,
        );

        Ciphertext { x, v, proof }
    }

    /// Verify the ciphertext `c`.
    pub fn verify(&self, c: &Ciphertext<E>) -> bool {
        // Implements the verification algorithm in the PKE2 scheme in the section 4, aka.
        // the algorithm `Ver` in the figure 5 of the paper.

        let Ciphertext { x, v, proof } = c;

        nizk::verify(&self.crs, proof, v, x)
    }

    /// Randomize the ciphertext `c` with a new randomness.
    ///
    /// ## Note
    /// This is different from the original approach in the paper, but serving the same purpose,
    /// that is to "randomize" the `v` and `x` in the ciphertext while keeping the verification equation holds.
    pub fn randomize<R: Rng>(&self, rng: &mut R, c: &Ciphertext<E>) -> Ciphertext<E> {
        let k = self.big_d.dim().1;

        let r = Array2::from_shape_fn((k, 1), |_| E::ScalarField::rand(rng));
        let s = Array2::from_shape_fn((k, 1), |_| E::ScalarField::rand(rng));

        // D* = (D^T, (a^T D)^T)^T
        let big_d_star = {
            let mut res = self.big_d.clone().reversed_axes(); // (k, k+1)
            let at_d_t = self.at_d.clone().reversed_axes(); // (k, 1)
            res.append(Axis(1), at_d_t.view()).unwrap(); // (k, k+2)
            res.reversed_axes() // (k+2, k)
        };

        // [x_cap]_1 = [x]_1 + [D*]_1 * r
        let x_delta = dot_1s::<E>(&big_d_star, &r);
        let x_cap = &c.x + &x_delta; // (k+2, 1)

        // [v_cap]_2 = [v]_2 + [E]_2 * s
        let v_delta = dot_2s::<E>(&self.big_e, &s);
        let v_cap = &c.v + &v_delta; // (k+1, 1)

        let proof = nizk::zkeval(rng, &self.crs, &c.proof, &v_cap, &v_delta, &x_cap, &x_delta);

        Ciphertext {
            x: x_cap,
            v: v_cap,
            proof,
        }
    }

    /// Implements the randomization algorithm in the PKE2 scheme in the section 4, aka.
    /// the algorithm `Rand` in the figure 5 of the paper.
    ///
    /// This function may be implemented incorrectly, or there are some mistakes in the paper.
    /// This function exists for debugging and studying purposes.
    #[allow(dead_code)]
    fn randomize_original<R: Rng>(
        &self,
        rng: &mut R,
        pp: &Params<E>,
        c: &Ciphertext<E>,
    ) -> Ciphertext<E> {
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
        let ft_d_r = dot_1s::<E>(&self.ft_d, &r);
        let big_ft_d_r = dot_1s::<E>(&self.big_ft_d, &r);
        let big_f_e_s = dot_2s::<E>(&self.big_f_e, &s);
        let pi1_cap = {
            let ones = Array::from_elem((ft_d_r.dim().0, 1), pp.p2);
            let part1 = dot_e::<E>(&ft_d_r.clone().reversed_axes(), &ones);
            let part2 = dot_e::<E>(&big_ft_d_r.clone().reversed_axes(), &v_cap);
            let part3 = dot_e::<E>(&u_t, &big_f_e_s);

            part1 + part2 + part3
        };

        // [pi2_cap]_T = [g^T E]_T * s + [[x_cap]_1^T [G^T E] * s]_T + [([big_g_d]] * r)^T v]_T
        let gt_e_s = dot_2s::<E>(&self.gt_e, &s);
        let big_gt_e_s = dot_2s::<E>(&self.big_gt_e, &s);
        let big_g_d_r = dot_1s::<E>(&self.big_g_d, &r);
        let pi2_cap = {
            let ones = Array::from_elem((1, gt_e_s.dim().0), pp.p1);
            let part1 = dot_e::<E>(&ones, &gt_e_s.clone());
            let part2 = dot_e::<E>(&x_cap.clone().reversed_axes(), &big_gt_e_s);
            let part3 = dot_e::<E>(&big_g_d_r.clone().reversed_axes(), &c.v);

            part1 + part2 + part3
        };

        let pi_cap_t = pi1_cap + pi2_cap;

        let proof = nizk::zkeval_original(
            &c.proof,
            &ft_d_r,
            &big_ft_d_r,
            &gt_e_s,
            &big_gt_e_s,
            &pi_cap_t,
        );

        Ciphertext {
            x: x_cap,
            v: v_cap,
            proof,
        }
    }
}
