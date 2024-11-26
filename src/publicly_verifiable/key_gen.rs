//! Defines Key generation for the publicly verifiable PKE scheme, PKE2.

use ark_ec::pairing::Pairing;
use ark_std::{rand::Rng, UniformRand};
use ndarray::{Array2, Axis};
use std::ops::Mul;

use crate::{publicly_verifiable::nizk, Params};

use super::{DecryptKey, EncryptKey};

/// Key generation algorithm for the PKE2 scheme.
///
/// # Example
///
/// ```rust
/// use ark_bls12_381::Bls12_381 as E;
/// use ark_ec::pairing::Pairing;
/// use ark_std::UniformRand;
/// use rand::thread_rng;
/// use rcca::{Params, pke2};
///
/// type G1 = <E as Pairing>::G1Affine;
///
/// let rng = &mut thread_rng();
/// let k = 3;
///
/// let pp = Params::<E>::rand(rng);
/// let (dk, ek) = rcca::pke2(rng, &pp, k);
///
/// let m = G1::rand(rng);
///
/// let ciphertext = ek.encrypt(rng, &pp, m);
/// assert!(ek.verify(&ciphertext));
///
/// let m_prime = dk.decrypt(&ciphertext).unwrap();
///
/// assert_eq!(m, m_prime);
///
/// let ciphertext2 = ek.randomize(rng, &ciphertext);
/// assert!(ciphertext != ciphertext2);
/// assert!(ek.verify(&ciphertext2));
///
/// let m_prime2 = dk.decrypt(&ciphertext2).unwrap();
/// assert_eq!(m, m_prime2);
/// ```
pub fn pke2<E: Pairing, R: Rng>(
    rng: &mut R,
    pp: &Params<E>,
    k: usize,
) -> (DecryptKey<E>, EncryptKey<E>) {
    let crs = nizk::init(rng, pp);
    let Params { p1, p2 } = pp;

    let a = Array2::from_shape_fn((k + 1, 1), |_| E::ScalarField::rand(rng));
    let f = Array2::from_shape_fn((k + 1, 1), |_| E::ScalarField::rand(rng));
    let g = Array2::from_shape_fn((k + 1, 1), |_| E::ScalarField::rand(rng));

    let big_f = Array2::from_shape_fn((k + 1, k + 1), |_| E::ScalarField::rand(rng));

    let big_g = Array2::from_shape_fn((k + 1, k + 2), |_| E::ScalarField::rand(rng));

    let big_d = Array2::from_shape_fn((k + 1, k), |_| E::ScalarField::rand(rng));
    let big_e = Array2::from_shape_fn((k + 1, k), |_| E::ScalarField::rand(rng));

    // transpose of [a]
    let a_t = a.clone().reversed_axes();

    // [D]_1
    let big_d_mat1 = big_d.mapv(|x| p1.mul(x));

    // [E]_2
    let big_e_mat2 = big_e.mapv(|x| p2.mul(x));

    // [a^T D]_1
    let at_d = a_t.dot(&big_d).mapv(|x| p1.mul(x));

    // [fT D]_1
    let ft_d = f.clone().reversed_axes().dot(&big_d).mapv(|x| p1.mul(x));

    // [F^T D]_1
    let big_ft_d = big_f
        .clone()
        .reversed_axes()
        .dot(&big_d)
        .mapv(|x| p1.mul(x));

    // [g^T E]_2
    let gt_e = g.clone().reversed_axes().dot(&big_e).mapv(|x| p2.mul(x));

    // [G^T E]_2
    let big_gt_e = big_g
        .clone()
        .reversed_axes()
        .dot(&big_e)
        .mapv(|x| p2.mul(x));

    // D* = (D^T, (a^T D)^T)^T
    let big_d_star = {
        let mut res = big_d.clone().reversed_axes(); // (k, k+1)
        let at_d_t = a_t.dot(&big_d).reversed_axes(); // (k, 1)
        res.append(Axis(1), at_d_t.view()).unwrap(); // (k, k+2)
        res.reversed_axes()
    };

    // [G D*]_1
    let big_g_d = big_g.dot(&big_d_star).mapv(|x| p1.mul(x));

    // [F E]_2
    let big_f_e = big_f.dot(&big_e).mapv(|x| p2.mul(x));

    (
        DecryptKey {
            a,
            crs: crs.clone(),
        },
        EncryptKey {
            big_d: big_d_mat1,
            big_e: big_e_mat2,
            at_d,
            ft_d,
            big_ft_d,
            gt_e,
            big_gt_e,
            big_g_d,
            big_f_e,
            crs,
        },
    )
}
