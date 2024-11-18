use ark_ec::pairing::Pairing;
use ark_std::{rand::Rng, UniformRand};
use ndarray::{Array2, Axis};
use std::ops::Mul;

use crate::{arith::dot_e, DecryptKey, EncryptKey, Params};

pub fn ken_gen<E: Pairing, R: Rng>(
    rng: &mut R,
    pp: &Params<E>,
    k: usize,
) -> (DecryptKey<E>, EncryptKey<E>) {
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

    // [fT D]_T
    let f_t_mat1 = Array2::from_shape_fn((1, k + 1), |(_, j)| p1.mul(f[[j, 0]]));
    let big_d_mat2 = Array2::from_shape_fn((k + 1, k), |(i, j)| p2.mul(big_d[[i, j]]));
    let ft_d = dot_e::<E>(&f_t_mat1, &big_d_mat2);

    // [F^T D]_1
    let big_ft_d = big_f
        .clone()
        .reversed_axes()
        .dot(&big_d)
        .mapv(|x| p1.mul(x));

    // // [g^T E]_T
    let g_t_mat1 = Array2::from_shape_fn((1, k + 1), |(_, j)| p1.mul(g[[j, 0]]));
    let gt_e = dot_e::<E>(&g_t_mat1, &big_e_mat2);

    // [G^T E]_2
    let big_gt_e_mat2 = big_g
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
    let big_g_d_mat1 = big_g.dot(&big_d_star).mapv(|x| p1.mul(x));

    // [F E]_2
    let big_f_e_mat2 = big_f.dot(&big_e).mapv(|x| p2.mul(x));

    (
        DecryptKey {
            a,
            f,
            g,
            big_f,
            big_g,
        },
        EncryptKey {
            big_d: big_d_mat1.mapv(|g1| g1.into()),
            big_e: big_e_mat2.mapv(|g2| g2.into()),
            at_d: at_d.mapv(|g1| g1.into()),
            ft_d,
            big_ft_d: big_ft_d.mapv(|g1| g1.into()),
            gt_e,
            big_gt_e: big_gt_e_mat2.mapv(|g2| g2.into()),
            big_g_d: big_g_d_mat1.mapv(|g1| g1.into()),
            big_f_e: big_f_e_mat2.mapv(|g2| g2.into()),
        },
    )
}
