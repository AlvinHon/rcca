use ark_ec::pairing::{Pairing, PairingOutput};
use ark_std::{rand::Rng, UniformRand, Zero};
use ndarray::{arr2, Array2};
use std::ops::Mul;

use crate::arith::dot_e;

pub(crate) type Com<G> = Array2<G>; // dim = (2, 1)

pub(crate) type Proof<E> = Array2<PairingOutput<E>>; // dim = (2, 2)

pub(crate) struct CommitmentKeys<E: Pairing> {
    g1: E::G1,
    g2: E::G2,
    // dim = (2, 1)
    u1: Array2<E::G1>,
    // dim = (2, 1)
    u2: Array2<E::G1>,
    // dim = (2, 1)
    v1: Array2<E::G2>,
    // dim = (2, 1)
    v2: Array2<E::G2>,
}

impl<E: Pairing> CommitmentKeys<E> {
    pub(crate) fn rand<R: Rng>(rng: &mut R) -> Self {
        let g1 = E::G1::rand(rng);
        let g2 = E::G2::rand(rng);
        let u1 = arr2(&[[E::G1::rand(rng)], [E::G1::zero()]]);
        let u2 = arr2(&[[E::G1::rand(rng)], [E::G1::zero()]]);
        let v1 = arr2(&[[E::G2::rand(rng)], [E::G2::zero()]]);
        let v2 = arr2(&[[E::G2::rand(rng)], [E::G2::zero()]]);

        Self {
            g1,
            g2,
            u1,
            u2,
            v1,
            v2,
        }
    }

    pub(crate) fn commit_vec_1<R: Rng>(&self, rng: &mut R, xs: Array2<E::G1>) -> Vec<Com<E::G1>> {
        let (m, n) = xs.dim();
        assert!(n == 1);

        let mut res = vec![];
        for i in 0..m {
            let x = xs[(i, 0)];
            let c = self.commit_1(rng, x);
            res.push(c);
        }

        res
    }

    /// Output a commitment (dim = (2, 1)) to x.
    pub(crate) fn commit_1<R: Rng>(&self, rng: &mut R, x: E::G1) -> Com<E::G1> {
        let (r1, r2) = (E::ScalarField::rand(rng), E::ScalarField::rand(rng));
        let zero = E::G1::zero();

        arr2(&[[x, zero]]).reversed_axes()
            + self.u1.mapv(|u| u.mul(r1))
            + self.u2.mapv(|u| u.mul(r2))
    }

    pub(crate) fn commit_vec_2<R: Rng>(&self, rng: &mut R, xs: Array2<E::G2>) -> Vec<Com<E::G2>> {
        let (m, n) = xs.dim();
        assert!(n == 1);

        let mut res = vec![];
        for i in 0..m {
            let x = xs[(i, 0)];
            let c = self.commit_2(rng, x);
            res.push(c);
        }

        res
    }

    /// Output a commitment (dim = (2, 1)) to x.
    pub(crate) fn commit_2<R: Rng>(&self, rng: &mut R, x: E::G2) -> Com<E::G2> {
        let (r1, r2) = (E::ScalarField::rand(rng), E::ScalarField::rand(rng));
        let zero = E::G2::zero();

        arr2(&[[x, zero]]).reversed_axes()
            + self.v1.mapv(|v| v.mul(r1))
            + self.v2.mapv(|v| v.mul(r2))
    }

    /// Output a commitment (dim = (2, 2)) to pi (dim = (1,1)).
    pub(crate) fn commit_t<R: Rng>(&self, rng: &mut R, pi: Array2<PairingOutput<E>>) -> Proof<E> {
        assert!(pi.dim() == (1, 1));
        let pi = pi[(0, 0)];
        let r = Array2::from_shape_fn((2, 2), |_| E::ScalarField::rand(rng));

        // [[pi, 0], [0, 0]] + r11 [u1 v1^T] + r12 [u1 v2^T] + r21 [u2 v1^T] + r22 [u2 v2^T]
        let zero = PairingOutput::zero();
        let v1_rev = self.v1.clone().reversed_axes();
        let v2_rev = self.v2.clone().reversed_axes();

        let l1 = arr2(&[[pi, zero], [zero, zero]]);
        let l2 = dot_e::<E>(&self.u1, &v1_rev).mapv(|x| x.mul(r[(0, 0)]));
        let l3 = dot_e::<E>(&self.u1, &v2_rev).mapv(|x| x.mul(r[(0, 1)]));
        let l4 = dot_e::<E>(&self.u2, &v1_rev).mapv(|x| x.mul(r[(1, 0)]));
        let l5 = dot_e::<E>(&self.u2, &v2_rev).mapv(|x| x.mul(r[(1, 1)]));

        l1 + l2 + l3 + l4 + l5
    }
}
