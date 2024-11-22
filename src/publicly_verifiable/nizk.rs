use ark_ec::{
    pairing::{Pairing, PairingOutput},
    PrimeGroup,
};
use ark_std::{rand::Rng, UniformRand, Zero};
use ndarray::{arr2, Array2, Axis};
use std::ops::Mul;

use crate::{
    arith::{dot_e, dot_s1, dot_s2},
    Params,
};

type Com<G> = Array2<G>; // dim = (m, 1)

type Pi<G> = Array2<G>; // dim = (2, 2)

pub(crate) struct Proof<E: Pairing> {
    c1: Com<E::G1>,
    d1: Com<E::G2>,
    c2: Com<E::G1>,
    d2: Com<E::G2>,
    theta: Pi<E::G1>,
    phi: Pi<E::G2>,
}

pub(crate) struct CRS<E: Pairing> {
    g1: E::G1,
    g2: E::G2,
    // dim = (2, 2)
    u: Array2<E::G1>,
    // dim = (2, 2)
    v: Array2<E::G2>,
}

impl<E: Pairing> CRS<E> {
    pub(crate) fn rand<R: Rng>(rng: &mut R, pp: &Params<E>) -> Self {
        let Params { p1: g1, p2: g2 } = pp.clone();

        let a1 = E::ScalarField::rand(rng);
        let a2 = E::ScalarField::rand(rng);
        let t1 = E::ScalarField::rand(rng);
        let t2 = E::ScalarField::rand(rng);

        let u = arr2(&[[g1, g1.mul(a1)], [g1.mul(t1), g1.mul(a1 * t1)]]);
        let v = arr2(&[[g2, g2.mul(a2)], [g2.mul(t2), g2.mul(a2 * t2)]]);

        Self { g1, g2, u, v }
    }
}

pub(crate) fn init<E: Pairing, R: Rng>(rng: &mut R, pp: &Params<E>) -> CRS<E> {
    CRS::rand(rng, pp)
}

/// Prove the equation (5) defined in Appendix B.1.
/// pi_t = e([f^T D] r, [1]) + e([F^T D] r, [v]) + e([1], [g^T E] s) + e([x], [G^T E] s)
/// where X1 = [f^T D], X2 = [F^T D], B1 = [1], B2 = [v], Y1 = [g^T E], Y2 = [G^T E], A1 = [1], A2 = [x]
pub(crate) fn prove<E: Pairing, R: Rng>(
    rng: &mut R,
    crs: &CRS<E>,
    ft_d_r: &Array2<E::G1>,     // X1
    big_ft_d_r: &Array2<E::G1>, // X2
    v: &Array2<E::G2>,          // B2
    gt_e_s: &Array2<E::G2>,     // Y1
    x: &Array2<E::G1>,          // A2
    big_gt_e_s: &Array2<E::G2>, // Y2
) -> Proof<E> {
    assert!(ft_d_r.dim() == v.dim());
    assert!(big_ft_d_r.dim() == v.dim());
    assert!(gt_e_s.dim() == x.dim());
    assert!(big_gt_e_s.dim() == x.dim());

    let (m, m_prime) = x.dim();
    let (n, n_prime) = v.dim();
    assert_eq!(m_prime, 1);
    assert_eq!(n_prime, 1);

    // randomness
    let r = Array2::from_shape_fn((m, 2), |_| E::ScalarField::rand(rng));
    let s = Array2::from_shape_fn((n, 2), |_| E::ScalarField::rand(rng));
    let z = Array2::from_shape_fn((2, 2), |_| E::ScalarField::rand(rng));

    // split equation into two parts:
    // 1. e([f^T D] r, [1]) + e([1], [g^T E] s)
    // 2. e([F^T D] r, [v]) + e([x], [G^T E] s)

    // Part 1.
    let c1 = commit_1(&crs, &r, ft_d_r);
    let d1 = commit_2(&crs, &s, gt_e_s);
    let const_a = Array2::from_shape_fn(gt_e_s.dim(), |_| crs.g1.clone());
    let const_b = Array2::from_shape_fn(ft_d_r.dim(), |_| crs.g2.clone());
    let proof1_1 = proof_1(&crs, &s, &const_a, &z);
    let proof1_2 = proof_2(&crs, &r, &const_b, &z);

    // Part 2.
    let c2 = commit_1(&crs, &r, big_ft_d_r);
    let d2 = commit_2(&crs, &s, big_gt_e_s);
    let proof2_1 = proof_1(&crs, &s, &x, &z);
    let proof2_2 = proof_2(&crs, &r, &v, &z);

    // Due to homomorphic behaviour of GS proof, combine the commitments and proofs into one
    let theta = proof1_1 + proof2_1;
    let phi = proof1_2 + proof2_2;

    Proof {
        c1,
        c2,
        d1,
        d2,
        theta,
        phi,
    }
}

pub(crate) fn verify<E: Pairing>(
    crs: &CRS<E>,
    proof: &Proof<E>,
    v: &Array2<E::G2>,
    x: &Array2<E::G1>,
    pi_t: &Array2<PairingOutput<E>>,
) -> bool {
    assert!(v.dim().1 == 1);
    assert!(x.dim().1 == 1);
    assert!(pi_t.dim() == (1, 1));

    let Proof {
        c1,
        c2,
        d1,
        d2,
        theta,
        phi,
    } = proof;

    let m = c1.dim().0;
    let n = d1.dim().0;

    let a1 = Array2::from_shape_fn((m, 1), |_| crs.g1.clone());
    let b1 = Array2::from_shape_fn((n, 1), |_| crs.g2.clone());

    // l(A1) d1 + l(A2) d2 + c1 l(B1) + c2 l(B2) = pi_t + u phi + theta v
    let lhs = dot_e::<E>(&l(&a1).reversed_axes(), d1)
        + dot_e(&x.clone().reversed_axes(), d2)
        + dot_e(&c1.clone().reversed_axes(), &l(&b1))
        + dot_e(&c2.clone().reversed_axes(), &v);
    let rhs = l_t(&pi_t[(0, 0)])
        + dot_e(&crs.u.clone().reversed_axes(), phi)
        + dot_e(&theta.clone().reversed_axes(), &crs.v);
    lhs == rhs
}

fn commit_1<E: Pairing>(crs: &CRS<E>, r: &Array2<E::ScalarField>, x: &Array2<E::G1>) -> Com<E::G1> {
    assert_eq!(x.dim().1, 1);

    // c = l(x) + Ru
    l(x) + dot_s1::<E>(r, &crs.u)
}

fn commit_2<E: Pairing>(crs: &CRS<E>, s: &Array2<E::ScalarField>, y: &Array2<E::G2>) -> Com<E::G2> {
    assert_eq!(y.dim().1, 1);

    // d = l(y) + Sv
    l(y) + dot_s2::<E>(s, &crs.v)
}

fn proof_1<E: Pairing>(
    crs: &CRS<E>,
    s: &Array2<E::ScalarField>,
    a: &Array2<E::G1>,
    z: &Array2<E::ScalarField>,
) -> Pi<E::G1> {
    assert_eq!(a.dim().1, 1);

    // assume gamma does not exist (i.e. all zeros) in the equation, we have:
    // theta = S^T l(a) + Zu
    dot_s1::<E>(&s.clone().reversed_axes(), &l(a)) + dot_s1::<E>(&z, &crs.u)
}

fn proof_2<E: Pairing>(
    crs: &CRS<E>,
    r: &Array2<E::ScalarField>,
    b: &Array2<E::G2>,
    z: &Array2<E::ScalarField>,
) -> Pi<E::G2> {
    assert_eq!(b.dim().1, 1);

    // assume gamma does not exist (i.e. all zeros) in the equation, we have:
    // theta = R^T l(b) - Z^T v
    dot_s2::<E>(&r.clone().reversed_axes(), &l(b)) - dot_s2::<E>(&z.clone().reversed_axes(), &crs.v)
}

/// l(a) = [a | 0]
fn l<G: PrimeGroup>(a: &Array2<G>) -> Array2<G> {
    let mut a = a.clone();
    let m = a.dim().0;
    let zeros = Array2::from_elem((m, 1), G::zero());
    a.append(Axis(1), zeros.view()).unwrap(); // dim = (m, 2)
    a
}

/// l(t) = [[t, 0], [0, 0]]
fn l_t<E: Pairing>(t: &PairingOutput<E>) -> Array2<PairingOutput<E>> {
    arr2(&[
        [t.clone(), PairingOutput::zero()],
        [PairingOutput::zero(), PairingOutput::zero()],
    ])
}

#[cfg(test)]
mod test {

    use ark_bls12_381::Bls12_381 as E;
    use ark_ec::pairing::Pairing;
    use ark_std::{test_rng, UniformRand};

    type G1 = <E as Pairing>::G1;
    type G2 = <E as Pairing>::G2;
    type Fr = <E as Pairing>::ScalarField;

    use super::*;

    #[test]
    fn test_ayxb_proof() {
        let rng = &mut test_rng();
        let pp = Params::<E>::rand(rng);
        let crs = CRS::rand(rng, &pp);

        // test simple equation e(a, y) e(x, b) = t
        let a = arr2(&[[G1::rand(rng)], [G1::rand(rng)]]);
        let x = arr2(&[[G1::rand(rng)], [G1::rand(rng)]]);
        let b = arr2(&[[G2::rand(rng)], [G2::rand(rng)]]);
        let y = arr2(&[[G2::rand(rng)], [G2::rand(rng)]]);

        let t =
            dot_e::<E>(&a.clone().reversed_axes(), &y) + dot_e::<E>(&x.clone().reversed_axes(), &b);
        let t = l_t(&t[(0, 0)]);

        let r = Array2::from_shape_fn((2, 2), |_| Fr::rand(rng)); // dim = (m, 2) = (2, 2)
        let s = Array2::from_shape_fn((2, 2), |_| Fr::rand(rng)); // dim = (n, 2) = (2, 2)
        let z = Array2::from_shape_fn((2, 2), |_| Fr::rand(rng));

        let c = commit_1(&crs, &r, &x);
        let d = commit_2(&crs, &s, &y);
        let theta = proof_1(&crs, &s, &a, &z);
        let phi = proof_2(&crs, &r, &b, &z);

        // check `verify` algorithm:
        // l(A) d + c l(B) = pi_t + u phi + theta v
        let lhs = dot_e(&l(&a).reversed_axes(), &d) + dot_e(&c.reversed_axes(), &l(&b));
        let rhs = t + dot_e(&crs.u.reversed_axes(), &phi) + dot_e(&theta.reversed_axes(), &crs.v);
        assert!(lhs == rhs);
    }
}
