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

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Proof<E: Pairing> {
    c1: Com<E::G1>,
    d1: Com<E::G2>,
    c2: Com<E::G1>,
    d2: Com<E::G2>,
    ct: Com<PairingOutput<E>>,
    theta: Pi<E::G1>,
    phi: Pi<E::G2>,
}

#[derive(Clone, Debug)]
pub(crate) struct Crs<E: Pairing> {
    g1: E::G1,
    g2: E::G2,
    // dim = (2, 2)
    u: Array2<E::G1>,
    // dim = (2, 2)
    v: Array2<E::G2>,
}

impl<E: Pairing> Crs<E> {
    fn rand<R: Rng>(rng: &mut R, pp: &Params<E>) -> Self {
        let Params { p1: g1, p2: g2 } = *pp;

        let a1 = E::ScalarField::rand(rng);
        let a2 = E::ScalarField::rand(rng);
        let t1 = E::ScalarField::rand(rng);
        let t2 = E::ScalarField::rand(rng);

        let u = arr2(&[[g1, g1.mul(a1)], [g1.mul(t1), g1.mul(a1 * t1)]]);
        let v = arr2(&[[g2, g2.mul(a2)], [g2.mul(t2), g2.mul(a2 * t2)]]);

        Self { g1, g2, u, v }
    }
}

pub(crate) fn init<E: Pairing, R: Rng>(rng: &mut R, pp: &Params<E>) -> Crs<E> {
    Crs::rand(rng, pp)
}

/// Prove the equation (5) defined in Appendix B.1.
///
/// ```text
/// pi_t = e([f^T D] r, [1]) + e([F^T D] r, [v]) + e([1], [g^T E] s) + e([x], [G^T E] s)
/// ```
///
/// where X1 = [f^T D], X2 = [F^T D], B1 = [1], B2 = [v], Y1 = [g^T E], Y2 = [G^T E], A1 = [1], A2 = [x]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove<E: Pairing, R: Rng>(
    rng: &mut R,
    crs: &Crs<E>,
    x: &Array2<E::G1>,          // A2
    gt_e_s: &Array2<E::G2>,     // Y1
    big_gt_e_s: &Array2<E::G2>, // Y2
    ft_d_r: &Array2<E::G1>,     // X1
    big_ft_d_r: &Array2<E::G1>, // X2
    v: &Array2<E::G2>,          // B2
    pi: &Array2<PairingOutput<E>>,
) -> Proof<E> {
    assert!(ft_d_r.dim() == (1, 1));
    assert!(big_ft_d_r.dim() == v.dim());
    assert!(gt_e_s.dim() == (1, 1));
    assert!(big_gt_e_s.dim() == x.dim());
    assert!(pi.dim() == (1, 1));

    // split equation into two parts:
    // 1. e([f^T D] r, [1]) + e([1], [g^T E] s)
    // 2. e([F^T D] r, [v]) + e([x], [G^T E] s)

    // Part 1.
    let const_a = Array2::from_shape_fn(gt_e_s.dim(), |_| crs.g1);
    let const_b = Array2::from_shape_fn(ft_d_r.dim(), |_| crs.g2);
    let (c1, d1, proof1_1, proof1_2) = prove_ayxb(rng, crs, &const_a, gt_e_s, ft_d_r, &const_b);

    // Part 2.
    let (c2, d2, proof2_1, proof2_2) = prove_ayxb(rng, crs, x, big_gt_e_s, big_ft_d_r, v);

    // Due to homomorphic behaviour of GS proof, combine the commitments and proofs into one
    let theta = proof1_1 + proof2_1;
    let phi = proof1_2 + proof2_2;

    // Commit to the proof pi
    let r_ct = Array2::from_shape_fn((2, 2), |_| E::ScalarField::rand(rng));
    let ct = commit_t(crs, pi[(0, 0)], &r_ct);
    // Adapt proof phi.
    // The calculation comes from the idea that the lhs of the verification equation will have the additional term in ct, that is r[u^T v].
    // In verification step, we applied the updated phi' = (phi - r[v]) to rhs of the equation.
    // so that the additional term in rhs will be -e([u], r[v]) which cancels out the additional term in ct.
    let phi = phi - dot_s2::<E>(&r_ct, &crs.v);

    Proof {
        c1,
        c2,
        d1,
        d2,
        ct,
        theta,
        phi,
    }
}

pub(crate) fn verify<E: Pairing>(
    crs: &Crs<E>,
    proof: &Proof<E>,
    v: &Array2<E::G2>,
    x: &Array2<E::G1>,
) -> bool {
    assert!(v.dim().1 == 1);
    assert!(x.dim().1 == 1);

    let Proof {
        c1,
        c2,
        d1,
        d2,
        theta,
        phi,
        ct,
    } = proof;

    let m = c1.dim().0;
    let n = d1.dim().0;

    let a1 = Array2::from_shape_fn((m, 1), |_| crs.g1);
    let b1 = Array2::from_shape_fn((n, 1), |_| crs.g2);

    // l(A1) d1 + l(A2) d2 + c1 l(B1) + c2 l(B2) = pi_t + u phi + theta v
    let lhs = dot_e::<E>(&l(&a1).reversed_axes(), d1)
        + dot_e(&l(x).reversed_axes(), d2)
        + dot_e(&c1.clone().reversed_axes(), &l(&b1))
        + dot_e(&c2.clone().reversed_axes(), &l(v));
    let rhs = ct
        + dot_e(&crs.u.clone().reversed_axes(), phi)
        + dot_e(&theta.clone().reversed_axes(), &crs.v);
    lhs == rhs
}

/// Implements the approach to randomize a proof in a ciphertext.
///
/// ## Note
/// This is different from the original approach in the paper, but serving the same purpose,
/// that is to "randomize" the `v` and `x` in the ciphertext while keeping the verification equation holds.
///
/// ## Calculation applied
///
/// In GS proof, we have to check the equation (simplified version)
///
/// ```text
/// l(A) * d + c * l(B) = pi_t + u * phi + theta * v
/// ```
///
/// Note that A and B are constants w.r.t. the terms 'x' and 'v' in the equation.
/// The algorithm defined in encryption in the paper requires adding additional terms to `x` and `v` in the ciphertext.
/// Hence, we have the additional term in lhs, that is,
///
/// ```text
/// c * l(B') = c * l(B) + c * l(B_delta)
/// l(A') * d = l(A) * d + l(A_delta) * d
/// ```
///
/// To cancel out the additional term in lhs, we need to add the term to the rhs of the equation.
/// Thus, the term `c_t` will be updated accordingly.
pub(crate) fn zkeval<E: Pairing>(
    proof: &Proof<E>,
    v_delta: &Array2<E::G2>,
    x_delta: &Array2<E::G1>,
) -> Proof<E> {
    let Proof {
        c1,
        c2,
        d1,
        d2,
        theta,
        phi,
        ct,
    } = proof.clone();

    let c_delta_term = dot_e::<E>(&c2.clone().reversed_axes(), &l(v_delta));
    let d_delta_term = dot_e::<E>(&l(x_delta).reversed_axes(), &d2);

    // TODO and then randomize the original GS proof system, i.e. the commitments and proofs.

    Proof {
        c1,
        c2,
        d1,
        d2,
        theta,
        phi,
        ct: ct + c_delta_term + d_delta_term,
    }
}

/// Implements the approach to randomize a proof in a ciphertext defined in Appendix B.1.
///
/// We can add to the commitments of `[f^T D] r` and `[F^T D] r` the values `[f^T D] r`
/// and `[F^T D] r` (similarly for the s components).
///
/// the commitmemt c_t can also be updated by adding the value pi_cap_t = pi1_cap + pi2_cap
/// where
///
/// ```text
/// pi1_cap = e([f^T D] r, [1]) + e([F^T D] r, [v_cap]) + e([u], [FE] s)
/// pi2_cap = e([1], [g^T E] s) + e([x_cap], [G^T E] s) + e([GD^T] r, [v])
/// ```
///
/// ## Note
/// This function may be implemented incorrectly, or there are some mistakes in the paper.
/// This function exists for debugging and studying purposes.
pub(crate) fn zkeval_original<E: Pairing>(
    proof: &Proof<E>,
    ft_d_r: &Array2<E::G1>,
    big_ft_d_r: &Array2<E::G1>,
    gt_e_s: &Array2<E::G2>,
    big_gt_e_s: &Array2<E::G2>,
    pi_cap_t: &Array2<PairingOutput<E>>,
) -> Proof<E> {
    assert!(pi_cap_t.dim() == (1, 1));

    let Proof {
        c1,
        c2,
        d1,
        d2,
        theta,
        phi,
        ct,
    } = proof;

    let c1 = c1 + l(ft_d_r);
    let c2 = c2 + l(big_ft_d_r);
    let d1 = d1 + l(gt_e_s);
    let d2 = d2 + l(big_gt_e_s);

    let ct = ct + l_t(&pi_cap_t[(0, 0)]);

    // TODO and then randomize the original GS proof system, i.e. the commitments and proofs.

    Proof {
        c1,
        c2,
        d1,
        d2,
        theta: theta.clone(),
        phi: phi.clone(),
        ct,
    }
}

#[allow(clippy::type_complexity)]
fn prove_ayxb<E: Pairing, R: Rng>(
    rng: &mut R,
    crs: &Crs<E>,
    a: &Array2<E::G1>,
    y: &Array2<E::G2>,
    x: &Array2<E::G1>,
    b: &Array2<E::G2>,
) -> (Com<E::G1>, Com<E::G2>, Pi<E::G1>, Pi<E::G2>) {
    assert!(a.dim() == y.dim());
    assert!(b.dim() == x.dim());
    let (m, m_prime) = x.dim();
    let (n, n_prime) = y.dim();
    assert_eq!(m_prime, 1);
    assert_eq!(n_prime, 1);

    let r = Array2::from_shape_fn((m, 2), |_| E::ScalarField::rand(rng));
    let s = Array2::from_shape_fn((n, 2), |_| E::ScalarField::rand(rng));
    let z = Array2::from_shape_fn((2, 2), |_| E::ScalarField::rand(rng));

    let c = commit_1(crs, &r, x);
    let d = commit_2(crs, &s, y);
    let theta = proof_1(crs, &s, a, &z);
    let phi = proof_2(crs, &r, b, &z);

    (c, d, theta, phi)
}

fn commit_1<E: Pairing>(crs: &Crs<E>, r: &Array2<E::ScalarField>, x: &Array2<E::G1>) -> Com<E::G1> {
    assert_eq!(x.dim().1, 1);

    // c = l(x) + Ru
    l(x) + dot_s1::<E>(r, &crs.u)
}

fn commit_2<E: Pairing>(crs: &Crs<E>, s: &Array2<E::ScalarField>, y: &Array2<E::G2>) -> Com<E::G2> {
    assert_eq!(y.dim().1, 1);

    // d = l(y) + Sv
    l(y) + dot_s2::<E>(s, &crs.v)
}

fn commit_t<E: Pairing>(
    crs: &Crs<E>,
    pi: PairingOutput<E>,
    rij: &Array2<E::ScalarField>,
) -> Com<PairingOutput<E>> {
    let r_v = dot_s2::<E>(rij, &crs.v);
    let u_rv = dot_e::<E>(&crs.u.clone().reversed_axes(), &r_v);
    l_t(&pi) + u_rv
}

fn proof_1<E: Pairing>(
    crs: &Crs<E>,
    s: &Array2<E::ScalarField>,
    a: &Array2<E::G1>,
    z: &Array2<E::ScalarField>,
) -> Pi<E::G1> {
    assert_eq!(a.dim().1, 1);

    // assume gamma does not exist (i.e. all zeros) in the equation, we have:
    // theta = S^T l(a) + Zu
    dot_s1::<E>(&s.clone().reversed_axes(), &l(a)) + dot_s1::<E>(z, &crs.u)
}

fn proof_2<E: Pairing>(
    crs: &Crs<E>,
    r: &Array2<E::ScalarField>,
    b: &Array2<E::G2>,
    z: &Array2<E::ScalarField>,
) -> Pi<E::G2> {
    assert_eq!(b.dim().1, 1);

    // assume gamma does not exist (i.e. all zeros) in the equation, we have:
    // phi = R^T l(b) - Z^T v
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
        [*t, PairingOutput::zero()],
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
        let crs = Crs::rand(rng, &pp);

        // test simple equation e(a, y) e(x, b) = t
        let a = arr2(&[[G1::rand(rng)], [G1::rand(rng)], [G1::rand(rng)]]);
        let x = arr2(&[[G1::rand(rng)], [G1::rand(rng)], [G1::rand(rng)]]);
        let b = arr2(&[[G2::rand(rng)], [G2::rand(rng)], [G2::rand(rng)]]);
        let y = arr2(&[[G2::rand(rng)], [G2::rand(rng)], [G2::rand(rng)]]);

        let t =
            dot_e::<E>(&a.clone().reversed_axes(), &y) + dot_e::<E>(&x.clone().reversed_axes(), &b);
        let t = l_t(&t[(0, 0)]);

        let r = Array2::from_shape_fn((3, 2), |_| Fr::rand(rng)); // dim = (m, 2) = (3, 2)
        let s = Array2::from_shape_fn((3, 2), |_| Fr::rand(rng)); // dim = (n, 2) = (3, 2)
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

    #[test]
    fn test_ayxb_proof_and_commit_t() {
        let rng = &mut test_rng();
        let pp = Params::<E>::rand(rng);
        let crs = Crs::rand(rng, &pp);

        // test simple equation e(a, y) e(x, b) = t
        let a = arr2(&[[G1::rand(rng)], [G1::rand(rng)], [G1::rand(rng)]]);
        let x = arr2(&[[G1::rand(rng)], [G1::rand(rng)], [G1::rand(rng)]]);
        let b = arr2(&[[G2::rand(rng)], [G2::rand(rng)], [G2::rand(rng)]]);
        let y = arr2(&[[G2::rand(rng)], [G2::rand(rng)], [G2::rand(rng)]]);

        let t =
            dot_e::<E>(&a.clone().reversed_axes(), &y) + dot_e::<E>(&x.clone().reversed_axes(), &b);

        let r = Array2::from_shape_fn((3, 2), |_| Fr::rand(rng)); // dim = (m, 2) = (3, 2)
        let s = Array2::from_shape_fn((3, 2), |_| Fr::rand(rng)); // dim = (n, 2) = (3, 2)
        let z = Array2::from_shape_fn((2, 2), |_| Fr::rand(rng));

        let c = commit_1(&crs, &r, &x);
        let d = commit_2(&crs, &s, &y);
        let theta = proof_1(&crs, &s, &a, &z);
        let phi = proof_2(&crs, &r, &b, &z);

        // commit t
        let r = Array2::from_shape_fn((2, 2), |_| Fr::rand(rng));
        let t = commit_t(&crs, t[(0, 0)], &r);

        // adapt proof phi
        let rv = dot_s2::<E>(&r, &crs.v);
        let phi = phi - rv;

        // check `verify` algorithm:
        // l(A) d + c l(B) = pi_t' + u phi' + theta v
        let lhs = dot_e(&l(&a).reversed_axes(), &d) + dot_e(&c.reversed_axes(), &l(&b));
        let rhs = t + dot_e(&crs.u.reversed_axes(), &phi) + dot_e(&theta.reversed_axes(), &crs.v);
        assert!(lhs == rhs);
    }

    #[test]
    fn test_l() {
        let rng = &mut test_rng();
        let a1 = arr2(&[[G1::rand(rng)], [G1::rand(rng)]]);
        let a2 = arr2(&[[G1::rand(rng)], [G1::rand(rng)]]);
        let a = &a1 + &a2;
        assert!(l(&a1) + l(&a2) == l(&a));

        let b = arr2(&[[G2::rand(rng)], [G2::rand(rng)]]);

        let e_a1b = dot_e::<E>(&l(&a1).reversed_axes(), &b);
        let e_a2b = dot_e::<E>(&l(&a2).reversed_axes(), &b);
        let e_ab = dot_e::<E>(&l(&a).reversed_axes(), &b);
        assert!(e_a1b + e_a2b == e_ab);
    }
}
