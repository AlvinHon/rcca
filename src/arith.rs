use ark_ec::pairing::{Pairing, PairingOutput};
use ark_std::Zero;
use ndarray::Array2;
use std::ops::Mul;

pub(crate) fn dot_e<E: Pairing>(a: &Array2<E::G1>, b: &Array2<E::G2>) -> Array2<PairingOutput<E>> {
    let (m, n_prime) = a.dim();
    let (m_prime, n) = b.dim();
    assert!(n_prime == m_prime);

    let mut res = Array2::from_elem((m, n), PairingOutput::zero());
    for i in 0..m {
        for j in 0..n {
            let mut sum = PairingOutput::zero();
            for k in 0..n_prime {
                sum += E::pairing(a[[i, k]], b[[k, j]]);
            }
            res[[i, j]] = sum;
        }
    }

    res
}

pub(crate) fn dot_e_rev<E: Pairing>(
    a: &Array2<E::G2>,
    b: &Array2<E::G1>,
) -> Array2<PairingOutput<E>> {
    let (m, n_prime) = a.dim();
    let (m_prime, n) = b.dim();
    assert!(n_prime == m_prime);

    let mut res = Array2::from_elem((m, n), PairingOutput::zero());
    for i in 0..m {
        for j in 0..n {
            let mut sum = PairingOutput::zero();
            for k in 0..n_prime {
                sum += E::pairing(b[[k, j]], a[[i, k]]);
            }
            res[[i, j]] = sum;
        }
    }

    res
}

pub(crate) fn dot_es<E: Pairing>(
    a: &Array2<PairingOutput<E>>,
    b: &Array2<E::ScalarField>,
) -> Array2<PairingOutput<E>> {
    let (m, n_prime) = a.dim();
    let (m_prime, n) = b.dim();
    assert!(n_prime == m_prime);

    let mut res = Array2::from_elem((m, n), PairingOutput::zero());
    for i in 0..m {
        for j in 0..n {
            let mut sum = PairingOutput::zero();
            for k in 0..n_prime {
                sum += a[[i, k]].mul(b[[k, j]]);
            }
            res[[i, j]] = sum;
        }
    }

    res
}

pub(crate) fn dot_1s<E: Pairing>(a: &Array2<E::G1>, b: &Array2<E::ScalarField>) -> Array2<E::G1> {
    let (m, n_prime) = a.dim();
    let (m_prime, n) = b.dim();
    assert!(n_prime == m_prime);

    let mut res = Array2::from_elem((m, n), E::G1::zero());
    for i in 0..m {
        for j in 0..n {
            let mut sum = E::G1::zero();
            for k in 0..n_prime {
                sum += a[[i, k]].mul(b[[k, j]]);
            }
            res[[i, j]] = sum;
        }
    }

    res
}

pub(crate) fn dot_s1<E: Pairing>(a: &Array2<E::ScalarField>, b: &Array2<E::G1>) -> Array2<E::G1> {
    let (m, n_prime) = a.dim();
    let (m_prime, n) = b.dim();
    assert!(n_prime == m_prime);

    let mut res = Array2::from_elem((m, n), E::G1::zero());
    for i in 0..m {
        for j in 0..n {
            let mut sum = E::G1::zero();
            for k in 0..n_prime {
                sum += b[[k, j]].mul(a[[i, k]]);
            }
            res[[i, j]] = sum;
        }
    }

    res
}

pub(crate) fn dot_2s<E: Pairing>(a: &Array2<E::G2>, b: &Array2<E::ScalarField>) -> Array2<E::G2> {
    let (m, n_prime) = a.dim();
    let (m_prime, n) = b.dim();
    assert!(n_prime == m_prime);

    let mut res = Array2::from_elem((m, n), E::G2::zero());
    for i in 0..m {
        for j in 0..n {
            let mut sum = E::G2::zero();
            for k in 0..n_prime {
                sum += a[[i, k]].mul(b[[k, j]]);
            }
            res[[i, j]] = sum;
        }
    }

    res
}

pub(crate) fn dot_s2<E: Pairing>(a: &Array2<E::ScalarField>, b: &Array2<E::G2>) -> Array2<E::G2> {
    let (m, n_prime) = a.dim();
    let (m_prime, n) = b.dim();
    assert!(n_prime == m_prime);

    let mut res = Array2::from_elem((m, n), E::G2::zero());
    for i in 0..m {
        for j in 0..n {
            let mut sum = E::G2::zero();
            for k in 0..n_prime {
                sum += b[[k, j]].mul(a[[i, k]]);
            }
            res[[i, j]] = sum;
        }
    }

    res
}

#[cfg(test)]
mod test {

    use ark_bls12_381::Bls12_381 as E;
    use ark_ec::pairing::Pairing;
    use ark_std::{test_rng, UniformRand};
    use ndarray::Array2;

    type Fr = <E as Pairing>::ScalarField;
    type G1 = <E as Pairing>::G1;
    type G2 = <E as Pairing>::G2;

    use super::*;

    #[test]
    fn test_associativity() {
        let rng = &mut test_rng();
        let m = 3;
        let n = 2;

        let a = Array2::from_shape_fn((m + 1, n), |_| Fr::rand(rng));
        let a_t = a.reversed_axes();
        let d = Array2::from_shape_fn((m + 1, m), |_| Fr::rand(rng));
        let s = Array2::from_shape_fn((m, n), |_| Fr::rand(rng));

        // ... Test Group 1 ...

        let p1 = G1::rand(rng);

        // [a^T d] s
        let at_d = a_t.dot(&d).mapv(|x| p1.mul(x));
        let at_d_s_1 = dot_1s::<E>(&at_d, &s);

        // a^T [d s]
        let d_s = d.dot(&s).mapv(|x| p1.mul(x));
        let at_d_s_2 = dot_s1::<E>(&a_t, &d_s);

        // [a^T d] s = a^T [d s]
        assert_eq!(at_d_s_1, at_d_s_2);

        // ... Test Group 2 ...

        let p2 = G2::rand(rng);

        // [a^T d] s
        let at_d = a_t.dot(&d).mapv(|x| p2.mul(x));
        let at_d_s_1 = dot_2s::<E>(&at_d, &s);

        // a^T [d s]
        let d_s = d.dot(&s).mapv(|x| p2.mul(x));
        let at_d_s_2 = dot_s2::<E>(&a_t, &d_s);

        // [a^T d] s = a^T [d s]
        assert_eq!(at_d_s_1, at_d_s_2);
    }

    #[test]
    fn test_pairing_output_scalar_multiplication() {
        let rng = &mut test_rng();
        let m = 3;
        let n = 2;

        let g1 = G1::rand(rng);
        let g2 = G2::rand(rng);
        let gt = E::pairing(g1, g2);

        let a = Array2::from_shape_fn((m, n), |_| Fr::rand(rng));
        let b = Array2::from_shape_fn((n, m), |_| Fr::rand(rng));

        // ... Test dot_e ...

        // e([a], [b])
        let a_g1 = a.mapv(|x| g1.mul(x));
        let b_g2 = b.mapv(|x| g2.mul(x));
        let ab_gt_1 = dot_e::<E>(&a_g1, &b_g2);

        // [a^T b]_T
        let ab = a.dot(&b);
        let ab_gt_2 = ab.mapv(|x| gt.mul(x));

        // e([a], [b]]) = [a^T b]_T
        assert_eq!(ab_gt_1, ab_gt_2);

        // ... Test dot_e_rev ...

        // e([a]^T, [b]^T)
        let a_g1_t = a.clone().reversed_axes().mapv(|x| g1.mul(x));
        let b_g2_t = b.clone().reversed_axes().mapv(|x| g2.mul(x));
        let ba_gt_1 = dot_e_rev::<E>(&b_g2_t, &a_g1_t);

        // [b a^T]_T
        let ba = b.clone().reversed_axes().dot(&a.clone().reversed_axes());
        let ba_gt_2 = ba.mapv(|x| gt.mul(x));

        // e([a]^T, [b]^T) = [b a^T]_T
        assert_eq!(ba_gt_1, ba_gt_2);
        // [a^T b]_T = [b a^T]_T ^ T
        assert_eq!(ab_gt_1, ba_gt_1.reversed_axes());

        // ... Test dot_es ...

        let s = Array2::from_shape_fn((m, n), |_| Fr::rand(rng));

        // e([a], [b]) s
        let ab_gt = dot_e::<E>(&a.mapv(|x| g1.mul(x)), &b.mapv(|x| g2.mul(x)));
        let ab_s_gt_1 = dot_es::<E>(&ab_gt, &s);

        // [a^T b]_T s
        let ab_s = a.dot(&b).dot(&s);
        let ab_s_gt_2 = ab_s.mapv(|x| gt.mul(x));

        // e([a], [b]) s = [a^T b]_T s
        assert_eq!(ab_s_gt_1, ab_s_gt_2);
    }
}
