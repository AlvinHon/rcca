use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr,
};
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
                sum = sum + E::pairing(a[[i, k]], b[[k, j]]);
            }
            res[[i, j]] = sum;
        }
    }

    res
}

pub(crate) fn dot_1s<E: Pairing>(
    a: &Array2<E::G1Affine>,
    b: &Array2<E::ScalarField>,
) -> Array2<E::G1Affine> {
    let (m, n_prime) = a.dim();
    let (m_prime, n) = b.dim();
    assert!(n_prime == m_prime);

    let mut res = Array2::from_elem((m, n), E::G1Affine::zero());
    for i in 0..m {
        for j in 0..n {
            let mut sum = E::G1Affine::zero();
            for k in 0..n_prime {
                sum = (sum + a[[i, k]].mul(b[[k, j]])).into();
            }
            res[[i, j]] = sum;
        }
    }

    res
}

pub(crate) fn dot_s1<E: Pairing>(
    a: &Array2<E::ScalarField>,
    b: &Array2<E::G1Affine>,
) -> Array2<E::G1Affine> {
    let (m, n_prime) = a.dim();
    let (m_prime, n) = b.dim();
    assert!(n_prime == m_prime);

    let mut res = Array2::from_elem((m, n), E::G1Affine::zero());
    for i in 0..m {
        for j in 0..n {
            let mut sum = E::G1Affine::zero();
            for k in 0..n_prime {
                sum = (sum + b[[k, j]].mul(a[[i, k]])).into();
            }
            res[[i, j]] = sum;
        }
    }

    res
}

pub(crate) fn dot_2s<E: Pairing>(
    a: &Array2<E::G2Affine>,
    b: &Array2<E::ScalarField>,
) -> Array2<E::G2Affine> {
    let (m, n_prime) = a.dim();
    let (m_prime, n) = b.dim();
    assert!(n_prime == m_prime);

    let mut res = Array2::from_elem((m, n), E::G2Affine::zero());
    for i in 0..m {
        for j in 0..n {
            let mut sum = E::G2Affine::zero();
            for k in 0..n_prime {
                sum = (sum + a[[i, k]].mul(b[[k, j]])).into();
            }
            res[[i, j]] = sum;
        }
    }

    res
}

#[cfg(test)]
mod test {

    use ark_bls12_381::Bls12_381 as E;
    use ark_ec::{pairing::Pairing, CurveGroup};
    use ark_std::{test_rng, UniformRand};
    use ndarray::Array2;

    type Fr = <E as Pairing>::ScalarField;
    type G1 = <E as Pairing>::G1;

    use super::*;

    #[test]
    fn test_commutativity() {
        let rng = &mut test_rng();
        let k = 3;

        let a = Array2::from_shape_fn((k + 1, 1), |_| Fr::rand(rng));
        let d = Array2::from_shape_fn((k + 1, k), |_| Fr::rand(rng));
        let s = Array2::from_shape_fn((k, 1), |_| Fr::rand(rng));

        let g = G1::rand(rng);

        // [a^T d] s
        let a_t = a.reversed_axes();
        let at_d = a_t.dot(&d).mapv(|x| g.mul(x).into_affine());
        let at_d_s_1 = dot_1s::<E>(&at_d, &s);

        // a^T [d s]
        let d_s = d.dot(&s).mapv(|x| g.mul(x).into_affine());
        let at_d_s_2 = dot_s1::<E>(&a_t, &d_s);

        // [a^T d] s = a^T [d s]
        assert_eq!(at_d_s_1, at_d_s_2);
    }
}
