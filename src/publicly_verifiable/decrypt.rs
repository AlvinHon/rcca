use ark_ec::pairing::Pairing;
use ark_std::One;
use ndarray::{Array2, Axis};

use crate::{arith::dot_s1, publicly_verifiable::nizk};

use super::{nizk::Crs, Ciphertext};

#[derive(Clone, Debug)]
pub struct DecryptKey<E: Pairing> {
    // dim = (k+1, 1)
    pub(crate) a: Array2<E::ScalarField>,
    pub(crate) crs: Crs<E>,
}

impl<E: Pairing> DecryptKey<E> {
    pub fn decrypt(&self, c: &Ciphertext<E>) -> Option<E::G1> {
        let Ciphertext { x, v, proof } = c;

        nizk::verify(&self.crs, proof, v, x).then(|| {
            // (-a^T, 1) * [x]_1
            let ones = Array2::from_elem((1, 1), E::ScalarField::one());
            let mut a_t = -self.a.clone().reversed_axes(); // (1, k+1)
            a_t.append(Axis(1), ones.view()).unwrap(); // (1, k+2)

            dot_s1::<E>(&a_t, x)[(0, 0)]
        })
    }
}
