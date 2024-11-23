use ark_ec::pairing::Pairing;
use ndarray::Array2;

use super::Proof;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ciphertext<E: Pairing> {
    // dim = (k+2, 1)
    pub(crate) x: Array2<E::G1>,
    // dim = (k+1, 1)
    pub(crate) v: Array2<E::G2>,
    pub(crate) proof: Proof<E>,
}
