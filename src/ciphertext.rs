use ark_ec::pairing::{Pairing, PairingOutput};
use ndarray::Array2;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ciphertext<E: Pairing> {
    // dim = (k+2, 1)
    pub(crate) x: Array2<E::G1Affine>,
    // dim = (k+1, 1)
    pub(crate) v: Array2<E::G2Affine>,
    pub(crate) pi: Array2<PairingOutput<E>>,
}
