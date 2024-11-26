//! Define the ciphertext struct.

use ark_ec::pairing::{Pairing, PairingOutput};
use ndarray::Array2;

/// Ciphertext that created by the encryption algorithm in PKE1 scheme.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ciphertext<E: Pairing> {
    // dim = (k+2, 1)
    pub(crate) x: Array2<E::G1>,
    // dim = (k+1, 1)
    pub(crate) v: Array2<E::G2>,
    pub(crate) pi: Array2<PairingOutput<E>>,
}
