//! Defines the `Ciphertext` struct for the PKE2 scheme.

use ark_ec::pairing::Pairing;
use ndarray::Array2;

use super::Proof;

/// Ciphertext that created by the encryption algorithm. Tt is also used as
/// the input and output of the randomization algorithm in PKE2 scheme.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ciphertext<E: Pairing> {
    // dim = (k+2, 1)
    pub(crate) x: Array2<E::G1>,
    // dim = (k+1, 1)
    pub(crate) v: Array2<E::G2>,
    pub(crate) proof: Proof<E>,
}
