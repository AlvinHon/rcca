pub(crate) mod arith;
pub mod ciphertext;
pub mod decrypt;
pub use decrypt::DecryptKey;
pub mod encrypt;
pub use encrypt::EncryptKey;
pub mod key_gen;
pub use key_gen::ken_gen;
pub mod params;
pub use params::Params;

#[cfg(test)]
mod test {

    use ark_bls12_381::Bls12_381 as E;
    use ark_ec::pairing::Pairing;
    use ark_std::{test_rng, UniformRand};

    type G1 = <E as Pairing>::G1Affine;

    use super::*;

    #[test]
    fn test() {
        let rng = &mut test_rng();
        let k = 3;

        let pp = Params::<E>::rand(rng);
        let (dk, ek) = ken_gen(rng, &pp, k);

        let m = G1::rand(rng);

        let ciphertext = ek.encrypt(rng, m);
        let m_prime = dk.decrypt(&pp, &ciphertext).unwrap();

        assert_eq!(m, m_prime);
    }
}
