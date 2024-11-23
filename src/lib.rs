pub(crate) mod arith;
pub mod ciphertext;
pub mod decrypt;
pub use decrypt::DecryptKey;
pub mod encrypt;
pub use encrypt::EncryptKey;
pub mod key_gen;
pub use key_gen::pke1;
pub mod params;
pub mod publicly_verifiable;
pub use params::Params;
pub use publicly_verifiable::pke2;

#[cfg(test)]
mod test {

    use ark_bls12_381::Bls12_381 as E;
    use ark_ec::pairing::Pairing;
    use ark_std::{test_rng, UniformRand};

    type G1 = <E as Pairing>::G1Affine;

    use super::*;

    #[test]
    fn test_pke1() {
        let rng = &mut test_rng();
        let k = 3;

        let pp = Params::<E>::rand(rng);
        let (dk, ek) = pke1(rng, &pp, k);

        let m = G1::rand(rng);

        let ciphertext = ek.encrypt(rng, m);
        let m_prime = dk.decrypt(&pp, &ciphertext).unwrap();

        assert_eq!(m, m_prime);

        let ciphertext2 = ek.randomize(rng, &ciphertext);
        assert!(ciphertext != ciphertext2);
        let m_prime2 = dk.decrypt(&pp, &ciphertext2).unwrap();

        assert_eq!(m, m_prime2);
    }

    #[test]
    fn test_pke2() {
        let rng = &mut test_rng();
        let k = 3;

        let pp = Params::<E>::rand(rng);
        let (dk, ek) = pke2(rng, &pp, k);

        let m = G1::rand(rng);

        let ciphertext = ek.encrypt(rng, &pp, m);

        let m_prime = dk.decrypt(&ciphertext).unwrap();

        assert_eq!(m, m_prime);
    }
}
