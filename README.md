# Re-randomizable RCCA-secure Public Key Encryption

Implementation of structure-preserving RCCA schemes defined in the paper [Structure-Preserving and Re-randomizable RCCA-secure Public Key Encryption and its Applications](https://eprint.iacr.org/2019/955.pdf), specifically the schemes `PKE1` (Rand-RCCA PKE scheme) and `PKE2` (Publicly-Verifiable Rand RCCA PKE).

These encryption schemes are replayable CCA-secure, and provide property of re-randomizable and publicly verifiable ciphertexts. In simple words, we can encrypt and decrypt messages as ordinary encryption scheme by using an asymmetric key. In addition, we can also re-randomize the ciphertexts, and verify its validity.

## Rand-RCCA PKE

The scheme is instantiated with the Matrix Diffie-Hellman Assumption, notated as `D_k-MDDH` where `k` is the the dimension of a matrix `D`. `k` determines the hardness of the MDDH problem. When setting `k` = 1, as stated in the paper, the scheme can be instantiated with the `SXDH` assumption.

```rust ignore
// Public Parameters
let pp = Params::<E>::rand(rng);
// Setup Scheme PKE1 with k = 3 and output decryption/encryption key
let (dk, ek) = rcca::pke1(rng, &pp, 3);

// A message
let m = G1::rand(rng);

// Encryption
let ciphertext = ek.encrypt(rng, m);

// Decryption
let m_prime = dk.decrypt(&pp, &ciphertext).unwrap();

assert_eq!(m, m_prime);

// Randomization
let ciphertext2 = ek.randomize(rng, &ciphertext);
// Outputs a different ciphertext
assert!(ciphertext != ciphertext2);

let m_prime2 = dk.decrypt(&pp, &ciphertext2).unwrap();
// Same message decrypted from different ciphertext
assert_eq!(m, m_prime2);
```

## Publicly-Verifiable Rand-RCCA PKE

Similar the previous scheme `PKE1`, the scheme `PKE2` provides additional method `verify` to verify the validity of the ciphertext.

```rust ignore
// Setup the scheme PKE2.
let (dk, ek) = rcca::pke2(rng, &pp, k);
// ...
assert!(ek.verify(&ciphertext));
```
