use ark_ec::pairing::Pairing;
use ndarray::{Array2, Axis};
use std::ops::Mul;

use crate::{
    arith::{dot_e, dot_e_rev, dot_s1, dot_s2},
    ciphertext::Ciphertext,
    Params,
};

pub struct DecryptKey<E: Pairing> {
    // dim = (k+1, 1)
    pub(crate) a: Array2<E::ScalarField>,
    // dim = (k+1, 1)
    pub(crate) f: Array2<E::ScalarField>,
    // dim = (k+1, 1)
    pub(crate) g: Array2<E::ScalarField>,
    // dim = (k+1, k+1)
    pub(crate) big_f: Array2<E::ScalarField>,
    // dim = (k+1, k+2)
    pub(crate) big_g: Array2<E::ScalarField>,
}

impl<E: Pairing> DecryptKey<E> {
    pub fn decrypt(&self, pp: &Params<E>, c: &Ciphertext<E>) -> Option<E::G1Affine> {
        let k = self.a.len() - 1;

        // parse x, [x^T]_1 = ([u^T], p)
        let (u, p) = {
            let mut x_t = c.x.clone().reversed_axes(); // (1, k+2)
            let p = x_t[(0, k + 1)];
            x_t.remove_index(Axis(1), k + 1);
            let u = x_t.reversed_axes(); // (1, k+1)
            (u, p)
        };

        // m = p - [a^T u]_1
        let a_t = self.a.clone().reversed_axes();
        let m = (p - dot_s1::<E>(&a_t, &u)[[0, 0]]).into();

        // [pi1]_T = [(f + Fv)^T u]_T
        let pi1 = {
            let l = self.f.mapv(|x| pp.p2.mul(x)) + dot_s2::<E>(&self.big_f, &c.v);
            let r = u.mapv(|x| x.into());
            dot_e_rev::<E>(&l.reversed_axes(), &r)
        };

        // [pi2]_T = [(g + Gx)^T v]_T
        let pi2 = {
            let l = self.g.mapv(|x| pp.p1.mul(x)) + dot_s1::<E>(&self.big_g, &c.x);
            let r = c.v.mapv(|x| x.into());
            dot_e::<E>(&l.reversed_axes(), &r)
        };

        let pi = pi1 + pi2;

        (pi == c.pi).then_some(m)
    }
}
