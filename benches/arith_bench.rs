#![feature(test)]
extern crate test;
use test::Bencher;

#[macro_use]
extern crate ndarray;
extern crate num_ru;
extern crate ndarray_parallel;

use ndarray::*;
use num_ru::math::arithmetic::*;

#[bench]
fn positive_bench_mid(b: &mut Bencher) {
    let pi = std::f64::consts::PI;
    let input_arr = Array::from_elem((50, 50, 50), 1.0);
    let expected_arr = Array::from_elem((50, 50, 50), -1.0);
    b.iter(|| {
        let res_arr = negative(&input_arr);
        assert_eq!(expected_arr, res_arr);
    });
}


