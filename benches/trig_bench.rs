#![feature(test)]
extern crate test;
use test::Bencher;

#[macro_use]
extern crate ndarray;
extern crate num_ru;
extern crate ndarray_parallel;

use ndarray::*;
use num_ru::math::trig::*;

#[bench]
fn sin_bench_mid(b: &mut Bencher) {
    let pi = std::f64::consts::PI;
    b.iter(|| {
        let input_arr = Array::from_elem((50, 50, 50), pi / 2.0);
        let expected_arr = Array::from_elem((50, 50, 50), 1.0);
        let res_arr = input_arr.sin().unwrap();
        assert_eq!(expected_arr, res_arr);
    });
}

#[bench]
fn sin_bench_mid_rayon(b: &mut Bencher) {
    let pi = std::f64::consts::PI;
    b.iter(|| {
        let input_arr = ArcArray::from_elem((50, 50, 50), pi / 2.0);
        let expected_arr = ArcArray::from_elem((50, 50, 50), 1.0);
        let res_arr = input_arr.sin().unwrap();
        assert_eq!(expected_arr, res_arr);
    });
}

#[bench]
fn arcsin_bench_large(b: &mut Bencher) {
    let pi = std::f64::consts::PI;
    b.iter(|| {
        let input_arr = Array::from_elem((50, 50, 50, 50), 1.0);
        let expected_arr = Array::from_elem((50, 50, 50, 50), pi / 2.0);
        let res_arr = input_arr.asin().unwrap();
        assert_eq!(expected_arr, res_arr);
    });
}

#[bench]
fn arcsin_bench_large_rayon(b: &mut Bencher) {
    let pi = std::f64::consts::PI;
    b.iter(|| {
        let input_arr = ArcArray::from_elem((50, 50, 50, 50), 1.0);
        let expected_arr = ArcArray::from_elem((50, 50, 50, 50), pi / 2.0);
        let res_arr = input_arr.asin().unwrap();
        assert_eq!(expected_arr, res_arr);
    });
}

#[bench]
fn cos_bench_large(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = Array::from_elem((50, 50, 50, 50), 0.0);
        let expected_arr = Array::from_elem((50, 50, 50, 50), 1.0);
        let res_arr = input_arr.cos().unwrap();
        assert_eq!(expected_arr, res_arr);
    });
}

#[bench]
fn cos_bench_large_rayon(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = ArcArray::from_elem((50, 50, 50, 50), 0.0);
        let expected_arr = ArcArray::from_elem((50, 50, 50, 50), 1.0);
        let res_arr = input_arr.cos().unwrap();
        assert_eq!(expected_arr, res_arr);
    });
}
//
#[bench]
fn arctan_bench_mid(b: &mut Bencher) {
    let pi = std::f64::consts::PI;
    b.iter(|| {
        let input_arr = Array::from_elem((50, 50, 50), 1.0);
        let expected_arr = Array::from_elem((50, 50, 50), pi / 4.0);
        let res_arr = input_arr.atan().unwrap();
        assert_eq!(expected_arr, res_arr);
    });
}

#[bench]
fn arctan_bench_mid_rayon(b: &mut Bencher) {
    let pi = std::f64::consts::PI;
    b.iter(|| {
        let input_arr = ArcArray::from_elem((50, 50, 50), 1.0);
        let expected_arr = ArcArray::from_elem((50, 50, 50), pi / 4.0);
        let res_arr = input_arr.atan().unwrap();
        assert_eq!(expected_arr, res_arr);
    });
}
