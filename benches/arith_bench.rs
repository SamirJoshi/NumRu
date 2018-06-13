#![feature(test)]
extern crate test;
use test::Bencher;

#[macro_use]
extern crate ndarray;
extern crate num_ru;
extern crate ndarray_parallel;

use ndarray::*;
use num_ru::math::arithmetic::*;
use num_ru::math::sumproddif::*;

#[bench]
fn negative_bench_mid(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = Array::from_elem((50, 50, 50), 1.0);
        let expected_arr = Array::from_elem((50, 50, 50), -1.0);
        let res_arr = input_arr.negative();
        assert_eq!(expected_arr, res_arr);
    });
}

#[bench]
fn negative_bench_large(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = Array::from_elem((50, 50, 50, 50), 1.0);
        let expected_arr = Array::from_elem((50, 50, 50, 50), -1.0);
        let res_arr = input_arr.negative();
        assert_eq!(expected_arr, res_arr);
    });
}

#[bench]
fn sum_bench_mid(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = Array::from_elem((50, 50, 50), 1.0);
        let res = input_arr.sum();
        assert_eq!(res, 125000.0);
    });
}

#[bench]
fn sum_bench_large(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = Array::from_elem((50, 50, 50, 50), 1.0);
        let res = input_arr.sum();
        assert_eq!(res, 6250000.0);
    });
}

#[bench]
fn prod_bench_mid(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = Array::from_elem((50, 50, 50), 1.0);
        let res = input_arr.prod();
        assert_eq!(res, 1.0);
    });
}

#[bench]
fn prod_bench_large(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = Array::from_elem((50, 50, 50, 50), 1.0);
        let res = input_arr.prod();
        assert_eq!(res, 1.0);
    });
}

#[bench]
fn negative_bench_mid_rayon(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = ArcArray::from_elem((50, 50, 50), 1.0);
        let expected_arr = ArcArray::from_elem((50, 50, 50), -1.0);
        let res_arr = input_arr.negative();
        assert_eq!(expected_arr, res_arr);
    });
}

#[bench]
fn negative_bench_large_rayon(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = ArcArray::from_elem((50, 50, 50, 50), 1.0);
        let expected_arr = ArcArray::from_elem((50, 50, 50, 50), -1.0);
        let res_arr = input_arr.negative();
        assert_eq!(expected_arr, res_arr);
    });
}

#[bench]
fn sum_bench_large_rayon(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = ArcArray::from_elem((50, 50, 50, 50), 1.0);
        let res = input_arr.sum();
        assert_eq!(res, 6250000.0);
    });
}

#[bench]
fn sum_bench_mid_rayon(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = ArcArray::from_elem((50, 50, 50), 1.0);
        let res = input_arr.sum();
        assert_eq!(res, 125000.0);
    });
}

#[bench]
fn prod_bench_mid_rayon(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = ArcArray::from_elem((50, 50, 50), 1.0);
        let res = input_arr.prod();
        assert_eq!(res, 1.0);
    });
}

#[bench]
fn prod_bench_large_rayon(b: &mut Bencher) {
    b.iter(|| {
        let input_arr = ArcArray::from_elem((50, 50, 50, 50), 1.0);
        let res = input_arr.prod();
        assert_eq!(res, 1.0);
    });
}
