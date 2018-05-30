#![feature(test)]
extern crate test;
use test::Bencher;

#[macro_use]
extern crate ndarray;
extern crate num_ru;
extern crate ndarray_parallel;

use ndarray::*;
use num_ru::*;
use num_ru::stats::order_stats::*;
use num_ru::stats::averages::*;

#[bench]
fn arr_max_bench_1d(b: &mut Bencher) {
    b.iter(|| {
        let arr = array![5, 3, 5, 2, 1];
        assert_eq!(amax(&arr), 5);
    });
}

//#[bench]
//fn arr_max_simple_1d_small(b: &mut Bencher) {
//    b.iter(|| {
//        let arr = array![5, 3, 5, 2, 1];
//        assert_eq!(amax_simple(&arr), 5);
//    });
//}

//#[bench]
//fn arr_max_parallel_1d_small(b: &mut Bencher) {
//    b.iter(|| {
//        let arr = array![5, 3, 5, 2, 1];
//        let m = arr.par_iter().max();
//        assert_eq!(m, 5);
//    });
//}

#[bench]
fn arr_max_mid(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((50, 50, 50));
        big_arr[[25, 25, 25]] = 15;
        assert_eq!(amax_simple(&big_arr), 15);
    });
}

#[bench]
fn arr_max_large(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((100, 100, 100, 100));
        big_arr[[25, 25, 25, 25]] = 15;
        assert_eq!(amax_simple(&big_arr), 15);
    });
}

#[bench]
fn arr_max_large_rayon_ref(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = RcArray::zeros((100, 100, 100, 100));
        big_arr[[25, 25, 25, 25]] = 15;
        assert_eq!(amax_simple_rayon_ref(&big_arr), 15);
        big_arr[[25, 25, 25, 25]] = 25;
    });
}

#[bench]
fn arr_max_large_rayon(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((100, 100, 100, 100));
        big_arr[[25, 25, 25, 25]] = 15;
        assert_eq!(amax_simple_rayon(big_arr), 15);
    });
}

#[bench]
fn arr_min_mid(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((50, 50, 50));
        big_arr[[25, 25, 25]] = -1;
        assert_eq!(amin(&big_arr), -1);
    });
}

#[bench]
fn arr_min_large(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((100, 100, 100, 100));
        big_arr[[25, 25, 25, 25]] = -1;
        assert_eq!(amin(&big_arr), -1);
    });
}

#[bench]
fn mean_mid(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((50, 50, 50));
        big_arr[[25, 25, 25]] = 125000.0;
        assert_eq!(mean(&big_arr), 1.0);
    });
}

#[bench]
fn mean_large(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((100, 100, 100, 100));
        big_arr[[25, 25, 25, 25]] = 100000000.0;
        assert_eq!(mean(&big_arr), 1.0);
    });
}

#[bench]
fn variance_mid(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((50, 50, 50));
        big_arr[[25, 25, 25]] = 0.0;
        assert_eq!(var(&big_arr), 0.0);
    });
}
