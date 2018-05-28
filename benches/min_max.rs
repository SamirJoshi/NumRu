#![feature(test)]
extern crate test;
use test::Bencher;

#[macro_use]
extern crate ndarray;
extern crate num_ru;

use ndarray::*;
use num_ru::*;
use num_ru::stats::basic_stats::*;

// remove 
extern crate crossbeam;
use std::thread;
use std::sync::{Arc, Mutex};



#[bench]
fn arr_max_bench_1d(b: &mut Bencher) {
    b.iter(|| {
        let arr = array![5, 3, 5, 2, 1];
        assert_eq!(amax(&arr), 5);
    });
}

#[bench]
fn arr_max_simple_1d_small(b: &mut Bencher) {
    b.iter(|| {
        let arr = array![5, 3, 5, 2, 1];
        assert_eq!(amax_simple(&arr), 5);
    });
}

#[bench]
fn arr_max_parallel_1d_small(b: &mut Bencher) {
    b.iter(|| {
        let arr = array![5, 3, 5, 2, 1];
        assert_eq!(amax_parallelized(&arr), 5);
    });
}

#[bench]
fn arr_max_simple(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((100, 100, 100, 100));
        big_arr[[2, 3, 4, 5]] = 15;
        assert_eq!(amax_simple(&big_arr), 15);
    });
}

#[bench]
fn arr_max_simple_with_arc(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((100, 100, 100, 100));
        big_arr[[2, 3, 4, 5]] = 15;
        let a = Arc::new(big_arr);
    });
}


#[bench]
fn arr_max_parallel(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((100, 100, 100, 100));
        big_arr[[2, 3, 4, 5]] = 15;
        assert_eq!(amax_parallelized(&big_arr), 15);
    });
}

// #[bench]
// fn arr_max_rayon(b: &mut Bencher) {
//     b.iter(|| {
//         let mut big_arr = Array::zeros((100, 100, 100, 100));
//         big_arr[[2, 3, 4, 5]] = 15;
//         assert_eq!(amax_simple_rayon(&big_arr), 15);
//     });
// }
