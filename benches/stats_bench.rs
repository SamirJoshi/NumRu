#![feature(test)]
extern crate test;
use test::Bencher;

#[macro_use]
extern crate ndarray;
extern crate num_ru;

use ndarray::*;
use num_ru::stats::order_stats::*;
use num_ru::stats::averages::*;

// #[bench]
// fn arr_max_bench_1d(b: &mut Bencher) {
//     b.iter(|| {
//         let arr = array![5, 3, 5, 2, 1];
//         assert_eq!(arr.amax(), 5.0);
//     });
// }

#[bench]
fn arr_max_mid(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((50, 50, 50));
        big_arr[[25, 25, 25]] = 15.0;
        assert_eq!(big_arr.amax(), 15.0);
    });
}

#[bench]
fn arr_max_large(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((50, 50, 50, 50));
        big_arr[[25, 25, 25, 25]] = 15.0;
        assert_eq!(big_arr.amax(), 15.0);
    });
}

#[bench]
fn arr_max_mid_rayon(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = ArcArray::zeros((50, 50, 50));
        big_arr[[25, 25, 25]] = 15.0;
        assert_eq!(big_arr.amax(), 15.0);
    });
}

#[bench]
fn arr_max_large_rayon(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = ArcArray::zeros((50, 50, 50, 50));
        big_arr[[25, 25, 25, 25]] = 15.0;
        assert_eq!(big_arr.amax(), 15.0);
    });
}

#[bench]
fn arr_min_mid(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((50, 50, 50));
        big_arr[[25, 25, 25]] = -1.0;
        assert_eq!(big_arr.amin(), -1.0);
    });
}

#[bench]
fn arr_min_mid_rayon(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = ArcArray::zeros((50, 50, 50));
        big_arr[[25, 25, 25]] = -1.0;
        assert_eq!(big_arr.amin(), -1.0);
    });
}

#[bench]
fn arr_min_large(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((50, 50, 50, 50));
        big_arr[[25, 25, 25, 25]] = -1.0;
        assert_eq!(big_arr.amin(), -1.0);
    });
}

#[bench]
fn arr_min_large_rayon(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = ArcArray::zeros((50, 50, 50, 50));
        big_arr[[25, 25, 25, 25]] = -1.0;
        assert_eq!(big_arr.amin(), -1.0);
    });
}

#[bench]
fn mean_mid(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((50, 50, 50));
        big_arr[[25, 25, 25]] = 125000.0;
        assert_eq!(big_arr.mean(), 1.0);
    });
}

#[bench]
fn mean_mid_rayon(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = ArcArray::zeros((50, 50, 50));
        big_arr[[25, 25, 25]] = 125000.0;
        assert_eq!(big_arr.mean(), 1.0);
    });
}

#[bench]
fn mean_large(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((50, 50, 50, 50));
        big_arr[[25, 25, 25, 25]] = 6250000.0;
        assert_eq!(big_arr.mean(), 1.0);
    });
}

#[bench]
fn mean_large_rayon(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = ArcArray::zeros((50, 50, 50, 50));
        big_arr[[25, 25, 25, 25]] = 6250000.0;
        assert_eq!(big_arr.mean(), 1.0);
    });
}

#[bench]
fn variance_mid(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((50, 50, 50));
        big_arr[[25, 25, 25]] = 0.0;
        assert_eq!(big_arr.var(), 0.0);
    });
}

#[bench]
fn variance_mid_rayon(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = ArcArray::zeros((50, 50, 50));
        big_arr[[25, 25, 25]] = 0.0;
        assert_eq!(big_arr.var(), 0.0);
    });
}

#[bench]
fn variance_large(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = Array::zeros((50, 50, 50, 50));
        big_arr[[25, 25, 25, 25]] = 0.0;
        assert_eq!(big_arr.var(), 0.0);
    });
}

#[bench]
fn variance_large_rayon(b: &mut Bencher) {
    b.iter(|| {
        let mut big_arr = ArcArray::zeros((50, 50, 50, 50));
        big_arr[[25, 25, 25, 25]] = 0.0;
        assert_eq!(big_arr.var(), 0.0);
    });
}
