#[macro_use]
extern crate ndarray;
extern crate num_ru;

use ndarray::*;
use num_ru::*;
use num_ru::stats::basic_stats::*;
use num_ru::math::trig::*;
//use ndarray::prelude::*;

pub fn main() {
    test_function();
    let arr = array![2, 3, 4];
    let mut arr_2d = array![[[5, 6], [7, 0]], [[1, 2], [3, 4]]];
    let mut arr_3d = array![[[11, 12], [13, 14]], [[15, 16], [17, 18]]];

    let mut big_arr = Array::zeros((5, 6, 7));
    big_arr[[2, 3, 4]] = 15;

    println!("arr: {}", arr);
    println!("arr: {}", arr[0]);
    let m = amax(&mut arr_2d);
    println!("max: {}", m);
    println!("max: {}", amax(&big_arr));
    println!("max: {}", amax_parallelized(&arr_3d));

    let pi = std::f64::consts::PI;
    let input_arr = array![pi, pi / 2.0];
    let res = num_ru::math::trig::sin(&input_arr);
    println!("sin res:{}", res);
    let res = sin(&input_arr);
    println!("sin res:{}", res);

}
