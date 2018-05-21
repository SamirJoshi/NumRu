#[macro_use]
extern crate ndarray;
extern crate num_ru;

use num_ru::*;
use num_ru::math::trig::*;
//use ndarray::prelude::*;

pub fn main() {
    test_function();
    let mut arr = array![2, 3, 4];
    print!("arr: {}", arr);
    print!("arr: {}", arr[0]);
    let m = amin(&mut arr);
    print!("min: {}", m);
    let pi = std::f64::consts::PI;
    let input_arr = array![pi, pi / 2.0];
    let res = num_ru::math::trig::sin(&input_arr);
    println!("sin res:{}", res);
    let res = sin(&input_arr);
    println!("sin res:{}", res);

}
