extern crate num_ru;
#[macro_use]
extern crate ndarray;

use num_ru::{test_function, amin};
//use ndarray::prelude::*;

pub fn main() {
    test_function();
    let mut arr = array![2, 3, 4];
    print!("arr: {}", arr);
    print!("arr: {}", arr[0]);
    let m = amin(&mut arr);
    print!("min: {}", m);
}
