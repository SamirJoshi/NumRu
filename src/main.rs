extern crate ndarray;
extern crate ndarray_rand;
extern crate num_ru;
extern crate num_traits;
extern crate rand;

use ndarray::*;
use num_ru::stats::order_stats::*;
use num_ru::stats::averages::*;
use num_ru::math::trig::*;
use num_ru::math::sumproddif::*;
use num_ru::math::arithmetic::*;
use ndarray_rand::RandomExt;
use rand::distributions::Range;

pub fn main() {
    for _i in 0..5 {
        stats_example();
        math_example();
    }
}

pub fn stats_example() {
    // create a random dataset
    let arr = ArcArray::random((100000, 50), Range::new(0., 1.));
    println!("Created array - beginning stats");

    // basic statistics
    let min_elem = arr.amin();
    let max_elem = arr.amax();
    let range = arr.ptp();
    println!("Minimum element: {}", min_elem);
    println!("Maximum element: {}", max_elem);
    println!("Range: {}", range);

    // more statistics
    let avg = arr.mean();
    let arr_std = arr.std_dev();
    let arr_sum = arr.sum();
    let arr_prod = arr.prod();
    println!("Mean: {}", avg);
    println!("Standard Deviation: {}", arr_std);
    println!("Sum: {}", arr_sum);
    println!("Product: {}", arr_prod);
    println!("\n");
}

pub fn math_example() {
    let math_arr = ArcArray::random((100000, 50), Range::new(-0.5, 0.5));
    let _math_arr_neg = math_arr.negative();
    let _math_arr_pos = math_arr.positive();
    
    let test_normal_sin = 1.57_f32.sin();
    println!("normal sin: {}", test_normal_sin);
    let _sin_arr = math_arr.sin();
    let _cos_arr = math_arr.cos();
    let tan_arr = math_arr.tan().unwrap();
    let _orig_arr = tan_arr.atan();
}
