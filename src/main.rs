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
    let min_elem = amin_rayon(&arr);
    let max_elem = amax_rayon(&arr);
    let range = ptp_rayon(&arr);
    println!("Minimum element: {}", min_elem);
    println!("Maximum element: {}", max_elem);
    println!("Range: {}", range);

    // more statistics
    let avg = mean_rayon(&arr);
    let arr_std = std_dev_rayon(&arr);
    let arr_sum = sum_rayon(&arr);
    let arr_prod = prod_rayon(&arr);
    println!("Mean: {}", avg);
    println!("Standard Deviation: {}", arr_std);
    println!("Sum: {}", arr_sum);
    println!("Product: {}", arr_prod);
    println!("\n");
}

pub fn math_example() {
    let math_arr = ArcArray::random((100000, 50), Range::new(-0.5, 0.5));
    let _math_arr_neg = negative_rayon(&math_arr);
    let _math_arr_pos = positive_rayon(&math_arr);

    let _sin_arr = math_arr.sin();
    let _cos_arr = math_arr.cos();
    let tan_arr = math_arr.tan().unwrap();
    let _orig_arr = arctan_rayon(&tan_arr);
}
