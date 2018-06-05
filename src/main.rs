#[macro_use]
extern crate ndarray;
extern crate num_ru;
extern crate num_traits;

use ndarray::*;
use num_ru::*;
use num_ru::stats::order_stats::*;
use num_ru::stats::averages::*;
use num_ru::math::trig::*;
// use num_traits::float::Float;
use num_traits::real::Real;

pub fn main() {
    test_function();
    let arr = array![2, 3, 4];
    let mut arr_2d = array![[[5, 6], [7, 0]], [[1, 2], [3, 4]]];

    let mut big_arr = Array::zeros((5, 6, 7));
    big_arr[[2, 3, 4]] = 15;

    println!("arr: {}", arr);
    println!("arr: {}", arr[0]);
    let m = amax(&mut arr_2d);
    println!("max: {}", m);
    println!("max: {}", amax(&big_arr));
//    println!("max: {}", amax_parallelized(&arr_3d));
    let arr_f = array![2.0, 3.0, 4.0];
    println!("mean: {}", mean(&arr_f));

    let pi = std::f64::consts::PI;
    let input_arr = array![pi, pi / 2.0];
    let res = num_ru::math::trig::sin(&input_arr);
    println!("sin res:{}", res);
    let res = sin(&input_arr);
    println!("sin res:{}", res);
    // cumsum_run();
    cumsum_no_axis();
}


fn exp_run() {
    println!("figuring out exp");
    let input_arr = array![0.0, 1.0, 5.0, 0.0];
    // let input_arr = array![[1,2,3],[1,2,3]];
    println!("array: {:?}",input_arr.dim());
    println!("exp: e^1.0: {}",1.0_f64.exp());
    let resarr = Array::from_iter(input_arr.into_iter().map(|x| x.exp()));
    println!("resarr: {:?}",resarr);

}

fn cumsum_no_axis() {
    let input_arr = array![[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                            [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]];
    let flat = Array::from_iter(input_arr.into_iter()).to_vec();
    println!("flat: {:?}",flat);
    let mut p = vec![*flat[0]];
    for i in 0..flat.len()-1 {
        let padd = p[p.len()-1]+flat[p.len()];
        p.push(padd);
    }
    let x = Array::from_iter(p.into_iter());
        // .into_shape(input_arr.raw_dim()).unwrap();

    // let padd = p[p.len()-1]+flat[p.len()];
    // p.push(padd);
    println!("p: {:?}",x);

    println!("kdjfd {:?}",math::sumproddif::cumsum(&input_arr));




}


fn cumsum_run() {
    println!("figuring out cumsum");
    let input_arr = array![[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]];
    let axis = 0;
    println!("input_arr: {:?}",input_arr);
    let shape = input_arr.shape();
    println!("shape: {:?}", shape);
    println!("shape len: {:?}",shape.len());
    let mut prod = 1;
    let mut prod_before = 1;
    let mut prod_after = 1;
    for i in 0..shape.len() {
        if i < axis {
            prod_before = prod_before * shape[i];
        } else if i > axis {
            prod_after = prod_after * shape[i];
        }
        prod = prod * shape[i];
    }
    println!("prod_before: {:?}", prod_before);
    println!("prod_after: {:?}", prod_after);

    // let mut val_vec = vec![];
    let mut res_vec = Array::from_iter(input_arr.into_iter()).to_vec();
    println!("dim: {:?}",input_arr.raw_dim());
    // let mut res_arr = output_arr.into_shape([prod]).unwrap();
    {
        let p = res_vec.get_mut(2).unwrap();
        let x = 123;
        // *p = &x.to_owned();
    }

    for i in 0..prod_after {
        println!("hello");
        // res_arr[i] = &(*res_arr[i]+1);
        // println!("{:?}",res_arr[i]);
    }

    // let res_arr = res_arr.into_shape(input_arr.raw_dim()).unwrap();
    // println!("index: {:?}",output_arr.s![1,1,2]);
    // output_arr[2][2][3] = output_arr[1][2][3] + output_arr[2][2][3];



    println!("vec: {:?}", res_vec);

    let input_arr_3 = array![[[1,2,3],[4,5,6]],
                             [[7,8,9],[10,11,12]]];
    println!("input_arr_3: {:?}",input_arr_3);
}
