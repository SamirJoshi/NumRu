use ndarray::*;
use ndarray_parallel::prelude::*;
use num_traits;
use std;

pub fn prod<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + std::ops::Mul + num_traits::identities::One,
{
    arr.iter().fold(A::one(), |acc, x| acc * *x)
}

pub fn sum<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + std::ops::Add + num_traits::identities::Zero,
{
    arr.iter().fold(A::zero(), |acc, x| acc + *x)
}

pub fn cumsum<A, D>(arr: &Array<A, D>) -> Array<A, Dim<[usize;1]>>
  where D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real + std::ops::Add,
    {
        let flat = Array::from_iter(arr.into_iter()).to_vec();
        let mut p = vec![*flat[0]];
        for i in 0..flat.len()-1 {
            let padd = p[p.len()-1]+ *flat[p.len()];
            p.push(padd);
        }
        let x = Array::from_iter(p.into_iter()).into_shape(flat.len()).unwrap();
        x
    }

pub fn cumprod<A, D>(arr: &Array<A, D>) -> Array<A, Dim<[usize;1]>>
  where D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real + std::ops::Add,
    {
        let flat = Array::from_iter(arr.into_iter()).to_vec();
        let mut p = vec![*flat[0]];
        for i in 0..flat.len()-1 {
            let padd = p[p.len()-1]* *flat[p.len()];
            p.push(padd);
        }
        let x = Array::from_iter(p.into_iter()).into_shape(flat.len()).unwrap();
        x
    }


/// Returns the product of an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::sumproddif::*;
/// # fn main(){
///     let arr = array![[[5.0, 6.0], [7.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]].into_shared();
///     assert_eq!(prod_rayon(&arr), 5040.0);
/// # }
/// ```
///
pub fn prod_rayon<A, D>(arr: &ArcArray<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::marker::Copy + std::marker::Sync + std::marker::Send +
          num_traits::real::Real,
{
    let arr_sum = arr.par_iter().cloned().reduce_with( |a, b| a * b);
    match arr_sum {
        Some(a) => a,
        None => panic!("Array of 0 elements")
    }
}


/// Returns the sum across an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::sumproddif::*;
/// # fn main(){
///     let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]].into_shared();
///     assert_eq!(sum_rayon(&arr), 28.0);
/// # }
/// ```
///
pub fn sum_rayon<A, D>(arr: &ArcArray<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::marker::Copy + std::marker::Sync + std::marker::Send +
          num_traits::real::Real + std::ops::Add,
{
    let arr_sum = arr.par_iter().cloned().reduce_with( |a, b| a + b);
    match arr_sum {
        Some(a) => a,
        None => panic!("Array of 0 elements")
    }
}


#[cfg(test)]
mod sumproddif_tests {
    use math::sumproddif::*;
    use ndarray::*;
    use num_traits;
    use std;

    #[test]
    fn prod_test() {
        assert_eq!(prod(&array![1.0,2.0,3.0]),6.0);
    }

    #[test]
    fn prod_test_int() {
        assert_eq!(prod(&array![1,2,3]),6);
    }

    #[test]
    fn sum_test() {
        assert_eq!(sum(&array![1.0,2.0,3.0,4.0]),10.0);
    }

    #[test]
    fn sum_test_int() {
        assert_eq!(sum(&array![1,2,3,4]),10);
    }

    #[test]
    fn cumsum_test_no_axis() {
        let input_arr = array![[1.0,2.0,3.0],
                               [4.0,5.0,6.0]];
        let res_arr = array![ 1.0,  3.0,  6.0, 10.0, 15.0, 21.0];

        assert!(compare_arrays(&res_arr, &cumsum(&input_arr)));
    }



    // fn cumsum_test_axis_0() {
    //     let input_arr = array![[1,2,3],[4,5,6]];
    //     let res_arr = array![[1,2,3],[5,7,9]];
    //
    // }

    // fn cumsum_test_axis_1() {
    //     let input_arr = array![[1,2,3],[4,5,6]];
    //     let res_arr = array![[1,3,6],[4,9,15]];
    // }

    // fn cumsum_test_3d_axis_2() {
    //     let input_arr = array![[[ 1,  2,  3],
    //     [ 4,  5,  6]],
    //    [[ 7,  8,  9],
    //     [10, 11, 12]]];
    //
    //     let res_arr = array![[[ 1,  3,  6],
    //     [ 4,  9, 15]],
    //    [[ 7, 15, 24],
    //     [10, 21, 33]]]
    // }

    // fn cumsum_test_3d_axis_1() {
    //     let input_arr = array![[[ 1,  2,  3],
    //         [ 4,  5,  6]],
    //        [[ 7,  8,  9],
    //         [10, 11, 12]]];
    //
    //     let res_arr = array![[[ 1,  2,  3],
    //     [ 5,  7,  9]],
    //    [[ 7,  8,  9],
    //     [17, 19, 21]]]
    // }

    #[test]
    fn cumprod_test() {
        let input_arr = array![[1.0,2.0,3.0],
                               [4.0,5.0,6.0]];
        let res_arr = array![  1.,   2.,   6.,  24., 120., 720.];

        assert!(compare_arrays(&res_arr, &cumprod(&input_arr)));
    }




    //TODO: put this in one spot
    fn compare_arrays<D>(expected_arr: &Array<f64, D>, res_arr: &Array<f64, D>) -> bool
        where D: Dimension,
    {
        let mut expected_iter = expected_arr.iter();
        let mut res_iter = res_arr.iter();

        while let Some(r) = res_iter.next() {
            let exp = expected_iter.next().unwrap();
            println!("Expected: {}, Res: {}", *exp, *r);
            if (*r - *exp).abs() > 1e-10 {
                return false;
            }
        }
        true
    }
}
