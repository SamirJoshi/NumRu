use ndarray::*;
use ndarray_parallel::prelude::*;
use num_traits;
use std;

/// Returns the product of an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::sumproddif::*;
/// # fn main(){
///     let arr = array![[[5.0, 6.0], [7.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]];
///     assert_eq!(prod(&arr), 5040.0);
/// # }
/// ```
///
pub fn prod<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + std::ops::Mul + num_traits::identities::One,
{
    arr.iter().fold(A::one(), |acc, x| acc * *x)
}


/// Returns the sum across an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::sumproddif::*;
/// # fn main(){
///     let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]];
///     assert_eq!(sum(&arr), 28.0);
/// # }
/// ```
///
pub fn sum<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + std::ops::Add + num_traits::identities::Zero,
{
    arr.iter().fold(A::zero(), |acc, x| acc + *x)
}

/// Returns the array that cumulatively sums across ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::test::*;
/// use num_ru::math::sumproddif::*;
/// # fn main(){
///     let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]];
///     let res_arr = array![5.0, 11.0, 18.0, 18.0, 19.0, 21.0, 24.0, 28.0];
///     assert!(compare_arrays(&cumsum(&arr),&res_arr));
/// # }
/// ```
///
pub fn cumsum<A, D>(arr: &Array<A, D>) -> Array<A, Dim<[usize; 1]>>
  where D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + std::ops::Add<Output = A> +
    std::ops::Add + num_traits::identities::Zero,
    {
        let mut flat = arr.iter();
        let mut prev = A::zero();
        let mut p = vec![];
        while let Some(a) = flat.next() {
            p.push(*a + prev);
            prev = *a + prev;
        }
        Array::from_vec(p)
    }

/// Returns the array that cumulatively multiplies across ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::test::*;
/// use num_ru::math::sumproddif::*;
/// # fn main(){
///     let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]];
///     let res_arr = array![5.0, 30.0, 210.0, 0.0, 0.0, 0.0, 0.0, 0.0];
///     assert!(compare_arrays(&cumprod(&arr),&res_arr));
/// # }
/// ```
///
pub fn cumprod<A, D>(arr: &Array<A, D>) -> Array<A, Dim<[usize;1]>>
  where D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + std::ops::Add<Output = A> + std::ops::Add + num_traits::identities::One,
    {
        let mut flat = arr.iter();
        let mut prev = A::one();
        let mut p = vec![];
        while let Some(a) = flat.next() {
            p.push(*a * prev);
            prev = *a * prev;
        }
        Array::from_vec(p)
    }

/// Returns a 1d array of the difference between each consecutive
/// element in the array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::sumproddif::*;
/// use num_ru::test::*;
/// # fn main(){
///     let arr = array![ 1.2 , 42.  ,  1.9 ,  3.56,  0.54,  9.4 ,  2.  ];
///     let res_arr = array![ 40.8 , -40.1 ,   1.66,  -3.02,   8.86,  -7.4 ];
///     assert!(compare_arrays(&ediff1d(&arr).unwrap(),&res_arr));
/// # }
/// ```
///
pub fn ediff1d<A,D>(arr: &Array<A, D>) -> Result<Array<A, Dim<[usize;1]>>,ShapeError>
    where D: Dimension,
          A: std::fmt::Debug + std::marker::Copy +
          num_traits::real::Real + std::ops::Add,
{
    let flat = Array::from_iter(arr.into_iter()).to_vec();
    let mut p = vec![*flat[1]-*flat[0]];
    for i in 0..flat.len()-2 {
        let padd = *flat[p.len()+1]-*flat[p.len()];
        p.push(padd);
    }
    let x = Array::from_iter(p.into_iter()).into_shape(flat.len()-1);
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
    use test::*;

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
        // println!("res_arr {:?}", )

        assert!(compare_arrays(&res_arr, &cumsum(&input_arr)));
    }

    #[test]
    fn cumsum_test_integers() {
        let input_arr = array![[1,2,3],[4,5,6]];
        let res_arr = array![1,3,6,10,15,21];

        assert_eq!(res_arr, cumsum(&input_arr));
    }

    #[test]
    fn ediff1d_test() {
        let input_arr = array![[1., 2., 3.],
                               [4., 5., 6.]];
        let res_arr = array![1., 1., 1., 1., 1.];
        assert!(compare_arrays(&res_arr, &ediff1d(&input_arr).unwrap()));
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

        assert_eq!(res_arr, cumprod(&input_arr));
    }

    #[test]
    fn cumprod_empty_test() {
        let input_arr: Array<f64,Dim<[usize;1]>> = array![];
        let res_arr = array![];
        assert_eq!(res_arr, cumprod(&input_arr));
    }
}
