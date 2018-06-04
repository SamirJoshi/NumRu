//! Arithmetics module implements Numpy routines listed at
//! https://docs.scipy.org/doc/numpy/reference/routines.math.html#arithmetic-operations
//!
//! add, multiply, divide and subtract already handled by ndarray lib

use ndarray::*;
use ndarray_parallel::prelude::*;
use num_traits;
use std;
use std::{fmt::Debug, marker::Copy, ops::Mul};


/// reciprocal
/// Return the reciprocal of the argument, element-wise.
/// Calculates 1/x.
pub trait Reciprocal<T, D>
where
    D: Dimension,
{
    fn reciprocal(&self) -> Array<T, D>;
}

macro_rules! impl_Reciprocal {
    (for $($t:ty),+) => {
        $(impl<D: Dimension> Reciprocal<$t, D> for Array<$t, D> {
            fn reciprocal(&self) -> Array<$t, D> {
                self.mapv(|x| 1 as $t / x)
            }
        })*
    };
}

impl_Reciprocal!{for usize, u32, u64, i32, i64, f32, f64}

/// Returns the numerical positive, element-wise of an ndarray Array
///
/// Same as absolute value
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::arithmetic::*;
/// # fn main(){
///     let arr = array![[[-5.0, 6.0], [7.0, -1.0]], [[1.0, -2.0], [-3.0, -4.0]]];
///     let expected_arr = array![[[5.0, 6.0], [7.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]];
///     assert_eq!(positive(&arr), expected_arr);
/// # }
/// ```
pub fn positive<A, D>(arr: &Array<A, D>) -> Array<A, D>
where
    D: Dimension,
    A: Debug + Copy + num_traits::sign::Signed,
{
    arr.mapv(|x| x.abs())
}

/// Returns the numerical positive, element-wise of an ndarray ArcArray
///
/// Same as absolute_rayon
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::arithmetic::*;
/// # fn main(){
///     let arr = array![[[-5.0, 6.0], [7.0, -1.0]], [[1.0, -2.0], [-3.0, -4.0]]].into_shared();
///     let expected_arr = array![[[5.0, 6.0], [7.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]].into_shared();
///     assert_eq!(positive_rayon(&arr), expected_arr);
/// # }
/// ```
pub fn positive_rayon<A, D>(arr: &ArcArray<A, D>) -> ArcArray<A, D>
where
    D: Dimension,
    A: Debug + Copy + num_traits::sign::Signed + std::marker::Sync + std::marker::Send,
{
    let mut pos_arr = arr.clone();
    Zip::from(&mut pos_arr).and(arr).par_apply(|pos_arr, &arr| {
        *pos_arr = arr.abs();
    });
    pos_arr
}

/// Returns the absolute value, element-wise of an ndarray Array
///
/// Same as positive
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::arithmetic::*;
/// # fn main(){
///     let arr = array![[[-5.0, 6.0], [7.0, -1.0]], [[1.0, -2.0], [-3.0, -4.0]]];
///     let expected_arr = array![[[5.0, 6.0], [7.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]];
///     assert_eq!(absolute(&arr), expected_arr);
/// # }
/// ```
pub fn absolute<A, D>(arr: &Array<A, D>) -> Array<A, D>
where
    D: Dimension,
    A: Debug + Copy + num_traits::sign::Signed,
{
    positive(arr)
}

/// Returns the absolute value, element-wise of an ndarray ArcArray
///
/// Same as positive_rayon
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::arithmetic::*;
/// # fn main(){
///     let arr = array![[[-5.0, 6.0], [7.0, -1.0]], [[1.0, -2.0], [-3.0, -4.0]]].into_shared();
///     let expected_arr = array![[[5.0, 6.0], [7.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]].into_shared();
///     assert_eq!(absolute_rayon(&arr), expected_arr);
/// # }
/// ```
pub fn absolute_rayon<A, D>(arr: &ArcArray<A, D>) -> ArcArray<A, D>
where
    D: Dimension,
    A: Debug + Copy + num_traits::sign::Signed + std::marker::Sync + std::marker::Send,
{
    positive_rayon(arr)
}

/// Returns the negative, element-wise of an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::arithmetic::*;
/// # fn main(){
///     let arr = array![[[-5.0, 6.0], [7.0, -1.0]], [[1.0, -2.0], [-3.0, -4.0]]];
///     let expected_arr = array![[[5.0, -6.0], [-7.0, 1.0]], [[-1.0, 2.0], [3.0, 4.0]]];
///     assert_eq!(negative(&arr), expected_arr);
/// # }
/// ```
pub fn negative<A, D>(arr: &Array<A, D>) -> Array<A, D>
where
    D: Dimension,
    A: Debug + Copy + num_traits::sign::Signed + Mul,
{
    arr.mapv(|x| x.neg())
}

/// Returns the negative, element-wise of an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::arithmetic::*;
/// # fn main(){
///     let arr = array![[[-5.0, 6.0], [7.0, -1.0]], [[1.0, -2.0], [-3.0, -4.0]]].into_shared();
///     let expected_arr = array![[[5.0, -6.0], [-7.0, 1.0]], [[-1.0, 2.0], [3.0, 4.0]]].into_shared();
///     assert_eq!(negative_rayon(&arr), expected_arr);
/// # }
/// ```
pub fn negative_rayon<A, D>(arr: &ArcArray<A, D>) -> ArcArray<A, D>
where
    D: Dimension,
    A: Debug + Copy + num_traits::sign::Signed + Mul + std::marker::Sync + std::marker::Send,
{
    let mut neg_arr = arr.clone();
    Zip::from(&mut neg_arr).and(arr).par_apply(|neg_arr, &arr| {
        *neg_arr = arr.neg();
    });
    neg_arr
}

/// power
/// First array elements raised to powers from second array, element-wise.
// fn power<A, D>(arr: &Array<A, D>, exp: usize) -> Array<A, D>
// where
//     D: Dimension,
//     A: Debug + Copy,
// {

// }

/// floor_divide
/// Return the largest integer smaller or equal to the division of the inputs.

/// float_power
/// First array elements raised to powers from second array, element-wise.

/// fmod
/// Return the element-wise remainder of division.

/// mod
/// Return element-wise remainder of division.
///

/// modf
/// Return the fractional and integral parts of an array, element-wise.

/// remainder
/// Return element-wise remainder of division.
pub trait Remainder<T, D>
where
    D: Dimension,
{
    fn remainder(&self, arr2: &Array<T, D>) -> Array<T, D>;
}

macro_rules! impl_Remainder {
    (for $($t:ty),+) => {
        $(impl<D: Dimension> Remainder<$t, D> for Array<$t, D> {
            fn remainder(&self, arr2: &Array<$t, D>) -> Array<$t, D> {
                let mut res = Array::from_elem(self.dim(), 0 as $t);
                Zip::from(&mut res)
                    .and(self)
                    .and(arr2)
                    .apply(|x, &y, &z| {
                        *x = &y % &z;
                    });
                res
            }
        })*
    };
}

impl_Remainder!{for usize, u32, u64, i32, i64, f32, f64}

/// divmod
/// Return element-wise quotient and remainder simultaneously.
///

#[cfg(test)]
mod arithmetic_tests {
    use super::{negative, positive, Reciprocal, Remainder};

    #[test]
    fn positive_test() {
        let input_arr = array![1.0, 0.0, -1.0];
        let expected_arr = array![1.0, 0.0, 1.0];
        assert_eq!(positive(&input_arr), expected_arr);

        // TODO: test limits
        // TODO: test other num types
    }

    #[test]
    fn negative_test() {
        let input_arr = array![1.0, 0.0, -1.0];
        let expected_arr = array![-1.0, 0.0, 1.0];
        assert_eq!(negative(&input_arr), expected_arr);

        // TODO: test limits
        // TODO: test other num types
    }

    #[test]
    fn reciprocal_test() {
        let input_arr = array![1.0, 2.0, 4.0];
        let expected_arr = array![1.0, 0.5, 0.25];
        assert_eq!(input_arr.reciprocal(), expected_arr);

        let input_arr_2 = array![1, 2, 4];
        let expected_arr_2 = array![1, 0, 0];
        assert_eq!(input_arr_2.reciprocal(), expected_arr_2);
    }

    #[test]
    fn remainder_test() {
        let arr1 = array![10.0, 11.0, 12.0];
        let arr2 = array![3.0, 4.0, 5.0];
        let arr3 = array![1.0, 3.0, 2.0];
        assert_eq!(arr1.remainder(&arr2), arr3);
    }
}
