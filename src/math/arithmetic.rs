//! Arithmetics module implements Numpy routines listed [here](https://docs.scipy.org/doc/numpy/reference/routines.math.html#arithmetic-operations)
//!
//! add, multiply, divide and subtract already handled by ndarray lib

use ndarray::*;
use ndarray_parallel::prelude::*;
use num_traits;
use std;
use std::{fmt::Debug, marker::Copy, ops::Mul};

/// Return the reciprocal of the argument, element-wise.
/// Calculates 1/x.
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::arithmetic::*;
/// # fn main(){
/// let input_arr = array![1.0, 2.0, 4.0];
/// let expected_arr = array![1.0, 0.5, 0.25];
/// assert_eq!(input_arr.reciprocal(), expected_arr);
/// # }
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

impl_Reciprocal!{ for usize, u8, u16, u32, u64, u128 }
impl_Reciprocal!{ for isize, i8, i16, i32, i64, i128 }
impl_Reciprocal!{ for f32, f64 }

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
/// let arr = array![[[-5.0, 6.0], [7.0, -1.0]], [[1.0, -2.0], [-3.0, -4.0]]];
/// let expected_arr = array![[[5.0, 6.0], [7.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]];
/// assert_eq!(positive(&arr), expected_arr);
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
/// let arr = array![[[-5.0, 6.0], [7.0, -1.0]], [[1.0, -2.0], [-3.0, -4.0]]].into_shared();
/// let expected_arr = array![[[5.0, 6.0], [7.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]].into_shared();
/// assert_eq!(positive_rayon(&arr), expected_arr);
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
/// let arr = array![[[-5.0, 6.0], [7.0, -1.0]], [[1.0, -2.0], [-3.0, -4.0]]];
/// let expected_arr = array![[[5.0, 6.0], [7.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]];
/// assert_eq!(absolute(&arr), expected_arr);
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
/// let arr = array![[[-5.0, 6.0], [7.0, -1.0]], [[1.0, -2.0], [-3.0, -4.0]]].into_shared();
/// let expected_arr = array![[[5.0, 6.0], [7.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]].into_shared();
/// assert_eq!(absolute_rayon(&arr), expected_arr);
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
/// let arr = array![[[-5.0, 6.0], [7.0, -1.0]], [[1.0, -2.0], [-3.0, -4.0]]];
/// let expected_arr = array![[[5.0, -6.0], [-7.0, 1.0]], [[-1.0, 2.0], [3.0, 4.0]]];
/// assert_eq!(negative(&arr), expected_arr);
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
/// let arr = array![[[-5.0, 6.0], [7.0, -1.0]], [[1.0, -2.0], [-3.0, -4.0]]].into_shared();
/// let expected_arr = array![[[5.0, -6.0], [-7.0, 1.0]], [[-1.0, 2.0], [3.0, 4.0]]].into_shared();
/// assert_eq!(negative_rayon(&arr), expected_arr);
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

/// First array elements raised to powers from second array, element-wise.
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::arithmetic::*;
///
/// # fn main(){
/// let arr1 = array![2.0, 3.0, 4.0];
/// let arr2 = array![-1.0, 2.0, 2.5];
/// let arr3 = array![0.5, 9.0, 32.0];
/// assert_eq!(arr1.power(&arr2), arr3);
/// # }
/// ```
pub trait Power<A, B, D>
where
    D: Dimension,
{
    fn power(&self, arr_pow: &Array<B, D>) -> Array<A, D>;
}

macro_rules! impl_Power {
    (for $($t:ty, $t2:ty, $pow:ident),+) => {
        $(
            impl<D: Dimension> Power<$t, $t2, D> for Array<$t, D> {
                fn power(&self, arr_pow: &Array<$t2, D>) -> Array<$t, D> {
                    let mut res = Array::from_elem(self.dim(), 0 as $t);
                    Zip::from(&mut res)
                        .and(self)
                        .and(arr_pow)
                        .apply(|x, &y, &z| {
                            *x = y.$pow(z);
                        });
                    res
                }
            }
        )*
    };
}

impl_Power!{ for usize, u32, pow, u8, u32, pow, u16, u32, pow, u32, u32, pow, u64, u32, pow, u128, u32, pow }
impl_Power!{ for isize, u32, pow, i8, u32, pow, i16, u32, pow, i32, u32, pow, i64, u32, pow, i128, u32, pow }
impl_Power!{ for f32, f32, powf, f64, f64, powf }

/// Return element-wise remainder of division.
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::arithmetic::*;
///
/// # fn main(){
/// let arr1 = array![10.0, 11.0, 12.0];
/// let arr2 = array![3.0, 4.0, 5.0];
/// let arr3 = array![1.0, 3.0, 2.0];
/// assert_eq!(arr1.remainder(&arr2), arr3);
/// # }
/// ```
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

impl_Remainder!{ for usize, u8, u16, u32, u64, u128 }
impl_Remainder!{ for isize, i8, i16, i32, i64, i128 }
impl_Remainder!{ for f32, f64 }

/// divmod
/// Return element-wise quotient and remainder simultaneously.
///

#[cfg(test)]
mod arithmetic_tests {
    use super::{negative, positive, Power, Reciprocal, Remainder};

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

    #[test]
    fn power_test() {
        let arr1 = array![2.0, 3.0, 4.0];
        let arr2 = array![-1.0, 2.0, 2.5];
        let arr3 = array![0.5, 9.0, 32.0];
        assert_eq!(arr1.power(&arr2), arr3);
    }
}
