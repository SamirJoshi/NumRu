//! Miscellaneous module implements Numpy routines listed [here](https://docs.scipy.org/doc/numpy/reference/routines.math.html#miscellaneous)
//!
//!

use ndarray::*;
use num_traits::identities::Zero;
use std::{cmp::{max, min, PartialOrd},
          fmt::Debug,
          marker::Copy,
          ops::{Add, Mul}};

const ONE_THIRD_F32: f32 = 1.0 / 3.0;
const ONE_THIRD_F64: f64 = 1.0 / 3.0;

/// Required by user to pass into convolve method,
/// determines the type of convolution to calculate
pub enum ConvolutionMode {
    Full,
    Same,
    Valid,
}

/// Returns the discrete, linear convolution of two one-dimensional sequences.
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::miscellaneous::{Convolve, ConvolutionMode};
///
/// # fn main(){
/// let arr1 = array![1.0, 2.0, 3.0];
/// let arr2 = array![0.0, 1.0, 0.5];
/// let arr3 = array![0.0, 1.0, 2.5, 4.0, 1.5];
/// assert_eq!(arr1.convolve(&arr2, ConvolutionMode::Full), arr3);
///
/// let arr4 = array![1.0, 2.5, 4.0];
/// assert_eq!(arr1.convolve(&arr2, ConvolutionMode::Same), arr4);
///
/// let arr5 = array![2.5];
/// assert_eq!(arr1.convolve(&arr2, ConvolutionMode::Valid), arr5);
/// # }
/// ```
pub trait Convolve<A> {
    fn convolve(
        &self,
        arr2: &Array<A, Dim<[usize; 1]>>,
        mode: ConvolutionMode,
    ) -> Array<A, Dim<[usize; 1]>>;
}

impl<A: Debug + Copy + PartialOrd + Add<Output = A> + Mul<Output = A> + Zero> Convolve<A>
    for Array<A, Dim<[usize; 1]>>
{
    fn convolve(
        &self,
        arr2: &Array<A, Dim<[usize; 1]>>,
        mode: ConvolutionMode,
    ) -> Array<A, Dim<[usize; 1]>> {
        // init vars and output arrays
        let m = self.len();
        let n = arr2.len();

        let out_size = m + n - 1;
        let mut out = Array1::<A>::zeros(out_size);

        // perform convolution calculation
        // TODO: Consider FFT convolution as implemented in scipy
        for i in 0..out_size {
            let mut elem: Option<A> = None;
            let x = min(i, m - 1);
            for a in (0..=x).rev() {
                let b = i - a;
                if b < n {
                    let to_add = self[[a]] * arr2[[b]];
                    match elem {
                        Some(x) => elem = Some(x + to_add),
                        None => elem = Some(to_add),
                    }
                }
            }

            out[[i]] = elem.unwrap();
        }

        // return convolution according to mode requested
        match mode {
            ConvolutionMode::Full => out,
            ConvolutionMode::Same => {
                let s = max(m, n);
                _centered(&out, out_size, s)
            }
            ConvolutionMode::Valid => {
                let s = max(m, n) - min(m, n) + 1;
                _centered(&out, out_size, s)
            }
        }
    }
}

// return the newshape portion of the array
fn _centered<A>(
    arr: &Array<A, Dim<[usize; 1]>>,
    arr_size: usize,
    mode_size: usize,
) -> Array<A, Dim<[usize; 1]>>
where
    A: Debug + Copy + Zero,
{
    let mut out = Array1::<A>::zeros(mode_size);
    let startind = (arr_size - mode_size) / 2;
    for i in 0..mode_size {
        out[[i]] = arr[[startind + i]];
    }
    out
}

/// Clip (limit) the values in an array.
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::miscellaneous::Clip;
///
/// # fn main(){
/// let arr1 = array![[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]];
/// let arr2 = array![[3, 3, 3, 4, 5], [6, 7, 8, 8, 8]];
/// assert_eq!(arr1.clip(3, 8), arr2);
///
/// let arr3 = array![[-1.0, -2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]];
/// let arr4 = array![[3.0, 3.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 8.0, 8.0]];
/// assert_eq!(arr3.clip(3.0, 8.0), arr4);
/// # }
/// ```
pub trait Clip<A, D> {
    fn clip(&self, min: A, max: A) -> Array<A, D>;
}
impl<A: Debug + Copy + PartialOrd, D: Dimension> Clip<A, D> for Array<A, D> {
    fn clip(&self, min: A, max: A) -> Array<A, D> {
        // error chain stuff here to check valid inputs

        // perform clipping
        // candidate for parallelization?
        self.mapv(|x| {
            if x < min {
                min
            } else if x > max {
                max
            } else {
                x
            }
        })
    }
}

/// Return the positive square-root of an array, element-wise.
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::miscellaneous::{Sqrt, ArrayComparisonFloat};
///
/// # fn main(){
/// let arr1 = array![1.0, 4.0, 9.0, 16.0];
/// let arr2 = array![1.0, 2.0, 3.0, 4.0];
/// assert!(arr2.array_comparison(&arr1.sqrt()));
/// # }
/// ```
pub trait Sqrt<A, D>
where
    D: Dimension,
{
    fn sqrt(&self) -> Array<A, D>;
}

macro_rules! impl_Sqrt {
    (for $($t:ty),+) => {
        $(impl<D: Dimension> Sqrt<$t, D> for Array<$t, D> {
            fn sqrt(&self) -> Array<$t, D> {
                self.mapv(|x| x.sqrt())
            }
        })*
    };
}

impl_Sqrt!{for f32, f64}

/// Return the cube-root of an array, element-wise.
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::miscellaneous::{Cbrt, ArrayComparisonFloat};
///
/// # fn main(){
/// let arr1 = array![1.0, 8.0, 27.0, 64.0];
/// let arr2 = array![1.0, 2.0, 3.0, 4.0];
/// assert!(arr2.array_comparison(&arr1.cbrt()));
/// # }
/// ```
pub trait Cbrt<A, D>
where
    D: Dimension,
{
    fn cbrt(&self) -> Array<A, D>;
}

macro_rules! impl_Cbrt {
    (for $($t:ty, $third:ident),+) => {
        $(impl<D: Dimension> Cbrt<$t, D> for Array<$t, D> {
            fn cbrt(&self) -> Array<$t, D> {
                self.mapv(|x| x.powf($third))
            }
        })*
    };
}

impl_Cbrt!{for f32, ONE_THIRD_F32, f64, ONE_THIRD_F64}

/// Return the element-wise square of the input.
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::miscellaneous::{Square, ArrayComparisonFloat};
///
/// # fn main(){
/// let arr1 = array![2.0, 4.0, 6.0, 8.0];
/// let arr2 = array![4.0, 16.0, 36.0, 64.0];
/// assert!(arr2.array_comparison(&arr1.square()));
///
/// let arr3 = array![2, 4, 6, 8];
/// let arr4 = array![4, 16, 36, 64];
/// assert_eq!(arr4, arr3.square());
/// # }
/// ```
pub trait Square<A, D>
where
    D: Dimension,
{
    fn square(&self) -> Array<A, D>;
}

macro_rules! impl_Square {
    (for $($t:ty, $t2:ty, $pow:ident),+) => {
        $(
            impl<D: Dimension> Square<$t, D> for Array<$t, D> {
                fn square(&self) -> Array<$t, D> {
                    self.mapv(|x| x.$pow(2 as $t2))
                }
            }
        )*
    };
}

impl_Square!{ for usize, u32, pow, u8, u32, pow, u16, u32, pow, u32, u32, pow, u64, u32, pow, u128, u32, pow }
impl_Square!{ for isize, u32, pow, i8, u32, pow, i16, u32, pow, i32, u32, pow, i64, u32, pow, i128, u32, pow }
impl_Square!{ for f32, f32, powf, f64, f64, powf }

/// Returns an element-wise indication of the sign of a number.
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::miscellaneous::{Sign, ArrayComparisonFloat};
///
/// # fn main(){
/// let arr1 = array![-0.0, 4.0, -6.0, 8.0];
/// let arr2 = array![0.0, 1.0, -1.0, 1.0];
/// assert!(arr2.array_comparison(&arr1.sign()));
///
/// let arr3 = array![-0, 4, -6, 8];
/// let arr4 = array![0, 1, -1, 1];
/// assert_eq!(arr4, arr3.sign());
/// # }
/// ```
pub trait Sign<A, D>
where
    D: Dimension,
{
    fn sign(&self) -> Array<A, D>;
}

macro_rules! impl_Sign {
    (for $($t:ty),+) => {
        $(
            impl<D: Dimension> Sign<$t, D> for Array<$t, D> {
                fn sign(&self) -> Array<$t, D> {
                    self.mapv(|x| {
                        if x == (0 as $t) {
                            0 as $t
                        } else if x > (0 as $t) {
                            1 as $t
                        } else {
                            -1 as $t
                        }
                    })
                }
            }
        )*
    };
}

impl_Sign!{ for isize, i8, i16, i32, i64, i128, f32, f64 }

/// heaviside
/// Compute the Heaviside step function.
///

/// interp
/// One-dimensional linear interpolation.
///

#[cfg(test)]
mod miscellaneous_tests {
    use super::{ArrayComparisonFloat, Cbrt, Clip, ConvolutionMode, Convolve, Sign, Sqrt, Square};

    #[test]
    fn convolve_test() {
        let arr1 = array![1.0, 2.0, 3.0];
        let arr2 = array![0.0, 1.0, 0.5];
        let arr3 = array![0.0, 1.0, 2.5, 4.0, 1.5];
        assert_eq!(arr1.convolve(&arr2, ConvolutionMode::Full), arr3);

        let arr4 = array![1.0, 2.5, 4.0];
        assert_eq!(arr1.convolve(&arr2, ConvolutionMode::Same), arr4);

        let arr5 = array![2.5];
        assert_eq!(arr1.convolve(&arr2, ConvolutionMode::Valid), arr5);
    }

    #[test]
    fn clip_test() {
        let arr1 = array![[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]];
        let arr2 = array![[3, 3, 3, 4, 5], [6, 7, 8, 8, 8]];
        assert_eq!(arr1.clip(3, 8), arr2);

        let arr3 = array![[-1.0, -2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]];
        let arr4 = array![[3.0, 3.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 8.0, 8.0]];
        assert_eq!(arr3.clip(3.0, 8.0), arr4);
    }

    #[test]
    fn sqrt_test() {
        let arr1 = array![1.0, 4.0, 9.0, 16.0];
        let arr2 = array![1.0, 2.0, 3.0, 4.0];
        assert!(arr2.array_comparison(&arr1.sqrt()));
    }

    #[test]
    fn cbrt_test() {
        let arr1 = array![1.0, 8.0, 27.0, 64.0];
        let arr2 = array![1.0, 2.0, 3.0, 4.0];
        assert!(arr2.array_comparison(&arr1.cbrt()));
    }

    #[test]
    fn square_test() {
        let arr1 = array![2.0, 4.0, 6.0, 8.0];
        let arr2 = array![4.0, 16.0, 36.0, 64.0];
        assert!(arr2.array_comparison(&arr1.square()));

        let arr3 = array![2, 4, 6, 8];
        let arr4 = array![4, 16, 36, 64];
        assert_eq!(arr4, arr3.square());
    }

    #[test]
    fn sign_test() {
        let arr1 = array![-0.0, 4.0, -6.0, 8.0];
        let arr2 = array![0.0, 1.0, -1.0, 1.0];
        assert!(arr2.array_comparison(&arr1.sign()));

        let arr3 = array![-0, 4, -6, 8];
        let arr4 = array![0, 1, -1, 1];
        assert_eq!(arr4, arr3.sign());
    }
}

/// Helper function used for testing comparison between 2 float arrays.
///
/// NOTE: Should probably be moved elsewhere
pub trait ArrayComparisonFloat<A, D>
where
    D: Dimension,
{
    fn array_comparison(&self, arr2: &Array<A, D>) -> bool;
}

macro_rules! impl_ArrayComparisonFloat {
    (for $($t:ty),+) => {
        $(impl<D: Dimension> ArrayComparisonFloat<$t, D> for Array<$t, D> {
            fn array_comparison(&self, arr2: &Array<$t, D>) -> bool
            {
                let mut iter1 = self.iter();
                let mut iter2 = arr2.iter();

                while let Some(r) = iter2.next() {
                    let exp = iter1.next().unwrap();
                    println!("Expected: {}, Res: {}", *exp, *r);
                    if (*r - *exp).abs() > (1e-10 as $t) {
                        return false;
                    }
                }
                true
            }
        })*
    };
}

impl_ArrayComparisonFloat!{for f32, f64}
