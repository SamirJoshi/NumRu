//! Miscellaneous module implements Numpy routines listed at
//! https://docs.scipy.org/doc/numpy/reference/routines.math.html#miscellaneous
//!
//!

use ndarray::*;
use num_traits::identities::Zero;
use std::{cmp::{max, min},
          fmt::Debug,
          marker::Copy,
          ops::{Add, Mul}};

pub enum ConvolutionMode {
    Full,
    Same,
    Valid,
}

/// convolve
/// Returns the discrete, linear convolution of two one-dimensional sequences.
pub fn convolve<A>(
    arr1: &Array<A, Dim<[usize; 1]>>,
    arr2: &Array<A, Dim<[usize; 1]>>,
    mode: ConvolutionMode,
) -> Array<A, Dim<[usize; 1]>>
where
    A: Debug + Copy + PartialOrd + Add<Output = A> + Mul<Output = A> + Zero,
{
    // init vars and output arrays
    let m = arr1.len();
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
                let to_add = arr1[[a]] * arr2[[b]];
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

// return the newshape portion of the array
fn _centered<A>(arr: &Array<A, Dim<[usize; 1]>>, arr_size: usize, mode_size: usize) -> Array<A, Dim<[usize; 1]>>
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

/// clip
/// Clip (limit) the values in an array.

/// sqrt
/// Return the positive square-root of an array, element-wise.

/// cbrt
/// Return the cube-root of an array, element-wise.

/// square
/// Return the element-wise square of the input.

/// absolute
/// Calculate the absolute value element-wise.

/// fabs
/// Compute the absolute values element-wise.

/// sign
/// Returns an element-wise indication of the sign of a number.

/// heaviside
/// Compute the Heaviside step function.

/// maximum
/// Element-wise maximum of array elements.

/// minimum
/// Element-wise minimum of array elements.

/// fmax
/// Element-wise maximum of array elements.

/// fmin
/// Element-wise minimum of array elements.

/// nan_to_num
/// Replace nan with zero and inf with large finite numbers.

/// real_if_close
/// If complex input returns a real array if complex parts are close to zero.
///

/// interp
/// One-dimensional linear interpolation.

#[cfg(test)]
mod miscellaneous_tests {
    use super::{convolve, ConvolutionMode};

    #[test]
    fn convolve_test() {
        let arr1 = array![1.0, 2.0, 3.0];
        let arr2 = array![0.0, 1.0, 0.5];
        let arr3 = array![0.0, 1.0, 2.5, 4.0, 1.5];
        assert_eq!(convolve(&arr1, &arr2, ConvolutionMode::Full), arr3);

        let arr4 = array![1.0 ,  2.5,  4.0];
        assert_eq!(convolve(&arr1, &arr2, ConvolutionMode::Same), arr4);

        let arr5 = array![2.5];
        assert_eq!(convolve(&arr1, &arr2, ConvolutionMode::Valid), arr5);
    }
}
