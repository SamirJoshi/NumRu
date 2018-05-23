//! Arithmetics module implements Numpy routines listed at
//! https://docs.scipy.org/doc/numpy/reference/routines.math.html#arithmetic-operations
//!
//!

use ndarray::*;
use num_traits;
use std;

/// add
/// takes two ndarrays and adds them
#[allow(unused)]
pub fn add<A, D>(arr1: &Array<A, D>, arr2: &Array<A, D>) -> Array<A, D>
where
    D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    unimplemented!()
}

/// reciprocal
/// Return the reciprocal of the argument, element-wise.
/// Calculates 1/x.
#[allow(unused)]
pub fn reciprocal<A, D>(arr: &Array<A, D>) -> Array<A, D>
where
    D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + std::ops::Div<A>,
    f64: std::ops::Div<A>,
{
    // arr.mapv(|x| 1.0 / x)
    unimplemented!()
}

/// positive
/// Numerical positive, element-wise.
pub fn positive<A, D>(arr: &Array<A, D>) -> Array<A, D>
where
    D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + num_traits::sign::Signed,
{
    arr.mapv(|x| x.abs())
}

/// negative
/// Numerical positive, element-wise.
pub fn negative<A, D>(arr: &Array<A, D>) -> Array<A, D>
where
    D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + num_traits::sign::Signed + std::ops::Mul<A>,
{
    arr.mapv(|x| x.abs().neg())
}

/// multiply
/// Multiply arguments element-wise.

/// divide
/// Returns a true division of the inputs, element-wise.

/// power
/// First array elements raised to powers from second array, element-wise.
#[allow(unused)]
pub fn power<A, D>(arr: &Array<A, D>, power: usize) -> Array<A, D>
where
    D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + num_traits::pow::Pow<usize>,
{
    // TODO
    // figure out how to get a type from Self::Output

    // arr.mapv(|x| x.pow(power))
    unimplemented!()
}

/// subtract
/// Subtract arguments, element-wise.

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

/// divmod
/// Return element-wise quotient and remainder simultaneously.
///

#[cfg(test)]
mod arithmetic_tests {
    use super::{negative, positive};
    use ndarray::*;
    use num_traits;
    use std;

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
        let expected_arr = array![-1.0, -0.0, -1.0];
        assert_eq!(negative(&input_arr), expected_arr);

        // TODO: test limits
        // TODO: test other num types
    }
}
