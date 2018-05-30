use ndarray::*;
use std;
//use std::sync::{Arc, Mutex};
//use crossbeam;
use ndarray_parallel::prelude::*;
use num_traits;

/// Retrieves the mean across an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::stats::averages::*;
/// # fn main(){
///     let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]];
///     assert_eq!(mean(&arr), 3.5);
/// # }
/// ```
///
pub fn mean<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::marker::Copy +
          num_traits::real::Real + std::ops::Add + std::ops::Div,
{
    let num_elem: A = A::from(arr.len()).unwrap();
    let arr_sum = arr.iter().fold(num_traits::zero(), |acc: A, x| acc + *x);
    arr_sum / num_elem
}
/// Returns the standard deviation of an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::stats::averages::*;
/// # fn main(){
///    let arr = array![2.0, 3.0, 4.0];
///    assert_eq!(std_dev(&arr), 1.0);
///    let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
///    assert_eq!(std_dev(&arr2), 0.0);
///    let arr3 = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
///    assert!((std_dev(&arr3)- 2.3898221691164) < 1e-10);
/// # }
/// ```
///
pub fn std_dev<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::marker::Copy +
          num_traits::real::Real + std::ops::Div + std::ops::Sub +
          std::marker::Sync + std::marker::Send + std::ops::Mul,
{
    var(&arr).sqrt()
}

/// Returns the variance of an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::stats::averages::*;
/// # fn main(){
///     let arr = array![1.0, 3.6, 5.9, 2.0, 0.2];
///     assert!((var(&arr)- 5.138) < 1e-10);
///     let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
///     assert_eq!(var(&arr2), 0.0);
///     let arr3 = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
///     assert_eq!(var(&arr3), 5.71125);
/// # }
/// ```
///
pub fn var<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::marker::Copy +
          num_traits::real::Real + std::ops::Div + std::ops::Sub +
          std::marker::Sync + std::marker::Send + std::ops::Mul,
{
    let avg = mean(arr);
    let num_elem: A = A::from(arr.len() - 1).unwrap();
    let arr_sum = arr.iter().fold(num_traits::zero(), |acc: A, x| acc + ((*x - avg) * (*x - avg)));
    arr_sum / num_elem
}


pub fn sort_elem<A, D>(arr: &Array<A, D>) -> Vec<&A>
    where D: Dimension,
          A: std::fmt::Debug + std::cmp::PartialOrd +  std::marker::Copy,
{
    let mut sorted_elem: Vec<&A> = arr.iter().collect();
    sorted_elem.sort_by(|a, b| (*a).partial_cmp(*b).unwrap());

    sorted_elem
}

/// Returns the median of an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::stats::averages::*;
/// # fn main(){
///     let arr = array![1.0, 3.6, 5.9, 2.0, 0.2];
///     assert_eq!(median(&arr), 2.0);
///     let arr2 = array![2.0, 4.0, 1.0, 3.0];
///     assert_eq!(median(&arr2), 2.5);
/// # }
/// ```
///
pub fn median<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::cmp::PartialOrd +  std::marker::Copy +
          num_traits::real::Real + std::ops::Add + std::ops::Div,
{
    let sorted_elem = sort_elem(arr);
    let num_elem = sorted_elem.len();
    if num_elem % 2 == 0 {
        let a = *sorted_elem[(num_elem / 2) as usize - 1];
        let b = *sorted_elem[(num_elem / 2) as usize];
        let denom: A = A::from(2.0).unwrap();
        (a + b) / denom
    } else {
        (*sorted_elem[(num_elem / 2) as usize]).clone()
    }
}

#[cfg(test)]
mod mean_tests {
    use super::mean;

    #[test]
    fn mean_test_1d() {
        let arr = array![2.0, 3.0, 4.0];
        assert_eq!(mean(&arr), 3.0);
        let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
        assert_eq!(mean(&arr2), 8.0);
        let arr3 = array![1.0, 3.0, 5.0, 2.0, 1.0];
        assert_eq!(mean(&arr3), 2.4);
        let arr5 = array![4.0, 3.0, -1.0, 2.0, 1.0];
        assert_eq!(mean(&arr5), 1.8);
    }

    #[test]
    fn mean_test_3d() {
        let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]];
        assert_eq!(mean(&arr), 3.5);
        let arr2 = array![[[-5.0, -6.0], [-7.0, 0.0]], [[-1.0, -2.0], [-3.0, -4.0]]];
        assert_eq!(mean(&arr2), -3.5);
    }
}


#[cfg(test)]
mod var_tests {
    use super::var;

    #[test]
    fn var_test_1d() {
        let arr = array![2.0, 3.0, 4.0];
        assert_eq!(var(&arr), 1.0);
        let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
        assert_eq!(var(&arr2), 0.0);
        let arr3 = array![1.0, 3.6, 5.9, 2.0, 0.2];
        assert!((var(&arr3)- 5.138) < 1e-10);
    }

    #[test]
    fn var_test_3d() {
        let arr = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
        assert_eq!(var(&arr), 5.71125);
    }
}

#[cfg(test)]
mod std_tests {
    use super::std_dev;

    #[test]
    fn std_test_1d() {
        let arr = array![2.0, 3.0, 4.0];
        assert_eq!(std_dev(&arr), 1.0);
        let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
        assert_eq!(std_dev(&arr2), 0.0);
        let arr3 = array![1.0, 3.6, 5.9, 2.0, 0.2];
        assert!((std_dev(&arr3)- 2.2667156857445) < 1e-10);
    }

    #[test]
    fn std_test_3d() {
        let arr = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
        assert!((std_dev(&arr)- 2.3898221691164) < 1e-10);
    }
}

#[cfg(test)]
mod median_tests {
    use super::median;

    #[test]
    fn median_test_1d() {
        let arr = array![2.0, 3.0, 4.0];
        assert_eq!(median(&arr), 3.0);
        let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
        assert_eq!(median(&arr2), 8.0);
        let arr3 = array![1.0, 3.6, 5.9, 2.0, 0.2];
        assert_eq!(median(&arr3), 2.0);
    }

    #[test]
    fn median_test_3d() {
        let arr = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
        assert_eq!(median(&arr), 3.5);
    }
}
