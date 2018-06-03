use ndarray::*;
use std;
use ndarray_parallel::prelude::*;
use num_traits;

/// Retrieves the min element from an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// # extern crate chrono;
/// use ndarray::*;
/// use num_ru::stats::order_stats::*;
/// use chrono::{NaiveDate, NaiveDateTime};
///
/// # fn main(){
/// let arr = array![[[5, 6], [7, 0]], [[1, 2], [3, 4]]];
/// assert_eq!(amin(&arr), 0);
/// let dt1: NaiveDateTime = NaiveDate::from_ymd(2016, 7, 8).and_hms(9, 10, 11);
/// let dt2: NaiveDateTime = NaiveDate::from_ymd(2018, 7, 8).and_hms(9, 10, 11);
/// let dt3: NaiveDateTime = NaiveDate::from_ymd(2016, 7, 8).and_hms(13, 10, 11);
/// let arr2 = array![dt3, dt1, dt2];
/// assert_eq!(amin(&arr2), dt1);
/// # }
/// ```
pub fn amin<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::cmp::PartialOrd + std::marker::Copy,
{
    println!("in simple - amax");
    let mut arr_iter = arr.iter();
    let first_elem = arr_iter.next().unwrap();
    let arr_max = arr_iter.fold(first_elem, |acc: &A, x: &A| {
        if *acc > *x {
            x
        } else {
            acc
        }
    });

    (*arr_max).clone()
}

/// Retrieves the min element from an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// # extern crate chrono;
/// use ndarray::*;
/// use num_ru::stats::order_stats::*;
/// use chrono::{NaiveDate, NaiveDateTime};
///
/// # fn main(){
/// let arr = array![[[5, 6], [7, 0]], [[1, 2], [3, 4]]].into_shared();
/// assert_eq!(amin_rayon(&arr), 0);
/// let dt1: NaiveDateTime = NaiveDate::from_ymd(2016, 7, 8).and_hms(9, 10, 11);
/// let dt2: NaiveDateTime = NaiveDate::from_ymd(2018, 7, 8).and_hms(9, 10, 11);
/// let dt3: NaiveDateTime = NaiveDate::from_ymd(2016, 7, 8).and_hms(13, 10, 11);
/// let arr2 = array![dt3, dt1, dt2].into_shared();
/// assert_eq!(amin_rayon(&arr2), dt1);
/// # }
/// ```
pub fn amin_rayon<A, D>(arr: &ArcArray<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::cmp::PartialOrd + std::marker::Copy + std::marker::Sync,
{
    let min_elem = arr.par_iter()
        .reduce_with(|a:&A, b: &A| {
            if a > b {
                b
            } else {
               a
            }
        });

    match min_elem {
        Some(m) => m.clone(),
        None => panic!("Array of 0 elements")
    }
}

/// Retrieves the max element from an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// # extern crate chrono;
/// use ndarray::*;
/// use num_ru::stats::order_stats::*;
/// use chrono::{NaiveDate, NaiveDateTime};
/// # fn main(){
///     let arr = array![[[5, 6], [7, 0]], [[1, 2], [3, 4]]];
///     assert_eq!(amax(&arr), 7);
///     let dt1: NaiveDateTime = NaiveDate::from_ymd(2016, 7, 8).and_hms(9, 10, 11);
///     let dt2: NaiveDateTime = NaiveDate::from_ymd(2018, 7, 8).and_hms(9, 10, 11);
///     let dt3: NaiveDateTime = NaiveDate::from_ymd(2016, 7, 8).and_hms(13, 10, 11);
///     let arr2 = array![dt3, dt1, dt2];
///     assert_eq!(amax(&arr2), dt2);
/// # }
/// ```
///
pub fn amax<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::cmp::PartialOrd + std::marker::Copy,
{
    if arr.len() < 1 {
        panic!("Array of 0 elements")
    }
    let mut arr_iter = arr.iter();
    let first_elem = arr_iter.next().unwrap();
    let arr_max = arr_iter.fold(first_elem, |acc: &A, x: &A| {
        if *acc < *x {
            x
        } else {
            acc
        }
    });

    (*arr_max).clone()
}

/// Retrieves the max element from an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// # extern crate chrono;
/// use ndarray::*;
/// use num_ru::stats::order_stats::*;
/// use chrono::{NaiveDate, NaiveDateTime};
/// # fn main(){
///     let arr = array![[[5, 6], [7, 0]], [[1, 2], [3, 4]]].into_shared();
///     assert_eq!(amax_rayon(&arr), 7);
///     let dt1: NaiveDateTime = NaiveDate::from_ymd(2016, 7, 8).and_hms(9, 10, 11);
///     let dt2: NaiveDateTime = NaiveDate::from_ymd(2018, 7, 8).and_hms(9, 10, 11);
///     let dt3: NaiveDateTime = NaiveDate::from_ymd(2016, 7, 8).and_hms(13, 10, 11);
///     let arr2 = array![dt3, dt1, dt2].into_shared();
///     assert_eq!(amax_rayon(&arr2), dt2);
/// # }
/// ```
///
pub fn amax_rayon<A, D>(arr: &ArcArray<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::cmp::PartialOrd + std::marker::Copy + std::marker::Sync,
{
    let max_elem = arr.par_iter()
        .reduce_with(|a:&A, b: &A| {
            if a < b {
                b
            } else {
               a
            }
        });

    match max_elem {
        Some(m) => m.clone(),
        None => panic!("Array of 0 elements")
    }
}
/// Returns the percentile of an element in an ndarray Array
/// Defaults to lower if the element between two values
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::stats::order_stats::*;
/// # fn main(){
///     let arr3d = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
///     assert_eq!(percentile(&arr3d, 2.5, None), 0.375);
/// # }
/// ```
///
pub fn percentile<A, D>(arr: &Array<A, D>, search_elem: A, _interpolation: Option<String>) -> f64
    where D: Dimension,
          A: std::fmt::Debug + std::cmp::PartialOrd +  std::marker::Copy,
{
    let num_elem = arr.len() as f64;
    let num_below = arr.iter().fold(0.0, |acc, x: &A| {
        if search_elem >= *x {
            acc + 1.0
        } else {
            acc
        }
    });

    num_below / num_elem
}

/// Returns the percentile of an element in an ndarray ArcArray
/// Defaults to lower if the element between two values
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::stats::order_stats::*;
/// # fn main(){
///     let arr3d = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]].into_shared();
///     assert_eq!(percentile_rayon(&arr3d, 2.5, None), 0.375);
/// # }
/// ```
///
pub fn percentile_rayon<A, D>(arr: &ArcArray<A, D>, search_elem: A, _interpolation: Option<String>) -> f64
    where D: Dimension,
          A: std::fmt::Debug + std::cmp::PartialOrd +  std::marker::Copy,
{
    let num_elem = arr.len() as f64;
    let num_below = arr.iter().fold(0.0, |acc, x: &A| {
        if search_elem >= *x {
            acc + 1.0
        } else {
            acc
        }
    });

    num_below / num_elem
}

/// Returns the range of an ndarray Array
/// For efficiency, this implementation does not use max or min
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::stats::order_stats::*;
/// # fn main(){
///     let arr = array![1.0, 3.6, 5.9, 2.0, 0.2];
///     assert_eq!(ptp(&arr), 5.7);
///     let arr2 = array![[[-5.1, -6.1], [-6.2, 5.8]], [[-1.0, -2.0], [-3.0, -4.0]]];
///     assert_eq!(ptp(&arr2), 12.0);
/// # }
/// ```
///
pub fn ptp<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::cmp::PartialOrd + std::marker::Copy +
          num_traits::real::Real + std::ops::Sub +
          std::marker::Sync + std::marker::Send,
{
    let max_elem = amax(arr);
    let min_elem = amin(arr);
    max_elem - min_elem
}

/// Returns the range of an ndarray ArcArray
/// For efficiency, this implementation does not use max or min
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::stats::order_stats::*;
/// # fn main(){
///     let arr = array![1.0, 3.6, 5.9, 2.0, 0.2].into_shared();
///     assert_eq!(ptp_rayon(&arr), 5.7);
///     let arr2 = array![[[-5.1, -6.1], [-6.2, 5.8]], [[-1.0, -2.0], [-3.0, -4.0]]].into_shared();
///     assert_eq!(ptp_rayon(&arr2), 12.0);
/// # }
/// ```
///
pub fn ptp_rayon<A, D>(arr: &ArcArray<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::cmp::PartialOrd + std::marker::Copy +
          num_traits::real::Real + std::ops::Sub +
          std::marker::Sync + std::marker::Send,
{
    let max_elem = amax_rayon(arr);
    let min_elem = amin_rayon(arr);
    max_elem - min_elem
}

#[cfg(test)]
mod amin_tests {
    use super::{ amin, amin_rayon };
    use chrono::{NaiveDate, NaiveDateTime};

    #[test]
    fn amin_test_1d(){
        let arr = array![5, 3, 5, 2, 1];
        assert_eq!(amin(&arr), 1);
        let arr2 = array![8, 8, 8, 8, 8];
        assert_eq!(amin(&arr2), 8);
        let arr3 = array![1, 3, 5, 2, 1];
        assert_eq!(amin(&arr3), 1);
        let arr5 = array![4, 3, -1, 2, 1];
        assert_eq!(amin(&arr5), -1);
    }

    #[test]
    fn amin_test_1d_rayon(){
        let arr = array![5, 3, 5, 2, 1].into_shared();
        assert_eq!(amin_rayon(&arr), 1);
        let arr2 = array![8, 8, 8, 8, 8].into_shared();
        assert_eq!(amin_rayon(&arr2), 8);
        let arr3 = array![1, 3, 5, 2, 1].into_shared();
        assert_eq!(amin_rayon(&arr3), 1);
        let arr5 = array![4, 3, -1, 2, 1].into_shared();
        assert_eq!(amin_rayon(&arr5), -1);
    }

    #[test]
    fn amin_test_2d() {
        let arr = array![[5, 3], [1, 2]];
        assert_eq!(amin(&arr), 1);
        let arr2 = array![[8, 8], [8, 8]];
        assert_eq!(amin(&arr2), 8);
    }

    #[test]
    fn amin_test_3d() {
        let arr = array![[[5, 6], [7, 0]], [[1, 2], [3, 4]]];
        assert_eq!(amin(&arr), 0);
        let arr2 = array![[[-5, -6], [-7, 0]], [[-1, -2], [-3, -4]]];
        assert_eq!(amin(&arr2), -7);
    }

    #[test]
    fn amin_not_int() {
        let dt1: NaiveDateTime = NaiveDate::from_ymd(2016, 7, 8).and_hms(9, 10, 11);
        let dt2: NaiveDateTime = NaiveDate::from_ymd(2018, 7, 8).and_hms(9, 10, 11);
        let dt3: NaiveDateTime = NaiveDate::from_ymd(2016, 7, 8).and_hms(13, 10, 11);
        let arr = array![dt3, dt1, dt2];
        assert_eq!(amin(&arr), dt1);
    }
}

#[cfg(test)]
mod amax_tests {
    use super::{ amax, amax_rayon };
    use ndarray::*;

    #[test]
    fn amax_rayon_test_1d(){
        let arr2 = array![5, 3, 5, 2, 1].into_shared();
        assert_eq!(amax_rayon(&arr2), 5);
    }

    #[test]
    fn amax_test_1d(){
        let arr = array![5.0, 3.0, 5.0, 2.0, 1.0];
        assert_eq!(amax(&arr), 5.0);
        let arr2 = array![8, 8, 8, 8, 8];
        assert_eq!(amax(&arr2), 8);
        let arr3 = array![1, 3, 5, 2, 1];
        assert_eq!(amax(&arr3), 5);
        let arr5 = array![4, 3, -1, 2, 1];
        assert_eq!(amax(&arr5), 4);
    }

    #[test]
    fn amax_test_3d() {
        let arr = array![[[5, 6], [7, 0]], [[1, 2], [3, 4]]];
        assert_eq!(amax(&arr), 7);
        let arr2 = array![[[-5, -6], [-7, 0]], [[-1, -2], [-3, -4]]];
        assert_eq!(amax(&arr2), 0);
    }
}

#[cfg(test)]
mod ptp_tests {
    use super::ptp;

    #[test]
    fn ptp_test_1d() {
        let arr = array![2.0, 3.0, 4.0];
        assert_eq!(ptp(&arr), 2.0);
        let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
        assert_eq!(ptp(&arr2), 0.0);
        let arr3 = array![1.0, 3.6, 5.9, 2.0, 0.2];
        assert_eq!(ptp(&arr3), 5.7);
    }

    #[test]
    fn ptp_test_3d() {
        let arr = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
        assert_eq!(ptp(&arr), 6.7);
        let arr2 = array![[[-5.1, -6.1], [-6.2, 5.8]], [[-1.0, -2.0], [-3.0, -4.0]]];
        assert_eq!(ptp(&arr2), 12.0);
    }
}

#[cfg(test)]
mod percentile_tests {
    use super::percentile;

    #[test]
    fn perc_test_1d() {
        let arr = array![1.0, 3.6, 5.9, 2.0, 0.2];
        assert_eq!(percentile(&arr, 0.0, None), 0.0);
        assert_eq!(percentile(&arr, 10.0, None), 1.0);
        assert_eq!(percentile(&arr, 1.5, None), 0.4);
        assert_eq!(percentile(&arr, 4.7, None), 0.8);
    }

    #[test]
    fn perc_test_3d() {
        let arr = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
        assert_eq!(percentile(&arr, 2.5, None), 0.375);
        let arr2 = array![[[-5.1, -6.1], [-6.2, 5.8]], [[-1.0, -2.0], [-3.0, -4.0]]];
        assert_eq!(percentile(&arr2, -1.0, None), 0.875);
    }
}
