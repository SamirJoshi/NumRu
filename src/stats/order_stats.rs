use ndarray::*;
use std;
//use std::sync::{Arc, Mutex};
//use crossbeam;
//use ndarray_parallel::prelude::*;
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
          A: std::fmt::Debug + std::cmp::PartialOrd +  std::marker::Copy,
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
          A: std::fmt::Debug + std::cmp::PartialOrd + std::marker::Copy + std::marker::Sync + std::marker::Send,
{
    println!("number of elements: {:?}", arr.len());
//    let num_elem = arr.len();
    amax_simple(arr)
}

pub fn amax_simple<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::cmp::PartialOrd +  std::marker::Copy,
{
    println!("in simple - amax");
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

//pub fn amax_simple_rayon<'a, A, D>(arr: &Array<A, D>) -> A
//    where D: Dimension,
//      A: std::fmt::Debug + std::cmp::Ord +  std::marker::Copy + std::marker::Sync + 'a,
//          Array<A, D> : NdarrayIntoParallelRefIterator<'a>,
//{
//    println!("in simple - amax");
//    let max_elem = arr.par_iter().max();
//    match max_elem {
//        Some(m) => m.clone(),
//        None => panic!("Array of 0 elements")
//    }
//
//    let mut a = Array2::<f64>::zeros((128, 128));
//
//     Parallel versions of regular array methods (ParMap trait)
//    a.par_map_inplace(|x| *x = x.exp());
//    a.par_mapv_inplace(f64::exp);
//
//     You can also use the parallel iterator directly
//    a.par_iter_mut().for_each(|x| *x = x.exp());
//    panic!("ASDF");
//}



//pub fn amax_parallelized<A, D>(arr: &Array<A, D>) -> A
//    where D: Dimension,
//      A: std::fmt::Debug + std::cmp::Ord +  std::marker::Copy + std::marker::Sync + std::marker::Send,
//{
//    let num_elem = arr.len();
//    let mut num_splits = 4; // TODO change to be log n??
//    if num_elem < 10 {
//        num_splits = 1;
//    }
//
//
//    let mut arr_iter = arr.iter();
//    let mut arr_arc = Arc::new(arr);
//    let thread_max = Arc::new(Mutex::new(arr_iter.next().unwrap()));
//
//    crossbeam::scope(|scope| {
//        for i in 0 .. num_splits {
//            let thread_max = thread_max.clone();
//            let skip_val = i * num_elem / num_splits as usize;
//            let mut end_val = (i + 1) * num_elem / num_splits as usize;
//            if i == (num_splits - 1) {
//                end_val = num_elem;
//            }
//            let mut offset_iter = arr_arc.iter().skip(skip_val);
//            scope.spawn(move || {
//                let mut max_elem = offset_iter.next().unwrap();
//                let mut thread_i = skip_val;
//                while let Some(curr_item) = offset_iter.next() {
//                    if thread_i >= end_val { break; }
//                    if curr_item > max_elem {
//                        max_elem = curr_item;
//                    }
//                    thread_i += 1;
//                }
//                let mut guard = thread_max.lock().unwrap();
//                if max_elem > (*guard)
//                {
//                    *guard = max_elem;
//                }
//            });
//        }
//    });
//
//    let mut max = thread_max.lock().unwrap();
//    (*max).clone()
//}


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
pub fn percentile<A, D>(arr: &Array<A, D>, search_elem: A, interpolation: Option<String>) -> f64
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

#[cfg(test)]
mod amin_tests {
    use super::amin;
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
    use super::amax;

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
