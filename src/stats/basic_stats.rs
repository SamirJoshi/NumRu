use ndarray::*;
use std;
use std::thread;
use std::sync::{Arc, Mutex};
use crossbeam;

/// Retrieves the min element from an ndarray Array
/// 
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// # extern crate chrono;
/// use ndarray::*;
/// use num_ru::stats::basic_stats::*;
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
      A: std::fmt::Debug + std::cmp::Ord +  std::marker::Copy,
{
    let mut arr_item = arr.iter();
    let mut min_elem = arr_item.next().unwrap();
    while let Some(curr_item) = arr_item.next() {
        if curr_item < min_elem {
            min_elem = curr_item;
        }
    }
    (*min_elem).clone()
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
/// use num_ru::stats::basic_stats::*;
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
      A: std::fmt::Debug + std::cmp::Ord +  std::marker::Copy + std::marker::Sync + std::marker::Send,
{
    println!("number of elements: {:?}", arr.len()); 
    let num_elem = arr.len();
    if num_elem < 100 {
        amax_simple(arr)
    } else {
        amax_parallelized(arr)
    }
}

pub fn amax_simple<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::cmp::Ord +  std::marker::Copy,
{
    println!("in simple - amax");
    let num_elem = arr.len();
    let mut arr_item = arr.iter();
    let mut max_elem = arr_item.next().unwrap();
    while let Some(curr_item) = arr_item.next() {
        if curr_item > max_elem {
            max_elem = curr_item;
        }
    }
    (*max_elem).clone()
}

pub fn amax_parallelized<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::cmp::Ord +  std::marker::Copy + std::marker::Sync + std::marker::Send,
{
    let num_elem = arr.len();
    let mut num_splits = 4; // TODO change to be log n??
    if num_elem < 10 {
        num_splits = 1;
    }

    // let num_splits = (num_elem as f64).log2() as usize; // TODO change to be log n??

    let mut arr_iter = arr.iter();
    let thread_max = Arc::new(Mutex::new(arr_iter.next().unwrap()));
    crossbeam::scope(|scope| {
        for i in 0 .. num_splits {
            let mut curr_iter = arr_iter.clone();
            let thread_max = thread_max.clone();
            let skip_val = i * num_elem / num_splits as usize;
            let mut end_val = (i + 1) * num_elem / num_splits as usize;
            if i == (num_splits - 1) {
                end_val = num_elem;
            }
            let mut offset_iter = curr_iter.skip(skip_val);
            scope.spawn(move || {
                let mut max_elem = offset_iter.next().unwrap();
                let mut thread_i = skip_val;
                while let Some(curr_item) = offset_iter.next() {
                    if thread_i >= end_val { break; }
                    if curr_item > max_elem {
                        max_elem = curr_item;
                    }
                    thread_i += 1;
                }
                let mut guard = thread_max.lock().unwrap();
                if max_elem > (*guard)
                {
                    *guard = max_elem;
                }
            });
        }
    });

    let mut guard = thread_max.lock().unwrap();
    (*guard).clone()
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
        let arr = array![5, 3, 5, 2, 1];
        assert_eq!(amax(&arr), 5);
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
