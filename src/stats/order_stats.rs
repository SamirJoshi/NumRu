use ndarray::*;
use std;
use ndarray_parallel::prelude::*;
use num_traits;

pub trait NumRuOrderStats {
    type Elt: std::fmt::Debug + std::marker::Copy + std::cmp::PartialOrd + 
        std::ops::Sub + num_traits::real::Real;

    fn amin(&self) -> Self::Elt;
    fn amax(&self) -> Self::Elt;
    fn percentile(&self, search_elem: Self::Elt, _interpolation: Option<String>) -> f64;
    fn ptp(&self) -> Self::Elt;
}

impl<A: std::fmt::Debug + std::marker::Copy + std::cmp::PartialOrd + 
    std::ops::Sub + num_traits::real::Real, D: Dimension> NumRuOrderStats
    for Array<A, D> {
    type Elt = A;

    /// Retrieves the min element from an ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::stats::order_stats::*;
    ///
    /// # fn main(){
    /// let arr = array![[[5., 6.], [7., 0.]], [[1., 2.], [3., 4.]]];
    /// assert_eq!(arr.amin(), 0.);
    /// # }
    /// ```
    fn amin(&self) -> Self::Elt
    {
        println!("in simple - amax");
        let mut arr_iter = self.iter();
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
    /// use ndarray::*;
    /// use num_ru::stats::order_stats::*;
    /// # fn main(){
    ///     let arr = array![[[5., 6.], [7., 0.]], [[1., 2.], [3., 4.]]];
    ///     assert_eq!(arr.amax(), 7.);
    /// # }
    /// ```
    ///
    fn amax(&self) -> Self::Elt
    {
        if self.len() < 1 {
            panic!("Array of 0 elements")
        }
        let mut arr_iter = self.iter();
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
    ///     assert_eq!(arr.ptp(), 5.7);
    ///     let arr2 = array![[[-5.1, -6.1], [-6.2, 5.8]], [[-1.0, -2.0], [-3.0, -4.0]]];
    ///     assert_eq!(arr2.ptp(), 12.0);
    /// # }
    /// ```
    ///
    fn ptp(&self) -> Self::Elt
    {
        let max_elem = self.amax();
        let min_elem = self.amin();
        max_elem - min_elem
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
    ///     assert_eq!(arr3d.percentile(2.5, None), 0.375);
    /// # }
    /// ```
    ///
    fn percentile(&self, search_elem: Self::Elt, _interpolation: Option<String>) -> f64
    {
        let num_elem = self.len() as f64;
        let num_below = self.iter().fold(0.0, |acc, x: &A| {
            if search_elem >= *x {
                acc + 1.0
            } else {
                acc
            }
        });

        num_below / num_elem
    }
}

impl<A: std::fmt::Debug + std::marker::Copy + std::cmp::PartialOrd + 
    std::marker::Sync + std::marker::Send + 
    std::ops::Sub + num_traits::real::Real, D: Dimension> NumRuOrderStats
    for ArcArray<A, D> {
    type Elt = A;

    /// Retrieves the min element from an ndarray ArcArray
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::stats::order_stats::*;
    ///
    /// # fn main(){
    /// let arr = array![[[5.0, 6.], [7., 0.]], [[1., 2.], [3., 4.]]].into_shared();
    /// assert_eq!(arr.amin(), 0.);
    /// # }
    /// ```
    fn amin(&self) -> Self::Elt
    {
        let min_elem = self.par_iter()
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

    /// Retrieves the max element from an ndarray ArcArray
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::stats::order_stats::*;
    /// # fn main(){
    ///     let arr = array![[[5., 6.], [7., 0.]], [[1., 2.], [3., 4.]]].into_shared();
    ///     assert_eq!(arr.amax(), 7.);
    /// # }
    /// ```
    fn amax(&self) -> Self::Elt
    {
        let max_elem = self.par_iter()
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
    ///     assert_eq!(arr.ptp(), 5.7);
    ///     let arr2 = array![[[-5.1, -6.1], [-6.2, 5.8]], [[-1.0, -2.0], [-3.0, -4.0]]].into_shared();
    ///     assert_eq!(arr2.ptp(), 12.0);
    /// # }
    /// ```
    ///
    fn ptp(&self) -> Self::Elt
    {
        let max_elem = self.amax();
        let min_elem = self.amin();
        max_elem - min_elem
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
    ///     assert_eq!(arr3d.percentile(2.5, None), 0.375);
    /// # }
    /// ```
    ///
    fn percentile(&self, search_elem: Self::Elt, _interpolation: Option<String>) -> f64
    {
        let num_elem = self.len() as f64;
        let num_below = self.iter().fold(0.0, |acc, x: &A| {
            if search_elem >= *x {
                acc + 1.0
            } else {
                acc
            }
        });

        num_below / num_elem
    }
}

#[cfg(test)]
mod amin_tests {
    use super::NumRuOrderStats;

    #[test]
    fn amin_test_1d(){
        let arr = array![5., 3., 5., 2., 1.];
        assert_eq!(arr.amin(), 1.);
        let arr2 = array![8., 8., 8., 8., 8.];
        assert_eq!(arr2.amin(), 8.);
        let arr3 = array![1., 3., 5., 2., 1.];
        assert_eq!(arr3.amin(), 1.);
        let arr5 = array![4., 3., -1., 2., 1.];
        assert_eq!(arr5.amin(), -1.);
    }

    #[test]
    fn amin_test_1d_rayon(){
        let arr = array![5., 3., 5., 2., 1.].into_shared();
        assert_eq!(arr.amin(), 1.);
        let arr2 = array![8., 8., 8., 8., 8.].into_shared();
        assert_eq!(arr2.amin(), 8.);
        let arr3 = array![1., 3., 5., 2., 1.].into_shared();
        assert_eq!(arr3.amin(), 1.);
        let arr5 = array![4., 3., -1., 2., 1.].into_shared();
        assert_eq!(arr5.amin(), -1.);
    }

    #[test]
    fn amin_test_2d() {
        let arr = array![[5., 3.], [1., 2.]];
        assert_eq!(arr.amin(), 1.);
        let arr2 = array![[8., 8.], [8., 8.]];
        assert_eq!(arr2.amin(), 8.);
    }

    #[test]
    fn amin_test_3d() {
        let arr = array![[[5., 6.], [7., 0.]], [[1., 2.], [3., 4.]]];
        assert_eq!(arr.amin(), 0.);
        let arr2 = array![[[-5., -6.], [-7., 0.]], [[-1., -2.], [-3., -4.]]];
        assert_eq!(arr2.amin(), -7.);
    }
}

#[cfg(test)]
mod amax_tests {
    use super::NumRuOrderStats;
    use ndarray::*;

    #[test]
    fn amax_rayon_test_1d(){
        let arr2 = array![5., 3., 5., 2., 1.].into_shared();
        assert_eq!(arr2.amax(), 5.);
    }

    #[test]
    fn amax_test_1d(){
        let arr = array![5.0, 3.0, 5.0, 2.0, 1.0];
        assert_eq!(arr.amax(), 5.0);
        let arr2 = array![8., 8., 8., 8., 8.];
        assert_eq!(arr2.amax(), 8.);
        let arr3 = array![1., 3., 5., 2., 1.];
        assert_eq!(arr3.amax(), 5.);
        let arr5 = array![4., 3., -1., 2., 1.];
        assert_eq!(arr5.amax(), 4.);
    }

    #[test]
    fn amax_test_3d() {
        let arr = array![[[5., 6.], [7., 0.]], [[1., 2.], [3., 4.]]];
        assert_eq!(arr.amax(), 7.);
        let arr2 = array![[[-5., -6.], [-7., 0.]], [[-1., -2.], [-3., -4.]]];
        assert_eq!(arr2.amax(), 0.);
    }
}

#[cfg(test)]
mod ptp_tests {
    use super::NumRuOrderStats;

    #[test]
    fn ptp_test_1d() {
        let arr = array![2.0, 3.0, 4.0];
        assert_eq!(arr.ptp(), 2.0);
        let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
        assert_eq!(arr2.ptp(), 0.0);
        let arr3 = array![1.0, 3.6, 5.9, 2.0, 0.2];
        assert_eq!(arr3.ptp(), 5.7);
    }

    #[test]
    fn ptp_test_3d() {
        let arr = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
        assert_eq!(arr.ptp(), 6.7);
        let arr2 = array![[[-5.1, -6.1], [-6.2, 5.8]], [[-1.0, -2.0], [-3.0, -4.0]]];
        assert_eq!(arr2.ptp(), 12.0);
    }
}

#[cfg(test)]
mod percentile_tests {
    use super::NumRuOrderStats;

    #[test]
    fn perc_test_1d() {
        let arr = array![1.0, 3.6, 5.9, 2.0, 0.2];
        assert_eq!(arr.percentile(0.0, None), 0.0);
        assert_eq!(arr.percentile(10.0, None), 1.0);
        assert_eq!(arr.percentile(1.5, None), 0.4);
        assert_eq!(arr.percentile(4.7, None), 0.8);
    }

    #[test]
    fn perc_test_3d() {
        let arr = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
        assert_eq!(arr.percentile(2.5, None), 0.375);
        let arr2 = array![[[-5.1, -6.1], [-6.2, 5.8]], [[-1.0, -2.0], [-3.0, -4.0]]];
        assert_eq!(arr2.percentile(-1.0, None), 0.875);
    }
}
