use ndarray::*;
use std;
use ndarray_parallel::prelude::*;
use num_traits;

pub trait NumRuAverages {
    type Elt: std::fmt::Debug + std::marker::Copy + std::cmp::PartialOrd + 
    num_traits::real::Real + std::ops::Add + std::ops::Div + std::ops::Mul + std::ops::Sub;

    fn mean(&self) -> Self::Elt;
    fn var(&self) -> Self::Elt;
    fn std_dev(&self) -> Self::Elt;
    fn sort_to_vec(&self) -> Vec<&Self::Elt>;
    fn median(&self) -> Self::Elt;
}

impl<A: std::fmt::Debug + std::marker::Copy + std::cmp::PartialOrd + 
    num_traits::real::Real + std::ops::Add + std::ops::Div, D: Dimension> NumRuAverages
    for Array<A, D> {
    type Elt = A;

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
    ///     assert_eq!(arr.mean(), 3.5);
    /// # }
    /// ```
    fn mean(&self) -> Self::Elt
    {
        let num_elem: A = A::from(self.len()).unwrap();
        let arr_sum = self.iter().fold(num_traits::zero(), |acc: A, x| acc + *x);
        arr_sum / num_elem
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
    ///     assert!((arr.var()- 5.138) < 1e-10);
    ///     let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
    ///     assert_eq!(arr2.var(), 0.0);
    ///     let arr3 = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
    ///     assert_eq!(arr3.var(), 5.71125);
    /// # }
    /// ```
    ///
    fn var(&self) -> Self::Elt
    {
        let avg = self.mean();
        let num_elem: A = A::from(self.len() - 1).unwrap();
        let arr_sum = self.iter().fold(num_traits::zero(), |acc: A, x| acc + ((*x - avg) * (*x - avg)));
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
    ///    assert_eq!(arr.std_dev(), 1.0);
    ///    let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
    ///    assert_eq!(arr2.std_dev(), 0.0);
    ///    let arr3 = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
    ///    assert!((arr3.std_dev()- 2.3898221691164) < 1e-10);
    /// # }
    /// ```
    ///
    fn std_dev(&self) -> Self::Elt
    {
        self.var().sqrt()
    }

    fn sort_to_vec(&self) -> Vec<&Self::Elt>
    {
        let mut sorted_elem : Vec<&A> = self.iter().collect();
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
    ///     assert_eq!(arr.median(), 2.0);
    ///     let arr2 = array![2.0, 4.0, 1.0, 3.0];
    ///     assert_eq!(arr2.median(), 2.5);
    /// # }
    /// ```
    ///
    fn median(&self) -> Self::Elt
    {
        let sorted_elem = self.sort_to_vec();
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
}

impl<A: std::fmt::Debug + std::marker::Copy + std::marker::Sync + std::marker::Send + std::cmp::PartialOrd + 
    num_traits::real::Real + std::ops::Add + std::ops::Div + std::ops::Mul + std::ops::Sub, D: Dimension> NumRuAverages
    for ArcArray<A, D> {
    type Elt = A;

    /// Retrieves the mean across an ndarray ArcArray
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::stats::averages::*;
    /// # fn main(){
    ///     let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]].into_shared();
    ///     assert_eq!(arr.mean(), 3.5);
    /// # }
    /// ```
    ///
    fn mean(&self) -> Self::Elt
    {
        let num_elem: A = A::from(self.len()).unwrap();
        let arr_sum = self.par_iter().cloned().reduce_with( |a, b| a + b);
        match arr_sum {
            Some(a) => a / num_elem,
            None => panic!("Array of 0 elements")
        }
    }

    /// Returns the variance of an ndarray ArcArray
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::stats::averages::*;
    /// # fn main(){
    ///     let arr = array![1.0, 3.6, 5.9, 2.0, 0.2].into_shared();
    ///     assert!((arr.var() - 5.138) < 1e-10);
    ///     let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0].into_shared();
    ///     assert_eq!(arr2.var(), 0.0);
    ///     let arr3 = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]].into_shared();
    ///     assert_eq!(arr3.var(), 5.71125);
    /// # }
    /// ```
    ///
    fn var(&self) -> Self::Elt
    {
        let avg = self.mean();
        let num_elem: A = A::from(self.len() - 1).unwrap();
        let arr_sum = self.iter().fold(num_traits::zero(), |acc: A, x| acc + ((*x - avg) * (*x - avg)));
        arr_sum / num_elem
    }

    /// Returns the standard deviation of an ndarray ArcArray
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::stats::averages::*;
    /// # fn main(){
    ///    let arr = array![2.0, 3.0, 4.0].into_shared();
    ///    assert_eq!(arr.std_dev(), 1.0);
    ///    let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0].into_shared();
    ///    assert_eq!(arr2.std_dev(), 0.0);
    ///    let arr3 = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]].into_shared();
    ///    assert!((arr3.std_dev()- 2.3898221691164) < 1e-10);
    /// # }
    /// ```
    ///
    fn std_dev(&self) -> Self::Elt
    {
        self.var().sqrt()
    }

    fn sort_to_vec(&self) -> Vec<&Self::Elt>
    {
        let mut sorted_elem : Vec<&A> = self.iter().collect();
        sorted_elem.sort_by(|a, b| (*a).partial_cmp(*b).unwrap());

        sorted_elem
    }

    /// Returns the median of an ndarray ArcArray
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::stats::averages::*;
    /// # fn main(){
    ///     let arr = array![1.0, 3.6, 5.9, 2.0, 0.2].into_shared();
    ///     assert_eq!(arr.median(), 2.0);
    ///     let arr2 = array![2.0, 4.0, 1.0, 3.0].into_shared();
    ///     assert_eq!(arr2.median(), 2.5);
    /// # }
    /// ```
    ///
    fn median(&self) -> Self::Elt
    {
        let sorted_elem = self.sort_to_vec();
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
}

//pub fn mean_along_axis<A, D>(arr: &Array<A, D>, axis: usize) -> Array<A, D::Smaller>
//    where D: Dimension + RemoveAxis,
//          A: std::fmt::Debug + std::marker::Copy +
//          num_traits::real::Real + std::ops::Add + std::ops::Div,
//{
//    arr.mean_axis(Axis(axis))
//}

#[cfg(test)]
mod mean_tests {
    use super::NumRuAverages;

    #[test]
    fn mean_test_1d() {
        let arr = array![2.0, 3.0, 4.0];
        assert_eq!(arr.mean(), 3.0);
        let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
        assert_eq!(arr2.mean(), 8.0);
        let arr3 = array![1.0, 3.0, 5.0, 2.0, 1.0];
        assert_eq!(arr3.mean(), 2.4);
        let arr5 = array![4.0, 3.0, -1.0, 2.0, 1.0];
        assert_eq!(arr5.mean(), 1.8);
    }

    #[test]
    fn mean_test_3d() {
        let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]];
        assert_eq!(arr.mean(), 3.5);
        let arr2 = array![[[-5.0, -6.0], [-7.0, 0.0]], [[-1.0, -2.0], [-3.0, -4.0]]];
        assert_eq!(arr2.mean(), -3.5);
    }
}


#[cfg(test)]
mod var_tests {
    use super::NumRuAverages;

    #[test]
    fn var_test_1d() {
        let arr = array![2.0, 3.0, 4.0];
        assert_eq!(arr.var(), 1.0);
        let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
        assert_eq!(arr2.var(), 0.0);
        let arr3 = array![1.0, 3.6, 5.9, 2.0, 0.2];
        assert!((arr3.var()- 5.138) < 1e-10);
    }

    #[test]
    fn var_test_3d() {
        let arr = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
        assert_eq!(arr.var(), 5.71125);
    }
}

#[cfg(test)]
mod std_tests {
    use super::NumRuAverages;

    #[test]
    fn std_test_1d() {
        let arr = array![2.0, 3.0, 4.0];
        assert_eq!(arr.std_dev(), 1.0);
        let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
        assert_eq!(arr2.std_dev(), 0.0);
        let arr3 = array![1.0, 3.6, 5.9, 2.0, 0.2];
        assert!((arr3.std_dev()- 2.2667156857445) < 1e-10);
    }

    #[test]
    fn std_test_3d() {
        let arr = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
        assert!((arr.std_dev()- 2.3898221691164) < 1e-10);
    }
}

#[cfg(test)]
mod median_tests {
    use super::NumRuAverages;

    #[test]
    fn median_test_1d() {
        let arr = array![2.0, 3.0, 4.0];
        assert_eq!(arr.median(), 3.0);
        let arr2 = array![8.0, 8.0, 8.0, 8.0, 8.0];
        assert_eq!(arr2.median(), 8.0);
        let arr3 = array![1.0, 3.6, 5.9, 2.0, 0.2];
        assert_eq!(arr3.median(), 2.0);
    }

    #[test]
    fn median_test_3d() {
        let arr = array![[[5.0, 6.0], [7.0, 0.3]], [[1.0, 2.0], [3.0, 4.0]]];
        assert_eq!(arr.median(), 3.5);
    }
}
