use ndarray::*;
use ndarray_parallel::prelude::*;
use num_traits;
use std;

pub fn prod<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    arr.iter().fold(num_traits::one(), |acc, x| acc * *x)
}

pub fn sum<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    arr.iter().fold(num_traits::zero(), |acc, x| acc + *x)
}

/// Returns the product of an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::sumproddif::*;
/// # fn main(){
///     let arr = array![[[5.0, 6.0], [7.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]].into_shared();
///     assert_eq!(prod_rayon(&arr), 5040.0);
/// # }
/// ```
///
pub fn prod_rayon<A, D>(arr: &ArcArray<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::marker::Copy + std::marker::Sync + std::marker::Send +
          num_traits::real::Real,
{
    let arr_sum = arr.par_iter().cloned().reduce_with( |a, b| a * b);
    match arr_sum {
        Some(a) => a,
        None => panic!("Array of 0 elements")
    }
}

/// Returns the sum across an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::sumproddif::*;
/// # fn main(){
///     let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]].into_shared();
///     assert_eq!(sum_rayon(&arr), 28.0);
/// # }
/// ```
///
pub fn sum_rayon<A, D>(arr: &ArcArray<A, D>) -> A
    where D: Dimension,
          A: std::fmt::Debug + std::marker::Copy + std::marker::Sync + std::marker::Send +
          num_traits::real::Real + std::ops::Add,
{
    let arr_sum = arr.par_iter().cloned().reduce_with( |a, b| a + b);
    match arr_sum {
        Some(a) => a,
        None => panic!("Array of 0 elements")
    }
}

#[cfg(test)]
mod tests {
    use math::sumproddif::*;
    use ndarray::*;
    use num_traits;
    use std;

    #[test]
    fn prod_test() {
        assert_eq!(prod(&array![1.0,2.0,3.0]),6.0);
    }

    #[test]
    fn sum_test() {
        assert_eq!(sum(&array![1.0,2.0,3.0,4.0]),10.0);
    }
}
