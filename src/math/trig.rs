use std;
use ndarray::*;
use ndarray_parallel::prelude::*;
use num_traits;


/// Computes element-wise sine on an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::trig::sin;
///
/// # fn main(){
/// let pi = std::f64::consts::PI;
/// let input_arr = array![pi, pi / 2.0, 0.004];
/// let res_arr = sin(&input_arr).unwrap();
/// # }
/// ```
pub fn sin<A, D>(arr: &Array<A, D>) -> Result<Array<A, D>,ShapeError>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    let sin_arr = Array::from_iter(arr.iter().map(|x| x.sin()));
    sin_arr.into_shape(arr.raw_dim())
}

/// Computes element-wise sine on an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::trig::sin_rayon;
///
/// # fn main(){
/// let pi = std::f64::consts::PI;
/// let input_arr = array![pi, pi / 2.0, 0.004].into_shared();
/// let res_arr = sin_rayon(&input_arr);
/// # }
/// ```
//TODO: Handle Unwrap
pub fn sin_rayon<A, D>(arr: &ArcArray<A, D>) -> ArcArray<A, D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real +
        std::marker::Sync + std::marker::Send,
{
    let mut sin_arr = ArcArray::from_elem(arr.dim(), A::from(0.0).unwrap());
    Zip::from(&mut sin_arr)
        .and(arr)
        .par_apply(|sin_arr, &arr| {
        *sin_arr = arr.sin();
    });
    sin_arr
}

/// Computes element-wise cosine on an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::trig::cos;
///
/// # fn main(){
/// let pi = std::f64::consts::PI;
/// let input_arr = array![[0.0, 3.0 * pi / 4.0, pi], [pi / 2.0, 0.004, pi / 4.0]];
/// let res_arr = cos(&input_arr).unwrap();
/// # }
/// ```
pub fn cos<A, D>(arr: &Array<A, D>) -> Result<Array<A, D>, ShapeError>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    //TODO: change to actually handle the error
    let res_arr = Array::from_iter(arr.iter().map(|x| x.cos()));
    res_arr.into_shape(arr.raw_dim())
}

/// Computes element-wise cosine on an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::trig::cos_rayon;
///
/// # fn main(){
/// let pi = std::f64::consts::PI;
/// let input_arr = array![[0.0, 3.0 * pi / 4.0, pi], [pi / 2.0, 0.004, pi / 4.0]].into_shared();
/// let res_arr = cos_rayon(&input_arr);
/// # }
/// ```
//TODO: Handle Unwrap
pub fn cos_rayon<A, D>(arr: &ArcArray<A, D>) -> ArcArray<A, D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real +
        std::marker::Sync + std::marker::Send,
{
    let mut cos_arr = ArcArray::from_elem(arr.dim(), A::from(0.0).unwrap());
    Zip::from(&mut cos_arr).and(arr).par_apply(|cos_arr, &arr| {
        *cos_arr = arr.cos();
    });
    cos_arr
}

/// Computes element-wise tangent on an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::trig::tan;
///
/// # fn main(){
/// let pi = std::f64::consts::PI;
/// let input_arr = array![[0.0, 3.0 * pi / 4.0, pi], [pi / 2.0, 0.004, pi / 4.0]];
/// let res_arr = tan(&input_arr).unwrap();
/// # }
/// ```
pub fn tan<A, D>(arr: &Array<A, D>) -> Result<Array<A, D>,ShapeError>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    //TODO: change to actually handle the error
    let res_arr = Array::from_iter(arr.iter().map(|x| {
        x.tan()
    }));
    res_arr.into_shape(arr.raw_dim())
}

/// Computes element-wise tangent on an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::trig::tan_rayon;
///
/// # fn main(){
/// let pi = std::f64::consts::PI;
/// let input_arr = array![[0.0, 3.0 * pi / 4.0, pi], [pi / 2.0, 0.004, pi / 4.0]].into_shared();
/// let res_arr = tan_rayon(&input_arr);
/// # }
/// ```
pub fn tan_rayon<A, D>(arr: &ArcArray<A, D>) -> ArcArray<A, D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real +
        std::marker::Sync + std::marker::Send,
{
    let mut tan_arr = ArcArray::from_elem(arr.dim(), A::from(0.0).unwrap());
    Zip::from(&mut tan_arr).and(arr).par_apply(|tan_arr, &arr| {
        *tan_arr = arr.tan();
    });
    tan_arr
}

/// Computes element-wise inverse sine on an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::trig::{sin_rayon, arcsin_rayon};
///
/// # fn main(){
/// let pi = std::f64::consts::PI;
/// let two: f64 = 2.0;
/// let input_arr = array![0.0, two.sqrt() / 2.0, 1.0].into_shared();
/// let res_arr = arcsin_rayon(&input_arr);
/// assert_eq!(input_arr, sin_rayon(&res_arr));
/// # }
/// ```
pub fn arcsin<A, D>(arr: &Array<A, D>) -> Result<Array<A, D>,ShapeError>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    //TODO: change to actually handle the error
    let res_arr = Array::from_iter(arr.iter().map(|x| x.asin()));
    res_arr.into_shape(arr.raw_dim())
}


/// Computes element-wise inverse cosine on an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::trig::{cos_rayon, arccos_rayon, compare_arc_arrays};
///
/// # fn main(){
/// let pi = std::f64::consts::PI;
/// let two: f64 = 2.0;
/// let three: f64 = 3.0;
/// let input_arr = array![0.0, two.sqrt() / 2.0, 0.5, three.sqrt() / 2.0, 1.0].into_shared();
/// let expect_arr = array![pi / 2.0, pi / 4.0, pi / 3.0, pi / 6.0, 0.0].into_shared();
/// let res_arr = arccos_rayon(&input_arr);
/// assert!(compare_arc_arrays(&expect_arr, &res_arr));
/// assert!(compare_arc_arrays(&input_arr, &cos_rayon(&res_arr)));
/// # }
/// ```
pub fn arcsin_rayon<A, D>(arr: &ArcArray<A, D>) -> ArcArray<A, D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real +
        std::marker::Sync + std::marker::Send,
{
    let mut res_arr = ArcArray::from_elem(arr.dim(), A::from(0.0).unwrap());
    Zip::from(&mut res_arr).and(arr).par_apply(|res_arr, &arr| {
        *res_arr = arr.asin();
    });
    res_arr
}

/// Computes element-wise inverse cosine on an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::trig::{cos, arccos, compare_arrays};
///
/// # fn main(){
/// let pi = std::f64::consts::PI;
/// let two: f64 = 2.0;
/// let three: f64 = 3.0;
/// let input_arr = array![0.0, two.sqrt() / 2.0, 0.5, three.sqrt() / 2.0, 1.0];
/// let expect_arr = array![pi / 2.0, pi / 4.0, pi / 3.0, pi / 6.0, 0.0];
/// let res_arr = arccos(&input_arr).unwrap();
/// assert!(compare_arrays(&expect_arr, &res_arr));
/// assert!(compare_arrays(&input_arr, &cos(&res_arr).unwrap()));
/// # }
/// ```
pub fn arccos<A, D>(arr: &Array<A, D>) -> Result<Array<A, D>,ShapeError>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    //TODO: change to actually handle the error
    let res_arr = Array::from_iter(arr.iter().map(|x| x.acos()));
    res_arr.into_shape(arr.raw_dim())
}

/// Computes element-wise inverse cosine on an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::trig::{cos_rayon, arccos_rayon, compare_arc_arrays};
///
/// # fn main(){
/// let pi = std::f64::consts::PI;
/// let two: f64 = 2.0;
/// let three: f64 = 3.0;
/// let input_arr = array![0.0, two.sqrt() / 2.0, 0.5, three.sqrt() / 2.0, 1.0].into_shared();
/// let expect_arr = array![pi / 2.0, pi / 4.0, pi / 3.0, pi / 6.0, 0.0].into_shared();
/// let res_arr = arccos_rayon(&input_arr);
/// assert!(compare_arc_arrays(&expect_arr, &res_arr));
/// assert!(compare_arc_arrays(&input_arr, &cos_rayon(&res_arr)));
/// # }
/// ```
pub fn arccos_rayon<A, D>(arr: &ArcArray<A, D>) -> ArcArray<A, D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real +
        std::marker::Sync + std::marker::Send,
{
    let mut res_arr = ArcArray::from_elem(arr.dim(), A::from(0.0).unwrap());
    Zip::from(&mut res_arr).and(arr).par_apply(|res_arr, &arr| {
        *res_arr = arr.acos();
    });
    res_arr
}

/// Computes element-wise inverse tangent on an ndarray Array
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::trig::{tan, arctan, compare_arrays};
///
/// # fn main(){
/// let pi = std::f64::consts::PI;
/// let two: f64 = 2.0;
/// let three: f64 = 3.0;
/// let input_arr = array![[0.0, 1.0], [three.sqrt(), three.sqrt() / 3.0]];
/// let expect_arr = array![[0.0, pi / 4.0], [pi / 3.0, pi / 6.0]];
/// let res_arr = arctan(&input_arr).unwrap();
/// assert!(compare_arrays(&expect_arr, &res_arr));
/// # }
/// ```
pub fn arctan<A, D>(arr: &Array<A, D>) -> Result<Array<A, D>,ShapeError>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    //TODO: change to actually handle the error
    let res_arr = Array::from_iter(arr.iter().map(|x| x.atan()));
    res_arr.into_shape(arr.raw_dim())
}

/// Computes element-wise inverse tangent on an ndarray ArcArray
///
/// # Examples
/// ```
/// # #[macro_use]
/// # extern crate ndarray;
/// # extern crate num_ru;
/// use ndarray::*;
/// use num_ru::math::trig::{arctan_rayon, compare_arc_arrays};
///
/// # fn main(){
/// let pi = std::f64::consts::PI;
/// let two: f64 = 2.0;
/// let three: f64 = 3.0;
/// let input_arr = array![[0.0, 1.0], [three.sqrt(), three.sqrt() / 3.0]].into_shared();
/// let expect_arr = array![[0.0, pi / 4.0], [pi / 3.0, pi / 6.0]].into_shared();
/// let res_arr = arctan_rayon(&input_arr);
/// assert!(compare_arc_arrays(&expect_arr, &res_arr));
/// # }
/// ```
pub fn arctan_rayon<A, D>(arr: &ArcArray<A, D>) -> ArcArray<A, D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real +
        std::marker::Sync + std::marker::Send,
{
    let mut res_arr = ArcArray::from_elem(arr.dim(), A::from(0.0).unwrap());
    Zip::from(&mut res_arr).and(arr).par_apply(|res_arr, &arr| {
        *res_arr = arr.atan();
    });
    res_arr
}

/// Convert from radians to degrees
pub fn deg2rad<D>(arr: &Array<f64, D>) -> Result<Array<f64, D>,ShapeError>
    where D: Dimension,
{
    //TODO: change to actually handle the error
    //TODO: implement for a generic type
    let conv_factor: f64 = 180.0 / std::f64::consts::PI;

    let res_arr = arr.clone();
    let res_arr_2 = &res_arr * conv_factor;
    res_arr_2.into_shape(arr.raw_dim())
}

/// Convert from degrees to radians
pub fn rad2deg<D>(arr: &Array<f64, D>) -> Result<Array<f64, D>,ShapeError>
    where D: Dimension,
{
    let conv_factor: f64 = std::f64::consts::PI / 180.0;

    let res_arr = arr.clone();
    let res_arr_2 = &res_arr * conv_factor;
    res_arr_2.into_shape(arr.raw_dim())
}

pub fn compare_arc_arrays<D>(expected_arr: &ArcArray<f64, D>, res_arr: &ArcArray<f64, D>) -> bool
    where D: Dimension,
{
    let mut expected_iter = expected_arr.iter();
    let mut res_iter = res_arr.iter();

    while let Some(r) = res_iter.next() {
        let exp = expected_iter.next().unwrap();
        println!("Expected: {}, Res: {}", *exp, *r);
        if (*r - *exp).abs() > 1e-10 {
            return false;
        }
    }
    true
}

pub fn compare_arrays<D>(expected_arr: &Array<f64, D>, res_arr: &Array<f64, D>) -> bool
    where D: Dimension,
{
    let mut expected_iter = expected_arr.iter();
    let mut res_iter = res_arr.iter();

    while let Some(r) = res_iter.next() {
        let exp = expected_iter.next().unwrap();
        println!("Expected: {}, Res: {}", *exp, *r);
        if (*r - *exp).abs() > 1e-10 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod compare_arrays_tests {
    use super::compare_arrays;

    #[test]
    fn match_test() {
        let arr1 = array![0.9, 0.1, 0.4];
        let arr2 = array![0.9, 0.1, 0.40000000000000000000001];
        assert!(compare_arrays(&arr1, &arr2));
    }

    #[test]
    fn no_match_test() {
        let arr1 = array![0.9, 0.1, 0.4];
        let arr2 = array![0.9, 0.1, 0.41000000000000000000001];
        assert!(!compare_arrays(&arr1, &arr2));
    }

    #[test]
    fn match_two_dims_test() {
        let arr1 = array![[0.9, 0.1], [0.399999999999999999999, 0.3]];
        let arr2 = array![[0.9, 0.1], [0.4, 0.3]];
        assert!(compare_arrays(&arr1, &arr2));
    }
}

#[cfg(test)]
mod trig_tests {
    use std;
    use super::{sin, sin_rayon, cos, cos_rayon, tan, tan_rayon, compare_arrays, compare_arc_arrays};

    const TAN_INF : f64 = 16331239353195370.0;

    #[test]
    fn sin_tests() {
        let pi = std::f64::consts::PI;
        let input_arr = array![pi, pi / 2.0];
        let expected_arr = array![0.0, 1.0];
        let res_arr = sin(&input_arr).unwrap();
        assert!(compare_arrays(&expected_arr, &res_arr));

    }

    #[test]
    fn sin_tests_rayon() {
        let pi = std::f64::consts::PI;
        let input_arr_arc = array![pi, pi / 2.0].into_shared();
        let expected_arr_arc = array![0.0, 1.0].into_shared();
        let res_arr_arc = sin_rayon(&input_arr_arc);
        assert!(compare_arc_arrays(&expected_arr_arc, &res_arr_arc));
    }

    #[test]
    fn cos_tests() {
        let pi = std::f64::consts::PI;
        let input_arr = array![pi, pi / 2.0, 0.0];
        let expected_arr = array![-1.0, 0.0, 1.0];
        let res_arr = cos(&input_arr).unwrap();
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn cos_tests_rayon() {
        let pi = std::f64::consts::PI;
        let input_arr_arc = array![pi, pi / 2.0, 0.0].into_shared();
        let expected_arr_arc = array![-1.0, 0.0, 1.0].into_shared();
        let res_arr_arc = cos_rayon(&input_arr_arc);
        assert!(compare_arc_arrays(&expected_arr_arc, &res_arr_arc));
    }

    #[test]
    fn tan_tests() {
        let pi = std::f64::consts::PI;
        let input_arr = array![0.0, pi / 4.0, pi / 2.0, pi];
        let expected_arr = array![0.0, 1.0, TAN_INF, 0.0];
        let res_arr = tan(&input_arr).unwrap();
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn tan_tests_rayon() {
        let pi = std::f64::consts::PI;
        let input_arr = array![0.0, pi / 4.0, pi / 2.0, pi].into_shared();
        let expected_arr = array![0.0, 1.0, TAN_INF, 0.0].into_shared();
        let res_arr = tan_rayon(&input_arr);
        assert!(compare_arc_arrays(&expected_arr, &res_arr));
    }
}
