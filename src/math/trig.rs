//! Trigonometric Math Module
//! Computes standard trig functions element wise

use std;
use ndarray::*;
use ndarray_parallel::prelude::*;
use num_traits;


pub trait NumRuTrig {
    // type Elt = std::fmt::Debug + std::marker::Copy;
    
    fn sin(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn cos(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn tan(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn asin(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn acos(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn atan(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn to_degrees(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn to_radians(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
}

impl<A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real, D: Dimension> NumRuTrig for Array<A,D> {
    /// Computes element-wise sine on an ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::trig::NumRuTrig;
    ///
    /// # fn main(){
    /// let pi = std::f64::consts::PI;
    /// let input_arr = array![pi, pi / 2.0, 0.004];
    /// let res_arr = input_arr.sin().unwrap();
    /// # }
    /// ```
    fn sin(&self) -> Result<Self, ShapeError> 
    {
        let sin_arr = Array::from_iter(self.iter().map(|x| x.sin()));
        sin_arr.into_shape(self.raw_dim())
    }

    /// Computes element-wise cosine on an ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::trig::NumRuTrig;
    ///
    /// # fn main(){
    /// let pi = std::f64::consts::PI;
    /// let input_arr = array![[0.0, 3.0 * pi / 4.0, pi], [pi / 2.0, 0.004, pi / 4.0]];
    /// let res_arr = input_arr.cos().unwrap();
    /// # }
    /// ```
    fn cos(&self) -> Result<Self, ShapeError>
    {
        let res_arr = Array::from_iter(self.iter().map(|x| x.cos()));
        res_arr.into_shape(self.raw_dim())
    }

    /// Computes element-wise tangent on an ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::trig::NumRuTrig;
    ///
    /// # fn main(){
    /// let pi = std::f64::consts::PI;
    /// let input_arr = array![[0.0, 3.0 * pi / 4.0, pi], [pi / 2.0, 0.004, pi / 4.0]];
    /// let res_arr = input_arr.tan().unwrap();
    /// # }
    /// ```
    fn tan(&self) -> Result<Self,ShapeError>
    {
        let res_arr = Array::from_iter(self.iter().map(|x| {
            x.tan()
        }));
        res_arr.into_shape(self.raw_dim())
    }

    /// Computes element-wise inverse sine on an ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::trig::NumRuTrig;
    ///
    /// # fn main(){
    /// let pi = std::f64::consts::PI;
    /// let two: f64 = 2.0;
    /// let input_arr = array![0.0, two.sqrt() / 2.0, 1.0];
    /// let res_arr = input_arr.asin().unwrap();
    /// assert_eq!(input_arr, res_arr.sin().unwrap());
    /// # }
    /// ```
    fn asin(&self) -> Result<Self,ShapeError>
    {
        let res_arr = Array::from_iter(self.iter().map(|x| x.asin()));
        res_arr.into_shape(self.raw_dim())
    }

    /// Computes element-wise inverse cosine on an ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::trig::{NumRuTrig, compare_arrays};
    ///
    /// # fn main(){
    /// let pi = std::f64::consts::PI;
    /// let two: f64 = 2.0;
    /// let three: f64 = 3.0;
    /// let input_arr = array![0.0, two.sqrt() / 2.0, 0.5, three.sqrt() / 2.0, 1.0];
    /// let expect_arr = array![pi / 2.0, pi / 4.0, pi / 3.0, pi / 6.0, 0.0];
    /// let res_arr = input_arr.acos().unwrap();
    /// assert!(compare_arrays(&expect_arr, &res_arr));
    /// assert!(compare_arrays(&input_arr, &res_arr.cos().unwrap()));
    /// # }
    /// ```
    fn acos(&self) -> Result<Self,ShapeError>
    {
        let res_arr = Array::from_iter(self.iter().map(|x| x.acos()));
        res_arr.into_shape(self.raw_dim())
    }
    
    /// Computes element-wise inverse tangent on an ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::trig::{NumRuTrig, compare_arrays};
    ///
    /// # fn main(){
    /// let pi = std::f64::consts::PI;
    /// let two: f64 = 2.0;
    /// let three: f64 = 3.0;
    /// let input_arr = array![[0.0, 1.0], [three.sqrt(), three.sqrt() / 3.0]];
    /// let expect_arr = array![[0.0, pi / 4.0], [pi / 3.0, pi / 6.0]];
    /// let res_arr = input_arr.atan().unwrap();
    /// assert!(compare_arrays(&expect_arr, &res_arr));
    /// # }
    /// ```
    fn atan(&self) -> Result<Self,ShapeError>
    {
        let res_arr = Array::from_iter(self.iter().map(|x| x.atan()));
        res_arr.into_shape(self.raw_dim())
    }

    /// Convert from radians to degrees element-wise for an ndarray ArcArray
    fn to_degrees(&self) -> Result<Self,ShapeError>
    {
        let conv_factor = A::from(180.0 / std::f64::consts::PI).unwrap();
        let mut res_arr = Array::from_elem(self.dim(), A::from(0.0).unwrap());
        Zip::from(&mut res_arr).and(self).apply(|res_arr, &s| {
            *res_arr = s * conv_factor;
        });
        Ok(res_arr)
    }

    /// Convert from degrees to radians element-wise for an ndarray ArcArray
    fn to_radians(&self) -> Result<Self,ShapeError>
    {
        let conv_factor = A::from(std::f64::consts::PI / 180.0).unwrap();
        let mut res_arr = Array::from_elem(self.dim(), A::from(0.0).unwrap());
        Zip::from(&mut res_arr).and(self).apply(|res_arr, &s| {
            *res_arr = s * conv_factor;
        });
        Ok(res_arr)
    }
}

impl<A: std::fmt::Debug + std::marker::Copy + std::marker::Sync + std::marker::Send + num_traits::real::Real, D: Dimension> NumRuTrig for ArcArray<A,D> {

    /// Computes element-wise sine on an ndarray ArcArray
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::trig::NumRuTrig;
    ///
    /// # fn main(){
    /// let pi = std::f64::consts::PI;
    /// let input_arr = array![pi, pi / 2.0, 0.004].into_shared();
    /// let res_arr = input_arr.sin();
    /// # }
    /// ```
    fn sin(&self) -> Result<Self, ShapeError>
    {
        let mut sin_arr = ArcArray::from_elem(self.dim(), A::from(0.0).unwrap());
        Zip::from(&mut sin_arr)
            .and(self)
            .par_apply(|sin_arr, &s| {
            *sin_arr = s.sin();
        });
        Ok(sin_arr)
    }

    /// Computes element-wise cosine on an ndarray ArcArray
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::trig::NumRuTrig;
    ///
    /// # fn main(){
    /// let pi = std::f64::consts::PI;
    /// let input_arr = array![[0.0, 3.0 * pi / 4.0, pi], [pi / 2.0, 0.004, pi / 4.0]].into_shared();
    /// let res_arr = input_arr.cos();
    /// # }
    /// ```
    fn cos(&self) -> Result<Self, ShapeError>
    {
        let mut cos_arr = ArcArray::from_elem(self.dim(), A::from(0.0).unwrap());
        Zip::from(&mut cos_arr).and(self).par_apply(|cos_arr, &s| {
            *cos_arr = s.cos();
        });
        Ok(cos_arr)
    }

    /// Computes element-wise tangent on an ndarray ArcArray
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::trig::NumRuTrig;
    ///
    /// # fn main(){
    /// let pi = std::f64::consts::PI;
    /// let input_arr = array![[0.0, 3.0 * pi / 4.0, pi], [pi / 2.0, 0.004, pi / 4.0]].into_shared();
    /// let res_arr = input_arr.tan();
    /// # }
    /// ```
    fn tan(&self) -> Result<Self, ShapeError>
    {
        let mut tan_arr = ArcArray::from_elem(self.dim(), A::from(0.0).unwrap());
        Zip::from(&mut tan_arr).and(self).par_apply(|tan_arr, &s| {
            *tan_arr = s.tan();
        });
        Ok(tan_arr)
    }

    /// Computes element-wise inverse sine on an ndarray ArcArray
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::trig::NumRuTrig;
    ///
    /// # fn main(){
    /// let pi = std::f64::consts::PI;
    /// let two: f64 = 2.0;
    /// let input_arr = array![0.0, two.sqrt() / 2.0, 1.0].into_shared();
    /// let res_arr = input_arr.asin().unwrap();
    /// assert_eq!(input_arr, res_arr.sin().unwrap());
    /// # }
    /// ```
    fn asin(&self) -> Result<Self, ShapeError>
    {
        let mut res_arr = ArcArray::from_elem(self.dim(), A::from(0.0).unwrap());
        Zip::from(&mut res_arr).and(self).par_apply(|res_arr, &s| {
            *res_arr = s.asin();
        });
        Ok(res_arr)
    }

    /// Computes element-wise inverse cosine on an ndarray ArcArray
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::trig::{NumRuTrig, compare_arc_arrays};
    ///
    /// # fn main(){
    /// let pi = std::f64::consts::PI;
    /// let two: f64 = 2.0;
    /// let three: f64 = 3.0;
    /// let input_arr = array![0.0, two.sqrt() / 2.0, 0.5, three.sqrt() / 2.0, 1.0].into_shared();
    /// let expect_arr = array![pi / 2.0, pi / 4.0, pi / 3.0, pi / 6.0, 0.0].into_shared();
    /// let res_arr = input_arr.acos().unwrap();
    /// assert!(compare_arc_arrays(&expect_arr, &res_arr));
    /// assert!(compare_arc_arrays(&input_arr, &res_arr.cos().unwrap()));
    /// # }
    /// ```
    fn acos(&self) -> Result<Self, ShapeError>
    {
        let mut res_arr = ArcArray::from_elem(self.dim(), A::from(0.0).unwrap());
        Zip::from(&mut res_arr).and(self).par_apply(|res_arr, &s| {
            *res_arr = s.acos();
        });
        Ok(res_arr)
    }

    /// Computes element-wise inverse tangent on an ndarray ArcArray
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::trig::{NumRuTrig, compare_arc_arrays};
    ///
    /// # fn main(){
    /// let pi = std::f64::consts::PI;
    /// let two: f64 = 2.0;
    /// let three: f64 = 3.0;
    /// let input_arr = array![[0.0, 1.0], [three.sqrt(), three.sqrt() / 3.0]].into_shared();
    /// let expect_arr = array![[0.0, pi / 4.0], [pi / 3.0, pi / 6.0]].into_shared();
    /// let res_arr = input_arr.atan().unwrap();
    /// assert!(compare_arc_arrays(&expect_arr, &res_arr));
    /// # }
    /// ```
    fn atan(&self) -> Result<Self, ShapeError>
    {
        let mut res_arr = ArcArray::from_elem(self.dim(), A::from(0.0).unwrap());
        Zip::from(&mut res_arr).and(self).par_apply(|res_arr, &s| {
            *res_arr = s.atan();
        });
        Ok(res_arr)
    }

    /// Convert from radians to degrees element-wise for an ndarray ArcArray
    fn to_degrees(&self) -> Result<Self,ShapeError>
    {
        let conv_factor = A::from(180.0 / std::f64::consts::PI).unwrap();
        let mut res_arr = ArcArray::from_elem(self.dim(), A::from(0.0).unwrap());
        Zip::from(&mut res_arr).and(self).par_apply(|res_arr, &s| {
            *res_arr = s * conv_factor;
        });
        Ok(res_arr)
    }

    /// Convert from degrees to radians element-wise for an ndarray ArcArray
    fn to_radians(&self) -> Result<Self,ShapeError>
    {
        let conv_factor = A::from(std::f64::consts::PI / 180.0).unwrap();
        let mut res_arr = ArcArray::from_elem(self.dim(), A::from(0.0).unwrap());
        Zip::from(&mut res_arr).and(self).par_apply(|res_arr, &s| {
            *res_arr = s * conv_factor;
        });
        Ok(res_arr)
    }

}

// Testing functions 
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
    use super::{compare_arrays, compare_arc_arrays, NumRuTrig};

    const TAN_INF : f64 = 16331239353195370.0;

    #[test]
    fn sin_tests() {
        let pi = std::f64::consts::PI;
        let input_arr = array![pi, pi / 2.0];
        let expected_arr = array![0.0, 1.0];
        let res_arr = input_arr.sin().unwrap();
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn sin_tests_rayon() {
        let pi = std::f64::consts::PI;
        let input_arr = array![pi, pi / 2.0].into_shared();
        let expected_arr = array![0.0, 1.0].into_shared();
        let res_arr = input_arr.sin().unwrap();
        assert!(compare_arc_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn cos_tests() {
        let pi = std::f64::consts::PI;
        let input_arr = array![pi, pi / 2.0, 0.0];
        let expected_arr = array![-1.0, 0.0, 1.0];
        let res_arr = input_arr.cos().unwrap();
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn cos_tests_rayon() {
        let pi = std::f64::consts::PI;
        let input_arr = array![pi, pi / 2.0, 0.0].into_shared();
        let expected_arr = array![-1.0, 0.0, 1.0].into_shared();
        let res_arr = input_arr.cos().unwrap();
        assert!(compare_arc_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn tan_tests() {
        let pi = std::f64::consts::PI;
        let input_arr = array![0.0, pi / 4.0, pi / 2.0, pi];
        let expected_arr = array![0.0, 1.0, TAN_INF, 0.0];
        let res_arr = input_arr.tan().unwrap();
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn tan_tests_rayon() {
        let pi = std::f64::consts::PI;
        let input_arr = array![0.0, pi / 4.0, pi / 2.0, pi].into_shared();
        let expected_arr = array![0.0, 1.0, TAN_INF, 0.0].into_shared();
        let res_arr = input_arr.tan().unwrap();
        assert!(compare_arc_arrays(&expected_arr, &res_arr));
    }
}
