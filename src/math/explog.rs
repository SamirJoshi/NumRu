use ndarray::*;
use std;
use num_traits;

pub trait NumRuEXP {
    fn exp(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn exp_m1(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn exp2(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn ln(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn log2(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn log10(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
    fn ln_1p(&self) -> Result<Self, ShapeError>
        where Self: std::marker::Sized;
}

impl<A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real ,D: Dimension> NumRuEXP for Array<A,D> {

    /// Returns an ndarray with .exp() applied to each element
    ///
    /// # Examples
    /// ```
    ///
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::explog::*;
    /// # fn main() {
    ///     let arr = array![[12.0,32.0],[0.0,1.0]];
    ///     assert_eq!(arr.exp().unwrap(),array![[12.0_f32.exp(),32.0_f32.exp()],
    ///                           [0.0_f32.exp(),1.0_f32.exp()]]);
    /// # }
    /// ```
    fn exp(&self) -> Result<Self,ShapeError>
    {
        let res_arr = Array::from_iter(self.iter().map(|x| x.exp()));
        res_arr.into_shape(self.raw_dim())
    }

    /// Returns an ndarray with .exp_m1() applied to each element
    ///
    /// # Examples
    /// ```
    ///
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::explog::*;
    /// # fn main() {
    ///     let arr = array![[0.4,0.8],[9.43,2.3]];
    ///     assert_eq!(arr.exp_m1().unwrap(),array![[0.4_f32.exp_m1(),0.8_f32.exp_m1()],
    ///                           [9.43_f32.exp_m1(),2.3_f32.exp_m1()]]);
    /// # }
    /// ```
    fn exp_m1(&self) -> Result<Self,ShapeError>
    {
        let res_arr = Array::from_iter(self.iter().map(|x| x.exp_m1()));
        res_arr.into_shape(self.raw_dim())
    }

    /// Returns an ndarray with .exp2() applied to each element
    ///
    /// # Examples
    /// ```
    ///
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::explog::*;
    /// # fn main() {
    ///     let arr = array![[3.8,1.24],[2.88,23.3]];
    ///     assert_eq!(arr.exp2().unwrap(),array![[3.8_f32.exp2(),1.24_f32.exp2()],
    ///                           [2.88_f32.exp2(),23.3_f32.exp2()]]);
    /// # }
    /// ```
    fn exp2(&self) -> Result<Self,ShapeError>
    {
        let res_arr = Array::from_iter(self.iter().map(|x| x.exp2()));
        res_arr.into_shape(self.raw_dim())
    }

    /// Returns an ndarray with .ln() applied to each element
    ///
    /// # Examples
    /// ```
    ///
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::explog::*;
    /// # fn main() {
    ///     let arr = array![[2.2,3.5,1.0,13.9],[3.04,93.1,0.0,1.0],[13.0,2.89,3.09,30.67]];
    ///     assert_eq!(arr.ln().unwrap(),array![[2.2_f32.ln(),3.5_f32.ln(),1.0_f32.ln(),13.9_f32.ln()],[3.04_f32.ln(),93.1_f32.ln(),0.0_f32.ln(),1.0_f32.ln()],[13.0_f32.ln(),2.89_f32.ln(),3.09_f32.ln(),30.67_f32.ln()]]);
    /// # }
    /// ```
    fn ln(&self) -> Result<Self, ShapeError>
    {
        let res_arr = Array::from_iter(self.iter().map(|x| x.ln()));
        res_arr.into_shape(self.raw_dim())
    }

    /// Returns an ndarray with .log2() applied to each element
    ///
    /// # Examples
    /// ```
    ///
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::explog::*;
    /// # fn main() {
    ///     let arr = array![[5.3,8.58,9.0],[2.2,44.4,2.9],[12.2,94.0,12.0]];
    ///     assert_eq!(arr.log2().unwrap(),array![[5.3_f32.log2(),8.58_f32.log2(),9.0_f32.log2()],[2.2_f32.log2(),44.4_f32.log2(),2.9_f32.log2()],[12.2_f32.log2(),94.0_f32.log2(),12.0_f32.log2()]]);
    /// # }
    /// ```
    fn log2(&self) -> Result<Self, ShapeError>
    {
        let res_arr = Array::from_iter(self.iter().map(|x| x.log2()));
        res_arr.into_shape(self.raw_dim())
    }

    /// Returns an ndarray with .log10() applied to each element
    ///
    /// # Examples
    /// ```
    ///
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::explog::*;
    /// # fn main() {
    ///     let arr = array![[3.5,4.95,2.49],[13.2,45.0,30.4]];
    ///     assert_eq!(arr.log10().unwrap(),array![[3.5_f32.log10(),4.95_f32.log10(),2.49_f32.log10()],[13.2_f32.log10(),45.0_f32.log10(),30.4_f32.log10()]]);
    /// # }
    /// ```
    fn log10(&self) -> Result<Self, ShapeError>
    {
        let res_arr = Array::from_iter(self.iter().map(|x| x.log10()));
        res_arr.into_shape(self.raw_dim())
    }

    /// Returns an ndarray with .ln_1p() applied to each element
    ///
    /// # Examples
    /// ```
    ///
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::explog::*;
    /// # fn main() {
    ///     let arr = array![[5.3,4.34],[18.2,0.43],[23.4,2.04]];
    ///     assert_eq!(arr.ln_1p().unwrap(),array![[5.3_f32.ln_1p(),4.34_f32.ln_1p()],[18.2_f32.ln_1p(),0.43_f32.ln_1p()],[23.4_f32.ln_1p(),2.04_f32.ln_1p()]]);
    /// # }
    /// ```
    fn ln_1p(&self) -> Result<Self, ShapeError>
    {
        let res_arr = Array::from_iter(self.iter().map(|x| x.ln_1p()));
        res_arr.into_shape(self.raw_dim())
    }
}

pub fn logaddexp<A>(x1: A, x2: A) -> A
  where A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
  {
      (x1.exp() + x2.exp()).ln()
  }

pub fn logaddexp2<A>(x1: A, x2: A) -> A
    where A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
    {
        (x1.exp2() + x2.exp2()).log2()
    }


#[cfg(test)]
mod tests {
    use math::explog::*;
    use ndarray::*;

    #[test]
    fn exp_test() {
        let input_arr = array![0.0, 1.0, 5.0, 0.0];
        let expected_arr = array![1.0, 1.0_f32.exp(), 5.0_f32.exp(), 1.0];
        let res_arr = input_arr.exp().unwrap();
        assert_eq!(&expected_arr, &res_arr);
    }

    #[test]
    fn exp_test_3_dim() {
        let input_arr = array![[[2.2,1.4],[1.3,1.7]],
                            [[13.2,1.8],[12.3,17.4]]];
        let expected_arr = array![[[2.2_f32.exp(),1.4_f32.exp()],[1.3_f32.exp(),1.7_f32.exp()]],
                            [[13.2_f32.exp(),1.8_f32.exp()],[12.3_f32.exp(),17.4_f32.exp()]]];
        let res_arr = input_arr.exp().unwrap();
        assert_eq!(&expected_arr, &res_arr);
    }

    #[test]
    fn expm1_test() {
        let input_arr = array![[23.2,35.4,34.2,3.4],
                               [5.2,4.7,5.8,33.0]];
        let expected_arr = array![[23.2_f32.exp_m1(),35.4_f32.exp_m1(),34.2_f32.exp_m1(),3.4_f32.exp_m1()],
                               [5.2_f32.exp_m1(),4.7_f32.exp_m1(),5.8_f32.exp_m1(),33.0_f32.exp_m1()]];
        let res_arr = input_arr.exp_m1().unwrap();
        assert_eq!(&expected_arr, &res_arr);
    }

    #[test]
    fn exp2_test() {
        let input_arr = array![[23.2,35.4,34.2,3.4],
                               [5.2,4.7,5.8,33.0]];
        let expected_arr = array![[23.2_f32.exp2(),35.4_f32.exp2(),34.2_f32.exp2(),3.4_f32.exp2()],
                               [5.2_f32.exp2(),4.7_f32.exp2(),5.8_f32.exp2(),33.0_f32.exp2()]];
        let res_arr = input_arr.exp2().unwrap();
        assert_eq!(&expected_arr, &res_arr);
    }

    #[test]
    fn ln_test() {
        let input_arr = array![[23.2,35.4,34.2,3.4],
                               [5.2,4.7,5.8,33.0]];
        let expected_arr = array![[23.2_f32.ln(),35.4_f32.ln(),34.2_f32.ln(),3.4_f32.ln()],
                               [5.2_f32.ln(),4.7_f32.ln(),5.8_f32.ln(),33.0_f32.ln()]];
        let res_arr = input_arr.ln().unwrap();
        assert_eq!(&expected_arr, &res_arr);
    }

    #[test]
    fn log2_test() {
        let input_arr = array![[23.2,35.4,34.2,3.4],
                               [5.2,4.7,5.8,33.0]];
        let expected_arr = array![[23.2_f32.log2(),35.4_f32.log2(),34.2_f32.log2(),3.4_f32.log2()],
                               [5.2_f32.log2(),4.7_f32.log2(),5.8_f32.log2(),33.0_f32.log2()]];
        let res_arr = input_arr.log2().unwrap();
        assert_eq!(&expected_arr, &res_arr);
    }

    #[test]
    fn log10_test() {
        let input_arr = array![[23.2,35.4,34.2,3.4],
                               [5.2,4.7,5.8,33.0]];
        let expected_arr = array![[23.2_f32.log10(),35.4_f32.log10(),34.2_f32.log10(),3.4_f32.log10()],
                               [5.2_f32.log10(),4.7_f32.log10(),5.8_f32.log10(),33.0_f32.log10()]];
        let res_arr = input_arr.log10().unwrap();
        assert_eq!(&expected_arr, &res_arr);
    }

    #[test]
    fn ln_1p_test() {
        let input_arr = array![[12.3,45.0,23.0,3.09],
                               [2.30,1.4,8.3,0.43]];
        let expected_arr = array![[12.3_f32.ln_1p(),45.0_f32.ln_1p(),23.0_f32.ln_1p(),3.09_f32.ln_1p()],
                               [2.30_f32.ln_1p(),1.4_f32.ln_1p(),8.3_f32.ln_1p(),0.43_f32.ln_1p()]];
        let res_arr = input_arr.ln_1p().unwrap();
        assert_eq!(&expected_arr,&res_arr);
    }



}
