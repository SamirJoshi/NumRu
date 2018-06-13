use ndarray::*;
use ndarray_parallel::prelude::*;
use num_traits;
use std;


pub trait NumRuSPD {
    type Elt: std::fmt::Debug + std::marker::Copy + 
    std::ops::Add + std::ops::Div + std::ops::Mul + std::ops::Sub;

    fn prod(&self) -> Self::Elt;
    fn sum(&self) -> Self::Elt;
    fn cumsum(&self) -> Array<Self::Elt, Dim<[usize; 1]>>;
    fn cumprod(&self) -> Array<Self::Elt, Dim<[usize;1]>>;
    fn ediff1d(&self) -> Array<Self::Elt, Dim<[usize;1]>>;
}
impl<A: std::fmt::Debug + std::marker::Copy + num_traits::identities::Zero + num_traits::identities::One + 
    std::ops::Add<Output=A> + std::ops::Div<Output=A> + std::ops::Mul<Output=A> + std::ops::Sub<Output=A>, D: Dimension> NumRuSPD
    for Array<A, D> {
    type Elt = A;

    /// Returns the product of an ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::sumproddif::*;
    /// # fn main(){
    ///     let arr = array![[[5.0, 6.0], [7.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]];
    ///     assert_eq!(arr.prod(), 5040.0);
    /// # }
    /// ```
    ///
    fn prod(&self) -> Self::Elt
    {
        self.iter().fold(A::one(), |acc, x| acc * *x)
    }

    /// Returns the sum across an ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::sumproddif::*;
    /// # fn main(){
    ///     let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]];
    ///     assert_eq!(arr.sum(), 28.0);
    /// # }
    /// ```
    ///
    fn sum(&self) -> Self::Elt
    {
        self.iter().fold(A::zero(), |acc, x| acc + *x)
    }

    /// Returns the array that cumulatively sums across ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::sumproddif::*;
    /// # fn main(){
    ///     let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]];
    ///     let res_arr = array![5.0, 11.0, 18.0, 18.0, 19.0, 21.0, 24.0, 28.0];
    ///     assert_eq!(arr.cumsum(), res_arr);
    /// # }
    /// ```
    ///
    fn cumsum(&self) -> Array<Self::Elt, Dim<[usize; 1]>>
    {
        let mut flat = self.iter();
        let mut prev = A::zero();
        let mut p = vec![];
        while let Some(a) = flat.next() {
            p.push(*a + prev);
            prev = *a + prev;
        }
        Array::from_vec(p)
    }

    /// Returns the array that cumulatively multiplies across ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::sumproddif::*;
    /// # fn main(){
    ///     let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]];
    ///     let res_arr = array![5.0, 30.0, 210.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    ///     assert_eq!(arr.cumprod(), res_arr);
    /// # }
    /// ```
    fn cumprod(&self) -> Array<Self::Elt, Dim<[usize;1]>>
    {
        let mut flat = self.iter();
        let mut prev = A::one();
        let mut p = vec![];
        while let Some(a) = flat.next() {
            p.push(*a * prev);
            prev = *a * prev;
        }
        Array::from_vec(p)
    }

    /// Returns a 1d array of the difference between each consecutive
    /// element in the array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::sumproddif::*;
    /// # fn main(){
    ///     let arr = array![ 1.2 , 42.  ,  1.9 ,  3.67,  0.54,  9.4 ,  2.  ];
    ///     let res_arr = array![ 40.8 , -40.1 ,   1.77,  -3.13,   8.86,  -7.4 ];
    ///     assert_eq!(arr.ediff1d(), res_arr);
    /// # }
    /// ```
    fn ediff1d(&self) -> Array<Self::Elt, Dim<[usize;1]>>
    {
        let mut flat = self.iter();
        let mut p = vec![];

        let first = flat.next();
        match first {
            Some(mut prev) => {
                while let Some(curr) = flat.next() {
                    p.push(*curr - *prev);
                    prev = curr;
                }
            },
            None => (),
        }
        Array::from_vec(p)
    }
}

impl<A: std::fmt::Debug + std::marker::Copy + num_traits::identities::Zero + num_traits::identities::One + 
    std::marker::Sync + std::marker::Send +
    std::ops::Add<Output=A> + std::ops::Div<Output=A> + std::ops::Mul<Output=A> + std::ops::Sub<Output=A>, D: Dimension> NumRuSPD
    for ArcArray<A, D> {
    type Elt = A;

    /// Returns the product of an ndarray Array
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
    ///     assert_eq!(arr.prod(), 5040.0);
    /// # }
    /// ```
    ///
    fn prod(&self) -> Self::Elt
    {
        let arr_sum = self.par_iter().cloned().reduce_with( |a, b| a * b);
        match arr_sum {
            Some(a) => a,
            None => panic!("Array of 0 elements")
        }
    }


    /// Returns the sum across an ndarray Array
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
    ///     assert_eq!(arr.sum(), 28.0);
    /// # }
    /// ```
    ///
    fn sum(&self) -> Self::Elt
    {
        let arr_sum = self.par_iter().cloned().reduce_with( |a, b| a + b);
        match arr_sum {
            Some(a) => a,
            None => panic!("Array of 0 elements")
        }
    }

    /// Returns the array that cumulatively sums across ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::sumproddif::*;
    /// # fn main(){
    ///     let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]];
    ///     let res_arr = array![5.0, 11.0, 18.0, 18.0, 19.0, 21.0, 24.0, 28.0];
    ///     assert_eq!(arr.cumsum(), res_arr);
    /// # }
    /// ```
    ///
    fn cumsum(&self) -> Array<Self::Elt, Dim<[usize; 1]>>
    {
        let mut flat = self.iter();
        let mut prev = A::zero();
        let mut p = vec![];
        while let Some(a) = flat.next() {
            p.push(*a + prev);
            prev = *a + prev;
        }
        Array::from_vec(p)
    }

    /// Returns the array that cumulatively multiplies across ndarray Array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::sumproddif::*;
    /// # fn main(){
    ///     let arr = array![[[5.0, 6.0], [7.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]]];
    ///     let res_arr = array![5.0, 30.0, 210.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    ///     assert_eq!(arr.cumprod(), res_arr);
    /// # }
    /// ```
    fn cumprod(&self) -> Array<Self::Elt, Dim<[usize;1]>>
    {
        let mut flat = self.iter();
        let mut prev = A::one();
        let mut p = vec![];
        while let Some(a) = flat.next() {
            p.push(*a * prev);
            prev = *a * prev;
        }
        Array::from_vec(p)
    }

    /// Returns a 1d array of the difference between each consecutive
    /// element in the array
    ///
    /// # Examples
    /// ```
    /// # #[macro_use]
    /// # extern crate ndarray;
    /// # extern crate num_ru;
    /// use ndarray::*;
    /// use num_ru::math::sumproddif::*;
    /// # fn main(){
    ///     let arr = array![ 1.2 , 42.  ,  1.9 ,  3.67,  0.54,  9.4 ,  2.  ];
    ///     let res_arr = array![ 40.8 , -40.1 ,   1.77,  -3.13,   8.86,  -7.4 ];
    ///     assert_eq!(arr.ediff1d(), res_arr);
    /// # }
    /// ```
    fn ediff1d(&self) -> Array<Self::Elt, Dim<[usize;1]>>
    {
        let mut flat = self.iter();
        let mut p = vec![];

        let first = flat.next();
        match first {
            Some(mut prev) => {
                while let Some(curr) = flat.next() {
                    p.push(*curr - *prev);
                    prev = curr;
                }
            },
            None => (),
        }
        Array::from_vec(p)
    }
}

#[cfg(test)]
mod sumproddif_tests {
    use math::sumproddif::*;

    #[test]
    fn prod_test() {
        assert_eq!(array![1.0,2.0,3.0].prod(),6.0);
    }

    #[test]
    fn prod_test_int() {
        assert_eq!(array![1,2,3].prod(),6);
    }

    #[test]
    fn sum_test() {
        assert_eq!(array![1.0,2.0,3.0,4.0].sum(),10.0);
    }

    #[test]
    fn sum_test_int() {
        assert_eq!(array![1,2,3,4].sum(),10);
    }

    #[test]
    fn cumsum_test_no_axis() {
        let input_arr = array![[1.0,2.0,3.0],
                               [4.0,5.0,6.0]];
        let res_arr = array![ 1.0,  3.0,  6.0, 10.0, 15.0, 21.0];
        assert_eq!(res_arr, input_arr.cumsum());
    }

    #[test]
    fn cumsum_test_integers() {
        let input_arr = array![[1,2,3],[4,5,6]];
        let res_arr = array![1,3,6,10,15,21];
        assert_eq!(res_arr, input_arr.cumsum());
    }

    #[test]
    fn ediff1d_test() {
        let input_arr = array![[1., 2., 3.],
                               [4., 5., 6.]];
        let res_arr = array![1., 1., 1., 1., 1.];
        assert_eq!(res_arr, input_arr.ediff1d());
    }

    #[test]
    fn ediff1d_test_different_difs() {
        let input_arr = array![1,3,6,10];
        let res_arr = array![2,3,4];
        assert_eq!(res_arr, input_arr.ediff1d());
    }

    #[test]
    fn ediff1d_test_empty() {
        let input_arr: Array<f64,Dim<[usize;1]>> = array![];
        let res_arr = array![];
        assert_eq!(res_arr, input_arr.ediff1d());
    }

    #[test]
    fn ediff1d_test_one() {
        let input_arr = array![1.0];
        let res_arr = array![];
        assert_eq!(res_arr, input_arr.ediff1d());
    }

    #[test]
    fn ediff1d_test_two() {
        let input_arr = array![1.0, 3.0];
        let res_arr = array![2.0];
        assert_eq!(res_arr, input_arr.ediff1d());
    }

    // fn cumsum_test_axis_0() {
    //     let input_arr = array![[1,2,3],[4,5,6]];
    //     let res_arr = array![[1,2,3],[5,7,9]];
    //
    // }

    // fn cumsum_test_axis_1() {
    //     let input_arr = array![[1,2,3],[4,5,6]];
    //     let res_arr = array![[1,3,6],[4,9,15]];
    // }

    // fn cumsum_test_3d_axis_2() {
    //     let input_arr = array![[[ 1,  2,  3],
    //     [ 4,  5,  6]],
    //    [[ 7,  8,  9],
    //     [10, 11, 12]]];
    //
    //     let res_arr = array![[[ 1,  3,  6],
    //     [ 4,  9, 15]],
    //    [[ 7, 15, 24],
    //     [10, 21, 33]]]
    // }

    // fn cumsum_test_3d_axis_1() {
    //     let input_arr = array![[[ 1,  2,  3],
    //         [ 4,  5,  6]],
    //        [[ 7,  8,  9],
    //         [10, 11, 12]]];
    //
    //     let res_arr = array![[[ 1,  2,  3],
    //     [ 5,  7,  9]],
    //    [[ 7,  8,  9],
    //     [17, 19, 21]]]
    // }

    #[test]
    fn cumprod_test() {
        let input_arr = array![[1.0,2.0,3.0],
                               [4.0,5.0,6.0]];
        let res_arr = array![  1.,   2.,   6.,  24., 120., 720.];

        assert_eq!(res_arr, input_arr.cumprod());
    }

    #[test]
    fn cumprod_empty_test() {
        let input_arr: Array<f64,Dim<[usize;1]>> = array![];
        let res_arr = array![];
        assert_eq!(res_arr, input_arr.cumprod());
    }
}
