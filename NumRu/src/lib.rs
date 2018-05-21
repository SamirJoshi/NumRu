#[macro_use]
extern crate ndarray;
extern crate num_traits;
use ndarray::*;

pub fn test_function() {
    println!("hello");
}

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

pub fn amax<A, D>(arr: &Array<A, D>) -> A 
    where D: Dimension,
      A: std::fmt::Debug + std::cmp::Ord +  std::marker::Copy,
{
    let mut arr_item = arr.iter();
    let mut max_elem = arr_item.next().unwrap();
    while let Some(curr_item) = arr_item.next() {
        if curr_item > max_elem {
            max_elem = curr_item;
        }
    }
    (*max_elem).clone()
}

/// Computes element-wise sin on an ndarray Array
pub fn sin<A, D>(arr: &Array<A, D>) -> Array<A, D> 
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{   
    //TODO: change to actually handle the error
    let sin_arr = Array::from_iter(arr.iter().map(|x| x.sin()));
    sin_arr.into_shape(arr.raw_dim()).unwrap()
}

#[cfg(test)]
mod amin_tests {
    use super::amin;

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

#[cfg(test)]
mod trig_tests {
    use super::{sin};
    use std;

    #[test]
    fn sin_tests() {
        let pi = std::f64::consts::PI;
        let input_arr = array![pi, pi / 2.0];
        let expected_arr = array![0.0, 1.0];
        let mut expected_iter = expected_arr.iter();
        let res_arr = sin(&input_arr);
        let mut res_iter = res_arr.iter();

        while let Some(r) = res_iter.next() {
            let exp = expected_iter.next().unwrap();
            assert!((r - exp) < 0.000001);
        }
    }
}


//#[cfg(test)]
//mod fast_matrix_multiply {


    //#[test]
    //fn two_by_two_test() {
        //let arr1 = array!([[1, 2], [3, 4]]);
        //let arr2 = array!([[5, 6], [7, 8]]);

        //assert_eq!(arr1 * arr2, array![[19, 22], [43, 50]]);
    //}
//}
