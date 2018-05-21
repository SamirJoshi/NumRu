#[macro_use]
extern crate ndarray;
extern crate num_traits;
use ndarray::*;


pub mod math;

pub fn test_function() {
    println!("hello");
}

/// Retrieves the min element from an ndarray Array
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

//#[cfg(test)]
//mod fast_matrix_multiply {


    //#[test]
    //fn two_by_two_test() {
        //let arr1 = array!([[1, 2], [3, 4]]);
        //let arr2 = array!([[5, 6], [7, 8]]);

        //assert_eq!(arr1 * arr2, array![[19, 22], [43, 50]]);
    //}
//}
