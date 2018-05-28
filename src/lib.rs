#[macro_use]
extern crate ndarray;
extern crate num_traits;
extern crate chrono;
extern crate crossbeam;

#[macro_use]
extern crate error_chain;

pub mod math;
pub mod stats;

pub fn test_function() {
    println!("hello");
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
