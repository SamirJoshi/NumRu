#![doc(html_root_url = "https://samirjoshi.github.io/NumRu/")]

#[macro_use]
extern crate ndarray;
extern crate num_traits;
extern crate chrono;
extern crate crossbeam;
extern crate rayon;
extern crate ndarray_parallel;

#[macro_use]
extern crate error_chain;

pub mod math;
pub mod stats;

pub fn test_function() {
    println!("hello");
}
