#![doc(html_root_url = "https://samirjoshi.github.io/NumRu/")]

//! A native Rust implementation of math and stats functions from NumPy
//! built on top of ndarray (https://docs.rs/ndarray/0.11.2/ndarray/)
//!
//! ### This crate is still under development
//!
//! # Notes
//! - Functions are implemented for both Array and ArcArray ndarrays
//! - We recommend using ArcArray to benefit from performance gains
//! - ArcArray implementations are parallelized using ndarray_parallel and rayon
//!

#[macro_use]
extern crate ndarray;
extern crate num_traits;
extern crate chrono;
extern crate rayon;
extern crate ndarray_parallel;

extern crate error_chain;

pub mod math;
pub mod stats;
pub mod test;
