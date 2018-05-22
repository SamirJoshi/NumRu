use ndarray::*;
use std;
use num_traits;


/// Computes element-wise sine on an ndarray Array
pub fn sin<A, D>(arr: &Array<A, D>) -> Array<A, D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    //TODO: change to actually handle the error
    let sin_arr = Array::from_iter(arr.iter().map(|x| x.sin()));
    sin_arr.into_shape(arr.raw_dim()).unwrap()
}

/// Computes element-wise cosine on an ndarray Array
pub fn cos<A, D>(arr: &Array<A, D>) -> Array<A, D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    //TODO: change to actually handle the error
    let res_arr = Array::from_iter(arr.iter().map(|x| x.cos()));
    res_arr.into_shape(arr.raw_dim()).unwrap()
}

/// Computes element-wise tangent on an ndarray Array
pub fn tan<A, D>(arr: &Array<A, D>) -> Array<A, D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    //TODO: change to actually handle the error
    let res_arr = Array::from_iter(arr.iter().map(|x| {
        x.tan()
    }));
    res_arr.into_shape(arr.raw_dim()).unwrap()
}

/// Computes element-wise arc sine on an ndarray Array
pub fn asin<A, D>(arr: &Array<A, D>) -> Array<A, D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    //TODO: change to actually handle the error
    let res_arr = Array::from_iter(arr.iter().map(|x| x.asin()));
    res_arr.into_shape(arr.raw_dim()).unwrap()
}

/// Computes element-wise arc cosine on an ndarray Array
pub fn acos<A, D>(arr: &Array<A, D>) -> Array<A, D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    //TODO: change to actually handle the error
    let res_arr = Array::from_iter(arr.iter().map(|x| x.acos()));
    res_arr.into_shape(arr.raw_dim()).unwrap()
}

/// Computes element-wise arc tangent on an ndarray Array
pub fn atan<A, D>(arr: &Array<A, D>) -> Array<A, D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    //TODO: change to actually handle the error
    let res_arr = Array::from_iter(arr.iter().map(|x| x.atan()));
    res_arr.into_shape(arr.raw_dim()).unwrap()
}

#[cfg(test)]
mod trig_tests {
    use ndarray::*;
    use num_traits;
    use std;
    use super::{sin, cos, tan};

    const TAN_INF : f64 = 16331239353195370.0;

    #[test]
    fn sin_tests() {
        let pi = std::f64::consts::PI;
        let input_arr = array![pi, pi / 2.0];
        let expected_arr = array![0.0, 1.0];
        let res_arr = sin(&input_arr);
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn cos_tests() {
        let pi = std::f64::consts::PI;
        let input_arr = array![pi, pi / 2.0, 0.0];
        let expected_arr = array![-1.0, 0.0, 1.0];
        let res_arr = cos(&input_arr);
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn tan_tests() {
        let pi = std::f64::consts::PI;
        let input_arr = array![0.0, pi / 4.0, pi / 2.0, pi];
        let expected_arr = array![0.0, 1.0, TAN_INF, 0.0];
        let res_arr = tan(&input_arr);
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    fn compare_arrays<D>(expected_arr: &Array<f64, D>, res_arr: &Array<f64, D>) -> bool
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
}
