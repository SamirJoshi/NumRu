use ndarray::*;
use std;
use num_traits::{real::Real};


pub fn exp<A, D>(arr: &Array<A, D>) -> Array<A,D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + Real,
{
    let res_arr = Array::from_iter(arr.into_iter().map(|x| x.exp()));
    res_arr.into_shape(arr.raw_dim()).unwrap()
}

pub fn expm1<A,D>(arr: &Array<A, D>) -> Array<A,D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + Real,
  {
      let res_arr = Array::from_iter(arr.into_iter().map(|x| x.exp()-A::one()));
      res_arr.into_shape(arr.raw_dim()).unwrap()
  }

pub fn exp2<A,D>(arr: &Array<A, D>) -> Array<A,D>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + Real,
      {
          let res_arr = Array::from_iter(arr.into_iter().map(|x| x.exp2()));
          res_arr.into_shape(arr.raw_dim()).unwrap()
      }

pub fn log<A,D>(arr: &Array<A, D>) -> Array<A,D>
  where D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + Real,
    {
        let res_arr = Array::from_iter(arr.into_iter().map(|x| x.ln()));
        res_arr.into_shape(arr.raw_dim()).unwrap()
    }

pub fn log2<A,D>(arr: &Array<A, D>) -> Array<A,D>
  where D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + Real,
    {
        let res_arr = Array::from_iter(arr.into_iter().map(|x| x.log2()));
        res_arr.into_shape(arr.raw_dim()).unwrap()
    }

pub fn log10<A,D>(arr: &Array<A, D>) -> Array<A,D>
  where D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + Real,
    {
        let res_arr = Array::from_iter(arr.into_iter().map(|x| x.log10()));
        res_arr.into_shape(arr.raw_dim()).unwrap()
    }

pub fn log1p<A,D>(arr: &Array<A, D>) -> Array<A,D>
  where D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + Real + std::ops::Add,
    {
        let res_arr = Array::from_iter(arr.into_iter().map(|x| (*x+A::one()).ln()));
        res_arr.into_shape(arr.raw_dim()).unwrap()
    }

pub fn logaddexp<A>(x1: A, x2: A) -> A
  where A: std::fmt::Debug + std::marker::Copy + Real,
  {
      (x1.exp() + x2.exp()).ln()
  }

pub fn logaddexp2<A>(x1: A, x2: A) -> A
    where A: std::fmt::Debug + std::marker::Copy + Real,
    {
        (x1.exp2() + x2.exp2()).log2()
    }


#[cfg(test)]
mod tests {
    use math::explog::*;
    use ndarray::*;
    use num_traits;
    use std;

    #[test]
    fn exp_test() {
        let input_arr = array![0.0, 1.0, 5.0, 0.0];
        let expected_arr = array![1.0, 1.0.exp(), 5.0.exp(), 1.0];
        let res_arr =exp(&input_arr);
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn exp_test_3_dim() {
        let input_arr = array![[[2.2,1.4],[1.3,1.7]],
                            [[13.2,1.8],[12.3,17.4]]];
        let expected_arr = array![[[2.2.exp(),1.4.exp()],[1.3.exp(),1.7.exp()]],
                            [[13.2.exp(),1.8.exp()],[12.3.exp(),17.4.exp()]]];
        let res_arr = exp(&input_arr);
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn expm1_test() {

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
