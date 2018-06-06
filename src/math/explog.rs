use ndarray::*;
use std;
use num_traits::{real::Real};


pub fn exp<A, D>(arr: &Array<A, D>) -> Result<Array<A,D>,ShapeError>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + Real,
{
    let res_arr = Array::from_iter(arr.into_iter().map(|x| x.exp()));
    res_arr.into_shape(arr.raw_dim())
}

pub fn expm1<A,D>(arr: &Array<A, D>) -> Result<Array<A,D>,ShapeError>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + Real,
  {
      let res_arr = Array::from_iter(arr.into_iter().map(|x| x.exp()-A::one()));
      res_arr.into_shape(arr.raw_dim())
  }

pub fn exp2<A,D>(arr: &Array<A, D>) -> Result<Array<A,D>,ShapeError>
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + Real,
      {
          let res_arr = Array::from_iter(arr.into_iter().map(|x| x.exp2()));
          res_arr.into_shape(arr.raw_dim())
      }

pub fn log<A,D>(arr: &Array<A, D>) -> Result<Array<A,D>,ShapeError>
  where D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + Real,
    {
        let res_arr = Array::from_iter(arr.into_iter().map(|x| x.ln()));
        res_arr.into_shape(arr.raw_dim())
    }

pub fn log2<A,D>(arr: &Array<A, D>) -> Result<Array<A,D>,ShapeError>
  where D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + Real,
    {
        let res_arr = Array::from_iter(arr.into_iter().map(|x| x.log2()));
        res_arr.into_shape(arr.raw_dim())
    }

pub fn log10<A,D>(arr: &Array<A, D>) -> Result<Array<A,D>,ShapeError>
  where D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + Real,
    {
        let res_arr = Array::from_iter(arr.into_iter().map(|x| x.log10()));
        res_arr.into_shape(arr.raw_dim())
    }

pub fn log1p<A,D>(arr: &Array<A, D>) -> Result<Array<A,D>,ShapeError>
  where D: Dimension,
    A: std::fmt::Debug + std::marker::Copy + Real + std::ops::Add,
    {
        let res_arr = Array::from_iter(arr.into_iter().map(|x| (*x+A::one()).ln()));
        res_arr.into_shape(arr.raw_dim())
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

    #[test]
    fn exp_test() {
        let input_arr = array![0.0, 1.0, 5.0, 0.0];
        let expected_arr = array![1.0, 1.0.exp(), 5.0.exp(), 1.0];
        let res_arr =exp(&input_arr).unwrap();
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn exp_test_3_dim() {
        let input_arr = array![[[2.2,1.4],[1.3,1.7]],
                            [[13.2,1.8],[12.3,17.4]]];
        let expected_arr = array![[[2.2.exp(),1.4.exp()],[1.3.exp(),1.7.exp()]],
                            [[13.2.exp(),1.8.exp()],[12.3.exp(),17.4.exp()]]];
        let res_arr = exp(&input_arr).unwrap();
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn expm1_test() {
        let input_arr = array![[23.2,35.4,34.2,3.4],
                               [5.2,4.7,5.8,33.0]];
        let expected_arr = array![[23.2.exp()-1.0,35.4.exp()-1.0,34.2.exp()-1.0,3.4.exp()-1.0],
                               [5.2.exp()-1.0,4.7.exp()-1.0,5.8.exp()-1.0,33.0.exp()-1.0]];
        let res_arr = expm1(&input_arr).unwrap();
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn exp2_test() {
        let input_arr = array![[23.2,35.4,34.2,3.4],
                               [5.2,4.7,5.8,33.0]];
        let expected_arr = array![[23.2.exp2(),35.4.exp2(),34.2.exp2(),3.4.exp2()],
                               [5.2.exp2(),4.7.exp2(),5.8.exp2(),33.0.exp2()]];
        let res_arr = exp2(&input_arr).unwrap();
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn log_test() {
        let input_arr = array![[23.2,35.4,34.2,3.4],
                               [5.2,4.7,5.8,33.0]];
        let expected_arr = array![[23.2.ln(),35.4.ln(),34.2.ln(),3.4.ln()],
                               [5.2.ln(),4.7.ln(),5.8.ln(),33.0.ln()]];
        let res_arr = log(&input_arr).unwrap();
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn log2_test() {
        let input_arr = array![[23.2,35.4,34.2,3.4],
                               [5.2,4.7,5.8,33.0]];
        let expected_arr = array![[23.2.log2(),35.4.log2(),34.2.log2(),3.4.log2()],
                               [5.2.log2(),4.7.log2(),5.8.log2(),33.0.log2()]];
        let res_arr = log2(&input_arr).unwrap();
        assert!(compare_arrays(&expected_arr, &res_arr));
    }

    #[test]
    fn log10_test() {
        let input_arr = array![[23.2,35.4,34.2,3.4],
                               [5.2,4.7,5.8,33.0]];
        let expected_arr = array![[23.2.log10(),35.4.log10(),34.2.log10(),3.4.log10()],
                               [5.2.log10(),4.7.log10(),5.8.log10(),33.0.log10()]];
        let res_arr = log10(&input_arr).unwrap();
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
