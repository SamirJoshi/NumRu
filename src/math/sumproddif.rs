use ndarray::*;
use std;
use num_traits;

pub fn prod<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    arr.iter().fold(num_traits::one(), |acc, x| acc * *x)
}

pub fn sum<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + num_traits::real::Real,
{
    arr.iter().fold(num_traits::zero(), |acc, x| acc + *x)
}



#[cfg(test)]
mod tests {
    use math::sumproddif::*;
    use ndarray::*;
    use num_traits;
    use std;

    #[test]
    fn prod_test() {
        assert_eq!(prod(&array![1.0,2.0,3.0]),6.0);
    }

    #[test]
    fn sum_test() {
        assert_eq!(sum(&array![1.0,2.0,3.0,4.0]),10.0);
    }
}
