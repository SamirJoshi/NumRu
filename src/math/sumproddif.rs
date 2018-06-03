use ndarray::*;
use std;
use num_traits;

pub fn prod<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + std::ops::Mul + num_traits::identities::One,
{
    arr.iter().fold(A::one(), |acc, x| acc * *x)
}

pub fn sum<A, D>(arr: &Array<A, D>) -> A
    where D: Dimension,
      A: std::fmt::Debug + std::marker::Copy + std::ops::Add + num_traits::identities::Zero,
{
    arr.iter().fold(A::zero(), |acc, x| acc + *x)
}




#[cfg(test)]
mod sumproddif_tests {
    use math::sumproddif::*;
    use ndarray::*;
    use num_traits;
    use std;

    #[test]
    fn prod_test() {
        assert_eq!(prod(&array![1.0,2.0,3.0]),6.0);
    }

    #[test]
    fn prod_test_int() {
        assert_eq!(prod(&array![1,2,3]),6);
    }

    #[test]
    fn sum_test() {
        assert_eq!(sum(&array![1.0,2.0,3.0,4.0]),10.0);
    }

    #[test]
    fn sum_test_int() {
        assert_eq!(sum(&array![1,2,3,4]),10);
    }
}
