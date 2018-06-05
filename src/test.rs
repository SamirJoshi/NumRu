use ndarray::*;


pub fn compare_arrays<D>(expected_arr: &Array<f64, D>, res_arr: &Array<f64, D>) -> bool
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
