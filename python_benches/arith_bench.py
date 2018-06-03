import numpy as np
import timeit

def negative_bench_mid():
    input_arr = np.full((50, 50, 50), 1.0)
    expected_arr = np.full((50, 50, 50), -1.0)
    assert np.array_equal(np.negative(input_arr), expected_arr)
    return

def negative_bench_large():
    input_arr = np.full((50, 50, 50, 50), 1.0)
    expected_arr = np.full((50, 50, 50, 50), -1.0)
    assert np.array_equal(np.negative(input_arr), expected_arr)
    return

def sum_bench_mid():
    input_arr = np.full((50, 50, 50), 1.0)
    assert np.sum(input_arr) == 125000.0
    return

def prod_bench_mid():
    input_arr = np.full((50, 50, 50), 1.0)
    assert np.prod(input_arr) == 1.0
    return

def prod_bench_large():
    input_arr = np.full((50, 50, 50, 50), 1.0)
    assert np.prod(input_arr) == 1.0
    return

def bench(function_name):
    setup = "from __main__ import " + function_name
    res_str = "test: "
    res_str += '{:<15}'.format(function_name) + "  ... bench: "
    res_str += '{:^,}'.format(timeit.timeit(function_name + "()", setup=setup, number=500) / 500 * 1000000000)
    res_str += " ns/iter"

    print res_str

def bench_arith():
    bench("negative_bench_mid")
    bench("negative_bench_large")
    bench("sum_bench_mid")
    bench("prod_bench_mid")
    bench("prod_bench_large")

if __name__ == "__main__":
    print "Arithmetic Python NumPy Benchmarks"
    bench_arith()
    print "\n"
 
