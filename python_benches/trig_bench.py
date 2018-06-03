import numpy as np
import timeit
import math

def sin_bench_mid():
    input_arr = np.full((50, 50, 50), math.pi / 2.0)
    expected_arr = np.full((50, 50, 50), 1.0)
    assert np.array_equal(np.sin(input_arr), expected_arr)
    return

def cos_bench_mid():
    input_arr = np.full((50, 50, 50), 0.0)
    expected_arr = np.full((50, 50, 50), 1.0)
    assert np.array_equal(np.cos(input_arr), expected_arr)
    return

def arctan_bench_mid():
    input_arr = np.full((50, 50, 50), 1.0)
    expected_arr = np.full((50, 50, 50), math.pi / 4.0)
    assert np.array_equal(np.arctan(input_arr), expected_arr)
    return

def bench(function_name):
    setup = "from __main__ import " + function_name
    res_str = "test: "
    res_str += '{:<15}'.format(function_name) + "  ... bench: " 
    res_str += '{:^,}'.format(timeit.timeit(function_name + "()", setup=setup, number=500) / 500 * 1000000000)
    res_str += " ns"

    print res_str

def bench_trig():
    bench("sin_bench_mid")
    bench("cos_bench_mid")
    bench("arctan_bench_mid")

if __name__ == "__main__":
    print "Trig Python NumPy Benchmarks"
    bench_trig()
    print "\n"
