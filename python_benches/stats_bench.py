import numpy as np
import timeit

def arr_max_bench_1d():
    arr = np.array([5, 3, 5, 2, 1])
    return np.amax(arr)

def arr_max_mid():
    arr = np.zeros((50, 50, 50))
    arr[25][25][25] = 15
    assert np.amax(arr) == 15
    return

def arr_max_large():
    arr = np.zeros((100, 100, 100, 100))
    arr[25][25][25][25]= 16
    assert np.amax(arr) == 16
    return

def arr_min_mid():
    arr = np.zeros((50, 50, 50))
    arr[25][25][25] = -1
    assert np.amin(arr) == -1
    return

def arr_min_large():
    arr = np.zeros((100, 100, 100, 100))
    arr[25][25][25][25] = -1
    assert np.amin(arr) == -1
    return

def mean_mid():
    arr = np.zeros((50, 50, 50))
    arr[25][25][25] = 125000
    assert np.mean(arr) == 1
    return

def mean_large():
    arr = np.zeros((100, 100, 100, 100))
    arr[25][25][25][25] = 100000000
    assert np.mean(arr) == 1
    return

def range_mid():
    arr = np.zeros((50, 50, 50))
    arr[25][25][25] = 5
    arr[26][26][26] = -5
    assert np.ptp(arr) == 10
    return

def variance_mid():
    arr = np.zeros((50, 50, 50))
    arr[25][25][25] = 0
    assert np.var(arr) == 0
    return

def bench(function_name):
    setup = "from __main__ import " + function_name
    res_str = "test: "
    res_str += '{:<15}'.format(function_name) + "  ... bench: "
    res_str += '{:^,}'.format(timeit.timeit(function_name + "()", setup=setup, number=500) / 500 * 1000000000)
    res_str += " ns/iter"

    print res_str

def bench_stats():
    bench("arr_max_bench_1d")
    bench("arr_max_mid")
    bench("arr_max_large")
    bench("arr_min_mid")
    bench("arr_min_large")
    bench("mean_mid")
    bench("mean_large")
    bench("range_mid")
    bench("variance_mid")

if __name__ == "__main__":
    bench_stats()
