import numpy as np
import timeit

def arr_max_1d():
    arr = np.array([5, 3, 5, 2, 1])
    return np.amax(arr)

def arr_max():
    arr = np.array([[5, 3, 5], [2, 1, 8]])
    return np.amax(arr)

def arr_max_mid():
    arr = np.zeros((50, 50, 50))
    arr[25][25][25] = 15
    assert np.amax(arr) == 15
    return

def arr_max_large():
    arr = np.zeros((100, 100, 100, 100))
    arr[25][25][25][25] = 15
    assert np.amax(arr) == 15
    return

def arr_min_small():
    arr = np.array([[5, 3, 5], [2, 1, 8]])
    assert np.amax(arr) == 1
    return

def arr_min_large():
    arr = np.rand((100, 100, 100, 100))
    arr[25][25][25][25] = -1
    assert np.amax(arr) == -1
    return

def bench(function_name):
    setup = "from __main__ import " + function_name
    res_str = "test: "
    res_str += '{:<15}'.format(function_name) + "  ... bench: " 
    res_str += '{:^,}'.format(timeit.timeit(function_name, setup=setup) * 1000000000)
    res_str += " ns"

    print res_str

if __name__ == "__main__":
    bench("arr_max_1d")
    bench("arr_max")
    bench("arr_max_mid")
    bench("arr_max_large")
    bench("arr_min_small")
    bench("arr_min_large")
