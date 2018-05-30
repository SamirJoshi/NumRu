import numpy as np
import timeit

def arr_trig():
    arr = np.array([5, 3, 5, 2, 1])
    return np.amax(arr)

def arr_inverse_trig():
    sum = 0
    for i in range(0, 10000):
        sum = i

    assert sum == 9999
    return


def trig_back_forth():
    sum = 0
    for i in range(0, 100):
        sum = i

    assert sum == 99
    return

def bench(function_name):
    setup = "from __main__ import " + function_name
    res_str = "test: "
    res_str += '{:<15}'.format(function_name) + "  ... bench: " 
    res_str += '{:^,}'.format(timeit.timeit(function_name + "()", setup=setup) * 1000000000)
    res_str += " ns"

    print res_str

def bench_trig():
    bench("arr_trig")
    bench("arr_inverse_trig")
    bench("trig_back_forth")

if __name__ == "__main__":
    bench_trig()
