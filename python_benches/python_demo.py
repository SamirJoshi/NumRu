import numpy as np
import time

def runner():
    for i in range(0, 5):
        stats_example()
        math_example()
    return 0

def stats_example():
    arr = np.random.rand(100000, 50)
    # basic statistics
    min_elem = np.amin(arr)
    max_elem = np.amax(arr)
    range = np.ptp(arr)
    print ("Minimum element: ", min_elem)
    print ("Maximum element: ", max_elem)
    print ("Range: ", range)

    # more statistics
    avg = np.mean(arr)
    arr_std = np.std(arr)
    arr_sum = np.sum(arr)
    arr_prod = np.prod(arr)
    print ("Mean: ", avg)
    print ("Standard Deviation: ", arr_std)
    print ("Sum: ", arr_sum)
    print ("Product: ", arr_prod)
    return

def math_example():
    #math
    math_arr = np.random.rand(100000, 50) - 0.5
    math_arr_neg = np.negative(math_arr)
    math_arr = np.positive(math_arr)

    sin_arr = np.sin(math_arr)
    cos_arr = np.cos(math_arr)
    tan_arr = np.tan(math_arr)
    same_math_arr = np.arctan(tan_arr)

    print ("\n")
    return

runner()