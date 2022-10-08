import time

def measure_time(fn, np_timer):
    """ Measures the time, which fn needs to be executed and returns it"""
    start = time.time()
    res = fn()
    end = time.time()
    np_timer += end - start
    return res