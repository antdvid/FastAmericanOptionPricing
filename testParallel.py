from multiprocessing import Pool
import time


def trial_func(x):
    return x * x

if __name__ == '__main__':
    pool = Pool(4)
    N = 50000000
    x = range(N)
    start = time.clock()
    res = pool.map(trial_func, x)
    print("total time cost = ", time.clock() - start)