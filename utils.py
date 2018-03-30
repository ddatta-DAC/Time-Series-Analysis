import matplotlib.pyplot as plt
from itertools import tee, izip
import numpy as np

# ---------------------------- #

def plot(x,y):
    # plt.xticks(range(min(x), max(x) + 1, 1000))
    plt.plot(x, y, 'r-.')
    plt.show()


def get_windowed_data(data, window_size):
    print ' window_size' , window_size
    def window(iterable, size):
        iters = tee(iterable, size)
        for i in xrange(1, size):
            for each in iters[i:]:
                next(each, None)
        return izip(*iters)

    op = []
    for w in window(data, window_size):
        w = np.reshape(w, [-1])

        op.append(w)

    op = np.asarray(op)
    return op