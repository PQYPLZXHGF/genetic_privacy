import timeit
from random import sample

import numpy as np

from recomb_helper import new_sequence
from diploid import Diploid

possible_starts = tuple(range(10000000))
starts = np.array(sorted(sample(possible_starts, 400)), dtype = np.uint32)
founders = np.array(list(range(400)), dtype = np.uint32)

# locations = list(sample(possible_starts, 20))
locations = np.array(list(sample(possible_starts, 20)), dtype = np.uint32)

diploid = Diploid(starts, 10000000, founders)

def speed_new_sequence():
    x = new_sequence(diploid, locations)


print("Speed of new_sequence:\t{}".format(timeit.timeit(speed_new_sequence)))
