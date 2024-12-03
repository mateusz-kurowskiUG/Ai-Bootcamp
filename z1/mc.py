import numpy as np
from random import randint

a, b, r = 1, 1, 1


def get_random_tuple():
    return randint(-1, 1), randint(-1, 1)


def gen_sets(n: int):
    return [get_random_tuple() for i in range(n)]


def mc_iter(x, y):
    in_circle = 0
    np.sum
    sets = np.dstack((x, y))
    for pair in sets:
        for x, y in pair:
            if (x**2 + y**2) ** 0.5:
                in_circle += 1
        return 4 * in_circle / sets.__len__()


x_set = np.random.uniform(-1, 1)
y_set = np.random.uniform(-1, 1)
print(x_set)
pi = mc_iter(x_set, y_set)
print(pi)
