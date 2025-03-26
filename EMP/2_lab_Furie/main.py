import numpy as np


def calculateCoefs(k):
    return (-1) ** (k + 1) * 32 / (np.pi * k) + (-1) ** k * 96 / ((np.pi * k) ** 3) + (-1) ** (k ) * 32 / (
                np.pi * k) + (128 * (k  % 2)) / ((k * np.pi) ** 3)


print(calculateCoefs(1))
print(calculateCoefs(2))
print(calculateCoefs(3))
print(calculateCoefs(4))
print(calculateCoefs(5))