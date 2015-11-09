import numpy as np
from matplotlib import pyplot


data = np.array([[2, 1.0], [3, 1.0], [4, 0.981], [5, 0.981], [6, 0.981],
                [7, 0.9630], [8, 0.9630], [9, 0.9630], [10, 0.963],
                [11, 0.9630]])
pyplot.scatter(data[:, 0], data[:, 1])
pyplot.show()
