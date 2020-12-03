import pickle
import numpy as np

na = np.zeros((10, 4))

q = [1, 2, 3, 4]

na[1, :] = np.asarray(q)

print(na[1])
