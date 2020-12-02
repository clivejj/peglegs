import pickle
import numpy as np

a = pickle.load(open("data/data.pickle", "rb"))

# print shape of each sentence embedding
for s in a[1]:
    print(np.shape(a[2][s]))

