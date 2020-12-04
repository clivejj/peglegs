import pickle
import numpy as np
import tensorflow as tf

h = np.zeros((4, 1))
t = np.ones((4, 1))

b = tf.concat([tf.transpose(h), tf.transpose], 0)

print(b)
