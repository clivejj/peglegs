import tensorflow as tf 
import numpy as np 
from tensorflow.keras import Model
from nltk.corpus import wordnet as wn
from preprocessing import get_vec

class Model(tf.keras.Model):
	def __init__(self):
		super(Model, self).__init__()

		#HyperParamaters 
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
		self.embedding_size = 300
		self.h1 = 300
		self.batch_size = 64
		#Trainable parameters
		self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = self.h1, 
			return_sequences = True, return_state = True))
		self.Embedding = get_vec()
		







