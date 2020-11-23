import tensorflow as tf 
import numpy as np 
from tensorflow.keras import Model
from preprocessing import get_vec

def main():
	word_2_vec = get_vec()
	print("Success!")
