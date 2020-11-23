import tensorflow as tf 
import numpy as np 
from tensorflow.keras import Model
from preprocessing import get_vec

def train(model, train_inputs, emotion_labels, sentiment_labels):
    train_inputs = np.array(train_inputs)
    emotion_labels = np.array(emotion_labels)
    sentiment_labels = np.array(sentiment_labels)


    random_indices = tf.random.shuffle(tf.range(train_inputs.shape[0]))
    train_inputs = tf.gather(train_inputs, random_indices)
    emotion_labels = tf.gather(emotion_labels, random_indices)
    sentiment_labels = tf.gather(sentiment_labels, random_indices)

    for batch_num in range(int(train_inputs.shape[0]/model.batch_size)):
    	batch_inputs, batch_emotion_labels, batch_sentiment_labels = get_batch(train_inputs, 
    		emotion_labels, sentiment_labels, batch_num, model.batch_size)

    	with tf.GradientTape() as tape:
    		emotion_logits, sentiment_logits = model.call(batch_inputs, None)
    		emotion_batch_loss = model.loss_function(emotion_logits, batch_emotion_labels)
    		sentiment_batch_loss = model.loss_function(sentiment_logits, batch_sentiment_labels)
    		batch_loss = emotion_batch_loss + sentiment_batch_loss

    	gradients = tape.gradient(batch_loss, model.trainable_parameters)
    	model.optimizer.apply_gradients(zip(gradients, model.trainable_parameters))

def test(model, test_inputs, emotion_labels, sentiment_labels):

	test_inputs = np.array(test_inputs)
	emotion_labels = np.array(emotion_labels)
	sentiment_labels = np.array(sentiment_labels)

	loss = tf.Variable([0], dtype=tf.float32)

	for batch_num in range(int(train_inputs.shape[0]/model.batch_size)):
		batch_inputs, batch_emotion_labels, batch_sentiment_labels = get_batch(test_inputs,
			emotion_labels, sentiment_labels, batch_num, model.batch_size)

			emotion_logits, sentiment_logits = model.call(batch_inputs, None)
			emotion_batch_loss = model.loss_function(emotion_logits, batch_emotion_labels)
			sentiment_batch_loss = model.loss_function(sentiment_logits, batch_sentiment_labels)

			loss = loss + batch_loss

	'''
	Calculate F1 score for emotion and sentiment, probably using tfa.metrics.F1Score
	and then print each

	Maybe make use of some visualization tool?

	'''

def main():
